"""Phase 0 feasibility check: how much of each head's dense attention a foveated key set keeps, with no training.

Pretrained DINOv3 runs densely. Each layer's attention input also goes through FoveatedAttention for every budget
and start level, and per head we record:

    fine_mass    share of the dense attention on patch keys that lands on patches kept at full resolution
    prefix_mass  share of all dense attention that goes to the CLS and register tokens, which are always kept
    cosine       cosine similarity of foveated and dense head outputs, averaged over all queries

Example:
    python -m dinov3.foveation.phase0 --weights weights/dinov3-vitl16-pretrain-lvd1689m --images images/ --out phase0.csv
"""

import argparse
import itertools
import logging
from typing import Callable
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2

from dinov3.data.transforms import make_base_transform
from dinov3.foveation.attention import (
    FoveatedAttention,
    FoveationConfig,
    Pyramid,
    RopeGrid,
    patch_coords,
    rope_patches,
)
from dinov3.hub import backbones
from dinov3.hub.hf_checkpoint import load_hf_checkpoint
from dinov3.models.vision_transformer import DinoVisionTransformer

logger = logging.getLogger("dinov3")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ARCHS = ["vits16", "vits16plus", "vitb16", "vitl16", "vitl16plus", "vith16plus", "vit7b16"]


def patch_aligned_size(height: int, width: int, long_side: int, patch_size: int) -> list[int]:
    """The size with a long side of about `long_side` pixels, both sides multiples of the patch size."""
    scale = long_side / max(height, width)
    return [max(1, round(side * scale / patch_size)) * patch_size for side in (height, width)]


def resize_image(image: Path | Tensor, long_side: int, patch_size: int) -> Tensor:
    """uint8 image, from a file or already decoded [3, H, W], resized to its patch-aligned size."""
    if isinstance(image, Path):
        image = decode_image(str(image), mode=ImageReadMode.RGB)
    size = patch_aligned_size(*image.shape[-2:], long_side, patch_size)
    return v2.functional.resize(image, size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True)


def load_model(arch: str, weights: str) -> DinoVisionTransformer:
    build = getattr(backbones, f"dinov3_{arch}")
    if weights.endswith(".pth"):
        return build(weights=weights)
    return load_hf_checkpoint(build(pretrained=False), weights)


def patch_grid(model: DinoVisionTransformer, image: Tensor) -> RopeGrid:
    H, W = (side // model.patch_size for side in image.shape[-2:])
    return RopeGrid(patch_coords(model.rope_embed, H, W), model.rope_embed.periods)


def each_layer(model: DinoVisionTransformer, image: Tensor, fn: Callable[[int, nn.Module, Tensor], None]) -> None:
    """Runs one dense forward pass over `image` [3, H, W], calling fn(layer, attn, x) on each attention input."""
    hooks = [
        block.attn.register_forward_pre_hook(lambda attn, args, layer=layer: fn(layer, attn, args[0]))
        for layer, block in enumerate(model.blocks)
    ]
    try:
        model.forward_features(make_base_transform()(image)[None])
    finally:
        for hook in hooks:
            hook.remove()


def spread_to_patches(cell_values: Tensor, cells: Tensor, pyramid: Pyramid, grid: RopeGrid) -> Tensor:
    """Spreads per-cell values [B, h, n] evenly over the patches each cell covers, giving [B, h, H, W]."""
    per_patch = cell_values / pyramid.counts[cells].clamp(min=1)
    by_cell = per_patch.new_zeros(*cells.shape[:2], len(pyramid.counts)).scatter(-1, cells, per_patch)
    *by_level, _null_cell = by_cell.split([h * w for h, w in pyramid.shapes] + [1], dim=-1)
    patches = sum(
        values.unflatten(-1, shape).repeat_interleave(2**level, -2).repeat_interleave(2**level, -1)
        for level, (values, shape) in enumerate(zip(by_level, pyramid.shapes))
    )
    H, W = grid.coords.shape[:2]
    return patches[..., :H, :W]


def dense_attention(q: Tensor, k: Tensor, v: Tensor, grid: RopeGrid, chunk: int) -> tuple[Tensor, Tensor]:
    """Dense head outputs [B, h, N, d] and the attention each key receives summed over all queries [B, h, N]."""
    q, k = rope_patches(q, grid), rope_patches(k, grid)
    scale = q.shape[-1] ** -0.5
    key_mass = sum((rows @ k.transpose(-1, -2) * scale).softmax(-1).sum(-2) for rows in q.split(chunk, dim=2))
    return F.scaled_dot_product_attention(q, k, v), key_mass


def head_stats(out, cells, pyramid: Pyramid, grid: RopeGrid, dense_out: Tensor, key_mass: Tensor) -> dict:
    """Per-head metrics [B, h] for one foveated attention result (`out`, `cells`, `pyramid`) against dense."""
    n_patches = grid.coords.shape[0] * grid.coords.shape[1]
    n_prefix = key_mass.shape[-1] - n_patches
    patch_mass = key_mass[..., n_prefix:].unflatten(-1, grid.coords.shape[:2])
    kept_fine = spread_to_patches((pyramid.level[cells] == 0).to(key_mass.dtype), cells, pyramid, grid)
    return dict(
        fine_mass=(patch_mass * kept_fine).sum((-2, -1)) / patch_mass.sum((-2, -1)),
        prefix_mass=key_mass[..., :n_prefix].sum(-1) / key_mass.sum(-1),
        cosine=F.cosine_similarity(out, dense_out, dim=-1).mean(-1),
    )


def measure(model: DinoVisionTransformer, image: Tensor, configs: list[FoveationConfig], chunk: int) -> list[dict]:
    """One row per layer, head and config, from a single dense forward pass over the uint8 `image` [3, H, W]."""
    grid = patch_grid(model, image)
    rows = []

    def record(layer: int, attn: nn.Module, x: Tensor):
        q, k, v = FoveatedAttention(attn, configs[0]).heads(x)
        dense_out, key_mass = dense_attention(q, k, v, grid, chunk)
        for config in configs:
            attended = FoveatedAttention(attn, config).attend(q, k, v, grid)
            stats = head_stats(*attended, grid, dense_out, key_mass)
            per_head = zip(*(values[0].tolist() for values in stats.values()))
            for head, values in enumerate(per_head):
                rows.append(
                    dict(layer=layer, head=head, start_level=config.start_level, budget=config.budget)
                    | dict(zip(stats, values))
                )

    each_layer(model, image, record)
    return rows


def feasible_configs(H: int, W: int, n_prefix: int, args) -> list[FoveationConfig]:
    configs = []
    for start_level, budget in itertools.product(args.start_levels, args.budgets):
        side = 2**start_level
        roots = -(-H // side) * -(-W // side)
        if roots > budget - n_prefix:
            logger.warning(
                f"Skipping budget {budget} at start level {start_level}: {H}x{W} patches give {roots} start cells"
            )
            continue
        score_level = start_level if args.score_level is None else min(args.score_level, start_level)
        configs.append(FoveationConfig(budget=budget, start_level=start_level, score_level=score_level))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=ARCHS, default="vitl16")
    parser.add_argument("--weights", required=True, help="Hugging Face model directory, or a Meta .pth path or URL")
    parser.add_argument("--images", type=Path, required=True, help="Directory of images")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1920], help="Long side in pixels")
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--start-levels", type=int, nargs="+", default=[2], help="Start cells 2**level patches wide")
    parser.add_argument("--score-level", type=int, help="Defaults to the start level")
    parser.add_argument("--chunk", type=int, default=256, help="Queries per chunk when summing dense attention")
    parser.add_argument("--out", type=Path, default=Path("phase0.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.arch, args.weights).to(device).eval()
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[: args.max_images]
    rows = []
    with torch.inference_mode():
        for path, size in itertools.product(paths, args.sizes):
            image = resize_image(path, size, model.patch_size).to(device)
            H, W = (side // model.patch_size for side in image.shape[-2:])
            logger.info(f"{path.name} at {size}px: {H}x{W} patches")
            if not (configs := feasible_configs(H, W, 1 + model.n_storage_tokens, args)):
                continue
            for row in measure(model, image, configs, args.chunk):
                rows.append(dict(image=path.name, size=size, patches_h=H, patches_w=W) | row)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    logger.info(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
