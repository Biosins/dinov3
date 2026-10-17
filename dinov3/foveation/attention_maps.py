"""Exports dense and foveated attention maps for one image, for visualising each head's layout.

Writes to --out:
    image.jpg   the image as the model saw it
    maps.png    grayscale tiles of H x W patches, one row of tiles per layer and three per head: the CLS token's
                dense attention, its foveated attention (each cell's weight spread over its patches), and the dense
                attention each patch receives from all queries. Pixel value b stands for
                log10(value) = lo + b / 255 * (hi - lo), with lo and hi per tile in meta.json
    meta.json   grid and pyramid shapes, the config, per-tile log ranges, per-head metrics, and under "cells" the
                base64 little-endian uint16 [layers, heads, n] pyramid index of every cell each head attends to

PNG and JSON keep the export loadable by a web page.

Example:
    python -m dinov3.foveation.attention_maps --weights weights/dinov3-vitl16-pretrain-lvd1689m --image photo.jpg
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.io import write_jpeg, write_png

from dinov3.foveation.attention import FoveatedAttention, FoveationConfig, gather_cells, rope_patches
from dinov3.foveation.phase0 import (
    ARCHS,
    dense_attention,
    each_layer,
    head_stats,
    load_model,
    patch_grid,
    resize_image,
    spread_to_patches,
)

LOG_DECADES = 4
"""Each map spans this many orders of magnitude below its maximum; anything fainter shows as zero."""


def to_log_bytes(maps: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """uint8 log-scaled maps [..., H, W] plus the log10 range [...] each byte scale spans."""
    log = maps.clamp(min=1e-30).log10()
    hi = log.amax((-2, -1))
    lo = hi - LOG_DECADES
    scaled = (log - lo[..., None, None]) / LOG_DECADES * 255
    return scaled.clamp(0, 255).round().to(torch.uint8), lo, hi


def export(model, image: Tensor, config: FoveationConfig, chunk: int, out: Path) -> None:
    grid = patch_grid(model, image)
    H, W = grid.coords.shape[:2]
    layers = []

    def record(layer: int, attn: nn.Module, x: Tensor):
        fov = FoveatedAttention(attn, config)
        q, k, v = fov.heads(x)
        n_prefix = q.shape[2] - H * W
        dense_out, key_mass = dense_attention(q, k, v, grid, chunk)
        out, cells, pyramid = fov.attend(q, k, v, grid)
        stats = head_stats(out, cells, pyramid, grid, dense_out, key_mass)

        cls_query = q[:, :, :1]  # CLS is never rotated
        dense_cls = (cls_query @ rope_patches(k, grid).transpose(-1, -2) * q.shape[-1] ** -0.5).softmax(-1)
        fov_keys = torch.cat([k[:, :, :n_prefix], gather_cells(pyramid.keys, cells)], dim=2)
        fov_bias = F.pad(pyramid.counts.log()[cells], (n_prefix, 0))
        fov_cls = (cls_query @ fov_keys.transpose(-1, -2) * q.shape[-1] ** -0.5 + fov_bias[:, :, None]).softmax(-1)
        maps = torch.stack(
            [
                dense_cls[:, :, 0, n_prefix:].unflatten(-1, (H, W)),
                spread_to_patches(fov_cls[:, :, 0, n_prefix:], cells, pyramid, grid),
                key_mass[..., n_prefix:].unflatten(-1, (H, W)) / key_mass.shape[-1],
            ],
            dim=2,
        )[0]
        layers.append(dict(maps=to_log_bytes(maps), cells=cells[0], stats=stats, pyramid=pyramid, n_prefix=n_prefix))

    each_layer(model, image, record)

    out.mkdir(parents=True, exist_ok=True)
    write_jpeg(image.cpu(), str(out / "image.jpg"), quality=90)
    tiles = torch.stack([layer["maps"][0] for layer in layers])  # [layers, heads, 3, H, W]
    write_png(tiles.permute(0, 3, 1, 2, 4).reshape(1, len(layers) * H, -1).cpu(), str(out / "maps.png"))
    cell_ids = np.stack([layer["cells"].cpu().numpy() for layer in layers]).astype("<u2")
    pyramid = layers[0]["pyramid"]
    meta = dict(
        patch_size=model.patch_size,
        grid=[H, W],
        n_prefix=layers[0]["n_prefix"],
        layers=len(layers),
        heads=layers[0]["cells"].shape[0],
        cells_per_head=layers[0]["cells"].shape[1],
        level_shapes=pyramid.shapes,
        config=dict(
            budget=config.budget,
            start_level=config.start_level,
            score_level=config.score_level,
            score=config.score,
            order=config.order,
        ),
        maps=["dense_cls", "foveated_cls", "key_mass"],
        log_decades=LOG_DECADES,
        log_lo=[layer["maps"][1].tolist() for layer in layers],
        log_hi=[layer["maps"][2].tolist() for layer in layers],
        stats={name: [layer["stats"][name][0].tolist() for layer in layers] for name in layers[0]["stats"]},
        cells=base64.b64encode(cell_ids.tobytes()).decode(),
    )
    (out / "meta.json").write_text(json.dumps(meta))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=ARCHS, default="vitl16")
    parser.add_argument("--weights", required=True, help="Hugging Face model directory, or a Meta .pth path or URL")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1920, help="Long side in pixels")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--start-level", type=int, default=2)
    parser.add_argument("--score-level", type=int, help="Defaults to the start level")
    parser.add_argument("--score", choices=["attention", "detail", "bound", "random"], default="detail")
    parser.add_argument("--order", choices=["per_level", "global", "tree"], default="per_level")
    parser.add_argument("--chunk", type=int, default=256, help="Queries per chunk when summing dense attention")
    parser.add_argument("--out", type=Path, default=Path("attention_maps"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.arch, args.weights).to(device).eval()
    score_level = args.start_level if args.score_level is None else args.score_level
    config = FoveationConfig(
        budget=args.budget, start_level=args.start_level, score_level=score_level, score=args.score, order=args.order
    )
    with torch.inference_mode():
        image = resize_image(args.image, args.size, model.patch_size).to(device)
        export(model, image, config, args.chunk, args.out)


if __name__ == "__main__":
    main()
