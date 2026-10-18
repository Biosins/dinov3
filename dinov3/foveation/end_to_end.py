"""End-to-end check: run DINOv3 with every layer foveated and compare it with dense DINOv3.

Unlike Phase 0, which feeds each layer the dense model's input, errors here carry from layer to layer. For each
image and layout we record the cosine similarity to dense of the residual stream after every block (averaged over
patch tokens), and of the final normalised CLS token and patch features.

Layouts: every selection order and score at each budget, plus uniform start cells with no splits.

Example:
    python -m dinov3.foveation.end_to_end --weights weights/dinov3-vitl16-pretrain-lvd1689m --images images/
"""

import argparse
import copy
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from dinov3.data.transforms import make_base_transform
from dinov3.foveation.attention import FoveationConfig, foveate
from dinov3.foveation.phase0 import ARCHS, IMAGE_SUFFIXES, load_model, resize_image
from dinov3.models.vision_transformer import DinoVisionTransformer

logger = logging.getLogger("dinov3")


def features(model: DinoVisionTransformer, image: Tensor) -> tuple[list[Tensor], dict[str, Tensor]]:
    """The residual stream after every block [N, C] and the final normalised features, for a uint8 image."""
    streams = []

    def record(block, args, out: list[Tensor]):  # forward_features runs blocks on lists of [B, N, C] tensors
        streams.append(out[0][0])

    hooks = [block.register_forward_hook(record) for block in model.blocks]
    try:
        final = model.forward_features(make_base_transform()(image)[None])
    finally:
        for hook in hooks:
            hook.remove()
    return streams, final


def compare(dense: tuple, foveated: tuple, n_prefix: int) -> list[dict]:
    (dense_streams, dense_final), (fov_streams, fov_final) = dense, foveated
    rows = [
        dict(where=f"block {i}", cosine=F.cosine_similarity(d[n_prefix:], f[n_prefix:], dim=-1).mean().item())
        for i, (d, f) in enumerate(zip(dense_streams, fov_streams))
    ]
    for key, name in [("x_norm_clstoken", "final CLS"), ("x_norm_patchtokens", "final patches")]:
        rows.append(
            dict(where=name, cosine=F.cosine_similarity(dense_final[key], fov_final[key], dim=-1).mean().item())
        )
    return rows


def layouts(budgets: list[int], start_level: int, roots: int, n_prefix: int) -> dict[str, FoveationConfig]:
    configs = {
        f"uniform {16 * 2**start_level}px (K={roots + n_prefix})": FoveationConfig(
            budget=roots + n_prefix, start_level=start_level, score_level=start_level
        )
    }
    for budget in budgets:
        for order, score, scoring in [
            ("per_level", "attention", "full"),
            ("per_level", "detail", "full"),
            ("tree", "detail", "full"),
            ("tree", "bound", "full"),
            ("tree", "variance", "full"),
            ("tree", "variance", "beam"),
            ("per_level", "random", "full"),
        ]:
            name = f"{order} {score}{' beam' if scoring == 'beam' else ''} K={budget}"
            configs[name] = FoveationConfig(
                budget=budget,
                start_level=start_level,
                score_level=start_level,
                order=order,
                score=score,
                scoring=scoring,
            )
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=ARCHS, default="vitl16")
    parser.add_argument("--weights", required=True, help="Hugging Face model directory, or a Meta .pth path or URL")
    parser.add_argument("--images", type=Path, required=True, help="Directory of images")
    parser.add_argument("--size", type=int, default=1920, help="Long side in pixels")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--start-level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="For the random layouts")
    parser.add_argument("--out", type=Path, default=Path("end_to_end.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dense = load_model(args.arch, args.weights).to(device).eval()
    n_prefix = 1 + dense.n_storage_tokens
    side = 2**args.start_level
    rows = []
    with torch.inference_mode():
        for path in sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            image = resize_image(path, args.size, dense.patch_size).to(device)
            H, W = (s // dense.patch_size for s in image.shape[-2:])
            logger.info(f"{path.name}: {H}x{W} patches")
            reference = features(dense, image)
            for name, config in layouts(
                args.budgets, args.start_level, -(-H // side) * -(-W // side), n_prefix
            ).items():
                torch.manual_seed(args.seed)
                foveated = foveate(copy.deepcopy(dense), config)
                for row in compare(reference, features(foveated, image), n_prefix):
                    rows.append(dict(image=path.name, layout=name) | row)
                del foveated
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    logger.info(f"Wrote {len(table)} rows to {args.out}")


if __name__ == "__main__":
    main()
