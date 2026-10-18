"""K vs N: how many keys a foveated DINOv3 needs as images grow.

Each image is resized to every long side in --sizes, giving N patches. For every largest cell size and budget rule,
the foveated model's final CLS and patch features are compared with dense DINOv3 at the same size (cosine
similarity, patches averaged), in float32. A budget rule is a fixed K such as 1024, or "<c>sqrt" for K = c sqrt(N),
so K grows with the image. Budgets that cannot hold the start cells plus the CLS and register tokens are skipped.

With --run, the foveated model carries that distill.py run's LoRA adapters, merged into its weights; dense
DINOv3 without them stays the reference.

Each configuration is then timed at every size on the first image, in bfloat16 with compile="dynamic", so one
compile serves every size (about 20% slower than compiling per size), next to dense attention.

Example:
    python -m dinov3.foveation.budget_sweep --arch vitb16 --weights weights/dinov3-vitb16-pretrain-lvd1689m \\
        --images data/sweep/dis5k_vd_4k --out data/results/budget_sweep.csv
"""

import argparse
import copy
import itertools
import logging
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from dinov3.data.transforms import make_base_transform
from dinov3.foveation.attention import FoveationConfig, foveate, start_cells
from dinov3.foveation.benchmark import measure
from dinov3.foveation.distill import with_adapters
from dinov3.foveation.phase0 import ARCHS, IMAGE_SUFFIXES, load_model, resize_image

logger = logging.getLogger("dinov3")


def final_features(model, image: Tensor) -> tuple[Tensor, Tensor]:
    out = model.forward_features(make_base_transform()(image)[None])
    return out["x_norm_clstoken"], out["x_norm_patchtokens"]


def budget(rule: str, N: int) -> int:
    """K for N patches under a budget rule: a fixed K such as "1024", or "<c>sqrt" for K = c sqrt(N)."""
    return round(float(rule.removesuffix("sqrt")) * math.sqrt(N)) if rule.endswith("sqrt") else int(rule)


def timed(model, x: Tensor, min_run_time: float) -> dict[str, float]:
    return dict(time_ms=measure(model.forward_features, x, torch.bfloat16, min_run_time)["time_ms"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=ARCHS, default="vitb16")
    parser.add_argument("--weights", required=True, help="Hugging Face model directory, or a Meta .pth path or URL")
    parser.add_argument("--images", type=Path, required=True, help="Directory of images")
    parser.add_argument("--run", type=Path, help="A distill.py run whose adapters the foveated model uses")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1280, 1920, 3840], help="Long sides in pixels")
    parser.add_argument("--budgets", nargs="+", default=["512", "1024", "2048"], help="e.g. 1024 or 10sqrt")
    parser.add_argument("--cell-sizes", type=int, nargs="+", default=[64, 128], help="Largest cell, in pixels")
    parser.add_argument("--order", default="tree")
    parser.add_argument("--score", default="variance")
    parser.add_argument("--scoring", default="beam")
    parser.add_argument("--min-run-time", type=float, default=1.0, help="Seconds of timing per configuration")
    parser.add_argument("--out", type=Path, default=Path("budget_sweep.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dense = load_model(args.arch, args.weights).cuda().eval()
    base = dense if args.run is None else with_adapters(dense, args.run).eval()
    n_prefix = 1 + dense.n_storage_tokens
    patch = dense.patch_size
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    # A rule that scales K with N gives every size its own budget, and so its own compiled graph
    torch._dynamo.config.recompile_limit = 64

    def config(cell: int, rule: str, H: int, W: int) -> FoveationConfig | None:
        """The configuration for this image, or None if the budget cannot hold the start cells."""
        level = round(math.log2(cell // patch))
        K = budget(rule, H * W)
        if start_cells(H, W, level) + n_prefix > K:
            return None
        return FoveationConfig(
            budget=K, start_level=level, score_level=level, order=args.order, score=args.score, scoring=args.scoring
        )

    rules = list(itertools.product(args.cell_sizes, args.budgets))

    quality = []
    with torch.inference_mode():
        for path, size in itertools.product(paths, args.sizes):
            image = resize_image(path, size, patch).cuda()
            H, W = image.shape[-2] // patch, image.shape[-1] // patch
            logger.info(f"{path.name} at {size}px: {H}x{W} patches")
            reference = final_features(dense, image)
            for cell, rule in rules:
                if (fov := config(cell, rule, H, W)) is None:
                    continue
                foveated = final_features(foveate(copy.deepcopy(base), fov), image)
                cls, patches = (F.cosine_similarity(f, r, dim=-1).mean().item() for f, r in zip(foveated, reference))
                row = dict(image=path.name, size=size, N=H * W, cell=cell, rule=rule, K=fov.budget)
                quality.append(row | dict(cls=cls, patches=patches))

    timing = []
    for size in args.sizes:
        image = resize_image(paths[0], size, patch).cuda()
        H, W = image.shape[-2] // patch, image.shape[-1] // patch
        x = make_base_transform()(image)[None]
        timing.append(dict(size=size, cell=None, rule="dense") | timed(dense, x, args.min_run_time))
        for cell, rule in rules:
            if (fov := config(cell, rule, H, W)) is not None:
                model = foveate(copy.deepcopy(base), replace(fov, compile="dynamic"))
                timing.append(dict(size=size, cell=cell, rule=rule) | timed(model, x, args.min_run_time))
                del model
                torch.cuda.empty_cache()

    quality, timing = pd.DataFrame(quality), pd.DataFrame(timing)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(args.out, index=False)
    timing.to_csv(args.out.with_suffix(".timing.csv"), index=False)
    summary = quality.groupby(["size", "cell", "rule"])[["N", "K", "cls", "patches"]].mean().reset_index()
    summary["K/N"] = summary.K / summary.N
    dense_ms = timing[timing.rule == "dense"].set_index("size").time_ms
    summary = summary.merge(timing[timing.rule != "dense"], on=["size", "cell", "rule"], how="left")
    summary["speedup"] = summary["size"].map(dense_ms) / summary.time_ms
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
