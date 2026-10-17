"""Speed, memory and FLOPs of foveated DINOv3 against dense, at several image sizes.

For each size, the dense model and each foveated budget run one forward pass over a synthetic image (timing does not
depend on content). We report the median wall-clock time, peak GPU memory, analytic FLOPs split into attention and
the per-token layers every token pays for regardless, and, for the foveated model, the GPU time of each stage.

At each size the start level rises from --start-level until the start cells use at most half the budget, so there is
budget left to split.

Example:
    python -m dinov3.foveation.benchmark --weights weights/dinov3-vitl16-pretrain-lvd1689m --sizes 1920 3840
"""

import argparse
import copy

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils import benchmark

from dinov3.data.transforms import make_base_transform
from dinov3.foveation.attention import FoveationConfig, foveate, level_schedule
from dinov3.foveation.phase0 import ARCHS, load_model
from dinov3.models.vision_transformer import DinoVisionTransformer

STAGES = ["foveation.pyramid", "foveation.select", "foveation.attend"]


def start_cells(H: int, W: int, level: int) -> int:
    side = 2**level
    return -(-H // side) * -(-W // side)


def flops(model: DinoVisionTransformer, H: int, W: int, config: FoveationConfig | None) -> dict[str, float]:
    """Multiply-adds x 2 for one forward pass, in TFLOPs. Dense attention costs 4 N^2 C per layer; foveated costs
    4 N K C for the final attention over K = budget keys (null slots included, since they still go through the
    matmul), plus 2 Q S C per scoring round for Q scoring queries and S scored keys. Tree order scores once against
    every fine key. Projections and MLPs cost 24 N C^2 per layer either way (MLP ratio 4)."""
    C, depth, n_prefix = model.embed_dim, len(model.blocks), 1 + model.n_storage_tokens
    N = n_prefix + H * W
    per_token = 24 * N * C**2
    if config is None:
        attention = 4 * N**2 * C
    else:
        roots = start_cells(H, W, config.start_level)
        scoring_queries = n_prefix + start_cells(H, W, config.score_level)
        if config.order == "tree":
            scored = [N]
        else:
            scored, keys = [], roots
            for splits in level_schedule(roots, config.budget - n_prefix, config.start_level):
                if splits:
                    scored.append(n_prefix + keys)
                    keys += 3 * splits
        attention = 4 * N * config.budget * C + sum(2 * scoring_queries * s * C for s in scored)
    return dict(attention_tflops=depth * attention / 1e12, per_token_tflops=depth * per_token / 1e12)


def measure(model: DinoVisionTransformer, x: torch.Tensor, dtype: torch.dtype, min_run_time: float) -> dict:
    def forward():
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            model.forward_features(x)

    forward()  # warm up kernels and caches
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    forward()
    torch.cuda.synchronize()
    peak_mb = (torch.cuda.max_memory_allocated() - base) / 2**20

    timer = benchmark.Timer(stmt="forward(); torch.cuda.synchronize()", globals=dict(forward=forward, torch=torch))
    time_ms = timer.blocked_autorange(min_run_time=min_run_time).median * 1e3

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        forward()
        torch.cuda.synchronize()
    device_ms = {e.key: e.device_time_total / 1e3 for e in prof.key_averages()}
    return (
        dict(time_ms=time_ms, peak_activation_mb=peak_mb)
        | {f"{name.split('.')[1]}_ms": device_ms.get(name, float("nan")) for name in STAGES}
        | dict(sdpa_ms=device_ms.get("aten::scaled_dot_product_attention", float("nan")))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=ARCHS, default="vitl16")
    parser.add_argument("--weights", required=True, help="Hugging Face model directory, or a Meta .pth path or URL")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1920, 3840], help="Long side in pixels, 16:9")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--start-level", type=int, default=2)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--min-run-time", type=float, default=3.0, help="Seconds of timing per model and size")
    parser.add_argument("--out", type=str, default="benchmark.csv")
    args = parser.parse_args()

    dtype = dict(bf16=torch.bfloat16, fp32=torch.float32)[args.dtype]
    dense = load_model(args.arch, args.weights).cuda().eval()
    rows = []
    for size in args.sizes:
        H, W = round(size * 9 / 16 / dense.patch_size), size // dense.patch_size
        image = torch.randint(0, 256, (3, H * dense.patch_size, W * dense.patch_size), dtype=torch.uint8)
        x = make_base_transform()(image)[None].cuda()
        common = dict(size=size, patches=f"{W}x{H}", dtype=args.dtype)
        rows.append(
            common | dict(model="dense") | flops(dense, H, W, None) | measure(dense, x, dtype, args.min_run_time)
        )
        for budget in args.budgets:
            level = args.start_level
            while start_cells(H, W, level) > budget // 2:
                level += 1
            for compile in (False, True):
                config = FoveationConfig(budget=budget, start_level=level, score_level=level, compile=compile)
                foveated = foveate(copy.deepcopy(dense), config)
                name = f"foveated K={budget}, {dense.patch_size * 2**level}px start" + (", compiled" if compile else "")
                rows.append(
                    common
                    | dict(model=name)
                    | flops(foveated, H, W, config)
                    | measure(foveated, x, dtype, args.min_run_time)
                )
                del foveated
                torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
