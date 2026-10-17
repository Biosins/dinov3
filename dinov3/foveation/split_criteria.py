"""Phase 0 test of split criteria: which per-cell score best predicts the output error of leaving a cell pooled?

For every layer, head and pyramid cell, the oracle is the true error in that head's output, summed over a random
sample of queries, when the cell's patches are replaced by its pooled token (with its log-count bonus) and every
other key stays exact. Each criterion is computed the way selection could compute it: from the pooled scoring
queries and the pyramid's pooled moments. Per level, we report the Spearman correlation between criterion and
oracle across cells, and how much oracle error the criterion's top 10% of cells hold relative to the oracle's own
top 10%.

Criteria. m is a cell's attention mass from the scoring queries over all fine keys, summed up the pyramid; spreads
are RMS distances from the cell mean, using RoPE-rotated keys so position counts:
    mass          m
    value_spread  m x value spread (the current "detail" score)
    bound         query norm x m x key spread x value spread / sqrt(d), bounding the first-order error
    kl            half the score variance within the cell, mass-weighted over queries (diagonal key covariance):
                  the information lost by spreading attention evenly over the cell
    covariance    norm of the mass-weighted covariance of scores and values within the cell, the first-order
                  output error; queries are summed inside the norm, so opposite errors can cancel
    random        baseline

Example:
    python -m dinov3.foveation.split_criteria --weights weights/dinov3-vitl16-pretrain-lvd1689m \\
        --images data/div2k/DIV2K_valid_HR --max-images 10
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dinov3.foveation.attention import (
    FoveatedAttention,
    FoveationConfig,
    Layout,
    Pyramid,
    RopeGrid,
    _rotate,
    _sum_pool,
    build_pyramid,
    cell_sums,
    pyramid_layout,
)
from dinov3.foveation.phase0 import ARCHS, IMAGE_SUFFIXES, each_layer, load_model, patch_grid, resize_image

logger = logging.getLogger("dinov3")

CRITERIA = ["mass", "value_spread", "bound", "kl", "covariance", "random"]


def on_patches(cell_values: Tensor, layout: Layout, level: int) -> Tensor:
    """Repeats each level-`level` cell's values [h, M, c] over its patches, on the padded patch grid [h, H', W', c]."""
    side = 2**level
    values = cell_values[:, layout.level_cells(level)].unflatten(1, layout.shapes[level])
    return values.repeat_interleave(side, 1).repeat_interleave(side, 2)


def criteria(q: Tensor, k: Tensor, v: Tensor, grid: RopeGrid, config: FoveationConfig) -> tuple[dict, Pyramid]:
    """Every criterion for every cell [h, M], for one image's heads [1, h, N, d], plus the pyramid."""
    H, W = grid.coords.shape[:2]
    n_prefix, d, levels = q.shape[2] - H * W, q.shape[-1], config.start_level
    layout = pyramid_layout(grid, levels)
    grid = grid._replace(layout=layout)
    patches = [t[:, :, n_prefix:] for t in (q, k, v)]
    pyramid = build_pyramid(*(t.unflatten(2, (H, W)) for t in patches), grid, levels, config.score_level)

    queries = torch.cat([q[:, :, :n_prefix], pyramid.queries], dim=2)[0]  # [h, Q, d]
    weights = F.pad(pyramid.query_counts, (n_prefix, 0), value=1.0)
    k_fine = _rotate(patches[1], layout.patch_rope)[0]  # [h, N, d]
    v_fine = patches[2][0]
    logits = queries @ torch.cat([k[0, :, :n_prefix], k_fine], dim=1).mT * d**-0.5
    probs = logits.softmax(-1)[..., n_prefix:] * weights[:, None]  # [h, Q, N], weighted by queries represented

    per_patch = torch.cat(
        [
            probs.sum(-2)[..., None],  # mass
            probs.mT @ queries.square(),  # for kl
            probs.mT @ queries * d**-0.5,  # for covariance
            probs.mT @ queries.norm(dim=-1, keepdim=True),  # for bound
            k_fine,
            k_fine.square(),
            v_fine,
            v_fine.square(),
        ],
        dim=-1,
    )
    sums = cell_sums(per_patch.unflatten(1, (H, W)), levels)
    mass, q_sq, q_dir, q_norm, k1, k2, v1, v2 = sums.split([1, d, d, 1, d, d, d, d], dim=-1)
    n = layout.counts.clamp(min=1)[:, None]
    k_mean, v_mean = k1 / n, v1 / n
    k_var, v_var = (k2 / n - k_mean.square()).clamp(min=0), (v2 / n - v_mean.square()).clamp(min=0)
    mass = mass[..., 0]
    scores = dict(
        mass=mass,
        value_spread=mass * v_var.sum(-1).sqrt(),
        bound=q_norm[..., 0] * k_var.sum(-1).sqrt() * v_var.sum(-1).sqrt() * d**-0.5,
        kl=0.5 * (k_var * q_sq).sum(-1) / d,
        covariance=covariance_error(k_fine, v_fine, q_dir, k_mean, v_mean, layout, H, W),
        random=torch.rand_like(mass),
    )
    return scores, pyramid


def covariance_error(k_fine, v_fine, q_dir, k_mean, v_mean, layout: Layout, H: int, W: int) -> Tensor:
    """|| (1/m) sum_i ((k_i - k_mean) . q_dir) (v_i - v_mean) || for every cell [h, M], where q_dir already holds the
    mass-weighted queries / sqrt(d) summed over the cell."""
    levels = len(layout.shapes) - 1
    pad = (0, 0, 0, layout.shapes[0][1] - W, 0, layout.shapes[0][0] - H)
    real = F.pad(torch.ones_like(k_fine[:1, :, :1]).unflatten(1, (H, W)), pad)  # zero on padding patches
    k_grid, v_grid = (F.pad(t.unflatten(1, (H, W)), pad) for t in (k_fine, v_fine))
    error = torch.zeros_like(layout.counts).expand(k_fine.shape[0], -1).clone()
    for level in range(1, levels + 1):
        along = ((k_grid - on_patches(k_mean, layout, level)) * on_patches(q_dir, layout, level)).sum(-1, keepdim=True)
        contrib = along * real * (v_grid - on_patches(v_mean, layout, level))
        for _ in range(level):
            contrib = _sum_pool(contrib)
        cells = layout.level_cells(level)
        error[:, cells] = contrib.flatten(1, 2).norm(dim=-1) / layout.counts[cells].clamp(min=1)
    return error


def oracle(q: Tensor, k: Tensor, v: Tensor, pyramid: Pyramid, H: int, W: int, n_queries: int, chunk=32) -> Tensor:
    """Per cell [h, M]: the output error, summed over n_queries random queries, of pooling that cell alone."""
    layout, levels = pyramid.layout, len(pyramid.shapes) - 1
    n_prefix, d = q.shape[2] - H * W, q.shape[-1]
    q_all, k_all = (torch.cat([t[0, :, :n_prefix], _rotate(t[0, :, n_prefix:], layout.patch_rope)], 1) for t in (q, k))
    v_patches = v[0, :, n_prefix:]
    sample = torch.randperm(q_all.shape[1], device=q.device)[:n_queries]
    log_counts = layout.counts.log()
    error = torch.zeros(q.shape[1], len(layout.counts), device=q.device)
    for head in range(q.shape[1]):
        for rows in sample.split(chunk):
            queries = q_all[head, rows]
            logits = queries @ k_all[head].mT * d**-0.5
            log_z = logits.logsumexp(-1, keepdim=True)  # the dense normaliser, held fixed
            p = (logits[:, n_prefix:] - log_z).exp()  # [S, N]
            exact = cell_sums((p[..., None] * v_patches[head]).unflatten(1, (H, W)), levels)  # [S, M, d]
            pooled_p = (queries @ pyramid.keys[0, head].mT * d**-0.5 + log_counts - log_z).exp()
            error[head] += (exact - pooled_p[..., None] * pyramid.values[0, head]).norm(dim=-1).sum(0)
    return error


def rank_stats(score: Tensor, truth: Tensor, top: float = 0.1) -> tuple[float, float]:
    """Spearman correlation, and the oracle error in the score's top cells relative to the oracle's top cells."""
    ranks = [t.argsort().argsort().float() for t in (score, truth)]
    spearman = torch.corrcoef(torch.stack(ranks))[0, 1].item()
    n_top = max(1, int(len(score) * top))
    capture = (truth[score.topk(n_top).indices].sum() / truth.topk(n_top).values.sum()).item()
    return spearman, capture


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=ARCHS, default="vitl16")
    parser.add_argument("--weights", required=True, help="Hugging Face model directory, or a Meta .pth path or URL")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--size", type=int, default=1920, help="Long side in pixels")
    parser.add_argument("--start-level", type=int, default=2)
    parser.add_argument("--n-queries", type=int, default=256, help="Random queries per head for the oracle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("split_criteria.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    torch.manual_seed(args.seed)
    model = load_model(args.arch, args.weights).cuda().eval()
    config = FoveationConfig(start_level=args.start_level, score_level=args.start_level)
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[: args.max_images]
    rows = []
    with torch.inference_mode():
        for path in paths:
            image = resize_image(path, args.size, model.patch_size).cuda()
            grid = patch_grid(model, image)
            logger.info(f"{path.name}: {grid.coords.shape[0]}x{grid.coords.shape[1]} patches")

            def record(layer: int, attn: nn.Module, x: Tensor):
                q, k, v = FoveatedAttention(attn, config).heads(x)
                scores, pyramid = criteria(q, k, v, grid, config)
                truth = oracle(q, k, v, pyramid, *grid.coords.shape[:2], args.n_queries)
                for level in range(1, config.start_level + 1):
                    cells = (pyramid.level == level) & (pyramid.counts > 0)
                    for head in range(q.shape[1]):
                        for name, score in scores.items():
                            spearman, capture = rank_stats(score[head, cells], truth[head, cells])
                            rows.append(
                                dict(image=path.name, layer=layer, head=head, level=level, criterion=name)
                                | dict(spearman=spearman, capture=capture)
                            )

            each_layer(model, image, record)
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    summary = table.groupby(["level", "criterion"])[["spearman", "capture"]].mean().unstack(0).round(3)
    print(summary.loc[CRITERIA].to_string())


if __name__ == "__main__":
    main()
