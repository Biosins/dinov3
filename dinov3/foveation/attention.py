"""Per-head foveated attention for DINOv3 ViTs.

Every fine patch still queries, but each head reads only a budget of keys drawn from a quadtree pyramid of
mean-pooled keys and values. A head chooses which cells to refine using its own attention, so attention cost
grows linearly with the number of patches rather than quadratically.
"""

import math
from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dinov3.layers.attention import SelfAttention, rope_apply
from dinov3.layers.rope_position_encoding import RopePositionEmbedding
from dinov3.models.vision_transformer import DinoVisionTransformer


@dataclass(frozen=True)
class FoveationConfig:
    budget: int = 1024
    """Most cells a head attends to, on top of the CLS and register tokens."""
    start_level: int = 2
    """Selection starts from cells 2**start_level patches wide (2 is 64px at 16px patches)."""
    score_level: int = 2
    """Scoring queries are mean-pooled over cells 2**score_level patches wide."""
    order: Literal["per_level", "global"] = "per_level"
    """per_level: one no-grad round per level, coarsest first, where cells only compete with cells of their own size.
    global: `rounds` rounds, each ranking every splittable cell by its score per patch covered."""
    score: Literal["attention", "detail", "random"] = "detail"
    """attention: the attention mass a cell receives. detail: mass times the spread of the cell's patch values, a
    rough estimate of the output error from leaving it unsplit. random: a baseline with the same budget."""
    rounds: int = 3
    """Scoring rounds for global order."""

    def __post_init__(self):
        if not 0 <= self.score_level <= self.start_level:
            raise ValueError("score_level must lie between 0 and start_level")


class RopeGrid(NamedTuple):
    """What a foveated model's RoPE module returns: positions rather than rotations, because pooled cells
    are rotated at their centroids."""

    coords: Tensor  # [H, W, 2] patch centres in RoPE's [-1, 1] space
    periods: Tensor  # [D_head // 4]


class Pyramid(NamedTuple):
    """All cells from fine (level 0) up to the start level in one flat buffer, plus a trailing empty null cell.

    Padding cells and the null cell cover no patches, so their log-count bias is -inf and they get no attention.
    """

    queries: Tensor  # [B, h, M, d] rotated at each cell's centroid
    keys: Tensor  # [B, h, M, d] rotated at each cell's centroid
    values: Tensor  # [B, h, M, d]
    spread: Tensor  # [B, h, M] RMS distance of a cell's patch values from their mean
    counts: Tensor  # [M] patches each cell covers
    level: Tensor  # [M], -1 for the null cell
    children: Tensor  # [M, 4] a cell's quadrants; splitting a fine cell keeps it and adds three null cells
    roots: Tensor  # [n0] the start-level cells
    shapes: list[tuple[int, int]]  # grid size of each level, fine first; level l cells are 2**l patches wide


def patch_coords(rope: RopePositionEmbedding, H: int, W: int) -> Tensor:
    """Patch centres as RopePositionEmbedding computes them, minus its train-time shift, jitter and rescale."""
    dd = dict(device=rope.periods.device, dtype=rope.dtype)
    denom_h, denom_w = {"max": (max(H, W),) * 2, "min": (min(H, W),) * 2, "separate": (H, W)}[rope.normalize_coords]
    coords_h = torch.arange(0.5, H, **dd) / denom_h
    coords_w = torch.arange(0.5, W, **dd) / denom_w
    return 2.0 * torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1) - 1.0


def rope_rotate(x: Tensor, coords: Tensor, periods: Tensor) -> Tensor:
    """Rotates x [..., D_head] at positions coords [..., 2], exactly as RopePositionEmbedding and rope_apply do."""
    angles = (2 * math.pi * coords[..., None] / periods).flatten(-2).tile(2)
    return rope_apply(x.to(angles.dtype), angles.sin(), angles.cos()).to(x.dtype)


def rope_patches(x: Tensor, grid: RopeGrid) -> Tensor:
    """Rotates the patch tokens of x [B, h, N, d], leaving the leading CLS and register tokens alone."""
    n_patches = grid.coords.shape[0] * grid.coords.shape[1]
    prefix, patches = x.split([x.shape[2] - n_patches, n_patches], dim=2)
    return torch.cat([prefix, rope_rotate(patches, grid.coords.flatten(0, 1), grid.periods)], dim=2)


def _sum_pool(x: Tensor) -> Tensor:
    """Sums 2x2 blocks of x [..., H, W, c]."""
    H, W = x.shape[-3:-1]
    return x.unflatten(-3, (H // 2, 2)).unflatten(-2, (W // 2, 2)).sum((-4, -2))


def build_pyramid(q: Tensor, k: Tensor, v: Tensor, grid: RopeGrid, levels: int) -> Pyramid:
    """Mean-pools patch queries, keys and values [B, h, H, W, d] into a quadtree `levels` levels deep."""
    side = 2**levels
    H, W = grid.coords.shape[:2]
    pad = (0, 0, 0, -W % side, 0, -H % side)  # pads H and W up to multiples of side
    ones = torch.ones_like(grid.coords[..., :1])
    d = v.shape[-1]
    sums = [F.pad(torch.cat([q, k, v, v.square().sum(-1, keepdim=True)], dim=-1), pad)]
    geo_sums = [F.pad(torch.cat([grid.coords, ones], dim=-1), pad)]  # coordinate sums and patch counts
    for _ in range(levels):
        sums.append(_sum_pool(sums[-1]))
        geo_sums.append(_sum_pool(geo_sums[-1]))

    null_cell = (0, 0, 0, 1)
    flat = F.pad(torch.cat([s.flatten(-3, -2) for s in sums], dim=-2), null_cell)
    geo = F.pad(torch.cat([s.flatten(0, 1) for s in geo_sums]), null_cell)
    counts = geo[:, 2]
    divisor = counts.clamp(min=1)[:, None]
    coords = geo[:, :2] / divisor
    queries, keys, values, value_sq = (flat / divisor.to(flat.dtype)).split([d, d, d, 1], dim=-1)
    spread = (value_sq[..., 0] - values.square().sum(-1)).clamp(min=0).sqrt()

    shapes = [tuple(s.shape[:2]) for s in geo_sums]
    null = len(counts) - 1
    flat_ids = torch.arange(null, device=counts.device).split([h * w for h, w in shapes])
    level = F.pad(torch.cat([torch.full_like(ids, lvl) for lvl, ids in enumerate(flat_ids)]), (0, 1), value=-1)
    grids = [ids.view(shape) for ids, shape in zip(flat_ids, shapes)]
    # The children of each level-l cell are a 2x2 block of the level l-1 grid.
    quadrants = [g.unflatten(0, (-1, 2)).unflatten(2, (-1, 2)).transpose(1, 2).reshape(-1, 4) for g in grids[:-1]]
    leaf_children = F.pad(grids[0].reshape(-1, 1), (0, 3), value=null)
    children = torch.cat([leaf_children, *quadrants, torch.full_like(leaf_children[:1], null)])

    return Pyramid(
        queries=rope_rotate(queries, coords, grid.periods),
        keys=rope_rotate(keys, coords, grid.periods),
        values=values,
        spread=spread,
        counts=counts,
        level=level,
        children=children,
        roots=flat_ids[-1],
        shapes=shapes,
    )


def _spare_splits(roots: int, budget: int) -> int:
    """Each split adds three cells to the start cells."""
    if roots > budget:
        raise ValueError(f"{roots} start cells exceed the budget of {budget}; raise start_level or the budget")
    return (budget - roots) // 3


def level_schedule(roots: int, budget: int, levels: int) -> list[int]:
    """Splits per level for per-level order, coarsest first, sharing the spare budget equally between levels.

    A level can only split the cells the level above created; whatever a level cannot take passes down. Equal
    shares are a provisional design decision, recorded in plan.md.
    """
    remaining, available, schedule = _spare_splits(roots, budget), roots, []
    for levels_left in range(levels, 0, -1):
        splits = min(available, -(-remaining // levels_left))
        schedule.append(splits)
        remaining -= splits
        available = 4 * splits
    return schedule


def round_schedule(roots: int, budget: int, rounds: int) -> list[int]:
    """Splits per round for global order: the spare budget spread evenly, each round splitting at most the cells
    it holds."""
    remaining, size, schedule = _spare_splits(roots, budget), roots, []
    for rounds_left in range(rounds, 0, -1):
        splits = min(size, -(-remaining // rounds_left))
        schedule.append(splits)
        remaining -= splits
        size += 3 * splits
    return schedule


def gather_cells(x: Tensor, cells: Tensor) -> Tensor:
    """Picks cells [B, h, n] from x [B, h, M, d]."""
    return x.gather(2, cells[..., None].expand(-1, -1, -1, x.shape[-1]))


@torch.no_grad()
def select_cells(
    pyramid: Pyramid, queries: Tensor, weights: Tensor, prefix_keys: Tensor, config: FoveationConfig
) -> Tensor:
    """Indices [B, h, n] of the cells each head attends to; together they tile the image exactly once.

    Each round, `queries` [B, h, Q, d], each standing in for `weights` [Q] fine queries, attend to the CLS and
    register tokens plus the current cells, which are then scored (see FoveationConfig.score) and the best split.
    """

    def scores(cells: Tensor) -> Tensor:
        n_prefix = prefix_keys.shape[2]
        keys = torch.cat([prefix_keys, gather_cells(pyramid.keys, cells)], dim=2)
        bias = F.pad(pyramid.counts.log()[cells], (n_prefix, 0))
        probs = (queries @ keys.transpose(-1, -2) * queries.shape[-1] ** -0.5 + bias[:, :, None]).softmax(-1)
        mass = (weights[:, None] * probs).sum(-2)[..., n_prefix:]
        if config.score == "random":
            return torch.rand_like(mass)
        if config.score == "detail":
            return mass * pyramid.spread.gather(-1, cells)
        return mass

    roots = pyramid.roots.expand(*queries.shape[:2], -1)
    if config.order == "global":
        return _select_global(pyramid, roots, scores, config)
    return _select_per_level(pyramid, roots, scores, config)


def _select_per_level(pyramid: Pyramid, roots: Tensor, scores, config: FoveationConfig) -> Tensor:
    """Coarsest level first, so cells only compete with cells of the same size: at each level the frontier of cells
    the level above created is scored alongside the settled cells, its best cells split into the next frontier, and
    the rest settle."""
    settled, frontier = roots[..., :0], roots
    for splits in level_schedule(roots.shape[-1], config.budget, config.start_level):
        if not splits:
            break
        frontier_scores = scores(torch.cat([settled, frontier], dim=-1))[..., settled.shape[-1] :]
        ranked = frontier.gather(-1, frontier_scores.argsort(dim=-1, descending=True))
        settled = torch.cat([settled, ranked[..., splits:]], dim=-1)
        frontier = pyramid.children[ranked[..., :splits]].flatten(-2)
    return torch.cat([settled, frontier], dim=-1)


def _select_global(pyramid: Pyramid, roots: Tensor, scores, config: FoveationConfig) -> Tensor:
    """Every round ranks all splittable cells together by score per patch, so cells of different sizes compete on
    equal terms. If a round runs out of splittable cells, it splits fine cells into themselves plus null cells."""
    cells = roots
    for splits in round_schedule(roots.shape[-1], config.budget, config.rounds):
        if not splits:
            break
        per_patch = scores(cells) / pyramid.counts[cells].clamp(min=1)
        per_patch = per_patch.masked_fill(pyramid.level[cells] <= 0, -math.inf)
        picked = per_patch.topk(splits).indices
        quads = pyramid.children[cells.gather(-1, picked)]
        cells = torch.cat([cells.scatter(-1, picked, quads[..., 0]), quads[..., 1:].flatten(-2)], dim=-1)
    return cells


class FoveatedAttention(nn.Module):
    """Drop-in replacement for SelfAttention that reuses its projections, so parameters and state-dict keys are
    unchanged and nothing is copied."""

    def __init__(self, attn: SelfAttention, config: FoveationConfig):
        super().__init__()
        self.qkv, self.proj, self.proj_drop = attn.qkv, attn.proj, attn.proj_drop
        self.num_heads = attn.num_heads
        self.config = config

    def forward(self, x: Tensor, attn_bias=None, rope: RopeGrid | None = None) -> Tensor:
        assert attn_bias is None and rope is not None, "use foveate() so the RoPE module returns a RopeGrid"
        out, _, _ = self.attend(*self.heads(x), rope)
        return self.proj_drop(self.proj(out.transpose(1, 2).flatten(2)))

    def heads(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Unrotated queries, keys and values [B, h, N, d] for tokens x [B, N, C]."""
        return self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4).unbind(0)

    def attend(self, q: Tensor, k: Tensor, v: Tensor, grid: RopeGrid) -> tuple[Tensor, Tensor, Pyramid]:
        """Head outputs [B, h, N, d], the cells each head attended to, and the pyramid they index into."""
        H, W = grid.coords.shape[:2]
        n_prefix = q.shape[2] - H * W
        (q_prefix, q_patches), (k_prefix, k_patches), (_, v_patches) = (
            t.split([n_prefix, H * W], dim=2) for t in (q, k, v)
        )
        pyramid = build_pyramid(
            *(t.unflatten(2, (H, W)) for t in (q_patches, k_patches, v_patches)), grid, self.config.start_level
        )
        scorers = pyramid.level == self.config.score_level
        cells = select_cells(
            pyramid,
            queries=torch.cat([q_prefix, pyramid.queries[:, :, scorers]], dim=2),
            weights=F.pad(pyramid.counts[scorers], (n_prefix, 0), value=1.0),
            prefix_keys=k_prefix,
            config=self.config,
        )
        keys = torch.cat([k_prefix, gather_cells(pyramid.keys, cells)], dim=2)
        values = torch.cat([v[:, :, :n_prefix], gather_cells(pyramid.values, cells)], dim=2)
        # A cell covering m patches stands in for m keys, so it gets log(m) added to its logits.
        log_counts = F.pad(pyramid.counts.log()[cells], (n_prefix, 0))
        out = F.scaled_dot_product_attention(
            rope_patches(q, grid), keys, values, attn_mask=log_counts[:, :, None].to(q.dtype)
        )
        return out, cells, pyramid


def _rope_grid(rope: RopePositionEmbedding, args, kwargs, output) -> RopeGrid:
    return RopeGrid(patch_coords(rope, kwargs["H"], kwargs["W"]), rope.periods)


def foveate(model: DinoVisionTransformer, config: FoveationConfig) -> DinoVisionTransformer:
    """Swaps every block's attention for FoveatedAttention, in place, sharing the original parameters."""
    if any(block.sample_drop_ratio > 0 for block in model.blocks):
        # Stochastic depth calls Attention.forward_list and indexes RoPE sin/cos, neither of which exists here.
        raise ValueError("Build the model with drop_path_rate=0 before foveating it")
    for block in model.blocks:
        block.attn = FoveatedAttention(block.attn, config)
    model.rope_embed.register_forward_hook(_rope_grid, with_kwargs=True)
    return model
