"""Per-head foveated attention for DINOv3 ViTs.

Every fine patch still queries, but each head reads only a budget of keys drawn from a quadtree pyramid of
mean-pooled keys and values. A head chooses which cells to refine using its own attention, so attention cost
grows linearly with the number of patches rather than quadratically.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.profiler import record_function

from dinov3.layers.attention import SelfAttention, rope_apply
from dinov3.layers.rope_position_encoding import RopePositionEmbedding
from dinov3.models.vision_transformer import DinoVisionTransformer


@dataclass(frozen=True)
class FoveationConfig:
    budget: int = 1024
    """Keys each head attends to, including the CLS and register tokens. Slots the layout does not fill hold empty
    null cells, so every head reads exactly this many keys; a multiple of 64 suits the attention kernels."""
    start_level: int = 2
    """Selection starts from cells 2**start_level patches wide (2 is 64px at 16px patches)."""
    score_level: int = 2
    """Scoring queries are mean-pooled over cells 2**score_level patches wide."""
    order: Literal["per_level", "global", "tree", "uniform"] = "per_level"
    """per_level: one no-grad round per level, coarsest first, where cells only compete with cells of their own size.
    global: `rounds` rounds, each ranking every splittable cell by its score per patch covered.
    tree: one scoring pass over all fine keys, then equal-slope allocation: each head splits exactly the cells whose
    error reduction per added key beats its own threshold, set so its keys fit the budget.
    uniform: no splits, so every head reads just the start cells, however large the budget. A uniform pooling
    baseline."""
    score: Literal["attention", "detail", "bound", "random"] = "detail"
    """attention: the attention mass a cell receives. detail: mass times the spread of the cell's patch values, a
    rough estimate of the output error from leaving it unsplit. bound: query norm x mass x key spread x value spread
    / sqrt(d), a bound on the first-order output error (tree order only). random: a baseline with the same budget."""
    rounds: int = 3
    """Scoring rounds for global order."""
    compile: bool = False
    """Compile the pyramid pooling and tree selection with torch.compile. Faster, but recompiles for every new image
    size."""

    def __post_init__(self):
        if not 0 <= self.score_level <= self.start_level:
            raise ValueError("score_level must lie between 0 and start_level")
        if self.score == "bound" and self.order in ("per_level", "global"):
            raise ValueError('score="bound" needs the one-pass scores of order="tree"')


class Layout(NamedTuple):
    """Everything about the pyramid that depends only on the patch grid, so all layers share it: all cells from fine
    (level 0) up to the start level in one flat index space, plus a trailing empty null cell.

    Padding cells and the null cell cover no patches, so their log-count bias is -inf and they get no attention.
    """

    counts: Tensor  # [M] patches each cell covers
    level: Tensor  # [M], -1 for the null cell
    children: Tensor  # [M, 4] a cell's quadrants; splitting a fine cell keeps it and adds three null cells
    roots: Tensor  # [n0] the start-level cells
    shapes: list[tuple[int, int]]  # grid size of each level, fine first; level l cells are 2**l patches wide
    cell_rope: tuple[Tensor, Tensor]  # RoPE sin and cos [M, d] at each cell's centroid
    patch_rope: tuple[Tensor, Tensor]  # RoPE sin and cos [H * W, d] at each patch centre

    def level_cells(self, level: int) -> slice:
        start = sum(h * w for h, w in self.shapes[:level])
        return slice(start, start + self.shapes[level][0] * self.shapes[level][1])


class RopeGrid(NamedTuple):
    """What a foveated model's RoPE module returns: positions rather than rotations, because pooled cells
    are rotated at their centroids. foveate() also precomputes the pyramid layout, once per forward pass."""

    coords: Tensor  # [H, W, 2] patch centres in RoPE's [-1, 1] space
    periods: Tensor  # [D_head // 4]
    layout: Layout | None = None


class Pyramid(NamedTuple):
    """One layer's pooled features over a Layout."""

    queries: Tensor  # [B, h, Q, d] queries pooled to the scoring level, rotated at each cell's centroid
    query_counts: Tensor  # [Q] patches each scoring query stands for
    keys: Tensor  # [B, h, M, d] rotated at each cell's centroid
    values: Tensor  # [B, h, M, d]
    spread: Tensor  # [B, h, M] RMS distance of a cell's patch values from their mean
    layout: Layout

    @property
    def counts(self) -> Tensor:
        return self.layout.counts

    @property
    def level(self) -> Tensor:
        return self.layout.level

    @property
    def children(self) -> Tensor:
        return self.layout.children

    @property
    def roots(self) -> Tensor:
        return self.layout.roots

    @property
    def shapes(self) -> list[tuple[int, int]]:
        return self.layout.shapes


def patch_coords(rope: RopePositionEmbedding, H: int, W: int) -> Tensor:
    """Patch centres as RopePositionEmbedding computes them, minus its train-time shift, jitter and rescale."""
    dd = dict(device=rope.periods.device, dtype=rope.dtype)
    denom_h, denom_w = {"max": (max(H, W),) * 2, "min": (min(H, W),) * 2, "separate": (H, W)}[rope.normalize_coords]
    coords_h = torch.arange(0.5, H, **dd) / denom_h
    coords_w = torch.arange(0.5, W, **dd) / denom_w
    return 2.0 * torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1) - 1.0


def rope_sincos(coords: Tensor, periods: Tensor) -> tuple[Tensor, Tensor]:
    """RoPE sin and cos [..., D_head] at positions coords [..., 2], exactly as RopePositionEmbedding computes them."""
    angles = (2 * math.pi * coords[..., None] / periods).flatten(-2).tile(2)
    return angles.sin(), angles.cos()


def _rotate(x: Tensor, sincos: tuple[Tensor, Tensor]) -> Tensor:
    sin, cos = sincos
    return rope_apply(x.to(sin.dtype), sin, cos).to(x.dtype)


def rope_rotate(x: Tensor, coords: Tensor, periods: Tensor) -> Tensor:
    """Rotates x [..., D_head] at positions coords [..., 2], exactly as RopePositionEmbedding and rope_apply do."""
    return _rotate(x, rope_sincos(coords, periods))


def rope_patches(x: Tensor, grid: RopeGrid) -> Tensor:
    """Rotates the patch tokens of x [B, h, N, d], leaving the leading CLS and register tokens alone."""
    n_patches = grid.coords.shape[0] * grid.coords.shape[1]
    prefix, patches = x.split([x.shape[2] - n_patches, n_patches], dim=2)
    return torch.cat([prefix, rope_rotate(patches, grid.coords.flatten(0, 1), grid.periods)], dim=2)


def _sum_pool(x: Tensor) -> Tensor:
    """Sums 2x2 blocks of x [..., H, W, c]."""
    H, W = x.shape[-3:-1]
    return x.unflatten(-3, (H // 2, 2)).unflatten(-2, (W // 2, 2)).sum((-4, -2))


def _pad_to(side: int, H: int, W: int) -> tuple[int, ...]:
    """F.pad widths that pad [..., H, W, c] up to multiples of side."""
    return (0, 0, 0, -W % side, 0, -H % side)


def pyramid_layout(grid: RopeGrid, levels: int) -> Layout:
    """The quadtree over grid's patches, `levels` levels deep."""
    H, W = grid.coords.shape[:2]
    ones = torch.ones_like(grid.coords[..., :1])
    geo_sums = [F.pad(torch.cat([grid.coords, ones], dim=-1), _pad_to(2**levels, H, W))]
    for _ in range(levels):
        geo_sums.append(_sum_pool(geo_sums[-1]))  # coordinate sums and patch counts
    geo = F.pad(torch.cat([s.flatten(0, 1) for s in geo_sums]), (0, 0, 0, 1))  # plus the null cell
    counts = geo[:, 2]
    coords = geo[:, :2] / counts.clamp(min=1)[:, None]

    shapes = [tuple(s.shape[:2]) for s in geo_sums]
    null = len(counts) - 1
    flat_ids = torch.arange(null, device=counts.device).split([h * w for h, w in shapes])
    level = F.pad(torch.cat([torch.full_like(ids, lvl) for lvl, ids in enumerate(flat_ids)]), (0, 1), value=-1)
    grids = [ids.view(shape) for ids, shape in zip(flat_ids, shapes)]
    # The children of each level-l cell are a 2x2 block of the level l-1 grid.
    quadrants = [g.unflatten(0, (-1, 2)).unflatten(2, (-1, 2)).transpose(1, 2).reshape(-1, 4) for g in grids[:-1]]
    leaf_children = F.pad(grids[0].reshape(-1, 1), (0, 3), value=null)
    return Layout(
        counts=counts,
        level=level,
        children=torch.cat([leaf_children, *quadrants, torch.full_like(leaf_children[:1], null)]),
        roots=flat_ids[-1],
        shapes=shapes,
        cell_rope=rope_sincos(coords, grid.periods),
        patch_rope=rope_sincos(grid.coords.flatten(0, 1), grid.periods),
    )


def cell_sums(x: Tensor, levels: int) -> Tensor:
    """Sums per-patch features x [..., H, W, c] over every cell of a `levels`-deep pyramid, giving [..., M, c] in
    Layout order: fine cells first, the null cell last."""
    H, W = x.shape[-3:-1]
    sums = [F.pad(x, _pad_to(2**levels, H, W))]
    for _ in range(levels):
        sums.append(_sum_pool(sums[-1]))
    return F.pad(torch.cat([s.flatten(-3, -2) for s in sums], dim=-2), (0, 0, 0, 1))


def _pool_features(
    q: Tensor, k: Tensor, v: Tensor, layout: Layout, levels: int, score_level: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Scoring queries pooled to score_level, plus keys, values and value spread pooled to every level, from patch
    q, k, v [B, h, H, W, d]. Queries and keys come out rotated at their cells' centroids."""
    H, W, d = k.shape[2], k.shape[3], k.shape[-1]
    pad = _pad_to(2**levels, H, W)
    divisor = layout.counts.clamp(min=1)[:, None].to(k.dtype)
    flat = cell_sums(torch.cat([k, v, v.square().sum(-1, keepdim=True)], dim=-1), levels)
    keys, values, value_sq = (flat / divisor).split([d, d, 1], dim=-1)
    spread = (value_sq[..., 0] - values.square().sum(-1)).clamp(min=0).sqrt()

    queries = F.pad(q, pad)
    for _ in range(score_level):
        queries = _sum_pool(queries)
    score_cells = layout.level_cells(score_level)
    queries = queries.flatten(-3, -2) / divisor[score_cells]
    score_rope = (layout.cell_rope[0][score_cells], layout.cell_rope[1][score_cells])
    return _rotate(queries, score_rope), _rotate(keys, layout.cell_rope), values, spread


_pool_features_compiled = torch.compile(_pool_features)


def build_pyramid(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    grid: RopeGrid,
    levels: int,
    score_level: int | None = None,
    compile: bool = False,
) -> Pyramid:
    """Mean-pools patch keys and values [B, h, H, W, d] into a quadtree `levels` levels deep, and queries to
    `score_level` (default: `levels`). Uses grid's precomputed layout when it has one."""
    score_level = levels if score_level is None else score_level
    layout = grid.layout if grid.layout is not None else pyramid_layout(grid, levels)
    assert len(layout.shapes) == levels + 1, "the grid's layout was built for a different start level"
    pool = _pool_features_compiled if compile else _pool_features
    queries, keys, values, spread = pool(q, k, v, layout, levels, score_level)
    return Pyramid(queries, layout.counts[layout.level_cells(score_level)], keys, values, spread, layout)


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
    """Indices [B, h, budget - prefix] of the cells each head attends to; together they tile the image exactly once,
    and any slots left over hold null cells, so with the CLS and register tokens every head reads exactly `budget`
    keys. Uniform order returns just the start cells instead.

    Each round, `queries` [B, h, Q, d], each standing in for `weights` [Q] fine queries, attend to the CLS and
    register tokens plus the current cells, which are then scored (see FoveationConfig.score) and the best split.
    """
    n_slots = config.budget - prefix_keys.shape[2]
    null = len(pyramid.counts) - 1

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
    if config.order == "uniform":
        return roots
    if config.order == "tree":
        tree_cells = _tree_cells_compiled if config.compile else _tree_cells
        return tree_cells(pyramid, queries, weights, prefix_keys, config.score, n_slots)
    if config.order == "global":
        cells = _select_global(pyramid, roots, scores, n_slots, config.rounds)
    else:
        cells = _select_per_level(pyramid, roots, scores, n_slots)
    return F.pad(cells, (0, n_slots - cells.shape[-1]), value=null)


def _select_per_level(pyramid: Pyramid, roots: Tensor, scores, n_slots: int) -> Tensor:
    """Coarsest level first, so cells only compete with cells of the same size: at each level the frontier of cells
    the level above created is scored alongside the settled cells, its best cells split into the next frontier, and
    the rest settle."""
    settled, frontier = roots[..., :0], roots
    for splits in level_schedule(roots.shape[-1], n_slots, len(pyramid.shapes) - 1):
        if not splits:
            break
        frontier_scores = scores(torch.cat([settled, frontier], dim=-1))[..., settled.shape[-1] :]
        ranked = frontier.gather(-1, frontier_scores.argsort(dim=-1, descending=True))
        settled = torch.cat([settled, ranked[..., splits:]], dim=-1)
        frontier = pyramid.children[ranked[..., :splits]].flatten(-2)
    return torch.cat([settled, frontier], dim=-1)


def _select_global(pyramid: Pyramid, roots: Tensor, scores, n_slots: int, rounds: int) -> Tensor:
    """Every round ranks all splittable cells together by score per patch, so cells of different sizes compete on
    equal terms. If a round runs out of splittable cells, it splits fine cells into themselves plus null cells."""
    cells = roots
    for splits in round_schedule(roots.shape[-1], n_slots, rounds):
        if not splits:
            break
        per_patch = scores(cells) / pyramid.counts[cells].clamp(min=1)
        per_patch = per_patch.masked_fill(pyramid.level[cells] <= 0, -math.inf)
        picked = per_patch.topk(splits).indices
        quads = pyramid.children[cells.gather(-1, picked)]
        cells = torch.cat([cells.scatter(-1, picked, quads[..., 0]), quads[..., 1:].flatten(-2)], dim=-1)
    return cells


def cell_errors(
    pyramid: Pyramid, queries: Tensor, weights: Tensor, prefix_keys: Tensor, score: str, chunk_scores: int = 2**25
) -> Tensor:
    """Estimated output error [B, h, M] from leaving each cell pooled, for every cell at once.

    The scoring queries attend once to the CLS and register tokens plus every fine key; each patch's attention (and
    its query-norm-weighted attention) is then summed up the pyramid, so a cell's mass is the sum of its children's.
    Queries go in chunks of about `chunk_scores` attention scores, to bound memory.
    """
    layout, d, n_prefix = pyramid.layout, queries.shape[-1], prefix_keys.shape[2]
    fine = layout.level_cells(0)
    fine_keys = pyramid.keys[:, :, fine]  # rotated; padding patches are masked by their -inf bias
    keys = torch.cat([prefix_keys, fine_keys], dim=2)
    # In the queries' precision: a float32 bias would promote every [chunk, N] score matrix to float32
    bias = F.pad(layout.counts[fine].log(), (n_prefix, 0)).to(queries.dtype)
    stats = torch.stack([weights.expand(queries.shape[:3]), weights * queries.norm(dim=-1)], dim=-1)  # [B, h, Q, 2]
    per_patch = 0
    chunk = max(1, chunk_scores // (queries.shape[0] * queries.shape[1] * keys.shape[2]))
    for rows in torch.arange(queries.shape[2], device=queries.device).split(chunk):
        probs = (queries[:, :, rows] @ keys.mT * d**-0.5 + bias).softmax(-1)[..., n_prefix:]
        per_patch = per_patch + (probs.mT @ stats[:, :, rows].to(probs.dtype)).float()  # [B, h, N', 2]
    features = torch.cat([per_patch, fine_keys, fine_keys.square().sum(-1, keepdim=True)], dim=-1)
    mass, q_mass, k_sum, k_sq = cell_sums(features.unflatten(2, layout.shapes[0]), len(layout.shapes) - 1).split(
        [1, 1, d, 1], dim=-1
    )
    n = layout.counts.clamp(min=1)[:, None]
    key_spread = (k_sq / n - (k_sum / n).square().sum(-1, keepdim=True)).clamp(min=0).sqrt()[..., 0]
    errors = dict(
        attention=lambda: mass[..., 0],
        detail=lambda: mass[..., 0] * pyramid.spread,
        bound=lambda: q_mass[..., 0] * key_spread * pyramid.spread * d**-0.5,
        random=lambda: torch.rand_like(mass[..., 0]),
    )
    return errors[score]()


def _quad_sum(x: Tensor) -> Tensor:
    """Sums 2x2 blocks of x [..., H, W]."""
    return _sum_pool(x[..., None])[..., 0]


def _upsample(x: Tensor) -> Tensor:
    """Repeats each element of x [..., H, W] over a 2x2 block."""
    return x.repeat_interleave(2, -2).repeat_interleave(2, -1)


def _blocks(x: Tensor, side: int) -> Tensor:
    """The side x side blocks of x [..., H, W], as [..., H / side, W / side, side**2]."""
    H, W = x.shape[-2:]
    return x.unflatten(-2, (H // side, side)).unflatten(-1, (W // side, side)).transpose(-3, -2).flatten(-2)


def _ratio(gain: Tensor, added: Tensor) -> Tensor:
    """Error removed per key added. A split that adds no keys is worth any price if it removes error."""
    free = torch.where(gain >= 0, math.inf, -math.inf)
    return torch.where(added > 0, gain / added.clamp(min=1), free)


def split_thresholds(layout: Layout, errors: Tensor) -> tuple[list[Tensor], list[Tensor]]:
    """For levels 1 and up: the price per key [B, h, H_l, W_l] below which the cheapest tiling splits each cell, and
    the keys [H_l, W_l] each split adds (one per real child, minus one). Index 0, the fine level, is None.

    This is cost-complexity pruning (Breiman et al., 1984). At price λ the cheapest tiling splits a cell when its
    error plus λ exceeds the cheapest tiling of its children. Every split adds keys, so that gap shrinks as λ grows
    and each cell has one threshold λ*, below which it splits: the tilings for decreasing λ are nested. λ* is the
    best ratio of error removed to keys added over the nested tilings of the cell's own subtree, built bottom-up
    from its descendants' thresholds. A cell is only reached if its ancestors split, so its effective threshold is
    the running minimum of λ* down from its start cell.
    """
    levels = len(layout.shapes) - 1
    real = [(layout.counts[layout.level_cells(level)] > 0).view(layout.shapes[level]) for level in range(levels + 1)]
    error = [errors[..., layout.level_cells(level)].unflatten(-1, layout.shapes[level]) for level in range(levels + 1)]
    gain, added, threshold = [None], [None], [None]
    for level in range(1, levels + 1):
        gain.append((error[level] - _quad_sum(error[level - 1] * real[level - 1])) * real[level])
        added.append(((_quad_sum(real[level - 1].to(errors.dtype)) - 1) * real[level]))
        best = _ratio(gain[level], added[level])
        # The cell's internal descendants, coarse first, each with the running minimum of thresholds below the cell.
        # Sorted by that, parents come before children (stable sort on ties), so every prefix is a nested tiling.
        reach, below = None, []
        for lower in range(level - 1, 0, -1):
            reach = threshold[lower] if reach is None else torch.minimum(threshold[lower], _upsample(reach))
            side = 2 ** (level - lower)
            below.append([_blocks(t.expand_as(reach), side) for t in (reach, gain[lower], added[lower])])
        if below:
            reach, gains, keys = (torch.cat(parts, dim=-1) for parts in zip(*below))
            order = reach.argsort(dim=-1, descending=True, stable=True)
            gains = gain[level][..., None] + gains.gather(-1, order).cumsum(-1)
            keys = added[level][..., None] + keys.gather(-1, order).cumsum(-1)
            best = torch.maximum(best, _ratio(gains, keys).amax(-1))
        threshold.append(best.masked_fill(~real[level], -math.inf))
    for level in range(levels - 1, 0, -1):
        threshold[level] = torch.minimum(threshold[level], _upsample(threshold[level + 1]))
    return threshold, added


def _select_tree(pyramid: Pyramid, errors: Tensor, n_slots: int) -> Tensor:
    """Equal-slope allocation in one pass. Sorting every cell by its split threshold (see split_thresholds), parents
    first on ties, lists the splits so that every prefix is a valid tiling, and each head takes the longest prefix
    whose keys fit its slots. A split adds one key per real child minus one, so at most two slots stay empty; they
    hold null cells, so shapes stay static."""
    layout = pyramid.layout
    levels, null = len(layout.shapes) - 1, len(layout.counts) - 1
    _spare_splits(len(layout.roots), n_slots)  # checks the start cells fit
    threshold, added = split_thresholds(layout, errors)
    coarse_first = range(levels, 0, -1)
    ranked = torch.cat([threshold[level].flatten(-2) for level in coarse_first], dim=-1)
    ranked = ranked.argsort(dim=-1, descending=True, stable=True)
    keys = torch.cat([added[level].flatten() for level in coarse_first])
    take = keys[ranked].cumsum(-1) <= n_slots - len(layout.roots)
    split = torch.zeros_like(take).scatter(-1, ranked, take)
    sizes = [layout.shapes[level][0] * layout.shapes[level][1] for level in coarse_first]
    split = dict(zip(coarse_first, split.split(sizes, dim=-1)))

    real = [(layout.counts[layout.level_cells(level)] > 0).view(layout.shapes[level]) for level in range(levels + 1)]
    chosen, reached = [], real[levels].expand(*errors.shape[:2], -1, -1)
    for level in coarse_first:
        splits = split[level].unflatten(-1, layout.shapes[level])
        chosen.append(reached & ~splits)
        reached = _upsample(reached & splits) & real[level - 1]
    chosen.append(reached)
    chosen = F.pad(torch.cat([c.flatten(-2) for c in reversed(chosen)], dim=-1), (0, 1))  # Layout order, null last
    order = (~chosen).to(torch.int8).argsort(dim=-1, stable=True)[..., :n_slots]  # chosen cells first
    used = torch.arange(n_slots, device=errors.device) < chosen.sum(-1, keepdim=True)
    return torch.where(used, order, null)


def _tree_cells(
    pyramid: Pyramid, queries: Tensor, weights: Tensor, prefix_keys: Tensor, score: str, n_slots: int
) -> Tensor:
    """Tree order's scoring pass and selection, in one function so torch.compile can fuse their small kernels."""
    return _select_tree(pyramid, cell_errors(pyramid, queries, weights, prefix_keys, score), n_slots)


_tree_cells_compiled = torch.compile(_tree_cells)


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
        with record_function("foveation.pyramid"):
            pyramid = build_pyramid(
                *(t.unflatten(2, (H, W)) for t in (q_patches, k_patches, v_patches)),
                grid,
                self.config.start_level,
                self.config.score_level,
                self.config.compile,
            )
        with record_function("foveation.select"):
            cells = select_cells(
                pyramid,
                queries=torch.cat([q_prefix, pyramid.queries], dim=2),
                weights=F.pad(pyramid.query_counts, (n_prefix, 0), value=1.0),
                prefix_keys=k_prefix,
                config=self.config,
            )
        with record_function("foveation.attend"):
            keys = torch.cat([k_prefix, gather_cells(pyramid.keys, cells)], dim=2)
            values = torch.cat([v[:, :, :n_prefix], gather_cells(pyramid.values, cells)], dim=2)
            # A cell covering m patches stands in for m keys, so it gets log(m) added to its logits.
            log_counts = F.pad(pyramid.counts.log()[cells], (n_prefix, 0))
            queries = torch.cat([q_prefix, _rotate(q_patches, pyramid.layout.patch_rope)], dim=2)
            out = F.scaled_dot_product_attention(queries, keys, values, attn_mask=log_counts[:, :, None].to(q.dtype))
        return out, cells, pyramid


def _rope_grid(rope: RopePositionEmbedding, args, kwargs, output, levels: int) -> RopeGrid:
    grid = RopeGrid(patch_coords(rope, kwargs["H"], kwargs["W"]), rope.periods)
    return grid._replace(layout=pyramid_layout(grid, levels))


def foveate(model: DinoVisionTransformer, config: FoveationConfig) -> DinoVisionTransformer:
    """Swaps every block's attention for FoveatedAttention, in place, sharing the original parameters."""
    if any(block.sample_drop_ratio > 0 for block in model.blocks):
        # Stochastic depth calls Attention.forward_list and indexes RoPE sin/cos, neither of which exists here.
        raise ValueError("Build the model with drop_path_rate=0 before foveating it")
    for block in model.blocks:
        block.attn = FoveatedAttention(block.attn, config)
    model.rope_embed.register_forward_hook(partial(_rope_grid, levels=config.start_level), with_kwargs=True)
    return model
