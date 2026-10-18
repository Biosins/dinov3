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
    score: Literal["attention", "detail", "bound", "variance", "random"] = "detail"
    """attention: the attention mass a cell receives. detail: mass times the spread of the cell's patch values, a
    rough estimate of the output error from leaving it unsplit. bound: query norm x mass x key spread x value spread
    / sqrt(d), a bound on the first-order output error (tree order only). variance: mass times the scalar variance of
    the cell's values (their mean squared distance from the cell mean). random: a baseline with the same budget."""
    scoring: Literal["full", "beam"] = "full"
    """How tree order estimates each cell's attention mass. full: the scoring queries attend once to every fine key.
    beam: top-down rounds against pooled cells, expanding only the cells that could matter (see beam_errors); needs
    score="variance"."""
    rounds: int = 3
    """Scoring rounds for global order."""
    compile: Literal["static", "dynamic"] | None = None
    """Compile the pyramid pooling and tree selection with torch.compile. static: one compile per image size, the
    fastest kernels (torch falls back to eager after 8 sizes). dynamic: dims that grow with the image are marked
    dynamic, so one compile serves every size, about 20% slower on ViT-B."""

    def __post_init__(self):
        if not 0 <= self.score_level <= self.start_level:
            raise ValueError("score_level must lie between 0 and start_level")
        if self.score == "bound" and self.order in ("per_level", "global"):
            raise ValueError('score="bound" needs the one-pass scores of order="tree"')
        if self.scoring == "beam" and (self.order != "tree" or self.score != "variance"):
            raise ValueError('scoring="beam" is for order="tree" with score="variance"')


class Layout(NamedTuple):
    """Everything about the pyramid that depends only on the patch grid, so all layers share it: all cells from fine
    (level 0) up to the start level in one flat index space, plus a trailing empty null cell.

    Padding cells and the null cell cover no patches, so their log-count bias is -inf and they get no attention.
    """

    counts: Tensor  # [M] patches each cell covers
    level: Tensor  # [M], -1 for the null cell
    children: Tensor  # [M, 4] a cell's quadrants; splitting a fine cell keeps it and adds three null cells
    roots: Tensor  # [n0] the start-level cells
    grids: list[Tensor]  # [H_l, W_l] cell indices of each level, fine first; level l cells are 2**l patches wide
    cell_rope: tuple[Tensor, Tensor]  # RoPE sin and cos [M, d] at each cell's centroid
    patch_rope: tuple[Tensor, Tensor]  # RoPE sin and cos [H * W, d] at each patch centre

    @property
    def shapes(self) -> list[torch.Size]:
        """Grid size of each level. Read from tensors, so torch.compile can keep them symbolic."""
        return [grid.shape for grid in self.grids]

    def level_cells(self, level: int) -> slice:
        start = sum(grid.numel() for grid in self.grids[:level])
        return slice(start, start + self.grids[level].numel())


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
        grids=grids,
        cell_rope=rope_sincos(coords, grid.periods),
        patch_rope=rope_sincos(grid.coords.flatten(0, 1), grid.periods),
    )


def _image_sized(x: Tensor, *dims: int) -> Tensor:
    """Marks dims that grow with the image as dynamic for compile="dynamic", so one compiled graph serves every
    image size while the budget, heads and head width stay static."""
    for dim in dims:
        torch._dynamo.maybe_mark_dynamic(x, dim)
    return x


def _image_sized_layout(layout: Layout) -> Layout:
    """Marks every cell and patch dim of the layout dynamic (see _image_sized)."""
    for x in (layout.counts, layout.level, layout.children, layout.roots, *layout.cell_rope, *layout.patch_rope):
        _image_sized(x, 0)
    for grid in layout.grids:
        _image_sized(grid, 0, 1)
    return layout


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


_pool_features_compiled = dict(
    static=torch.compile(_pool_features, dynamic=False), dynamic=torch.compile(_pool_features)
)


def build_pyramid(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    grid: RopeGrid,
    levels: int,
    score_level: int | None = None,
    compile: Literal["static", "dynamic"] | None = None,
) -> Pyramid:
    """Mean-pools patch keys and values [B, h, H, W, d] into a quadtree `levels` levels deep, and queries to
    `score_level` (default: `levels`). Uses grid's precomputed layout when it has one."""
    score_level = levels if score_level is None else score_level
    layout = grid.layout if grid.layout is not None else pyramid_layout(grid, levels)
    assert len(layout.shapes) == levels + 1, "the grid's layout was built for a different start level"
    if compile == "dynamic":
        q, k, v = (_image_sized(t, 2, 3) for t in (q, k, v))
        layout = _image_sized_layout(layout)
    pool = _pool_features_compiled[compile] if compile else _pool_features
    queries, keys, values, spread = pool(q, k, v, layout, levels, score_level)
    query_counts = layout.counts[layout.level_cells(score_level)]
    if compile == "dynamic":  # these feed the compiled tree selection
        queries, keys, values, spread = (_image_sized(t, 2) for t in (queries, keys, values, spread))
        query_counts = _image_sized(query_counts, 0)
    return Pyramid(queries, query_counts, keys, values, spread, layout)


def start_cells(H: int, W: int, level: int) -> int:
    """Start cells, 2**level patches wide, covering an H x W patch grid."""
    side = 2**level
    return -(-H // side) * -(-W // side)


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
        if config.score == "variance":
            return mass * pyramid.spread.gather(-1, cells).square()
        if config.score == "detail":
            return mass * pyramid.spread.gather(-1, cells)
        return mass

    roots = pyramid.roots.expand(*queries.shape[:2], -1)
    if config.order == "uniform":
        return roots
    if config.order == "tree":
        if config.compile is None:
            return _tree_cells(pyramid, queries, weights, prefix_keys, config, n_slots)
        if config.compile == "dynamic":
            queries, weights = _image_sized(queries, 2), _image_sized(weights, 0)
        # prefix_keys is a view into all keys, so its strides change with the image size: a copy keeps them fixed
        return _tree_cells_compiled[config.compile](
            pyramid, queries, weights, prefix_keys.contiguous(), config, n_slots
        )
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
    pyramid: Pyramid, queries: Tensor, weights: Tensor, prefix_keys: Tensor, score: str, chunks: int = 4
) -> Tensor:
    """Estimated output error [B, h, M] from leaving each cell pooled, for every cell at once.

    The scoring queries attend once to the CLS and register tokens plus every fine key; each patch's attention (and
    its query-norm-weighted attention) is then summed up the pyramid, so a cell's mass is the sum of its children's.
    Queries go in a fixed number of chunks, which bounds memory and keeps the loop the same for every image size.
    """
    layout, d, n_prefix = pyramid.layout, queries.shape[-1], prefix_keys.shape[2]
    fine = layout.level_cells(0)
    keys = torch.cat([prefix_keys, pyramid.keys[:, :, fine]], dim=2)  # padding patches get no mass: -inf bias
    # In the queries' precision: a float32 bias would promote every [chunk, N] score matrix to float32
    bias = F.pad(layout.counts[fine].log(), (n_prefix, 0)).to(queries.dtype)
    stats = torch.stack([weights.expand(queries.shape[:3]), weights * queries.norm(dim=-1)], dim=-1)  # [B, h, Q, 2]
    # Padding to a multiple of the chunks, with queries of zero weight, keeps every chunk the same size
    pad = (0, 0, 0, -queries.shape[2] % chunks)
    queries, stats = (F.pad(t, pad).unflatten(2, (chunks, -1)).unbind(2) for t in (queries, stats))
    per_patch = 0
    for chunk, chunk_stats in zip(queries, stats):
        probs = (chunk @ keys.mT * d**-0.5 + bias).softmax(-1)[..., n_prefix:]
        per_patch = per_patch + (probs.mT @ chunk_stats.to(probs.dtype)).float()  # [B, h, N', 2]
    mass, q_mass = cell_sums(per_patch.unflatten(2, layout.shapes[0]), len(layout.shapes) - 1).unbind(-1)
    errors = dict(
        attention=lambda: mass,
        detail=lambda: mass * pyramid.spread,
        variance=lambda: mass * pyramid.spread.square(),
        bound=lambda: q_mass * key_spread(pyramid) * pyramid.spread * d**-0.5,
        random=lambda: torch.rand_like(mass),
    )
    return errors[score]()


def key_spread(pyramid: Pyramid) -> Tensor:
    """RMS distance [B, h, M] of each cell's rotated patch keys from their mean. In float32: key means are large next
    to their spread, so bfloat16 sums would cancel."""
    layout = pyramid.layout
    keys = pyramid.keys[:, :, layout.level_cells(0)].float()
    sums = cell_sums(
        torch.cat([keys, keys.square().sum(-1, keepdim=True)], -1).unflatten(2, layout.shapes[0]),
        len(layout.shapes) - 1,
    )
    mean, square = (sums / layout.counts.clamp(min=1)[:, None]).split([keys.shape[-1], 1], dim=-1)
    return (square[..., 0] - mean.square().sum(-1)).clamp(min=0).sqrt()


def beam_errors(
    pyramid: Pyramid, queries: Tensor, weights: Tensor, prefix_keys: Tensor, n_slots: int
) -> tuple[Tensor, Tensor]:
    """Estimated errors [B, h, M], mass times value variance, for the cells a top-down beam reaches, and which cells
    may split [B, h, M], without attending to every fine key.

    Top-down, round by round from the start cells, the scoring queries attend to the CLS and register tokens plus
    the current cells, each with the log-count bias. A cell's error bounds what any splitting beneath it can
    recover, since its variance is over all its patches (law of total variance). So each round expands the cells
    with the largest errors, as many as a head can split in total, (n_slots - start cells) // 3, and their children
    are scored next round. Expanded cells may split, and so may 32px cells, whose fine children are exact.

    Then bottom-up, each expanded cell's mass becomes the sum of its children's, as in full scoring. A split's gain
    is a small difference between a cell's error and its children's, so masses estimated in different rounds, each
    with its own error, would swamp it.
    """
    layout, d, n_prefix = pyramid.layout, queries.shape[-1], prefix_keys.shape[2]
    levels, null = len(layout.shapes) - 1, len(layout.counts) - 1
    beam = (n_slots - len(layout.roots)) // 3
    variance = pyramid.spread.float().square()
    masses = torch.zeros_like(variance)
    splittable = (layout.level == 1).expand_as(masses).clone()
    cells = pyramid.roots.expand(*queries.shape[:2], -1)
    for level in range(levels, 0, -1):
        keys = torch.cat([prefix_keys, gather_cells(pyramid.keys, cells)], dim=2)
        bias = F.pad(layout.counts.log()[cells], (n_prefix, 0))[:, :, None]
        probs = (queries @ keys.mT * d**-0.5 + bias).softmax(-1)
        mass = (weights[:, None] * probs).sum(-2)[..., n_prefix:]
        masses = masses.scatter(-1, cells, mass)
        if level == 1:
            break
        candidates = (mass * variance.gather(-1, cells)).masked_fill(layout.level[cells] != level, -math.inf)
        best = candidates.topk(min(beam, cells.shape[-1]), dim=-1)
        picked = cells.gather(-1, best.indices)
        expand = best.values.isfinite()
        parents = torch.where(expand, picked, null)
        splittable = splittable.scatter(-1, parents, True)
        quads = layout.children[parents]  # all null for the null cell
        quads[..., 0] = torch.where(expand, quads[..., 0], picked)  # so picks that are not expanded stay put
        cells = torch.cat([cells.scatter(-1, best.indices, quads[..., 0]), quads[..., 1:].flatten(-2)], dim=-1)
    for level in range(2, levels + 1):
        summed = masses[..., layout.children].sum(-1)
        masses = torch.where(splittable & (layout.level == level), summed, masses)
    errors = masses * variance
    errors[..., -1], splittable[..., -1] = 0, False  # the null cell; -1 keeps the index static
    return errors, splittable


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


def split_thresholds(
    layout: Layout, errors: Tensor, splittable: Tensor | None = None
) -> tuple[list[Tensor], list[Tensor]]:
    """For levels 1 and up: the price per key [B, h, H_l, W_l] below which the cheapest tiling splits each cell, and
    the keys each split adds (one per real child, minus one). Index 0, the fine level, is None. Cells outside
    `splittable` [B, h, M] (default: all real cells) never split: their threshold is -inf and they add nothing.

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
        can_split = real[level]
        if splittable is not None:
            can_split = can_split & splittable[..., layout.level_cells(level)].unflatten(-1, layout.shapes[level])
        gain.append((error[level] - _quad_sum(error[level - 1] * real[level - 1])) * can_split)
        added.append(((_quad_sum(real[level - 1].to(errors.dtype)) - 1) * can_split))
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
        threshold.append(best.masked_fill(~can_split, -math.inf))
    for level in range(levels - 1, 0, -1):
        threshold[level] = torch.minimum(threshold[level], _upsample(threshold[level + 1]))
    return threshold, added


def _select_tree(pyramid: Pyramid, errors: Tensor, n_slots: int, splittable: Tensor | None = None) -> Tensor:
    """Equal-slope allocation in one pass. Sorting every cell by its split threshold (see split_thresholds), parents
    first on ties, lists the splits so that every prefix is a valid tiling, and each head takes the longest prefix
    whose keys fit its slots, short of cells that never split. A split adds one key per real child minus one, so at
    most two slots stay empty while splits remain; empty slots hold null cells, so shapes stay static."""
    layout = pyramid.layout
    levels, null = len(layout.shapes) - 1, len(layout.counts) - 1
    _spare_splits(len(layout.roots), n_slots)  # checks the start cells fit
    threshold, added = split_thresholds(layout, errors, splittable)
    coarse_first = range(levels, 0, -1)
    thresholds = torch.cat([threshold[level].flatten(-2) for level in coarse_first], dim=-1)
    ranked = thresholds.argsort(dim=-1, descending=True, stable=True)
    keys = torch.cat([added[level].expand_as(threshold[level]).flatten(-2) for level in coarse_first], dim=-1)
    take = keys.gather(-1, ranked).cumsum(-1) <= n_slots - len(layout.roots)
    take &= thresholds.gather(-1, ranked) > -math.inf
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
    pyramid: Pyramid, queries: Tensor, weights: Tensor, prefix_keys: Tensor, config: FoveationConfig, n_slots: int
) -> Tensor:
    """Tree order's scoring and selection, in one function so torch.compile can fuse their small kernels."""
    if config.scoring == "beam":
        errors, splittable = beam_errors(pyramid, queries, weights, prefix_keys, n_slots)
        return _select_tree(pyramid, errors, n_slots, splittable)
    return _select_tree(pyramid, cell_errors(pyramid, queries, weights, prefix_keys, config.score), n_slots)


_tree_cells_compiled = dict(static=torch.compile(_tree_cells, dynamic=False), dynamic=torch.compile(_tree_cells))


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


def _rope_grid(rope: RopePositionEmbedding, args, kwargs, output, model: DinoVisionTransformer) -> RopeGrid:
    grid = RopeGrid(patch_coords(rope, kwargs["H"], kwargs["W"]), rope.periods)
    return grid._replace(layout=pyramid_layout(grid, model.blocks[0].attn.config.start_level))


def foveate(model: DinoVisionTransformer, config: FoveationConfig) -> DinoVisionTransformer:
    """Swaps every block's attention for FoveatedAttention, in place, sharing the original parameters."""
    if any(block.sample_drop_ratio > 0 for block in model.blocks):
        # Stochastic depth calls Attention.forward_list and indexes RoPE sin/cos, neither of which exists here.
        raise ValueError("Build the model with drop_path_rate=0 before foveating it")
    for block in model.blocks:
        block.attn = FoveatedAttention(block.attn, config)
    model.rope_embed.register_forward_hook(partial(_rope_grid, model=model), with_kwargs=True)
    return model


def reconfigure(model: nn.Module, config: FoveationConfig) -> None:
    """Switches every FoveatedAttention in a foveated model, or a wrapper around one, to config, e.g. to train with
    a different budget for each image."""
    for module in model.modules():
        if isinstance(module, FoveatedAttention):
            module.config = config
