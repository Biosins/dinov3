import copy

import pytest
import torch

from dinov3.foveation import FoveatedAttention, FoveationConfig, RopeGrid, foveate, patch_coords
from dinov3.foveation.attention import build_pyramid, level_schedule, rope_rotate, round_schedule
from dinov3.foveation.phase0 import spread_to_patches
from dinov3.hub.backbones import dinov3_vits16
from dinov3.layers.attention import SelfAttention, rope_apply
from dinov3.layers.rope_position_encoding import RopePositionEmbedding

DIM, HEADS, PREFIX = 96, 4, 5


def padded_patches(H: int, W: int, level: int) -> int:
    side = 2**level
    return -(-H // side) * side * -(-W // side) * side


def sharpen(attn: SelfAttention) -> None:
    """Random-init weights give near-uniform attention, under which wrong positions or keys go unnoticed."""
    torch.nn.init.normal_(attn.qkv.weight, std=0.15)
    torch.nn.init.normal_(attn.qkv.bias, std=0.5)


def leaf_patches(cells: list[int], pyramid) -> list[int]:
    """The fine patches under the given cells, padding excluded."""
    leaves = []
    while cells:
        cell = cells.pop()
        if pyramid.level[cell] > 0:
            cells.extend(pyramid.children[cell].tolist())
        elif pyramid.counts[cell] > 0:
            leaves.append(cell)
    return leaves


@pytest.fixture
def layer() -> tuple[SelfAttention, RopePositionEmbedding]:
    torch.manual_seed(0)
    attn = SelfAttention(DIM, num_heads=HEADS, qkv_bias=True)
    sharpen(attn)
    return attn.eval(), RopePositionEmbedding(DIM, num_heads=HEADS, base=100.0, dtype=torch.float32)


@pytest.mark.parametrize("normalize", ["min", "max", "separate"])
def test_rope_rotate_matches_rope_embedding(normalize):
    rope = RopePositionEmbedding(DIM, num_heads=HEADS, base=100.0, normalize_coords=normalize, dtype=torch.float32)
    x = torch.randn(2, HEADS, 6 * 10, DIM // HEADS)
    expected = rope_apply(x, *rope(H=6, W=10))
    torch.testing.assert_close(rope_rotate(x, patch_coords(rope, 6, 10).flatten(0, 1), rope.periods), expected)


@pytest.mark.parametrize("order", ["per_level", "global", "tree"])
@pytest.mark.parametrize("H, W", [(8, 8), (6, 10)])  # 6x10 exercises the padding
def test_full_budget_reproduces_dense_layer(layer, H, W, order):
    attn, rope = layer
    config = FoveationConfig(budget=padded_patches(H, W, 2) + PREFIX, start_level=2, score_level=1, order=order)
    x = torch.randn(2, PREFIX + H * W, DIM)
    expected = attn(x, rope=rope(H=H, W=W))
    actual = FoveatedAttention(attn, config)(x, rope=RopeGrid(patch_coords(rope, H, W), rope.periods))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_log_count_bonus_makes_uniform_cells_exact(layer):
    """When keys and values are constant within each start cell, a cell covering m patches is the same as m
    identical keys, so attending to the unsplit cells with the log-count bonus must equal dense attention."""
    attn, rope = layer
    H, W, B, d = 6, 10, 2, DIM // HEADS
    no_rotation = RopeGrid(patch_coords(rope, H, W), torch.full_like(rope.periods, torch.inf))
    q, k, v = torch.randn(3, B, HEADS, PREFIX + H * W, d)
    cell_k, cell_v = torch.randn(2, B, HEADS, 2, 3, d)  # 4x4 cells, the last row and column clipped by the image
    k[:, :, PREFIX:], v[:, :, PREFIX:] = (
        t.repeat_interleave(4, 2).repeat_interleave(4, 3)[:, :, :H, :W].flatten(2, 3) for t in (cell_k, cell_v)
    )
    fov = FoveatedAttention(attn, FoveationConfig(budget=6 + PREFIX, start_level=2))
    out, cells, _ = fov.attend(q, k, v, no_rotation)
    assert cells.shape[-1] == 6, "the budget allows no splits"
    torch.testing.assert_close(out, torch.nn.functional.scaled_dot_product_attention(q, k, v))


@pytest.mark.parametrize("order", ["per_level", "global", "tree"])
@pytest.mark.parametrize("score", ["attention", "detail", "bound", "random"])
@pytest.mark.parametrize("budget", [17, 35, 66])
def test_layout_tiles_image_within_budget(layer, budget, score, order):
    if score == "bound" and order != "tree":
        pytest.skip("the bound needs tree order's one-pass scores")
    attn, rope = layer
    H, W = 10, 13
    fov = FoveatedAttention(attn, FoveationConfig(budget=budget, start_level=2, score=score, order=order))
    x = torch.randn(2, PREFIX + H * W, DIM, requires_grad=True)
    out, cells, pyramid = fov.attend(*fov.heads(x), RopeGrid(patch_coords(rope, H, W), rope.periods))

    roots, slots = 12, budget - PREFIX  # 10x13 patches give 3x4 start cells
    assert cells.shape[-1] == slots, "every head reads exactly `budget` keys"
    real_cells = (pyramid.counts[cells] > 0).sum(-1)
    if order == "tree":  # fills its slots, up to two left over, while splits remain
        assert (real_cells >= slots - 2).all() and (real_cells <= slots).all()
    else:  # splits by quadrant, so edge splits leave padding cells in their slots
        assert (real_cells <= roots + 3 * ((slots - roots) // 3)).all()
    patches = ((pyramid.level == 0) & (pyramid.counts > 0)).nonzero().flatten().tolist()
    for head_cells in cells.flatten(0, 1):
        assert sorted(leaf_patches(head_cells.tolist(), pyramid)) == patches, "cells must tile the image exactly once"
    grid = RopeGrid(patch_coords(rope, H, W), rope.periods)
    torch.testing.assert_close(
        spread_to_patches(pyramid.counts[cells], cells, pyramid, grid), torch.ones(2, HEADS, H, W)
    )
    out.sum().backward()
    assert x.grad is not None and x.grad.isfinite().all()


def test_spread_is_rms_distance_from_cell_mean(layer):
    _, rope = layer
    H, W, d = 6, 10, DIM // HEADS
    q, k, v = torch.randn(3, 1, HEADS, H, W, d)
    pyramid = build_pyramid(q, k, v, RopeGrid(patch_coords(rope, H, W), rope.periods), levels=2)
    root = pyramid.roots[-1]  # the bottom-right cell, clipped to 2x2 patches by the image edge
    patches = v[:, :, 4:6, 8:10].flatten(2, 3)
    expected = (patches - patches.mean(2, keepdim=True)).square().sum(-1).mean(-1).sqrt()
    torch.testing.assert_close(pyramid.spread[:, :, root], expected)


def test_level_schedule_shares_budget_equally_between_levels():
    assert level_schedule(roots=510, budget=1024, levels=2) == [86, 85]  # HD at 64px start cells
    assert level_schedule(roots=4, budget=40, levels=2) == [4, 8]  # what the top level can't take passes down
    assert level_schedule(roots=4, budget=64, levels=2) == [4, 16]  # splits everything down to fine
    with pytest.raises(ValueError):
        level_schedule(roots=40, budget=32, levels=2)


def test_round_schedule_never_splits_more_cells_than_a_round_holds():
    assert round_schedule(roots=4, budget=64, rounds=3) == [4, 8, 8]
    assert round_schedule(roots=4, budget=64, rounds=1) == [4]


def test_foveated_model_reproduces_dense_model():
    torch.manual_seed(0)
    dense = dinov3_vits16(pretrained=False).eval()
    for block in dense.blocks:
        sharpen(block.attn)
    H, W = 8, 6
    fov = foveate(copy.deepcopy(dense), FoveationConfig(budget=padded_patches(H, W, 2) + PREFIX, start_level=2))
    assert fov.state_dict().keys() == dense.state_dict().keys()

    image = torch.randn(1, 3, H * 16, W * 16)
    with torch.no_grad():
        expected, actual = dense.forward_features(image), fov.forward_features(image)
    for key in ("x_norm_clstoken", "x_norm_patchtokens"):
        torch.testing.assert_close(actual[key], expected[key], atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiles for the GPU")
def test_compiled_pooling_matches_eager(layer):
    attn, rope = layer
    attn, rope = attn.cuda(), rope.cuda()
    H, W = 10, 13
    x = torch.randn(2, PREFIX + H * W, DIM, device="cuda")
    grid = RopeGrid(patch_coords(rope, H, W), rope.periods)
    eager, compiled = (
        FoveatedAttention(attn, FoveationConfig(budget=35, start_level=2, compile=flag)) for flag in (False, True)
    )
    with torch.no_grad():
        torch.testing.assert_close(compiled(x, rope=grid), eager(x, rope=grid), atol=1e-5, rtol=1e-4)


def test_oracle_and_spread_criteria_vanish_on_uniform_cells(layer):
    from dinov3.foveation.split_criteria import criteria, oracle

    _, rope = layer
    H, W, d = 8, 12, DIM // HEADS
    grid = RopeGrid(patch_coords(rope, H, W), torch.full_like(rope.periods, torch.inf))  # no rotation
    q = torch.randn(1, HEADS, PREFIX + H * W, d)
    k, v = torch.randn(2, 1, HEADS, PREFIX + H * W, d)
    cell_k, cell_v = torch.randn(2, 1, HEADS, H // 4, W // 4, d)
    k[:, :, PREFIX:], v[:, :, PREFIX:] = (
        t.repeat_interleave(4, 2).repeat_interleave(4, 3).flatten(2, 3) for t in (cell_k, cell_v)
    )
    scores, pyramid = criteria(q, k, v, grid, FoveationConfig(start_level=2, score_level=2))
    truth = oracle(q, k, v, pyramid, H, W, n_queries=PREFIX + H * W)
    cells = pyramid.layout.level_cells(2)
    torch.testing.assert_close(truth[:, cells], torch.zeros_like(truth[:, cells]), atol=1e-5, rtol=0)
    for name in ("value_spread", "bound", "kl", "covariance"):
        torch.testing.assert_close(scores[name][:, cells], torch.zeros_like(truth[:, cells]), atol=1e-5, rtol=0)
