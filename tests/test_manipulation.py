"""Tests for the Stage 1 manipulation branch (Member 4).

Uses only mock patch features (``torch.randn``) -- this module has no
dependence on real DINOv2 features or a real backbone.
"""

from __future__ import annotations

import pytest
import torch

from src.models.manipulation import (
    ManipulationHead,
    patch_logits_to_heatmap,
    topk_manipulation_probability,
)

BATCH_SIZE = 8
NUM_PATCHES = 256
EMBEDDING_DIM = 384


def make_patch_features(batch_size: int = BATCH_SIZE) -> torch.Tensor:
    return torch.randn(batch_size, NUM_PATCHES, EMBEDDING_DIM)


def test_output_shapes() -> None:
    head = ManipulationHead()
    output = head(make_patch_features())

    assert output["manipulation_probability"].shape == (BATCH_SIZE,)
    assert output["patch_mask_logits"].shape == (BATCH_SIZE, NUM_PATCHES)
    assert output["heatmap"].shape == (BATCH_SIZE, 1, 224, 224)


def test_manipulation_probability_in_unit_interval() -> None:
    head = ManipulationHead()
    output = head(make_patch_features())

    probabilities = output["manipulation_probability"]
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_heatmap_dimensions_configurable() -> None:
    head = ManipulationHead(heatmap_size=112)
    output = head(make_patch_features())
    assert output["heatmap"].shape == (BATCH_SIZE, 1, 112, 112)


def test_heatmap_is_genuine_bilinear_upsample() -> None:
    """A non-constant 16x16 grid must interpolate smoothly, not nearest-neighbour."""
    grid_size = 16
    logits = torch.zeros(1, grid_size * grid_size)
    # Turn on a single patch so the surrounding upsampled region shows a smooth
    # gradient rather than the hard step nearest-neighbour would produce.
    center_index = (grid_size // 2) * grid_size + (grid_size // 2)
    logits[0, center_index] = 10.0

    heatmap = patch_logits_to_heatmap(logits, patch_grid_size=grid_size, heatmap_size=224)

    # Direct comparison against F.interpolate ensures we're using real bilinear
    # interpolation with align_corners=False, not a hand-rolled approximation.
    expected = torch.nn.functional.interpolate(
        logits.reshape(1, 1, grid_size, grid_size),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    assert torch.allclose(heatmap, expected)

    # Bilinear upsampling of a single hot cell produces intermediate (non-binary)
    # values in the transition region -- nearest-neighbour would not.
    row = heatmap[0, 0, 112]
    unique_values = torch.unique(row.round(decimals=4))
    assert unique_values.numel() > 2


def test_configurable_top_k() -> None:
    head_k4 = ManipulationHead(top_k=4)
    head_k64 = ManipulationHead(top_k=64)
    assert head_k4.top_k == 4
    assert head_k64.top_k == 64


def test_topk_aggregation_matches_manual_computation() -> None:
    torch.manual_seed(0)
    logits = torch.randn(BATCH_SIZE, NUM_PATCHES)
    top_k = 16

    result = topk_manipulation_probability(logits, top_k=top_k)

    probabilities = torch.sigmoid(logits)
    expected = torch.topk(probabilities, k=top_k, dim=-1).values.mean(dim=-1)
    assert torch.allclose(result, expected)


def test_topk_differs_from_plain_max() -> None:
    torch.manual_seed(1)
    logits = torch.randn(BATCH_SIZE, NUM_PATCHES)

    topk_result = topk_manipulation_probability(logits, top_k=16)
    plain_max = torch.sigmoid(logits).max(dim=-1).values

    # With random, non-degenerate logits, mean-of-top-16 must not collapse to
    # the single highest patch -- that would mean top-k degenerated into max.
    assert not torch.allclose(topk_result, plain_max)
    assert torch.all(topk_result <= plain_max)


def test_top_k_of_one_equals_plain_max() -> None:
    """Sanity check: top_k=1 is mathematically equivalent to max."""
    torch.manual_seed(2)
    logits = torch.randn(BATCH_SIZE, NUM_PATCHES)

    topk_result = topk_manipulation_probability(logits, top_k=1)
    plain_max = torch.sigmoid(logits).max(dim=-1).values
    assert torch.allclose(topk_result, plain_max)


def test_invalid_rank_is_rejected() -> None:
    head = ManipulationHead()
    with pytest.raises(ValueError):
        head(torch.randn(BATCH_SIZE, NUM_PATCHES, EMBEDDING_DIM, 1))
    with pytest.raises(ValueError):
        head(torch.randn(NUM_PATCHES, EMBEDDING_DIM))


def test_invalid_patch_count_is_rejected() -> None:
    head = ManipulationHead()
    with pytest.raises(ValueError):
        head(torch.randn(BATCH_SIZE, 100, EMBEDDING_DIM))


def test_invalid_embedding_dimension_is_rejected() -> None:
    head = ManipulationHead()
    with pytest.raises(ValueError):
        head(torch.randn(BATCH_SIZE, NUM_PATCHES, 512))


@pytest.mark.parametrize("bad_top_k", [0, -1, 257])
def test_invalid_top_k_is_rejected_at_construction(bad_top_k: int) -> None:
    with pytest.raises(ValueError):
        ManipulationHead(top_k=bad_top_k)


def test_invalid_top_k_is_rejected_in_helper() -> None:
    logits = torch.randn(BATCH_SIZE, NUM_PATCHES)
    with pytest.raises(ValueError):
        topk_manipulation_probability(logits, top_k=0)
    with pytest.raises(ValueError):
        topk_manipulation_probability(logits, top_k=NUM_PATCHES + 1)


def test_gradient_flows_into_head_parameters() -> None:
    head = ManipulationHead()
    patch_features = make_patch_features()

    output = head(patch_features)
    loss = output["manipulation_probability"].sum() + output["heatmap"].sum()
    loss.backward()

    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert torch.any(parameter.grad != 0), f"zero gradient for {name}"


def test_no_dependence_on_real_dinov2_features() -> None:
    """Purely random mock features must produce valid outputs end-to-end."""
    head = ManipulationHead()
    output = head(torch.randn(2, NUM_PATCHES, EMBEDDING_DIM))
    assert output["manipulation_probability"].shape == (2,)
    assert torch.all(torch.isfinite(output["manipulation_probability"]))
