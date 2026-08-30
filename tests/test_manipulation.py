"""Tests for the manipulation branch (Member 4): Stage 1 model + Stage 2 training
+ Stage 3 heatmap visualization.

Uses only mock patch features (``torch.randn``) -- this module has no
dependence on real DINOv2 features or a real backbone.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import manifests
from src.models.manipulation import (
    ManipulationHead,
    patch_logits_to_heatmap,
    topk_manipulation_probability,
)
from src.models.manipulation_visualization import (
    create_manipulation_overlay,
    heatmap_to_probabilities,
)
from src.training.train_manipulation import (
    bce_mask_loss,
    dice_loss,
    filter_manipulation_batch,
    manipulation_loss,
    resize_mask_to_patch_grid,
    train_one_epoch,
)

BATCH_SIZE = 8
NUM_PATCHES = 256
EMBEDDING_DIM = 384
PATCH_GRID_SIZE = 16


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


# ---------------------------------------------------------------------------
# Stage 2: mask resizing / alignment
# ---------------------------------------------------------------------------


def test_resize_mask_to_patch_grid_shape() -> None:
    mask = torch.zeros(BATCH_SIZE, 1, 224, 224)
    target = resize_mask_to_patch_grid(mask, patch_grid_size=PATCH_GRID_SIZE)
    assert target.shape == (BATCH_SIZE, NUM_PATCHES)


def test_resize_mask_to_patch_grid_is_binary() -> None:
    """Resized targets must remain valid {0,1} manipulation targets."""
    torch.manual_seed(3)
    mask = (torch.rand(BATCH_SIZE, 1, 224, 224) > 0.5).float()
    target = resize_mask_to_patch_grid(mask, patch_grid_size=PATCH_GRID_SIZE)
    unique_values = set(torch.unique(target).tolist())
    assert unique_values <= {0.0, 1.0}


def test_resize_mask_all_zero_stays_all_zero() -> None:
    mask = torch.zeros(BATCH_SIZE, 1, 224, 224)
    target = resize_mask_to_patch_grid(mask, patch_grid_size=PATCH_GRID_SIZE)
    assert torch.all(target == 0.0)


def test_resize_mask_all_one_stays_all_one() -> None:
    mask = torch.ones(BATCH_SIZE, 1, 224, 224)
    target = resize_mask_to_patch_grid(mask, patch_grid_size=PATCH_GRID_SIZE)
    assert torch.all(target == 1.0)


def test_resize_mask_preserves_small_sub_patch_manipulated_region() -> None:
    """A tampered region covering well under 50% of one patch's area must
    still mark that patch positive (any-pixel-positive semantics, not a
    majority-vote threshold)."""
    mask = torch.zeros(1, 1, 224, 224)
    patch_pixels = 224 // PATCH_GRID_SIZE  # 14
    row, col = 5, 5
    row_start, col_start = row * patch_pixels, col * patch_pixels
    # 2x2 = 4 pixels out of 14x14 = 196 -> ~2% of the patch's area.
    mask[0, 0, row_start : row_start + 2, col_start : col_start + 2] = 1.0

    target = resize_mask_to_patch_grid(mask, patch_grid_size=PATCH_GRID_SIZE)

    positive_patch_index = row * PATCH_GRID_SIZE + col
    assert target[0, positive_patch_index] == 1.0
    # No other patch contains any manipulated pixel.
    assert target.sum().item() == 1.0


def test_resize_mask_accepts_batch_hw_without_channel_dim() -> None:
    mask = torch.zeros(BATCH_SIZE, 224, 224)
    target = resize_mask_to_patch_grid(mask, patch_grid_size=PATCH_GRID_SIZE)
    assert target.shape == (BATCH_SIZE, NUM_PATCHES)


def test_resize_mask_rejects_bad_rank() -> None:
    with pytest.raises(ValueError):
        resize_mask_to_patch_grid(torch.zeros(BATCH_SIZE, 3, 224, 224))
    with pytest.raises(ValueError):
        resize_mask_to_patch_grid(torch.zeros(224, 224))


# ---------------------------------------------------------------------------
# Stage 2: BCE / Dice / combined loss
# ---------------------------------------------------------------------------


def test_bce_loss_is_finite() -> None:
    torch.manual_seed(4)
    logits = torch.randn(BATCH_SIZE, NUM_PATCHES, requires_grad=True)
    target = (torch.rand(BATCH_SIZE, NUM_PATCHES) > 0.5).float()
    loss = bce_mask_loss(logits, target)
    assert torch.isfinite(loss)
    assert loss.shape == ()


def test_dice_loss_is_finite() -> None:
    torch.manual_seed(5)
    probabilities = torch.sigmoid(torch.randn(BATCH_SIZE, NUM_PATCHES))
    target = (torch.rand(BATCH_SIZE, NUM_PATCHES) > 0.5).float()
    loss = dice_loss(probabilities, target)
    assert torch.isfinite(loss)
    assert loss.shape == ()


def test_dice_loss_handles_all_zero_target_without_nan() -> None:
    """Authentic (label 0) samples have an all-zero target -- must not NaN."""
    probabilities = torch.sigmoid(torch.randn(BATCH_SIZE, NUM_PATCHES))
    target = torch.zeros(BATCH_SIZE, NUM_PATCHES)
    loss = dice_loss(probabilities, target)
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)


def test_dice_loss_all_zero_target_and_prediction_is_near_zero() -> None:
    """A correctly-predicted empty mask should land near zero once smoothed,
    without special-casing -- confirms the epsilon smoothing behaves sanely
    rather than merely avoiding a crash."""
    probabilities = torch.zeros(BATCH_SIZE, NUM_PATCHES)
    target = torch.zeros(BATCH_SIZE, NUM_PATCHES)
    loss = dice_loss(probabilities, target)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-3


def test_combined_manipulation_loss_is_finite_with_pixel_mask() -> None:
    torch.manual_seed(6)
    logits = torch.randn(BATCH_SIZE, NUM_PATCHES)
    mask = (torch.rand(BATCH_SIZE, 1, 224, 224) > 0.5).float()
    result = manipulation_loss(logits, mask)
    assert torch.isfinite(result.total)
    assert torch.isfinite(result.bce)
    assert torch.isfinite(result.dice)
    assert result.total.shape == ()
    assert result.bce.shape == ()
    assert result.dice.shape == ()


def test_combined_manipulation_loss_equals_bce_plus_dice_by_default() -> None:
    torch.manual_seed(7)
    logits = torch.randn(BATCH_SIZE, NUM_PATCHES)
    target = (torch.rand(BATCH_SIZE, NUM_PATCHES) > 0.5).float()
    result = manipulation_loss(logits, target)
    assert torch.allclose(result.total, result.bce + result.dice)


def test_combined_manipulation_loss_handles_all_authentic_batch() -> None:
    """A batch of only authentic (all-zero mask) samples must not NaN."""
    torch.manual_seed(8)
    logits = torch.randn(BATCH_SIZE, NUM_PATCHES)
    mask = torch.zeros(BATCH_SIZE, 1, 224, 224)
    result = manipulation_loss(logits, mask)
    assert torch.isfinite(result.total)


def test_combined_manipulation_loss_handles_all_tampered_batch() -> None:
    torch.manual_seed(9)
    logits = torch.randn(BATCH_SIZE, NUM_PATCHES)
    mask = torch.ones(BATCH_SIZE, 1, 224, 224)
    result = manipulation_loss(logits, mask)
    assert torch.isfinite(result.total)


def test_combined_manipulation_loss_handles_mixed_batch() -> None:
    torch.manual_seed(10)
    logits = torch.randn(BATCH_SIZE, NUM_PATCHES)
    mask = torch.zeros(BATCH_SIZE, 1, 224, 224)
    mask[: BATCH_SIZE // 2] = 1.0  # half tampered, half authentic
    result = manipulation_loss(logits, mask)
    assert torch.isfinite(result.total)


# ---------------------------------------------------------------------------
# Stage 2: label 1 exclusion
# ---------------------------------------------------------------------------


def test_filter_batch_accepts_label_0() -> None:
    patch_features = make_patch_features(4)
    mask = torch.zeros(4, 1, 224, 224)
    label = torch.full((4,), manifests.LABEL_AUTHENTIC)
    filtered = filter_manipulation_batch(patch_features, mask, label)
    assert filtered is not None
    assert filtered[0].shape[0] == 4


def test_filter_batch_accepts_label_2() -> None:
    patch_features = make_patch_features(4)
    mask = torch.ones(4, 1, 224, 224)
    label = torch.full((4,), manifests.LABEL_LOCALLY_TAMPERED)
    filtered = filter_manipulation_batch(patch_features, mask, label)
    assert filtered is not None
    assert filtered[0].shape[0] == 4


def test_filter_batch_excludes_label_1() -> None:
    patch_features = make_patch_features(4)
    mask = torch.zeros(4, 1, 224, 224)
    label = torch.full((4,), manifests.LABEL_FULLY_SYNTHETIC)
    filtered = filter_manipulation_batch(patch_features, mask, label)
    assert filtered is None


def test_filter_batch_mixed_labels_excludes_only_label_1() -> None:
    patch_features = make_patch_features(6)
    mask = torch.zeros(6, 1, 224, 224)
    label = torch.tensor([0, 1, 2, 1, 0, 2])

    filtered_features, filtered_mask, filtered_label = filter_manipulation_batch(
        patch_features, mask, label
    )

    assert filtered_label.shape[0] == 4
    assert torch.all(filtered_label != manifests.LABEL_FULLY_SYNTHETIC)
    assert set(filtered_label.tolist()) <= {manifests.LABEL_AUTHENTIC, manifests.LABEL_LOCALLY_TAMPERED}
    assert filtered_features.shape[0] == 4
    assert filtered_mask.shape[0] == 4
    # Rows must stay aligned: original indices 0,2,4,5 survive in order.
    expected_features = patch_features[[0, 2, 4, 5]]
    assert torch.equal(filtered_features, expected_features)


def test_filter_batch_all_label_1_is_handled_explicitly() -> None:
    """A batch containing only fully-synthetic samples has nothing eligible."""
    patch_features = make_patch_features(5)
    mask = torch.zeros(5, 1, 224, 224)
    label = torch.full((5,), manifests.LABEL_FULLY_SYNTHETIC)
    assert filter_manipulation_batch(patch_features, mask, label) is None


def test_filter_batch_rejects_invalid_label_values() -> None:
    patch_features = make_patch_features(3)
    mask = torch.zeros(3, 1, 224, 224)
    label = torch.tensor([0, 2, 7])
    with pytest.raises(ValueError):
        filter_manipulation_batch(patch_features, mask, label)


def test_filter_batch_rejects_mismatched_batch_sizes() -> None:
    patch_features = make_patch_features(4)
    mask = torch.zeros(3, 1, 224, 224)
    label = torch.tensor([0, 2, 0, 1])
    with pytest.raises(ValueError):
        filter_manipulation_batch(patch_features, mask, label)


# ---------------------------------------------------------------------------
# Stage 2: gradient flow and end-to-end mock training
# ---------------------------------------------------------------------------


def test_gradients_flow_through_combined_manipulation_loss() -> None:
    head = ManipulationHead()
    patch_features = make_patch_features()
    mask = (torch.rand(BATCH_SIZE, 1, 224, 224) > 0.5).float()

    output = head(patch_features)
    result = manipulation_loss(output["patch_mask_logits"], mask)
    result.total.backward()

    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert torch.any(parameter.grad != 0), f"zero gradient for {name}"


def test_train_one_epoch_completes_forward_backward_with_mock_features() -> None:
    torch.manual_seed(11)
    head = ManipulationHead()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.01)

    batches = [
        {
            "patch_features": make_patch_features(4),
            "mask": (torch.rand(4, 1, 224, 224) > 0.5).float(),
            "label": torch.tensor([0, 2, 0, 2]),
        },
        {
            "patch_features": make_patch_features(4),
            "mask": torch.zeros(4, 1, 224, 224),
            "label": torch.tensor([0, 0, 0, 0]),
        },
    ]

    stats = train_one_epoch(head, batches, optimizer)

    assert stats.num_batches == 2
    assert stats.num_skipped_batches == 0
    assert stats.mean_total_loss == pytest.approx(stats.mean_bce_loss + stats.mean_dice_loss)
    assert stats.mean_total_loss >= 0.0


def test_train_one_epoch_skips_all_label_1_batches() -> None:
    torch.manual_seed(12)
    head = ManipulationHead()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.01)

    batches = [
        {
            "patch_features": make_patch_features(4),
            "mask": torch.zeros(4, 1, 224, 224),
            "label": torch.tensor([1, 1, 1, 1]),
        },
        {
            "patch_features": make_patch_features(4),
            "mask": (torch.rand(4, 1, 224, 224) > 0.5).float(),
            "label": torch.tensor([0, 2, 0, 2]),
        },
    ]

    stats = train_one_epoch(head, batches, optimizer)
    assert stats.num_batches == 1
    assert stats.num_skipped_batches == 1


def test_train_one_epoch_raises_when_nothing_eligible() -> None:
    head = ManipulationHead()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
    batches = [
        {
            "patch_features": make_patch_features(4),
            "mask": torch.zeros(4, 1, 224, 224),
            "label": torch.tensor([1, 1, 1, 1]),
        }
    ]
    with pytest.raises(ValueError):
        train_one_epoch(head, batches, optimizer)


# ---------------------------------------------------------------------------
# Stage 3: heatmap -> probability conversion
# ---------------------------------------------------------------------------


def test_heatmap_to_probabilities_matches_sigmoid() -> None:
    torch.manual_seed(20)
    logits = torch.randn(16, 16) * 5.0
    probabilities = heatmap_to_probabilities(logits, is_logits=True)
    assert torch.allclose(probabilities, torch.sigmoid(logits))


def test_heatmap_to_probabilities_accepts_channel_dim() -> None:
    torch.manual_seed(21)
    logits = torch.randn(1, 16, 16)
    probabilities = heatmap_to_probabilities(logits)
    assert probabilities.shape == (16, 16)
    assert torch.allclose(probabilities, torch.sigmoid(logits.squeeze(0)))


def test_heatmap_to_probabilities_no_double_sigmoid() -> None:
    """is_logits=False must pass an already-[0,1] map through unchanged."""
    already_probabilities = torch.rand(16, 16)
    result = heatmap_to_probabilities(already_probabilities, is_logits=False)
    assert torch.allclose(result, already_probabilities)


def test_heatmap_to_probabilities_rejects_mislabelled_probabilities() -> None:
    """Values outside [0,1] under is_logits=False indicate mislabelled semantics."""
    logits = torch.tensor([[-3.0, 5.0], [0.2, -0.1]])
    with pytest.raises(ValueError):
        heatmap_to_probabilities(logits, is_logits=False)


def test_heatmap_to_probabilities_rejects_nan_and_inf() -> None:
    bad = torch.zeros(16, 16)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        heatmap_to_probabilities(bad)

    bad_inf = torch.zeros(16, 16)
    bad_inf[0, 0] = float("inf")
    with pytest.raises(ValueError):
        heatmap_to_probabilities(bad_inf)


def test_heatmap_to_probabilities_rejects_batched_input() -> None:
    with pytest.raises(ValueError):
        heatmap_to_probabilities(torch.randn(4, 1, 16, 16))


def test_heatmap_to_probabilities_rejects_empty() -> None:
    with pytest.raises(ValueError):
        heatmap_to_probabilities(torch.zeros(0, 0))


def test_heatmap_to_probabilities_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError):
        heatmap_to_probabilities([[0.1, 0.2], [0.3, 0.4]])


def test_heatmap_to_probabilities_detaches_from_autograd() -> None:
    logits = torch.randn(16, 16, requires_grad=True)
    probabilities = heatmap_to_probabilities(logits)
    assert not probabilities.requires_grad


def test_heatmap_to_probabilities_accepts_numpy() -> None:
    logits_np = np.random.randn(16, 16).astype(np.float32)
    result = heatmap_to_probabilities(logits_np)
    expected = torch.sigmoid(torch.from_numpy(logits_np))
    assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# Stage 3: overlay rendering
# ---------------------------------------------------------------------------


def make_test_image(size: tuple[int, int] = (64, 48), color: tuple[int, int, int] = (120, 130, 140)) -> Image.Image:
    return Image.new("RGB", size, color)


def test_overlay_returns_pil_image() -> None:
    image = make_test_image()
    heatmap = torch.zeros(16, 16)
    overlay = create_manipulation_overlay(image, heatmap)
    assert isinstance(overlay, Image.Image)


def test_overlay_matches_original_dimensions() -> None:
    image = make_test_image(size=(100, 40))
    heatmap = torch.zeros(16, 16)
    overlay = create_manipulation_overlay(image, heatmap)
    assert overlay.size == image.size


def test_overlay_does_not_mutate_original_image() -> None:
    image = make_test_image()
    original_bytes = image.tobytes()
    heatmap = torch.randn(16, 16)
    create_manipulation_overlay(image, heatmap)
    assert image.tobytes() == original_bytes


def test_overlay_accepts_hw_heatmap() -> None:
    image = make_test_image()
    heatmap = torch.zeros(16, 16)
    overlay = create_manipulation_overlay(image, heatmap)
    assert overlay.size == image.size


def test_overlay_accepts_1hw_heatmap() -> None:
    image = make_test_image()
    heatmap_hw = torch.randn(16, 16)
    heatmap_1hw = heatmap_hw.unsqueeze(0)
    overlay_hw = create_manipulation_overlay(image, heatmap_hw)
    overlay_1hw = create_manipulation_overlay(image, heatmap_1hw)
    assert np.array_equal(np.array(overlay_hw), np.array(overlay_1hw))


def test_overlay_accepts_torch_heatmap() -> None:
    image = make_test_image()
    heatmap = torch.randn(16, 16)
    overlay = create_manipulation_overlay(image, heatmap)
    assert isinstance(overlay, Image.Image)


def test_overlay_accepts_numpy_heatmap() -> None:
    image = make_test_image()
    torch.manual_seed(22)
    heatmap_t = torch.randn(16, 16)
    overlay_t = create_manipulation_overlay(image, heatmap_t)
    overlay_np = create_manipulation_overlay(image, heatmap_t.numpy())
    assert np.array_equal(np.array(overlay_t), np.array(overlay_np))


def test_overlay_resizes_heatmap_to_original_image() -> None:
    """A small heatmap must be upsampled to the full image size, not left mismatched."""
    image = make_test_image(size=(64, 64), color=(0, 0, 0))
    heatmap = torch.full((4, 4), -10.0)  # near-zero probability everywhere
    heatmap[0, 0] = 10.0  # one strongly positive corner

    overlay = create_manipulation_overlay(image, heatmap, alpha=1.0)
    pixels = np.array(overlay)
    assert pixels.shape[:2] == (64, 64)

    # The corner corresponding to the hot patch should differ far more from
    # black than the opposite (near-zero-probability) corner.
    hot_corner = pixels[0:4, 0:4].astype(np.float32)
    cold_corner = pixels[-4:, -4:].astype(np.float32)
    assert hot_corner.mean() > cold_corner.mean() + 20


def test_overlay_alpha_zero_is_unchanged_original() -> None:
    image = make_test_image()
    heatmap = torch.randn(16, 16) * 10.0
    overlay = create_manipulation_overlay(image, heatmap, alpha=0.0)
    assert np.array_equal(np.array(overlay), np.array(image.convert("RGB")))


@pytest.mark.parametrize("bad_alpha", [-0.1, 1.1, float("nan")])
def test_overlay_rejects_invalid_alpha(bad_alpha: float) -> None:
    image = make_test_image()
    heatmap = torch.zeros(16, 16)
    with pytest.raises(ValueError):
        create_manipulation_overlay(image, heatmap, alpha=bad_alpha)


def test_overlay_rejects_invalid_heatmap_rank() -> None:
    image = make_test_image()
    with pytest.raises(ValueError):
        create_manipulation_overlay(image, torch.randn(2, 1, 16, 16))
    with pytest.raises(ValueError):
        create_manipulation_overlay(image, torch.randn(16))


def test_overlay_rejects_nan_inf_heatmap() -> None:
    image = make_test_image()
    heatmap = torch.zeros(16, 16)
    heatmap[3, 3] = float("nan")
    with pytest.raises(ValueError):
        create_manipulation_overlay(image, heatmap)


def test_overlay_rejects_non_pil_image() -> None:
    heatmap = torch.zeros(16, 16)
    with pytest.raises(TypeError):
        create_manipulation_overlay(np.zeros((10, 10, 3), dtype=np.uint8), heatmap)


def test_overlay_low_probability_changes_less_than_high_probability() -> None:
    image = make_test_image(color=(0, 0, 0))
    low_logits = torch.full((16, 16), -8.0)   # sigmoid ~ 0.0003
    high_logits = torch.full((16, 16), 8.0)   # sigmoid ~ 0.9997

    low_overlay = np.array(create_manipulation_overlay(image, low_logits)).astype(np.float32)
    high_overlay = np.array(create_manipulation_overlay(image, high_logits)).astype(np.float32)
    original = np.array(image.convert("RGB")).astype(np.float32)

    low_change = np.abs(low_overlay - original).mean()
    high_change = np.abs(high_overlay - original).mean()
    assert low_change < high_change


def test_overlay_uniform_low_confidence_is_not_stretched_to_look_suspicious() -> None:
    """A uniformly low-probability heatmap must stay unobtrusive, not be
    min-max-normalized into a falsely alarming full-strength overlay."""
    image = make_test_image(color=(10, 10, 10))
    uniform_low_logits = torch.full((16, 16), -6.0)  # sigmoid ~ 0.0025 everywhere

    overlay = np.array(create_manipulation_overlay(image, uniform_low_logits)).astype(np.float32)
    original = np.array(image.convert("RGB")).astype(np.float32)

    mean_change = np.abs(overlay - original).mean()
    # If this were min-max normalized per-image, a uniform map would be
    # stretched to look maximally suspicious; it must instead stay small.
    assert mean_change < 5.0
