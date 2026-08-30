"""
Tests for the TraceLens-R Member 3 reliability system.

These tests use synthetic tensors only.
They do not require SID-Set or a trained Member 2 checkpoint.
"""

import torch
import pytest

from src.models.reliability import (
    ReliabilityHead,
    TraceLensReliability,
    weighted_patch_aggregation,
    combine_logits,
    correct_class_evidence,
    compute_survival_target,
    survival_loss,
)


BATCH = 4
NUM_PATCHES = 256
FEATURE_DIM = 384


def make_mock_inputs():
    """Create deterministic tensors with the required shapes."""

    torch.manual_seed(42)

    patch_features = torch.randn(
        BATCH,
        NUM_PATCHES,
        FEATURE_DIM,
    )

    patch_logits = torch.randn(
        BATCH,
        NUM_PATCHES,
    )

    global_logit = torch.randn(BATCH)

    labels = torch.tensor([0, 1, 0, 1])

    return (
        patch_features,
        patch_logits,
        global_logit,
        labels,
    )


# ---------------------------------------------------------------------
# Reliability head
# ---------------------------------------------------------------------

def test_reliability_head_shape():
    patch_features, _, _, _ = make_mock_inputs()

    model = ReliabilityHead()

    reliability = model(patch_features)

    assert reliability.shape == (
        BATCH,
        NUM_PATCHES,
    )


def test_reliability_values_are_between_zero_and_one():
    patch_features, _, _, _ = make_mock_inputs()

    model = ReliabilityHead()

    reliability = model(patch_features)

    assert torch.all(reliability >= 0.0)
    assert torch.all(reliability <= 1.0)


# ---------------------------------------------------------------------
# Full Member 3 output
# ---------------------------------------------------------------------

def test_full_model_output_shapes():
    patch_features, patch_logits, global_logit, _ = make_mock_inputs()

    model = TraceLensReliability()

    outputs = model(
        patch_features,
        patch_logits,
        global_logit,
    )

    assert outputs["reliability"].shape == (
        BATCH,
        NUM_PATCHES,
    )

    assert outputs["weighted_patch_logit"].shape == (
        BATCH,
    )

    assert outputs["final_logit"].shape == (
        BATCH,
    )

    assert outputs["aigc_probability"].shape == (
        BATCH,
    )

    assert outputs["mean_reliability"].shape == (
        BATCH,
    )


def test_aigc_probability_is_between_zero_and_one():
    patch_features, patch_logits, global_logit, _ = make_mock_inputs()

    model = TraceLensReliability()

    outputs = model(
        patch_features,
        patch_logits,
        global_logit,
    )

    probability = outputs["aigc_probability"]

    assert torch.all(probability >= 0.0)
    assert torch.all(probability <= 1.0)


def test_mean_reliability_is_correct():
    patch_features, patch_logits, global_logit, _ = make_mock_inputs()

    model = TraceLensReliability()

    outputs = model(
        patch_features,
        patch_logits,
        global_logit,
    )

    expected = outputs["reliability"].mean(dim=1)

    assert torch.allclose(
        outputs["mean_reliability"],
        expected,
    )


# ---------------------------------------------------------------------
# Weighted aggregation
# ---------------------------------------------------------------------

def test_uniform_reliability_equals_unweighted_mean():
    """
    If every patch has identical reliability, the weighted mean
    must equal the ordinary patch mean.
    """

    patch_logits = torch.randn(
        BATCH,
        NUM_PATCHES,
    )

    reliability = torch.ones_like(
        patch_logits
    )

    weighted = weighted_patch_aggregation(
        patch_logits,
        reliability,
    )

    ordinary_mean = patch_logits.mean(
        dim=1
    )

    assert torch.allclose(
        weighted,
        ordinary_mean,
        atol=1e-6,
    )


def test_low_reliability_patch_contributes_less():
    """
    A high-logit patch with very low reliability should have
    much less influence than it has under ordinary averaging.
    """

    patch_logits = torch.tensor(
        [[10.0, 0.0]]
    )

    reliability = torch.tensor(
        [[0.01, 1.0]]
    )

    weighted = weighted_patch_aggregation(
        patch_logits,
        reliability,
    )

    ordinary_mean = patch_logits.mean(
        dim=1
    )

    assert weighted.item() < ordinary_mean.item()


def test_zero_reliability_is_safe():
    """
    All-zero reliability must not cause division by zero or NaN.
    """

    patch_logits = torch.randn(
        2,
        NUM_PATCHES,
    )

    reliability = torch.zeros_like(
        patch_logits
    )

    result = weighted_patch_aggregation(
        patch_logits,
        reliability,
    )

    assert torch.isfinite(result).all()

    assert torch.allclose(
        result,
        torch.zeros(2),
        atol=1e-6,
    )


# ---------------------------------------------------------------------
# Final logit
# ---------------------------------------------------------------------

def test_final_logit_uses_half_global_half_patch():
    global_logit = torch.tensor(
        [2.0, -2.0]
    )

    weighted_patch_logit = torch.tensor(
        [4.0, -4.0]
    )

    result = combine_logits(
        global_logit,
        weighted_patch_logit,
    )

    expected = torch.tensor(
        [3.0, -3.0]
    )

    assert torch.allclose(
        result,
        expected,
    )


# ---------------------------------------------------------------------
# Correct-class evidence
# ---------------------------------------------------------------------

def test_correct_class_evidence():
    """
    Label 1:
        correct evidence = raw logit

    Label 0:
        correct evidence = negative raw logit
    """

    patch_logits = torch.tensor(
        [
            [2.0, -3.0],
            [2.0, -3.0],
        ]
    )

    labels = torch.tensor(
        [1, 0]
    )

    evidence = correct_class_evidence(
        patch_logits,
        labels,
    )

    expected = torch.tensor(
        [
            [2.0, -3.0],
            [-2.0, 3.0],
        ]
    )

    assert torch.allclose(
        evidence,
        expected,
    )


def test_label_two_is_rejected():
    """
    Locally tampered images (label 2) must not be treated
    as fully AI-generated for the AIGC reliability target.
    """

    patch_logits = torch.randn(
        BATCH,
        NUM_PATCHES,
    )

    labels = torch.tensor(
        [0, 1, 2, 1]
    )

    with pytest.raises(ValueError):
        correct_class_evidence(
            patch_logits,
            labels,
        )


# ---------------------------------------------------------------------
# Survival targets
# ---------------------------------------------------------------------

def test_survival_target_shape():
    clean_logits = torch.randn(
        BATCH,
        NUM_PATCHES,
    )

    degraded_logits = torch.randn(
        BATCH,
        NUM_PATCHES,
    )

    labels = torch.tensor(
        [0, 1, 0, 1]
    )

    target, weight = compute_survival_target(
        clean_logits,
        degraded_logits,
        labels,
    )

    assert target.shape == (
        BATCH,
        NUM_PATCHES,
    )

    assert weight.shape == (
        BATCH,
        NUM_PATCHES,
    )


def test_survival_target_is_between_zero_and_one():
    clean_logits = torch.randn(
        BATCH,
        NUM_PATCHES,
    )

    degraded_logits = torch.randn(
        BATCH,
        NUM_PATCHES,
    )

    labels = torch.tensor(
        [0, 1, 0, 1]
    )

    target, weight = compute_survival_target(
        clean_logits,
        degraded_logits,
        labels,
    )

    assert torch.all(target >= 0.0)
    assert torch.all(target <= 1.0)

    assert torch.all(weight >= 0.0)
    assert torch.all(weight <= 1.0)


def test_survival_target_is_detached():
    clean_logits = torch.randn(
        BATCH,
        NUM_PATCHES,
        requires_grad=True,
    )

    degraded_logits = torch.randn(
        BATCH,
        NUM_PATCHES,
        requires_grad=True,
    )

    labels = torch.tensor(
        [0, 1, 0, 1]
    )

    target, weight = compute_survival_target(
        clean_logits,
        degraded_logits,
        labels,
    )

    assert target.requires_grad is False
    assert weight.requires_grad is False


def test_survival_target_rejects_label_two():
    clean_logits = torch.randn(
        BATCH,
        NUM_PATCHES,
    )

    degraded_logits = torch.randn(
        BATCH,
        NUM_PATCHES,
    )

    labels = torch.tensor(
        [0, 1, 2, 1]
    )

    with pytest.raises(ValueError):
        compute_survival_target(
            clean_logits,
            degraded_logits,
            labels,
        )


# ---------------------------------------------------------------------
# Survival loss
# ---------------------------------------------------------------------

def test_survival_loss_is_finite():
    prediction = torch.rand(
        BATCH,
        NUM_PATCHES,
        requires_grad=True,
    )

    target = torch.rand(
        BATCH,
        NUM_PATCHES,
    )

    weight = torch.rand(
        BATCH,
        NUM_PATCHES,
    )

    loss = survival_loss(
        prediction,
        target,
        weight,
    )

    assert torch.isfinite(loss)


def test_survival_loss_has_gradient():
    prediction = torch.rand(
        BATCH,
        NUM_PATCHES,
        requires_grad=True,
    )

    target = torch.rand(
        BATCH,
        NUM_PATCHES,
    )

    weight = torch.ones(
        BATCH,
        NUM_PATCHES,
    )

    loss = survival_loss(
        prediction,
        target,
        weight,
    )

    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(
        prediction.grad
    ).all()


# ---------------------------------------------------------------------
# Gradient flow through reliability model
# ---------------------------------------------------------------------

def test_reliability_model_has_gradient_flow():
    patch_features, patch_logits, global_logit, _ = (
        make_mock_inputs()
    )

    model = TraceLensReliability()

    outputs = model(
        patch_features,
        patch_logits,
        global_logit,
    )

    loss = outputs["final_logit"].mean()

    loss.backward()

    found_gradient = False

    for parameter in model.parameters():

        if parameter.grad is not None:

            found_gradient = True

            assert torch.isfinite(
                parameter.grad
            ).all()

    assert found_gradient


# ---------------------------------------------------------------------
# Checkpoint compatibility
# ---------------------------------------------------------------------

def test_checkpoint_state_dict_can_be_loaded():
    """
    A saved reliability state_dict must load into an identical
    reliability architecture.
    """

    model_a = TraceLensReliability()

    state_dict = model_a.state_dict()

    model_b = TraceLensReliability()

    model_b.load_state_dict(
        state_dict
    )

    for key in state_dict:

        assert torch.equal(
            state_dict[key],
            model_b.state_dict()[key],
        )