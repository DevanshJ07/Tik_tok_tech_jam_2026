"""Unit tests for src/models/baseline.py (Member 2).

Pure-tensor tests -- no model download, no backbone. Random CLS / patch
features stand in for real DINOv2 output.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.baseline import (  # noqa: E402
    EMBED_DIM,
    NUM_PATCHES,
    OUTPUT_KEYS,
    BaselineAIGCDetector,
)


def make_features(batch: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    cls_features = torch.randn(batch, EMBED_DIM, generator=gen)
    patch_features = torch.randn(batch, NUM_PATCHES, EMBED_DIM, generator=gen)
    return cls_features, patch_features


@pytest.fixture
def model() -> BaselineAIGCDetector:
    torch.manual_seed(0)
    m = BaselineAIGCDetector()
    m.eval()  # disable dropout for deterministic assertions
    return m


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------
def test_output_has_exactly_the_required_keys(model):
    out = model(*make_features(4))
    assert set(out.keys()) == set(OUTPUT_KEYS)
    assert set(out.keys()) == {
        "global_logit",
        "patch_logits",
        "patch_mean_logit",
        "final_logit",
        "aigc_probability",
    }


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_output_shapes(model, batch):
    out = model(*make_features(batch))
    assert out["global_logit"].shape == (batch,)
    assert out["patch_logits"].shape == (batch, NUM_PATCHES)
    assert out["patch_mean_logit"].shape == (batch,)
    assert out["final_logit"].shape == (batch,)
    assert out["aigc_probability"].shape == (batch,)


# ---------------------------------------------------------------------------
# Numerical properties
# ---------------------------------------------------------------------------
def test_probabilities_between_zero_and_one(model):
    out = model(*make_features(16, seed=1))
    p = out["aigc_probability"]
    assert torch.all(p >= 0.0)
    assert torch.all(p <= 1.0)


def test_probability_is_sigmoid_of_final_logit(model):
    out = model(*make_features(8, seed=2))
    assert torch.allclose(out["aigc_probability"], torch.sigmoid(out["final_logit"]), atol=1e-6)


def test_patch_mean_logit_is_mean_over_patches(model):
    out = model(*make_features(5, seed=3))
    assert torch.allclose(out["patch_mean_logit"], out["patch_logits"].mean(dim=1), atol=1e-6)


def test_final_logit_is_half_global_plus_half_patch_mean(model):
    out = model(*make_features(6, seed=4))
    expected = 0.5 * out["global_logit"] + 0.5 * out["patch_mean_logit"]
    assert torch.allclose(out["final_logit"], expected, atol=1e-6)


def test_custom_fusion_weights_are_respected():
    torch.manual_seed(0)
    m = BaselineAIGCDetector(global_weight=0.3, patch_weight=0.7)
    m.eval()
    out = m(*make_features(4, seed=5))
    expected = 0.3 * out["global_logit"] + 0.7 * out["patch_mean_logit"]
    assert torch.allclose(out["final_logit"], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradients: only the lightweight heads train
# ---------------------------------------------------------------------------
def test_backward_populates_head_gradients():
    torch.manual_seed(0)
    m = BaselineAIGCDetector()
    m.train()
    cls_features, patch_features = make_features(4, seed=6)
    out = m(cls_features, patch_features)
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["final_logit"], target)
    loss.backward()
    grads = [p.grad for p in m.parameters()]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)


def test_head_parameter_count_is_small():
    m = BaselineAIGCDetector()
    n_params = sum(p.numel() for p in m.parameters())
    # Two small MLPs (384->128->1 each): well under 200k params.
    assert n_params < 200_000


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls_shape, patch_shape",
    [
        ((4, EMBED_DIM + 1), (4, NUM_PATCHES, EMBED_DIM)),   # wrong cls dim
        ((4, EMBED_DIM), (4, NUM_PATCHES - 1, EMBED_DIM)),   # wrong patch count
        ((4, EMBED_DIM), (4, NUM_PATCHES, EMBED_DIM + 1)),   # wrong patch dim
        ((4, EMBED_DIM), (3, NUM_PATCHES, EMBED_DIM)),       # batch mismatch
    ],
)
def test_rejects_bad_input_shapes(model, cls_shape, patch_shape):
    with pytest.raises(AssertionError):
        model(torch.randn(*cls_shape), torch.randn(*patch_shape))


def test_predict_proba_returns_probability_vector(model):
    p = model.predict_proba(*make_features(7, seed=8))
    assert p.shape == (7,)
    assert torch.all((p >= 0.0) & (p <= 1.0))
