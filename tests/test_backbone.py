"""Unit tests for src/models/backbone.py (Member 2).

The real ``facebook/dinov2-small`` checkpoint is NOT downloaded here. A tiny
stub with genuine ``nn.Parameter`` tensors is injected through the
``_load_hf_backbone`` seam, which is enough to verify:

* every backbone parameter is frozen (``requires_grad is False``),
* CLS features are ``[B, 384]`` and patch features are ``[B, 256, 384]``,
* input / output shape assertions fire,
* execution works on CPU and returns detached tensors.

An optional test against the real model runs only when
``TRACELENS_RUN_SLOW=1`` is set in the environment.
"""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.models import backbone as backbone_mod  # noqa: E402
from src.models.backbone import (  # noqa: E402
    EMBED_DIM,
    IMAGE_SIZE,
    NUM_PATCHES,
    PATCH_GRID,
    DINOv2Backbone,
)


# ---------------------------------------------------------------------------
# Tiny stand-in for transformers' Dinov2Model
# ---------------------------------------------------------------------------
class _StubOutput:
    def __init__(self, last_hidden_state: "torch.Tensor") -> None:
        self.last_hidden_state = last_hidden_state


class _StubDINOv2(nn.Module):
    """Returns ``last_hidden_state`` of shape ``[B, 1 + num_patches, EMBED_DIM]``.

    Carries real parameters so the freezing test is meaningful. Output is a
    deterministic linear function of a global-pooled input so shapes are
    exercised without any randomness.
    """

    def __init__(self, num_patches: int = NUM_PATCHES) -> None:
        super().__init__()
        self.num_tokens = num_patches + 1
        self.proj = nn.Linear(3, EMBED_DIM)
        self.token_bias = nn.Parameter(torch.randn(self.num_tokens, EMBED_DIM))

    def forward(self, pixel_values=None, return_dict=True, **_kwargs):  # noqa: D401
        assert pixel_values is not None
        b = pixel_values.shape[0]
        pooled = pixel_values.mean(dim=(2, 3))          # [B, 3]
        base = self.proj(pooled).unsqueeze(1)           # [B, 1, EMBED_DIM]
        hidden = base + self.token_bias.unsqueeze(0)    # [B, num_tokens, EMBED_DIM]
        return _StubOutput(hidden)


@pytest.fixture
def stub_backbone(monkeypatch) -> DINOv2Backbone:
    monkeypatch.setattr(backbone_mod, "_load_hf_backbone", lambda name: _StubDINOv2())
    return DINOv2Backbone(model_name="stub-dinov2-small", device="cpu")


def _dummy_input(batch: int) -> "torch.Tensor":
    return torch.randn(batch, 3, IMAGE_SIZE, IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------
def test_all_backbone_parameters_are_frozen(stub_backbone):
    assert stub_backbone.is_frozen is True
    params = list(stub_backbone.model.parameters())
    assert len(params) > 0
    assert all(p.requires_grad is False for p in params)


def test_backbone_stays_in_eval_even_after_train_call(stub_backbone):
    stub_backbone.train()  # user toggles the wrapper
    assert stub_backbone.model.training is False


def test_no_trainable_parameters_exposed(stub_backbone):
    trainable = [p for p in stub_backbone.parameters() if p.requires_grad]
    assert trainable == []


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch", [1, 2, 5])
def test_feature_shapes(stub_backbone, batch):
    cls_features, patch_features = stub_backbone(_dummy_input(batch))
    assert cls_features.shape == (batch, EMBED_DIM)
    assert patch_features.shape == (batch, NUM_PATCHES, EMBED_DIM)
    assert PATCH_GRID * PATCH_GRID == NUM_PATCHES


def test_extract_features_matches_forward(stub_backbone):
    x = _dummy_input(3)
    a_cls, a_patch = stub_backbone(x)
    b_cls, b_patch = stub_backbone.extract_features(x)
    assert torch.allclose(a_cls, b_cls)
    assert torch.allclose(a_patch, b_patch)


# ---------------------------------------------------------------------------
# Input assertions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        torch.randn(3, IMAGE_SIZE, IMAGE_SIZE),          # missing batch dim
        torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE),       # wrong channel count
        torch.randn(2, 3, 128, 128),                     # wrong spatial size
        torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE + 1),   # off-by-one
    ],
)
def test_forward_rejects_bad_input_shape(stub_backbone, bad):
    with pytest.raises(AssertionError):
        stub_backbone(bad)


def test_forward_rejects_non_tensor(stub_backbone):
    with pytest.raises(TypeError):
        stub_backbone([[0.0] * 10])


# ---------------------------------------------------------------------------
# Output assertions (wrong token count from the underlying model)
# ---------------------------------------------------------------------------
def test_output_assertion_on_wrong_token_count(monkeypatch):
    monkeypatch.setattr(
        backbone_mod, "_load_hf_backbone", lambda name: _StubDINOv2(num_patches=100)
    )
    bad_backbone = DINOv2Backbone(model_name="stub", device="cpu")
    with pytest.raises(AssertionError):
        bad_backbone(_dummy_input(2))


# ---------------------------------------------------------------------------
# CPU execution
# ---------------------------------------------------------------------------
def test_runs_on_cpu_and_returns_detached_tensors(stub_backbone):
    cls_features, patch_features = stub_backbone(_dummy_input(2))
    assert cls_features.device.type == "cpu"
    assert patch_features.device.type == "cpu"
    # forward() is wrapped in torch.no_grad() -> no autograd history
    assert cls_features.requires_grad is False
    assert patch_features.requires_grad is False
    assert cls_features.grad_fn is None


# ---------------------------------------------------------------------------
# Optional: real model (network + download). Off unless explicitly enabled.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.getenv("TRACELENS_RUN_SLOW") != "1",
    reason="set TRACELENS_RUN_SLOW=1 to test against the real facebook/dinov2-small",
)
def test_real_dinov2_small_shapes_and_freezing():
    pytest.importorskip("transformers")
    real = DINOv2Backbone(model_name="facebook/dinov2-small", device="cpu")
    assert real.is_frozen is True
    cls_features, patch_features = real(_dummy_input(2))
    assert cls_features.shape == (2, EMBED_DIM)
    assert patch_features.shape == (2, NUM_PATCHES, EMBED_DIM)
