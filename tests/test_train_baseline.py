"""Checkpoint and cache-dir tests for Member 2 training helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.models import backbone as backbone_mod  # noqa: E402
from src.models.backbone import DINOv2Backbone  # noqa: E402
from src.models.baseline import EMBED_DIM, NUM_PATCHES, BaselineAIGCDetector  # noqa: E402
from src.training.train_baseline import (  # noqa: E402
    CACHE_GROUP_CLEAN,
    CACHE_GROUP_TRANSFORMED,
    CachedFeatureDataset,
    MixedCacheGroupsError,
    TrainConfig,
    load_checkpoint,
    save_checkpoint,
)


class _StubOutput:
    def __init__(self, last_hidden_state: "torch.Tensor") -> None:
        self.last_hidden_state = last_hidden_state


class _StubDINOv2(nn.Module):
    def __init__(self, num_patches: int = NUM_PATCHES) -> None:
        super().__init__()
        self.num_tokens = num_patches + 1
        self.proj = nn.Linear(3, EMBED_DIM)
        self.token_bias = nn.Parameter(torch.randn(self.num_tokens, EMBED_DIM))

    def forward(self, pixel_values=None, return_dict=True, **_kwargs):
        pooled = pixel_values.mean(dim=(2, 3))
        base = self.proj(pooled).unsqueeze(1)
        hidden = base + self.token_bias.unsqueeze(0)
        return _StubOutput(hidden)


def _fixed_features(batch: int = 4, seed: int = 11):
    gen = torch.Generator().manual_seed(seed)
    cls_features = torch.randn(batch, EMBED_DIM, generator=gen)
    patch_features = torch.randn(batch, NUM_PATCHES, EMBED_DIM, generator=gen)
    return cls_features, patch_features


def _write_feature(path: Path, *, transform_name: str, label: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_id": path.stem,
            "label": label,
            "transform_name": transform_name,
            "transform_severity": None,
            "cls_features": torch.zeros(EMBED_DIM),
            "patch_features": torch.zeros(NUM_PATCHES, EMBED_DIM),
        },
        path,
    )


def test_checkpoint_roundtrip_preserves_hparams_predictions_and_frozen_backbone(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(backbone_mod, "_load_hf_backbone", lambda name: _StubDINOv2())
    backbone = DINOv2Backbone(model_name="stub-dinov2-small", device="cpu")
    assert backbone.is_frozen is True

    torch.manual_seed(0)
    model = BaselineAIGCDetector(
        hidden_dim=64,
        dropout=0.0,
        global_weight=0.5,
        patch_weight=0.5,
    )
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = TrainConfig(hidden_dim=64, dropout=0.0, global_weight=0.5, patch_weight=0.5)

    cls_features, patch_features = _fixed_features()
    before = model(cls_features, patch_features)

    ckpt_path = tmp_path / "baseline.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        global_step=17,
        config=cfg,
    )

    backbone.train()
    loaded = load_checkpoint(ckpt_path, map_location="cpu")
    restored = loaded["model"]
    restored.eval()

    assert loaded["model_hparams"]["embed_dim"] == model.embed_dim
    assert loaded["model_hparams"]["num_patches"] == model.num_patches
    assert loaded["model_hparams"]["hidden_dim"] == cfg.hidden_dim
    assert loaded["model_hparams"]["dropout"] == cfg.dropout
    assert loaded["model_hparams"]["global_weight"] == model.global_weight
    assert loaded["model_hparams"]["patch_weight"] == model.patch_weight
    assert restored.embed_dim == model.embed_dim
    assert restored.num_patches == model.num_patches
    assert restored.global_weight == model.global_weight
    assert restored.patch_weight == model.patch_weight

    after = restored(cls_features, patch_features)
    for key in before:
        assert torch.allclose(before[key], after[key], atol=1e-6)

    assert backbone.is_frozen is True
    assert all(not p.requires_grad for p in backbone.parameters())
    assert backbone.model.training is False


def test_mixed_cache_dir_raises_without_subset(tmp_path: Path) -> None:
    _write_feature(tmp_path / "clean" / "a.pt", transform_name="clean")
    _write_feature(tmp_path / "transformed" / "a.pt", transform_name="jpeg")
    with pytest.raises(MixedCacheGroupsError, match="mixes"):
        CachedFeatureDataset(tmp_path)


def test_cache_subset_selects_one_group(tmp_path: Path) -> None:
    _write_feature(tmp_path / "clean" / "a.pt", transform_name="clean")
    _write_feature(tmp_path / "transformed" / "b.pt", transform_name="jpeg")
    ds = CachedFeatureDataset(tmp_path, cache_subset=CACHE_GROUP_CLEAN)
    assert len(ds) == 1
    assert ds.files[0].name == "a.pt"


def test_single_group_cache_directory_is_accepted(tmp_path: Path) -> None:
    clean_dir = tmp_path / CACHE_GROUP_CLEAN / "train"
    _write_feature(clean_dir / "a.pt", transform_name="clean")
    _write_feature(clean_dir / "b.pt", transform_name="clean")
    ds = CachedFeatureDataset(clean_dir)
    assert len(ds) == 2


def test_flat_mixed_transform_names_require_subset(tmp_path: Path) -> None:
    _write_feature(tmp_path / "a.pt", transform_name="clean")
    _write_feature(tmp_path / "b.pt", transform_name="gaussian_blur")
    with pytest.raises(MixedCacheGroupsError, match="mixes"):
        CachedFeatureDataset(tmp_path)
    ds = CachedFeatureDataset(tmp_path, cache_subset=CACHE_GROUP_TRANSFORMED)
    assert len(ds) == 1
    assert ds.files[0].name == "b.pt"
