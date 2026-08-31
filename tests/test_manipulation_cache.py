"""Manipulation feature-cache tests. DINOv2 is never downloaded."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.data import manifests  # noqa: E402
from src.models.backbone import DINOv2Backbone, EMBED_DIM, NUM_PATCHES  # noqa: E402
from src.training.manipulation_cache import (  # noqa: E402
    CachedManipulationDataset,
    cache_manipulation_split,
)
from src.training.train_manipulation import ManipulationMaskError  # noqa: E402


class _StubOutput:
    def __init__(self, last_hidden_state: "torch.Tensor") -> None:
        self.last_hidden_state = last_hidden_state


class _StubDINOv2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, EMBED_DIM)
        self.token_bias = nn.Parameter(torch.zeros(1 + NUM_PATCHES, EMBED_DIM))

    def forward(self, pixel_values=None, return_dict=True, **_kwargs):
        pooled = pixel_values.mean(dim=(2, 3))
        base = self.proj(pooled).unsqueeze(1)
        return _StubOutput(base + self.token_bias.unsqueeze(0))


def _stub_backbone() -> DINOv2Backbone:
    return DINOv2Backbone(model_name="stub", device="cpu", model=_StubDINOv2())


def _write_manifest(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "sid"
    (root / "authentic").mkdir(parents=True)
    (root / "locally_tampered").mkdir(parents=True)
    (root / "locally_tampered_masks").mkdir(parents=True)
    Image.new("RGB", (32, 32), (10, 20, 30)).save(root / "authentic" / "a.jpg")
    Image.new("RGB", (32, 32), (40, 50, 60)).save(root / "locally_tampered" / "t.jpg")
    mask = Image.new("L", (32, 32), 0)
    for x in range(8, 20):
        for y in range(8, 20):
            mask.putpixel((x, y), 255)
    mask.save(root / "locally_tampered_masks" / "t.png")
    csv = tmp_path / "manifest.csv"
    csv.write_text(
        "image_id,image_path,label,mask_path,split,source,protected\n"
        "authentic_a,authentic/a.jpg,0,,train,authentic,False\n"
        "locally_tampered_t,locally_tampered/t.jpg,2,locally_tampered_masks/t.png,train,locally_tampered,False\n",
        encoding="utf-8",
    )
    return csv, root


def test_cache_refuses_test_split(tmp_path: Path) -> None:
    csv, root = _write_manifest(tmp_path)
    with pytest.raises(ValueError, match="test split"):
        cache_manipulation_split(
            manifest=csv,
            dataset_root=root,
            split="test",
            out_dir=tmp_path / "out",
            backbone=_stub_backbone(),
        )


def test_cache_allow_test_is_opt_in(tmp_path: Path) -> None:
    csv, root = _write_manifest(tmp_path)
    csv.write_text(
        "image_id,image_path,label,mask_path,split,source,protected\n"
        "authentic_a,authentic/a.jpg,0,,test,authentic,False\n"
        "locally_tampered_t,locally_tampered/t.jpg,2,locally_tampered_masks/t.png,test,locally_tampered,False\n",
        encoding="utf-8",
    )
    out = tmp_path / "test-cache"
    stats = cache_manipulation_split(
        manifest=csv,
        dataset_root=root,
        split="test",
        out_dir=out,
        backbone=_stub_backbone(),
        allow_test=True,
    )
    assert stats["computed"] == 2
    assert stats["failed"] == 0


def test_cache_keeps_labels_0_and_2_and_masks(tmp_path: Path) -> None:
    csv, root = _write_manifest(tmp_path)
    out = tmp_path / "cache"
    stats = cache_manipulation_split(
        manifest=csv,
        dataset_root=root,
        split="train",
        out_dir=out,
        backbone=_stub_backbone(),
    )
    assert stats["computed"] == 2
    assert stats["failed"] == 0
    ds = CachedManipulationDataset(out)
    assert len(ds) == 2
    labels = sorted(int(ds[i]["label"]) for i in range(len(ds)))
    assert labels == [0, 2]
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["patch_features"].shape == (NUM_PATCHES, EMBED_DIM)
        assert sample["mask"].shape[0] == 1
        if int(sample["label"]) == manifests.LABEL_AUTHENTIC:
            assert float(sample["mask"].sum()) == 0.0
        else:
            assert float(sample["mask"].sum()) > 0.0


def test_cache_rejects_empty_label2_mask(tmp_path: Path) -> None:
    csv, root = _write_manifest(tmp_path)
    Image.new("L", (32, 32), 0).save(root / "locally_tampered_masks" / "t.png")
    with pytest.raises(RuntimeError, match="rejected"):
        cache_manipulation_split(
            manifest=csv,
            dataset_root=root,
            split="train",
            out_dir=tmp_path / "cache",
            backbone=_stub_backbone(),
        )


def test_cached_dataset_drops_label_1(tmp_path: Path) -> None:
    rec = {
        "image_id": "syn",
        "label": 1,
        "patch_features": torch.zeros(NUM_PATCHES, EMBED_DIM),
        "mask": torch.zeros(1, 8, 8),
    }
    torch.save(rec, tmp_path / "syn.pt")
    rec0 = {
        "image_id": "auth",
        "label": 0,
        "patch_features": torch.zeros(NUM_PATCHES, EMBED_DIM),
        "mask": torch.zeros(1, 8, 8),
    }
    torch.save(rec0, tmp_path / "auth.pt")
    ds = CachedManipulationDataset(tmp_path)
    assert len(ds) == 1
    assert int(ds[0]["label"]) == 0
