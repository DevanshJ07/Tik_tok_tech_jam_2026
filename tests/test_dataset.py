"""Tests for src/data/dataset.py and src/data/manifests.py using synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import raw_record
from src.data import manifests
from src.data.dataset import MissingMaskError, TraceLensDataset
from src.data.manifests import ProtectedDataError, build_manifest, assign_splits, validate_manifest, ManifestValidationError
from src.data.transforms import official_transforms

IMAGE_SIZE = 224


def _small_balanced_manifest(tmp_path, n_per_class: int = 10) -> pd.DataFrame:
    records = []
    for label in manifests.VALID_LABELS:
        for i in range(n_per_class):
            records.append(
                raw_record(tmp_path, image_id=f"cls{label}_{i:03d}", label=label, color=(10 * label, 50, 200))
            )
    df = build_manifest(records)
    return assign_splits(df, seed=42)


# ---------------------------------------------------------------------------
# Dataset sample shape / content
# ---------------------------------------------------------------------------


def test_dataset_sample_shapes_and_label(tmp_path):
    df = _small_balanced_manifest(tmp_path, n_per_class=10)
    dataset = TraceLensDataset(df, split="train", dataset_root=tmp_path)

    assert len(dataset) > 0
    sample = dataset[0]

    assert sample["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert sample["mask"].shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert sample["label"] in manifests.VALID_LABELS
    assert isinstance(sample["image_path"], str)
    assert isinstance(sample["image_id"], str)


def test_all_samples_have_valid_labels(tmp_path):
    df = _small_balanced_manifest(tmp_path, n_per_class=6)
    for split in manifests.VALID_SPLITS:
        dataset = TraceLensDataset(df, split=split, dataset_root=tmp_path)
        for i in range(len(dataset)):
            assert dataset[i]["label"] in manifests.VALID_LABELS


def test_zero_mask_for_authentic_and_synthetic(tmp_path):
    df = _small_balanced_manifest(tmp_path, n_per_class=8)
    for split in manifests.VALID_SPLITS:
        dataset = TraceLensDataset(df, split=split, dataset_root=tmp_path)
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample["label"] in (manifests.LABEL_AUTHENTIC, manifests.LABEL_FULLY_SYNTHETIC):
                assert float(sample["mask"].sum()) == 0.0


def test_tampered_mask_is_nonzero_with_identity_transform(tmp_path):
    df = _small_balanced_manifest(tmp_path, n_per_class=8)
    found_tampered = False
    for split in manifests.VALID_SPLITS:
        dataset = TraceLensDataset(df, split=split, dataset_root=tmp_path)
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample["label"] == manifests.LABEL_LOCALLY_TAMPERED:
                found_tampered = True
                assert float(sample["mask"].sum()) > 0.0
    assert found_tampered


def test_mask_values_are_binary(tmp_path):
    df = _small_balanced_manifest(tmp_path, n_per_class=6)
    dataset = TraceLensDataset(df, split="train", dataset_root=tmp_path, transform_pool=official_transforms())
    for i in range(len(dataset)):
        mask = dataset[i]["mask"]
        values = set(torch_unique(mask))
        assert values <= {0.0, 1.0}


def torch_unique(tensor):
    return set(tensor.unique().tolist())


def test_transform_metadata_present(tmp_path):
    df = _small_balanced_manifest(tmp_path, n_per_class=6)
    dataset = TraceLensDataset(df, split="train", dataset_root=tmp_path, transform_pool=official_transforms())
    sample = dataset[0]
    metadata = sample["transform_metadata"]
    assert set(metadata.keys()) == {"transform_name", "severity", "is_geometric"}


# ---------------------------------------------------------------------------
# Deterministic transform selection
# ---------------------------------------------------------------------------


def test_dataset_transform_selection_is_deterministic(tmp_path):
    df = _small_balanced_manifest(tmp_path, n_per_class=6)
    pool = official_transforms()

    ds_a = TraceLensDataset(df, split="train", dataset_root=tmp_path, transform_pool=pool, seed=42)
    ds_b = TraceLensDataset(df, split="train", dataset_root=tmp_path, transform_pool=pool, seed=42)

    for i in range(len(ds_a)):
        sample_a, sample_b = ds_a[i], ds_b[i]
        assert sample_a["transform_metadata"] == sample_b["transform_metadata"]
        assert torch_allclose(sample_a["image"], sample_b["image"])
        assert torch_allclose(sample_a["mask"], sample_b["mask"])


def torch_allclose(a, b):
    import torch

    return torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Deterministic, group-aware split assignment
# ---------------------------------------------------------------------------


def test_split_assignment_is_deterministic_for_seed_42(tmp_path):
    records = [raw_record(tmp_path, image_id=f"img_{i:03d}", label=i % 3) for i in range(60)]
    df = build_manifest(records)

    split_a = assign_splits(df, seed=42)
    split_b = assign_splits(df, seed=42)

    assert split_a.set_index("image_id")["split"].equals(split_b.set_index("image_id")["split"])


def test_split_assignment_respects_80_10_10_roughly(tmp_path):
    records = [raw_record(tmp_path, image_id=f"img_{i:03d}", label=i % 3) for i in range(150)]
    df = build_manifest(records)
    split_df = assign_splits(df, seed=42)

    counts = split_df["split"].value_counts(normalize=True)
    assert counts["train"] == pytest.approx(0.8, abs=0.05)
    assert counts["val"] == pytest.approx(0.1, abs=0.05)
    assert counts["test"] == pytest.approx(0.1, abs=0.05)


def test_duplicate_image_id_rejected_at_build(tmp_path):
    r1 = raw_record(tmp_path, image_id="dup", label=0)
    r2 = raw_record(tmp_path, image_id="dup", label=1)
    with pytest.raises(ManifestValidationError):
        build_manifest([r1, r2])


def test_duplicate_content_hash_never_crosses_splits(tmp_path):
    # Two different image_ids pointing at byte-identical image content.
    records = [raw_record(tmp_path, image_id=f"img_{i:03d}", label=i % 3, color=(1, 2, 3)) for i in range(40)]

    dup_a = raw_record(tmp_path, image_id="dup_a", label=0, color=(250, 10, 10))
    dup_b_path = tmp_path / "dup_b.png"
    dup_b_path.write_bytes((tmp_path / "dup_a.png").read_bytes())
    dup_b = manifests.RawRecord(image_id="dup_b", image_path=str(dup_b_path), label=0, source="synthetic")

    df = build_manifest(records + [dup_a, dup_b])
    split_df = assign_splits(df, seed=42)

    split_by_id = split_df.set_index("image_id")["split"]
    assert split_by_id["dup_a"] == split_by_id["dup_b"]


def test_duplicate_content_hash_with_different_labels_never_crosses_splits(tmp_path):
    """Regression test: a byte-identical image saved under two different
    image_ids *and* two different labels must always land in the same
    split. Stratifying by label before grouping duplicate-hash rows used to
    let such a pair be assigned independently in each label's stratum,
    which could put one copy in "train" and the other in "test" -- a direct
    train/test leak. Swept across many seeds since any single seed could
    pass by chance."""
    for seed in range(25):
        # image_ids are suffixed with `seed` so every iteration's files
        # coexist in the same tmp_path without needing a subdirectory.
        records = [
            raw_record(tmp_path, image_id=f"img_{seed}_{i:03d}", label=i % 3, color=(1, 2, 3))
            for i in range(40)
        ]

        # Byte-identical image content, different image_ids, different labels.
        dup_a = raw_record(tmp_path, image_id=f"dup_a_{seed}", label=0, color=(250, 10, 10))
        dup_b_path = tmp_path / f"dup_b_{seed}.png"
        dup_b_path.write_bytes((tmp_path / f"dup_a_{seed}.png").read_bytes())
        dup_b = manifests.RawRecord(
            image_id=f"dup_b_{seed}", image_path=str(dup_b_path), label=1, source="synthetic"
        )

        df = build_manifest(records + [dup_a, dup_b])
        split_df = assign_splits(df, seed=seed)

        split_by_id = split_df.set_index("image_id")["split"]
        assert split_by_id[f"dup_a_{seed}"] == split_by_id[f"dup_b_{seed}"], (
            f"seed={seed}: byte-identical images with different labels landed in "
            f"different splits ({split_by_id[f'dup_a_{seed}']!r} vs {split_by_id[f'dup_b_{seed}']!r})"
        )


def test_no_image_id_crosses_splits(tmp_path):
    records = [raw_record(tmp_path, image_id=f"img_{i:03d}", label=i % 3) for i in range(30)]
    df = build_manifest(records)
    split_df = assign_splits(df, seed=42)
    # One row per image_id by construction -- assert that invariant plus a
    # valid, single split value per id.
    assert not split_df["image_id"].duplicated().any()
    assert split_df["split"].isin(manifests.VALID_SPLITS).all()


# ---------------------------------------------------------------------------
# Protected-data rejection
# ---------------------------------------------------------------------------


def test_protected_source_rejected_by_default(tmp_path):
    record = raw_record(tmp_path, image_id="wf_1", label=1, source="wildfake_v2")
    with pytest.raises(ProtectedDataError):
        build_manifest([record])


def test_protected_data_never_assigned_to_train(tmp_path):
    protected_records = [raw_record(tmp_path, image_id=f"wf_{i}", label=1, source="wildfake") for i in range(10)]
    normal_records = [raw_record(tmp_path, image_id=f"norm_{i}", label=1, source="synthetic") for i in range(10)]

    df = build_manifest(protected_records + normal_records, allow_protected=True)
    split_df = assign_splits(df, seed=42)

    protected_splits = split_df.loc[split_df["protected"], "split"]
    assert "train" not in set(protected_splits)


def test_dataset_rejects_manifest_with_protected_row_in_train(tmp_path):
    df = _small_balanced_manifest(tmp_path, n_per_class=4)
    # Simulate a hand-edited manifest sneaking protected data into train.
    df = df.copy()
    df.loc[df.index[0], "split"] = "train"
    df.loc[df.index[0], "protected"] = True

    with pytest.raises(ProtectedDataError):
        TraceLensDataset(df, split="train", dataset_root=tmp_path)


# ---------------------------------------------------------------------------
# Missing-mask handling for label 2
# ---------------------------------------------------------------------------


def test_missing_mask_file_raises_explicitly(tmp_path):
    record = raw_record(tmp_path, image_id="tampered_missing", label=manifests.LABEL_LOCALLY_TAMPERED)
    df = build_manifest([record])
    df.loc[df.index[0], "split"] = "train"
    # Point mask_path at a file that doesn't actually exist on disk.
    df.loc[df.index[0], "mask_path"] = str(tmp_path / "does_not_exist.png")

    dataset = TraceLensDataset(df, split="train", dataset_root=tmp_path)
    with pytest.raises(MissingMaskError):
        dataset[0]


def test_null_mask_path_for_tampered_sample_rejected_by_validation(tmp_path):
    record = raw_record(tmp_path, image_id="tampered_no_mask", label=manifests.LABEL_LOCALLY_TAMPERED, with_mask=False)
    df = build_manifest([record])
    df.loc[df.index[0], "split"] = "train"

    with pytest.raises(ManifestValidationError):
        validate_manifest(df)
