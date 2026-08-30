"""Tests for the committed SID-Set operational_1000 manifest.

The manifest CSV (data/manifests/manifest_operational_1000.csv) is
committed to Git and always available; the 3,000 images + 1,000 masks it
references are not (see docs/DATASET.md) and only exist locally once
pulled from the shared team Drive folder. Schema/count/path checks below
always run. Checks that need actual pixel data (mask non-zero-pixel
verification, checksum re-validation) skip cleanly when the dataset root
isn't present on the current machine, mirroring tests/conftest.py's
existing "SID-Set is not available in this environment" pattern.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data import manifests

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "manifest_operational_1000.csv"
DATASET_ROOT = REPO_ROOT / "data" / "raw" / "sid_set_operational_1000"
CHECKSUMS_PATH = DATASET_ROOT / "_checksums.json"

TARGET_PER_LABEL = 1000
EXPECTED_SPLIT_COUNTS = {"train": 800, "val": 100, "test": 100}


def _load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        pytest.skip(f"committed manifest not found at {MANIFEST_PATH}")
    df = pd.read_csv(MANIFEST_PATH, dtype={"image_id": str})
    df["protected"] = df["protected"].astype(bool)
    df["mask_path"] = df["mask_path"].where(df["mask_path"].notna(), None)
    return df


# ---------------------------------------------------------------------------
# Schema / counts
# ---------------------------------------------------------------------------


def test_manifest_passes_shared_schema_validation():
    df = _load_manifest()
    manifests.validate_manifest(df)  # raises on any schema/label/protected-in-train violation


def test_manifest_has_exactly_3000_rows_1000_per_label():
    df = _load_manifest()
    assert len(df) == 3000
    counts = df["label"].value_counts().sort_index()
    assert counts.to_dict() == {0: TARGET_PER_LABEL, 1: TARGET_PER_LABEL, 2: TARGET_PER_LABEL}


def test_manifest_splits_are_800_100_100_per_label():
    df = _load_manifest()
    cross = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    for label in manifests.VALID_LABELS:
        assert cross.loc[label, "train"] == EXPECTED_SPLIT_COUNTS["train"]
        assert cross.loc[label, "val"] == EXPECTED_SPLIT_COUNTS["val"]
        assert cross.loc[label, "test"] == EXPECTED_SPLIT_COUNTS["test"]


def test_manifest_has_no_duplicate_image_ids():
    df = _load_manifest()
    assert not df["image_id"].duplicated().any()


# ---------------------------------------------------------------------------
# Relative paths
# ---------------------------------------------------------------------------


def test_manifest_paths_are_relative():
    df = _load_manifest()
    assert not df["image_path"].astype(str).str.startswith("/").any()
    assert not df["image_path"].astype(str).str.match(r"^[A-Za-z]:\\").any()
    mask_paths = df["mask_path"].dropna().astype(str)
    assert not mask_paths.str.startswith("/").any()
    assert not mask_paths.str.match(r"^[A-Za-z]:\\").any()


def test_every_label_2_row_has_a_mask_path_and_no_other_label_does():
    df = _load_manifest()
    tampered = df[df["label"] == manifests.LABEL_LOCALLY_TAMPERED]
    assert tampered["mask_path"].notna().all()
    non_tampered = df[df["label"] != manifests.LABEL_LOCALLY_TAMPERED]
    assert non_tampered["mask_path"].isna().all()


# ---------------------------------------------------------------------------
# WildFake exclusion
# ---------------------------------------------------------------------------


def test_manifest_contains_no_wildfake_sample():
    df = _load_manifest()
    assert (df["protected"] == False).all()  # noqa: E712
    combined = (
        df["image_id"].astype(str)
        + " " + df["image_path"].astype(str)
        + " " + df["mask_path"].fillna("").astype(str)
        + " " + df["source"].astype(str)
    )
    assert not combined.str.lower().str.contains("wildfake|wild_fake|wild-fake", regex=True).any()
    assert not df["image_id"].astype(str).apply(manifests.is_protected_source).any()
    assert not df["source"].astype(str).apply(manifests.is_protected_source).any()


# ---------------------------------------------------------------------------
# Real-data checks (skip cleanly if the dataset folder isn't present locally)
# ---------------------------------------------------------------------------


def _require_dataset_root() -> Path:
    if not DATASET_ROOT.exists():
        pytest.skip(
            f"SID-Set operational subset not present at {DATASET_ROOT} "
            "(pull it from the shared team Drive folder; see docs/DATASET.md)"
        )
    return DATASET_ROOT


def test_all_manifest_paths_resolve_against_dataset_root():
    df = _load_manifest()
    root = _require_dataset_root()
    missing_images = [p for p in df["image_path"] if not (root / p).exists()]
    assert missing_images == []
    mask_paths = df["mask_path"].dropna()
    missing_masks = [p for p in mask_paths if not (root / p).exists()]
    assert missing_masks == []


def test_every_label_2_mask_is_readable_and_has_a_non_zero_pixel():
    df = _load_manifest()
    root = _require_dataset_root()
    tampered = df[df["label"] == manifests.LABEL_LOCALLY_TAMPERED]
    assert len(tampered) == TARGET_PER_LABEL
    for mask_path in tampered["mask_path"]:
        img = Image.open(root / mask_path).convert("L")
        arr = np.array(img)
        assert (arr > 0).any(), f"mask has no non-zero pixel: {mask_path}"


def test_checksums_match_files_on_disk():
    root = _require_dataset_root()
    if not CHECKSUMS_PATH.exists():
        pytest.skip(f"_checksums.json not present at {CHECKSUMS_PATH}")
    checksums = json.loads(CHECKSUMS_PATH.read_text())
    assert len(checksums) > 0
    for entry in checksums:
        path = root / entry["path"]
        assert path.exists(), f"checksummed file missing on disk: {entry['path']}"
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                hasher.update(chunk)
        assert hasher.hexdigest() == entry["sha256"], f"checksum mismatch: {entry['path']}"
