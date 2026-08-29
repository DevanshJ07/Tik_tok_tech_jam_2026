#!/usr/bin/env python3
"""Build a TraceLens-R manifest CSV for one dataset stage (smoke/initial/final).

SID-Set is not available in this environment, so this script does not assume
any specific real SID-Set directory layout. It supports two input modes:

  1. ``--index-csv``: a raw index CSV you (or someone with SID-Set access)
     produce, with columns ``image_id,image_path,label,source`` and
     optionally ``mask_path,protected``. Paths may be absolute or relative
     to ``--dataset-root``. This is the recommended, layout-agnostic mode.

  2. ``--auto-scan``: assumes a conventional directory layout under
     ``--dataset-root``:

         <dataset_root>/authentic/*.{jpg,jpeg,png}
         <dataset_root>/fully_synthetic/*.{jpg,jpeg,png}
         <dataset_root>/locally_tampered/*.{jpg,jpeg,png}
         <dataset_root>/locally_tampered_masks/<same-stem>.{png,jpg}

     This is a reasonable convention, not a claim about SID-Set's real
     layout -- adjust ``discover_records_from_directory`` once the real
     layout is known.

Either way, any record whose source matches a protected-data keyword (e.g.
"wildfake") makes the whole run fail immediately unless ``--allow-protected``
is passed explicitly, and even then it can never land in the "train" split
(enforced by ``manifests.assign_splits``).

Usage:
    python scripts/prepare_dataset.py \\
        --dataset-root /path/to/sid_set \\
        --output-dir data/manifests \\
        --stage smoke \\
        --index-csv /path/to/raw_index.csv
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import manifests  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_CLASS_FOLDERS = {
    "authentic": manifests.LABEL_AUTHENTIC,
    "fully_synthetic": manifests.LABEL_FULLY_SYNTHETIC,
    "locally_tampered": manifests.LABEL_LOCALLY_TAMPERED,
}
_MASK_FOLDER = "locally_tampered_masks"


def discover_records_from_directory(dataset_root: Path) -> list[manifests.RawRecord]:
    """Scan the conventional ``<root>/<class_folder>/*`` layout described above."""
    records: list[manifests.RawRecord] = []
    for folder_name, label in _CLASS_FOLDERS.items():
        folder = dataset_root / folder_name
        if not folder.is_dir():
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = None
            if label == manifests.LABEL_LOCALLY_TAMPERED:
                candidate_dir = dataset_root / _MASK_FOLDER
                for ext in IMAGE_EXTENSIONS:
                    candidate = candidate_dir / f"{image_path.stem}{ext}"
                    if candidate.exists():
                        mask_path = str(candidate)
                        break
            records.append(
                manifests.RawRecord(
                    image_id=f"{folder_name}_{image_path.stem}",
                    image_path=str(image_path),
                    label=label,
                    source=folder_name,
                    mask_path=mask_path,
                )
            )
    return records


def load_records_from_index_csv(index_csv: Path, dataset_root: Path) -> list[manifests.RawRecord]:
    df = pd.read_csv(index_csv, dtype=str)
    required = {"image_id", "image_path", "label", "source"}
    missing = required - set(df.columns)
    if missing:
        raise manifests.ManifestValidationError(f"--index-csv is missing required columns: {sorted(missing)}")

    records = []
    for _, row in df.iterrows():
        image_path = row["image_path"]
        if not Path(image_path).is_absolute():
            image_path = str(dataset_root / image_path)

        mask_path = row.get("mask_path")
        if pd.isna(mask_path) or mask_path in (None, ""):
            mask_path = None
        elif not Path(mask_path).is_absolute():
            mask_path = str(dataset_root / mask_path)

        protected_raw = row.get("protected")
        protected = None if pd.isna(protected_raw) or protected_raw in (None, "") else str(protected_raw).lower() in ("1", "true", "yes")

        records.append(
            manifests.RawRecord(
                image_id=str(row["image_id"]),
                image_path=image_path,
                label=int(row["label"]),
                source=str(row["source"]),
                mask_path=mask_path,
                protected=protected,
            )
        )
    return records


def subsample_per_class(df: pd.DataFrame, count_per_class: int, seed: int) -> pd.DataFrame:
    """Deterministically cap each label to at most ``count_per_class`` rows."""
    parts = []
    for label, sub in df.groupby("label"):
        if len(sub) <= count_per_class:
            parts.append(sub)
            continue
        rng = random.Random(manifests.stable_seed(seed, "subsample", label))
        chosen_idx = rng.sample(list(sub.index), count_per_class)
        parts.append(sub.loc[chosen_idx])
    return pd.concat(parts).reset_index(drop=True)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Root directory containing SID-Set images (configurable; never hardcoded).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write the manifest CSV into.")
    parser.add_argument("--stage", required=True, choices=sorted(manifests.STAGE_COUNTS_PER_CLASS), help="Dataset stage; determines images-per-class cap.")
    parser.add_argument("--index-csv", type=Path, default=None, help="Raw index CSV (image_id,image_path,label,source[,mask_path,protected]).")
    parser.add_argument("--auto-scan", action="store_true", help="Discover images from the conventional directory layout under --dataset-root instead of --index-csv.")
    parser.add_argument("--seed", type=int, default=manifests.DEFAULT_SEED, help="Deterministic seed for subsampling and split assignment.")
    parser.add_argument("--allow-protected", action="store_true", help="Acknowledge that protected (e.g. WildFake) records are present. They still can never enter the train split.")
    return parser.parse_args()


def main() -> None:
    args = build_args()

    if bool(args.index_csv) == bool(args.auto_scan):
        raise SystemExit("Specify exactly one of --index-csv or --auto-scan.")

    if args.index_csv is not None:
        records = load_records_from_index_csv(args.index_csv, args.dataset_root)
    else:
        records = discover_records_from_directory(args.dataset_root)

    if not records:
        raise SystemExit(f"No images discovered under {args.dataset_root}. Nothing to do.")

    raw_manifest = manifests.build_manifest(records, allow_protected=args.allow_protected)

    count_per_class = manifests.STAGE_COUNTS_PER_CLASS[args.stage]
    sampled = subsample_per_class(raw_manifest, count_per_class, args.seed)

    split_manifest = manifests.assign_splits(sampled, seed=args.seed)
    manifests.validate_manifest(split_manifest)

    output_path = args.output_dir / f"manifest_{args.stage}.csv"
    manifests.save_manifest(split_manifest, output_path)

    print(f"Wrote {len(split_manifest)} rows to {output_path}")
    print(split_manifest.groupby(["label", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
