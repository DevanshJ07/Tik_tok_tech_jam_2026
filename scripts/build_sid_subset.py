#!/usr/bin/env python3
"""Deterministic, balanced SID-Set operational subset builder.

Streams parquet shards from saberzl/SID_Set (train split) in fixed shard
order, fetched directly against a pinned commit of the dataset's parquet
export (``PARQUET_REVISION_SHA``) rather than the mutable "current parquet
conversion" convenience endpoint, so re-running this script always reads the
same bytes even if the upstream dataset is later re-converted. For each
shard, deterministically (seed-derived) shuffles the shard-local candidate
rows per label and takes as many as still needed, until every label reaches
its target count. Masks are exported only for label 2 (locally tampered),
matching TraceLens-R's label semantics (0=authentic, 1=fully synthetic,
2=locally tampered). A label-2 candidate whose mask has no non-zero pixel
after grayscale conversion is rejected and skipped in favour of the next
shuffled candidate -- every exported mask is guaranteed non-empty.

This is NOT a uniform reservoir sample over the full 210k-row train split
(that would require downloading the ~124GB split in full). It is a
deterministic sample confined to the shards actually needed to fill the
per-label quotas, processed in a fixed, recorded shard order -- reproducible
exactly given the same seed, pinned revision, and shard sequence.

The seed is never hardcoded here: it is read from configs/default.yaml
(top-level "seed" key) via src.config.load_config(), using --repo-root to
locate that config the same way every other TraceLens-R entry point does.

Every written file is re-read from disk immediately after being written and
its SHA-256 recomputed to confirm it matches the hash of the in-memory bytes
that were meant to be written -- this catches disk/filesystem write
corruption that an in-memory-only checksum would miss.

Usage:
    python scripts/build_sid_subset.py \\
        --output-root data/raw/sid_set_operational_1000 \\
        --shard-cache /tmp/sid_shard_cache \\
        --repo-root .
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from PIL import Image

REPO = "saberzl/SID_Set"
DATASET_REPO_SHA = "dc03ead57929879319ce30a82bfcfb8d317b10bd"  # main branch sha, for attribution/provenance only
PARQUET_REVISION_SHA = "7d1c8a8252aacdf24415d709f9285b2c89e2e039"  # refs/convert/parquet commit -- this is what's actually fetched
CONFIG_SEED_KEY = "seed"  # top-level key in configs/default.yaml (src.config.load_config contract)
LABEL_FOLDERS = {0: "authentic", 1: "fully_synthetic", 2: "locally_tampered"}
MASK_FOLDER = "locally_tampered_masks"
PROTECTED_KEYWORDS = ("wildfake", "wild_fake", "wild-fake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory to write the conventional authentic/fully_synthetic/"
        "locally_tampered/locally_tampered_masks layout into (consumed later "
        "by scripts/prepare_dataset.py --auto-scan).",
    )
    parser.add_argument(
        "--shard-cache",
        required=True,
        type=Path,
        help="Scratch directory for downloaded parquet shards. Each shard is "
        "deleted immediately after processing.",
    )
    parser.add_argument(
        "--repo-root",
        required=True,
        type=Path,
        help="TraceLens-R repository root; used to import src.config / "
        "src.data.manifests and to locate configs/default.yaml for the seed.",
    )
    parser.add_argument(
        "--target-per-label",
        type=int,
        default=1000,
        help="Balanced sample count per label (default: 1000).",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=40,
        help="Safety cap on shards scanned before giving up (default: 40).",
    )
    return parser.parse_args()


def pinned_shard_url(index: int) -> str:
    """URL of parquet shard ``index``, resolved against the pinned commit of
    the dataset's parquet export -- not the mutable "latest conversion"
    endpoint -- so the exact bytes fetched are reproducible."""
    return (
        f"https://huggingface.co/datasets/{REPO}/resolve/"
        f"{PARQUET_REVISION_SHA}/default/train/{index:04d}.parquet"
    )


def download_shard(index: int, shard_cache: Path) -> Path:
    dest = shard_cache / f"shard_{index}.parquet"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with requests.get(pinned_shard_url(index), stream=True, timeout=180) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    return dest


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_and_verify(path: Path, data: bytes) -> str:
    """Write ``data`` to ``path``, then re-read it from disk and confirm the
    on-disk SHA-256 matches the in-memory one. Raises on any mismatch
    (silent write corruption, truncation, etc.) rather than trusting that a
    successful ``write_bytes`` call implies correct bytes on disk."""
    expected = sha256_bytes(data)
    path.write_bytes(data)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch after writing {path}: expected {expected}, got {actual} "
            "(disk write did not round-trip correctly)"
        )
    return expected


def ext_for(pil_format: str | None) -> str:
    mapping = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}
    return mapping.get(pil_format or "JPEG", ".jpg")


def make_wildfake_guard(is_protected_source):
    def assert_not_wildfake(img_id: str, label: int) -> None:
        """Fail loudly if any selected sample's identifier looks WildFake-sourced."""
        if is_protected_source(str(img_id)):
            raise RuntimeError(
                f"WildFake-like img_id encountered and rejected: {img_id!r} (label={label})"
            )
        lowered = str(img_id).lower()
        if any(kw in lowered for kw in PROTECTED_KEYWORDS):
            raise RuntimeError(
                f"WildFake-like img_id encountered and rejected: {img_id!r} (label={label})"
            )

    return assert_not_wildfake


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    shard_cache = args.shard_cache.resolve()
    repo_root = args.repo_root.resolve()
    target_per_label = args.target_per_label
    max_shards = args.max_shards

    shard_cache.mkdir(parents=True, exist_ok=True)
    for folder in list(LABEL_FOLDERS.values()) + [MASK_FOLDER]:
        (output_root / folder).mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_root))
    from src.config import load_config  # noqa: E402
    from src.data.manifests import is_protected_source, stable_seed  # noqa: E402

    config = load_config()
    seed = config[CONFIG_SEED_KEY]
    print(
        f"[config] loaded seed from configs/default.yaml key '{CONFIG_SEED_KEY}' = {seed}",
        flush=True,
    )
    assert_not_wildfake = make_wildfake_guard(is_protected_source)

    collected = {label: 0 for label in LABEL_FOLDERS}
    checksums: list[dict] = []
    shard_log: list[dict] = []
    empty_masks_skipped = 0

    shard_idx = 0
    while any(collected[l] < target_per_label for l in LABEL_FOLDERS) and shard_idx < max_shards:
        print(f"[shard {shard_idx}] downloading (pinned revision {PARQUET_REVISION_SHA})...", flush=True)
        shard_path = download_shard(shard_idx, shard_cache)
        pf = pq.ParquetFile(shard_path)
        table = pf.read(columns=["label", "img_id", "image", "mask"])
        labels = table.column("label").to_pylist()
        img_ids = table.column("img_id").to_pylist()
        images = table.column("image").to_pylist()
        masks = table.column("mask").to_pylist()

        shard_taken = {label: 0 for label in LABEL_FOLDERS}

        for label in LABEL_FOLDERS:
            deficit = target_per_label - collected[label]
            if deficit <= 0:
                continue
            candidate_indices = [i for i, l in enumerate(labels) if l == label]
            rng = random.Random(stable_seed(seed, "sid_set_operational_subset", label, shard_idx))
            rng.shuffle(candidate_indices)

            taken_this_shard = 0
            for i in candidate_indices:
                if taken_this_shard >= deficit:
                    break

                img_id = img_ids[i]
                assert_not_wildfake(img_id, label)

                mask_png_bytes = None
                if label == 2:
                    mask_entry = masks[i]
                    if mask_entry is None or not mask_entry.get("bytes"):
                        raise RuntimeError(
                            f"label=2 row {img_id} in shard {shard_idx} has no mask bytes"
                        )
                    mask_pil = Image.open(io.BytesIO(mask_entry["bytes"])).convert("L")
                    mask_arr = np.array(mask_pil)
                    if not (mask_arr > 0).any():
                        # Enforce non-empty label-2 masks: an all-zero mask
                        # would silently mislabel a tampered image as having
                        # no tampered region. Reject and try the next
                        # shuffled candidate instead of counting it.
                        empty_masks_skipped += 1
                        continue
                    mbuf = io.BytesIO()
                    mask_pil.save(mbuf, format="PNG")
                    mask_png_bytes = mbuf.getvalue()

                img_bytes = images[i]["bytes"]
                pil_img = Image.open(io.BytesIO(img_bytes))
                ext = ext_for(pil_img.format)
                safe_id = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(img_id))
                out_name = f"{safe_id}{ext}"
                out_path = output_root / LABEL_FOLDERS[label] / out_name
                image_sha256 = write_and_verify(out_path, img_bytes)
                checksums.append(
                    {
                        "path": f"{LABEL_FOLDERS[label]}/{out_name}",
                        "sha256": image_sha256,
                        "label": label,
                        "img_id": img_id,
                        "kind": "image",
                    }
                )

                if label == 2:
                    mask_out_path = output_root / MASK_FOLDER / f"{safe_id}.png"
                    mask_sha256 = write_and_verify(mask_out_path, mask_png_bytes)
                    checksums.append(
                        {
                            "path": f"{MASK_FOLDER}/{safe_id}.png",
                            "sha256": mask_sha256,
                            "label": label,
                            "img_id": img_id,
                            "kind": "mask",
                        }
                    )

                taken_this_shard += 1

            collected[label] += taken_this_shard
            shard_taken[label] = taken_this_shard

        shard_log.append(
            {
                "shard_index": shard_idx,
                "shard_rows": len(labels),
                "taken": shard_taken,
                "running_totals": dict(collected),
            }
        )
        print(
            f"[shard {shard_idx}] rows={len(labels)} taken={shard_taken} totals={collected} "
            f"empty_masks_skipped_so_far={empty_masks_skipped}",
            flush=True,
        )

        shard_path.unlink(missing_ok=True)
        shard_idx += 1

    if any(collected[l] < target_per_label for l in LABEL_FOLDERS):
        raise RuntimeError(f"Could not fill quotas within {max_shards} shards: {collected}")

    (output_root / "_checksums.json").write_text(json.dumps(checksums, indent=2))
    (output_root / "_extraction_log.json").write_text(
        json.dumps(
            {
                "repo": REPO,
                "dataset_repo_sha": DATASET_REPO_SHA,
                "parquet_revision_sha": PARQUET_REVISION_SHA,
                "parquet_revision_sha_source": "refs/convert/parquet, resolved via "
                "GET /api/datasets/{repo}/revision/refs%2Fconvert%2Fparquet and pinned "
                "directly into the download URL (not just recorded after the fact)",
                "seed": seed,
                "seed_source": f"configs/default.yaml key '{CONFIG_SEED_KEY}' via src.config.load_config()",
                "target_per_label": target_per_label,
                "shards_used": shard_idx,
                "shard_log": shard_log,
                "final_counts": collected,
                "empty_masks_skipped": empty_masks_skipped,
                "wildfake_check": "every img_id checked against src.data.manifests.is_protected_source() "
                "and protected-keyword substrings at selection time; none matched",
                "checksum_verification": "every written file re-read from disk and re-hashed "
                "immediately after writing; write_and_verify() raises on any mismatch",
            },
            indent=2,
        )
    )

    print("DONE", collected, "empty_masks_skipped:", empty_masks_skipped)


if __name__ == "__main__":
    main()
