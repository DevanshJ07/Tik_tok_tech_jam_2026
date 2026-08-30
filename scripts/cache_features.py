"""Pre-extract frozen DINOv2 features to disk (Member 2 component).

Running the backbone is the expensive part of training on CPU. This script
runs ``facebook/dinov2-small`` **once** per sample and writes a small
PyTorch ``.pt`` file so later training / evaluation can skip the backbone
entirely.

One cached file per sample::

    {
        "image_id":           str,
        "label":              int,           # 0 / 1 for task=aigc
        "transform_name":     str,           # "clean" for the clean pass
        "transform_severity": value,         # float | "none" | None (as produced by the dataset)
        "cls_features":       Tensor[384],   # float32, CPU
        "patch_features":     Tensor[256, 384],
    }

Layout (clean and transformed kept separate, linked by ``image_id``)::

    <out-dir>/
        clean/<image_id>.pt
        transformed/<image_id>.pt      # one deterministic transform per image

Key properties
--------------
* CPU by default (``--device cpu``).
* Existing files are skipped unless ``--overwrite`` -- and the check happens
  *before* the backbone runs (the id comes from the manifest, not the image),
  so nothing is recomputed unnecessarily.
* Every feature tensor's shape is validated before it is written.
* Writes are atomic (temp file + ``os.replace``).

Examples
--------
::

    python scripts/cache_features.py \
        --manifest data/manifests/manifest.csv \
        --dataset-root /path/to/SID-Set \
        --out-dir data/cache/features \
        --splits train val \
        --include-transformed
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

# Make ``src`` importable when run as a plain script (python scripts/...).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.backbone import EMBED_DIM, NUM_PATCHES, DINOv2Backbone  # noqa: E402

CLEAN_SUBDIR = "clean"
TRANSFORMED_SUBDIR = "transformed"


# ===========================================================================
# Record construction / validation
# ===========================================================================
def validate_feature_shapes(cls_features: torch.Tensor, patch_features: torch.Tensor) -> None:
    """Raise ValueError if a single-sample feature pair has the wrong shape."""
    if tuple(cls_features.shape) != (EMBED_DIM,):
        raise ValueError(
            f"cls_features must be [{EMBED_DIM}], got {tuple(cls_features.shape)}"
        )
    if tuple(patch_features.shape) != (NUM_PATCHES, EMBED_DIM):
        raise ValueError(
            f"patch_features must be [{NUM_PATCHES}, {EMBED_DIM}], "
            f"got {tuple(patch_features.shape)}"
        )
    if not torch.isfinite(cls_features).all() or not torch.isfinite(patch_features).all():
        raise ValueError("non-finite value in extracted features")


def build_record(
    *,
    image_id: str,
    label: int,
    transform_name: str,
    transform_severity: Any,
    cls_features: torch.Tensor,
    patch_features: torch.Tensor,
) -> Dict[str, Any]:
    """Assemble (and validate) one cache record."""
    cls_features = cls_features.detach().to("cpu", torch.float32).contiguous()
    patch_features = patch_features.detach().to("cpu", torch.float32).contiguous()
    validate_feature_shapes(cls_features, patch_features)
    return {
        "image_id": str(image_id),
        "label": int(label),
        "transform_name": str(transform_name),
        "transform_severity": transform_severity,
        "cls_features": cls_features,
        "patch_features": patch_features,
    }


def atomic_save(record: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(record, tmp)
    os.replace(tmp, path)


def load_cached_feature(path: str | Path) -> Dict[str, Any]:
    """Load one cache file (helper for downstream members / tests)."""
    rec = torch.load(Path(path), map_location="cpu", weights_only=False)
    validate_feature_shapes(
        torch.as_tensor(rec["cls_features"]), torch.as_tensor(rec["patch_features"])
    )
    return rec


# ===========================================================================
# Caching one dataset
# ===========================================================================
def cache_dataset(
    dataset,
    backbone: DINOv2Backbone,
    out_dir: Path,
    *,
    batch_size: int,
    overwrite: bool,
    limit: Optional[int] = None,
    label_prefix: str = "",
) -> Dict[str, int]:
    """Extract + save features for every sample in ``dataset``.

    Manual mini-batching: output paths are resolved from ``dataset.df`` (no
    image decode), so already-cached samples are skipped before any backbone
    compute happens.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(dataset)
    if limit is not None:
        n_total = min(n_total, limit)

    computed = skipped = failed = 0
    pending: List[int] = []
    t0 = time.time()

    def flush() -> None:
        nonlocal computed, failed
        if not pending:
            return
        samples = []
        for idx in pending:
            try:
                samples.append(dataset[idx])
            except Exception as exc:  # noqa: BLE001 - keep going, report at end
                failed_ids.append((idx, repr(exc)))
        if not samples:
            pending.clear()
            return
        images = torch.stack([s["image"] for s in samples], dim=0).float()
        cls_batch, patch_batch = backbone.extract_features(images)
        for i, sample in enumerate(samples):
            image_id = str(sample["image_id"])
            meta = sample.get("transform_metadata", {}) or {}
            try:
                record = build_record(
                    image_id=image_id,
                    label=int(sample["label"]),
                    transform_name=meta.get("transform_name", "clean"),
                    transform_severity=meta.get("severity", None),
                    cls_features=cls_batch[i],
                    patch_features=patch_batch[i],
                )
                atomic_save(record, out_dir / f"{image_id}.pt")
                computed += 1
            except Exception as exc:  # noqa: BLE001
                failed_ids.append((image_id, repr(exc)))
        pending.clear()

    failed_ids: List[Any] = []
    for idx in range(n_total):
        image_id = str(dataset.df.iloc[idx]["image_id"])
        out_path = out_dir / f"{image_id}.pt"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        pending.append(idx)
        if len(pending) >= batch_size:
            flush()
        if (idx + 1) % max(batch_size * 10, 100) == 0:
            print(
                f"  [{label_prefix}] {idx + 1}/{n_total} "
                f"(computed={computed} skipped={skipped}) {time.time() - t0:.1f}s"
            )
    flush()

    failed = len(failed_ids)
    for fid, err in failed_ids[:10]:
        print(f"  [{label_prefix}] FAILED {fid}: {err}")
    print(
        f"  [{label_prefix}] done: computed={computed} skipped={skipped} "
        f"failed={failed} in {time.time() - t0:.1f}s"
    )
    return {"computed": computed, "skipped": skipped, "failed": failed, "total": n_total}


# ===========================================================================
# CLI
# ===========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cache frozen DINOv2 features for TraceLens-R.")
    p.add_argument("--manifest", required=True, help="Path to a manifest CSV.")
    p.add_argument("--dataset-root", required=True,
                   help="Root dir that manifest image_path entries resolve against.")
    p.add_argument("--out-dir", default="data/cache/features",
                   help="Base output directory (clean/ and transformed/ created inside).")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--task", default="aigc", choices=["aigc", "manipulation"],
                   help="Label filter applied by TraceLensDataset. Baseline training uses 'aigc'.")
    p.add_argument("--include-transformed", action="store_true",
                   help="Also cache one deterministic official transform per image.")
    p.add_argument("--clean", default=True, action=argparse.BooleanOptionalAction,
                   help="Cache the clean (un-transformed) pass. On by default (use --no-clean to skip).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--backbone-name", default="facebook/dinov2-small")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="Recompute even if the .pt file exists.")
    p.add_argument("--limit", type=int, default=None, help="Cap samples per split per pass (debug).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)

    # Lazy imports: only needed for the real run, keep --help / import cheap.
    from src.data.dataset import TraceLensDataset
    from src.data.transforms import CLEAN, official_transforms

    out_base = Path(args.out_dir)
    backbone = DINOv2Backbone(model_name=args.backbone_name, device=args.device)
    print(f"[cache] backbone={args.backbone_name} device={args.device} frozen={backbone.is_frozen}")

    results: Dict[str, Any] = {}
    for split in args.splits:
        print(f"[cache] === split={split} ===")

        if args.clean:
            clean_ds = TraceLensDataset(
                manifest=args.manifest,
                split=split,
                dataset_root=args.dataset_root,
                task=args.task,
                transform_pool=[CLEAN],
                seed=args.seed,
                normalize=True,
            )
            results[f"{split}/clean"] = cache_dataset(
                clean_ds, backbone, out_base / CLEAN_SUBDIR / split,
                batch_size=args.batch_size, overwrite=args.overwrite,
                limit=args.limit, label_prefix=f"{split}/clean",
            )

        if args.include_transformed:
            transformed_ds = TraceLensDataset(
                manifest=args.manifest,
                split=split,
                dataset_root=args.dataset_root,
                task=args.task,
                transform_pool=official_transforms(),
                seed=args.seed,
                normalize=True,
            )
            results[f"{split}/transformed"] = cache_dataset(
                transformed_ds, backbone, out_base / TRANSFORMED_SUBDIR / split,
                batch_size=args.batch_size, overwrite=args.overwrite,
                limit=args.limit, label_prefix=f"{split}/transformed",
            )

    print(f"[cache] summary: {results}")
    return results


if __name__ == "__main__":  # pragma: no cover
    main()
