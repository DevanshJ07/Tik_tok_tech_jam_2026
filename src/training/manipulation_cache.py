"""Cached patch features + aligned masks for manipulation training.

Reuses Member 1 ``TraceLensDataset`` (clean 224×224 RGB + mask alignment) and
the frozen Member 2 backbone. Does not implement DINOv2, transforms, or
mask-alignment logic. AIGC caches under ``data/cache/features`` must not be
used here: they store labels 0/1 and have no masks.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset

from src.data.dataset import TASK_MANIPULATION, TraceLensDataset
from src.data.transforms import CLEAN
from src.models.backbone import DINOv2Backbone, EMBED_DIM, NUM_PATCHES
from src.training.train_manipulation import (
    ManipulationMaskError,
    validate_manipulation_masks,
)

__all__ = [
    "CachedManipulationDataset",
    "cache_manipulation_split",
]


def _atomic_save(record: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(record, tmp)
    os.replace(tmp, path)


def cache_manipulation_split(
    *,
    manifest: str | Path,
    dataset_root: str | Path,
    split: str,
    out_dir: str | Path,
    backbone: DINOv2Backbone,
    seed: int = 42,
    batch_size: int = 8,
    overwrite: bool = False,
    allow_test: bool = False,
) -> Dict[str, int]:
    """Extract frozen patch features and aligned masks for one split.

    Only ``task="manipulation"`` rows (labels 0 and 2) are cached. Label-2
    empty masks are rejected. The backbone is never trained.
    """
    if split == "test" and not allow_test:
        raise ValueError(
            "Refusing to cache the test split during training. "
            "Cache test only after model selection if evaluation requires it."
        )
    if not getattr(backbone, "is_frozen", False):
        raise RuntimeError("DINOv2 backbone must remain frozen while caching.")

    dataset = TraceLensDataset(
        manifest=manifest,
        split=split,
        dataset_root=dataset_root,
        task=TASK_MANIPULATION,
        transform_pool=[CLEAN],
        seed=seed,
        normalize=True,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    computed = skipped = failed = 0
    pending_idx: list[int] = []
    t0 = time.time()
    failures: list[tuple[Any, str]] = []

    def flush() -> None:
        nonlocal computed
        if not pending_idx:
            return
        samples = [dataset[i] for i in pending_idx]
        images = torch.stack([s["image"] for s in samples], dim=0).float()
        _cls, patch_batch = backbone.extract_features(images)
        for i, sample in enumerate(samples):
            image_id = str(sample["image_id"])
            label = int(sample["label"])
            mask = sample["mask"].detach().to(dtype=torch.float32).contiguous()
            try:
                validate_manipulation_masks(mask.unsqueeze(0), torch.tensor([label]))
            except ManipulationMaskError as exc:
                failures.append((image_id, str(exc)))
                continue
            if tuple(patch_batch[i].shape) != (NUM_PATCHES, EMBED_DIM):
                failures.append((image_id, f"bad patch shape {tuple(patch_batch[i].shape)}"))
                continue
            record = {
                "image_id": image_id,
                "label": label,
                "image_path": str(sample["image_path"]),
                "patch_features": patch_batch[i].detach().to("cpu", torch.float32).contiguous(),
                "mask": mask.cpu(),
            }
            _atomic_save(record, out_dir / f"{image_id}.pt")
            computed += 1
        pending_idx.clear()

    for idx in range(len(dataset)):
        image_id = str(dataset.df.iloc[idx]["image_id"])
        dest = out_dir / f"{image_id}.pt"
        if dest.exists() and not overwrite:
            skipped += 1
            continue
        pending_idx.append(idx)
        if len(pending_idx) >= batch_size:
            flush()
        if (idx + 1) % max(batch_size * 10, 100) == 0:
            print(
                f"  [manip/{split}] {idx + 1}/{len(dataset)} "
                f"(computed={computed} skipped={skipped}) {time.time() - t0:.1f}s",
                flush=True,
            )
    flush()
    failed = len(failures)
    for fid, err in failures[:10]:
        print(f"  [manip/{split}] FAILED {fid}: {err}", flush=True)
    if failed:
        raise RuntimeError(
            f"Manipulation cache rejected {failed} sample(s) (empty/missing masks or bad shapes)."
        )
    print(
        f"  [manip/{split}] done: computed={computed} skipped={skipped} "
        f"failed={failed} in {time.time() - t0:.1f}s",
        flush=True,
    )
    return {"computed": computed, "skipped": skipped, "failed": failed, "total": len(dataset)}


class CachedManipulationDataset(Dataset):
    """Reads ``.pt`` files written by :func:`cache_manipulation_split`."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"manipulation cache-dir does not exist: {self.cache_dir}")
        self.files = sorted(self.cache_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files under {self.cache_dir}")
        kept: list[Path] = []
        skipped = 0
        for path in self.files:
            rec = torch.load(path, map_location="cpu", weights_only=False)
            label = int(rec["label"])
            if label == 1:
                skipped += 1
                continue
            if label not in (0, 2):
                skipped += 1
                continue
            kept.append(path)
        self.files = kept
        if not self.files:
            raise ValueError(f"No label-0/2 manipulation cache files under {self.cache_dir}")
        if skipped:
            print(f"[manip-cache] {len(self.files)} usable files, {skipped} skipped (label 1 / other).")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        patch = torch.as_tensor(rec["patch_features"], dtype=torch.float32)
        mask = torch.as_tensor(rec["mask"], dtype=torch.float32)
        label = torch.tensor(int(rec["label"]), dtype=torch.long)
        if patch.shape != (NUM_PATCHES, EMBED_DIM):
            raise ValueError(f"{self.files[idx].name}: patch_features shape {tuple(patch.shape)}")
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] != 1:
            raise ValueError(f"{self.files[idx].name}: mask must be [1,H,W], got {tuple(mask.shape)}")
        return {
            "patch_features": patch,
            "mask": mask,
            "label": label,
        }
