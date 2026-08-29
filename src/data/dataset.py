"""PyTorch ``Dataset`` for TraceLens-R (Member 1 component).

Wraps a manifest (see ``manifests.py``) and the official transforms (see
``transforms.py``) into the single interface Members 2-4 consume. Every
sample is a dict:

    {
        "image": FloatTensor [3, 224, 224]         # RGB, ImageNet/DINOv2 normalized
        "label": int                                # 0=authentic, 1=fully synthetic, 2=locally tampered
        "mask": FloatTensor [1, 224, 224]           # binary {0.0, 1.0}; all-zero for labels 0/1
        "image_path": str
        "image_id": str
        "transform_metadata": {"transform_name": str, "severity": float | None, "is_geometric": bool}
    }

Transform selection is deterministic: each sample's transform is chosen (and
parameterized, e.g. noise/jitter factors) from a seed derived from
``(seed, image_id)``, so repeated reads of the same sample are always
identical, independent of iteration order or ``DataLoader`` worker count.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import manifests
from .transforms import (
    DEFAULT_IMAGE_SIZE,
    IDENTITY,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MissingMaskError,
    TransformSpec,
    apply_transform,
    binarize_mask,
    zero_mask,
)

__all__ = ["TraceLensDataset", "MissingMaskError"]


def _derive_seed(seed: int, key: str) -> int:
    """Stable (seed, key) -> int, independent of Python's hash randomization."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _to_normalized_tensor(image: Image.Image, normalize: bool) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor


class TraceLensDataset(Dataset):
    """Deterministic dataset over one split of a TraceLens-R manifest.

    Parameters
    ----------
    manifest:
        Either a path to a manifest CSV (see ``manifests.save_manifest``) or
        an already-validated in-memory DataFrame.
    split:
        One of "train", "val", "test". Only rows matching this split are kept.
    dataset_root:
        Base directory that relative ``image_path`` / ``mask_path`` entries
        in the manifest are resolved against. Absolute paths are used as-is.
        Deliberately configurable (never hardcoded) since the real SID-Set
        location varies per machine.
    transform_pool:
        Candidate transforms to sample from (see ``transforms.official_transforms()``).
        Defaults to ``[IDENTITY]`` (no augmentation) if omitted.
    seed:
        Deterministic seed governing transform selection/parameterization.
    image_size:
        Output height/width; the shared model contract is 224.
    normalize:
        Apply ImageNet/DINOv2 mean/std normalization to the image tensor.
    """

    def __init__(
        self,
        manifest: Union[str, Path, pd.DataFrame],
        split: str,
        dataset_root: Union[str, Path],
        transform_pool: Optional[Sequence[TransformSpec]] = None,
        seed: int = manifests.DEFAULT_SEED,
        image_size: int = DEFAULT_IMAGE_SIZE,
        normalize: bool = True,
    ) -> None:
        if split not in manifests.VALID_SPLITS:
            raise ValueError(f"split must be one of {manifests.VALID_SPLITS}, got {split!r}")

        if isinstance(manifest, (str, Path)):
            df = manifests.load_manifest(manifest)
        else:
            df = manifest.copy()
            manifests.validate_manifest(df)

        # Defense in depth: even a hand-edited manifest can't smuggle
        # protected data into the train split without raising here.
        manifests.assert_no_protected_in_train(df)

        self.df = df[df["split"] == split].reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform_pool = list(transform_pool) if transform_pool else [IDENTITY]
        self.seed = seed
        self.image_size = image_size
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, raw_path: Union[str, Path]) -> Path:
        p = Path(raw_path)
        return p if p.is_absolute() else self.dataset_root / p

    def _load_mask(self, row: pd.Series) -> Image.Image:
        label = int(row["label"])
        if label != manifests.LABEL_LOCALLY_TAMPERED:
            # Labels 0 (authentic) and 1 (fully synthetic) never have a real
            # tampering mask -- always an all-zero mask, regardless of
            # whatever mask_path might be present in the manifest.
            return zero_mask((self.image_size, self.image_size))

        mask_path = row["mask_path"]
        if pd.isna(mask_path):
            raise MissingMaskError(
                f"image_id={row['image_id']!r} has label=2 (locally tampered) but "
                "mask_path is missing. Refusing to substitute a zero mask, since "
                "that would silently mislabel a real tampering mask as untampered."
            )
        resolved = self._resolve_path(mask_path)
        if not resolved.exists():
            raise MissingMaskError(
                f"image_id={row['image_id']!r} mask_path does not exist on disk: {resolved}"
            )
        return Image.open(resolved).convert("L")

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        image_path = self._resolve_path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(
                f"image_id={row['image_id']!r} image_path does not exist on disk: {image_path}"
            )

        base_image = Image.open(image_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        base_mask = self._load_mask(row).resize((self.image_size, self.image_size), Image.NEAREST)
        base_mask = binarize_mask(base_mask)

        local_seed = _derive_seed(self.seed, str(row["image_id"]))
        rng = random.Random(local_seed)
        spec = (
            self.transform_pool[rng.randrange(len(self.transform_pool))]
            if len(self.transform_pool) > 1
            else self.transform_pool[0]
        )

        transformed_image, transformed_mask, metadata = apply_transform(
            base_image, base_mask, spec, rng, image_size=self.image_size
        )

        image_tensor = _to_normalized_tensor(transformed_image, normalize=self.normalize)
        mask_arr = np.array(transformed_mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "label": int(row["label"]),
            "mask": mask_tensor,
            "image_path": str(image_path),
            "image_id": str(row["image_id"]),
            "transform_metadata": metadata,
        }
