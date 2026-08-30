"""Synthetic-data helpers shared by Member 1's tests.

SID-Set is not available in this environment, so every test in this package
builds tiny synthetic images/masks on the fly with PIL/NumPy instead of
relying on any real dataset.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from src.data import manifests


def make_synthetic_image(
    path: Path,
    size: Tuple[int, int] = (64, 64),
    color: Tuple[int, int, int] = (120, 60, 200),
    unique_key: Optional[str] = None,
) -> None:
    """Solid-color image. When ``unique_key`` is given, a corner pixel is
    stamped from its hash so otherwise-identical solid-color images (e.g.
    many samples sharing a per-class color) still hash differently -- tests
    that want genuine duplicate-hash behaviour create it explicitly by
    copying file bytes instead."""
    image = Image.new("RGB", size, color)
    if unique_key is not None:
        digest = hashlib.sha256(unique_key.encode("utf-8")).digest()
        image.putpixel((0, 0), (digest[0], digest[1], digest[2]))
    image.save(path)


def make_synthetic_mask(path: Path, size: Tuple[int, int] = (64, 64), box: Optional[Tuple[int, int, int, int]] = None) -> None:
    """``box`` = (x0, y0, x1, y1) marks the "on" (255) region; else all-zero."""
    width, height = size
    arr = np.zeros((height, width), dtype=np.uint8)
    if box is not None:
        x0, y0, x1, y1 = box
        arr[y0:y1, x0:x1] = 255
    Image.fromarray(arr, mode="L").save(path)


def raw_record(
    tmp_path: Path,
    image_id: str,
    label: int,
    *,
    source: str = "synthetic",
    color: Tuple[int, int, int] = (120, 60, 200),
    mask_box: Optional[Tuple[int, int, int, int]] = None,
    protected: Optional[bool] = None,
    size: Tuple[int, int] = (64, 64),
    with_mask: bool = True,
) -> manifests.RawRecord:
    """Write a synthetic image (+ mask, if label==2) under tmp_path."""
    image_path = tmp_path / f"{image_id}.png"
    make_synthetic_image(image_path, size=size, color=color, unique_key=image_id)

    mask_path = None
    if label == manifests.LABEL_LOCALLY_TAMPERED and with_mask:
        mask_path = tmp_path / f"{image_id}_mask.png"
        default_box = (0, 0, size[0] // 2, size[1] // 2)
        make_synthetic_mask(mask_path, size=size, box=mask_box or default_box)

    return manifests.RawRecord(
        image_id=image_id,
        image_path=str(image_path),
        label=label,
        source=source,
        mask_path=str(mask_path) if mask_path else None,
        protected=protected,
    )
