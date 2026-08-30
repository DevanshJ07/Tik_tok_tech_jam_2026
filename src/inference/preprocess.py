"""Single-image preprocess matching Member 1 training tensors.

Inference applies the clean (no-augmentation) path: RGB convert, resize to
224×224, then ImageNet / DINOv2 normalisation. Official robustness
transforms are not applied here; those belong to dataset construction.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.data.transforms import (
    CLEAN,
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    apply_transform,
    zero_mask,
)

IMAGE_SIZE = DEFAULT_IMAGE_SIZE


class ImagePreprocessError(ValueError):
    """Raised when an image cannot be prepared for the model."""


def load_rgb_image(image_path: str | Path) -> Image.Image:
    path = Path(image_path)
    if not path.is_file():
        raise ImagePreprocessError(f"Image not found: {path}")
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.load()
            return rgb
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as exc:
        raise ImagePreprocessError(f"Unreadable image: {path}") from exc


def image_to_model_tensor(image: Image.Image, *, normalize: bool = True) -> torch.Tensor:
    """Convert a 224×224 RGB PIL image to ``[3, 224, 224]`` float32.

    Matches ``src.data.dataset._to_normalized_tensor`` (Member 1).
    """
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        raise ImagePreprocessError(
            f"Expected a {IMAGE_SIZE}x{IMAGE_SIZE} RGB image, got {image.size}"
        )
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor


def preprocess_image_path(
    image_path: str | Path,
    *,
    image_size: int = IMAGE_SIZE,
    normalize: bool = True,
) -> torch.Tensor:
    """Return a training-compatible tensor ``[3, 224, 224]``."""
    if image_size != IMAGE_SIZE:
        raise ImagePreprocessError(f"image_size must be {IMAGE_SIZE}, got {image_size}")
    image = load_rgb_image(image_path)
    dummy_mask = zero_mask(image.size)
    resized, _, _ = apply_transform(
        image,
        dummy_mask,
        CLEAN,
        random.Random(0),
        image_size=image_size,
    )
    return image_to_model_tensor(resized, normalize=normalize)
