"""Official TraceLens-R image/mask transformations.

Every transform here is a pure function of (image, mask, parameters, rng) so
that behaviour is fully deterministic given a seeded ``random.Random``. Rules
enforced throughout this module (per the shared spec):

  * Pixel-value transforms (JPEG, blur, noise, colour jitter) are NEVER
    applied to the mask -- only geometric transforms (resize-degrade,
    centre-crop) touch the mask, and they apply the identical geometry to
    both image and mask.
  * Masks are always resized with nearest-neighbour interpolation and
    re-binarized afterwards, so they never pick up intermediate grey values.
  * Every transform ultimately returns an image and mask at the requested
    ``image_size`` (default 224x224).
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

DEFAULT_IMAGE_SIZE = 224

# ImageNet / DINOv2 normalization statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MissingMaskError(FileNotFoundError):
    """Raised when a label=2 (locally-tampered) sample has no usable mask.

    This must never be handled by silently substituting a zero mask --
    that would mislabel a real tampering mask as "nothing tampered", which
    is exactly the failure mode the shared spec calls out. Callers that
    truly want a permissive fallback must catch this exception explicitly.
    """


@dataclass(frozen=True)
class TransformSpec:
    """One official transform + severity level, plus whether it is geometric.

    ``is_geometric`` transforms change spatial layout (crop/resize) and thus
    must be applied identically to the mask. Non-geometric transforms only
    ever touch the image.
    """

    name: str
    severity: Optional[float]
    is_geometric: bool


IDENTITY = TransformSpec(name="identity", severity=None, is_geometric=False)

JPEG_QUALITIES = (90, 70, 50, 30)
BLUR_SIGMAS = (0.5, 1.0, 2.0)
RESIZE_SCALES = (0.5, 0.25)
NOISE_SIGMAS = (0.02, 0.05, 0.10)
COLOR_JITTER_FRACTION = 0.2
CENTER_CROP_RETAIN = 0.8


def official_transforms() -> list[TransformSpec]:
    """The full grid of official transforms x severities (excludes identity)."""
    specs: list[TransformSpec] = []
    specs += [TransformSpec("jpeg", q, False) for q in JPEG_QUALITIES]
    specs += [TransformSpec("gaussian_blur", s, False) for s in BLUR_SIGMAS]
    specs += [TransformSpec("resize_degrade", s, True) for s in RESIZE_SCALES]
    specs += [TransformSpec("gaussian_noise", s, False) for s in NOISE_SIGMAS]
    specs += [TransformSpec("color_jitter", COLOR_JITTER_FRACTION, False)]
    specs += [TransformSpec("center_crop", CENTER_CROP_RETAIN, True)]
    return specs


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------


def zero_mask(size: Tuple[int, int]) -> Image.Image:
    """All-zero binary mask for authentic (0) / fully-synthetic (1) samples."""
    return Image.new("L", size, 0)


def binarize_mask(mask: Image.Image, threshold: int = 127) -> Image.Image:
    """Force a mask back to strict {0, 255} after any resampling."""
    if mask.mode != "L":
        mask = mask.convert("L")
    arr = np.array(mask)
    arr = np.where(arr >= threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


# ---------------------------------------------------------------------------
# Individual transforms (image-only, non-geometric)
# ---------------------------------------------------------------------------


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode the image through an in-memory JPEG at the given quality."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """PIL's GaussianBlur radius is used directly as the blur sigma."""
    return image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def apply_gaussian_noise(image: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    """Add zero-mean Gaussian noise (sigma in [0,1] pixel-intensity units)."""
    seed = rng.randrange(2**32)
    generator = np.random.default_rng(seed)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noise = generator.normal(loc=0.0, scale=float(sigma), size=arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).round().astype(np.uint8), mode="RGB")


def apply_color_jitter(image: Image.Image, fraction: float, rng: random.Random) -> Image.Image:
    """Independently jitter brightness, contrast, and saturation by +/-fraction."""
    image = image.convert("RGB")
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = rng.uniform(1.0 - fraction, 1.0 + fraction)
        image = enhancer_cls(image).enhance(factor)
    return image


# ---------------------------------------------------------------------------
# Individual transforms (geometric: image + mask)
# ---------------------------------------------------------------------------


def apply_resize_degrade(
    image: Image.Image, mask: Image.Image, scale: float, image_size: int = DEFAULT_IMAGE_SIZE
) -> Tuple[Image.Image, Image.Image]:
    """Downscale by ``scale`` then upscale back to ``image_size`` (lossy)."""
    w, h = image.size
    small_size = (max(1, round(w * scale)), max(1, round(h * scale)))

    small_image = image.resize(small_size, Image.BILINEAR)
    out_image = small_image.resize((image_size, image_size), Image.BILINEAR)

    small_mask = mask.resize(small_size, Image.NEAREST)
    out_mask = small_mask.resize((image_size, image_size), Image.NEAREST)
    return out_image, out_mask


def apply_center_crop(
    image: Image.Image, mask: Image.Image, retain: float, image_size: int = DEFAULT_IMAGE_SIZE
) -> Tuple[Image.Image, Image.Image]:
    """Crop the centre ``retain`` fraction of each dimension, then resize back up."""
    w, h = image.size
    crop_w, crop_h = max(1, round(w * retain)), max(1, round(h * retain))
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    box = (left, top, left + crop_w, top + crop_h)

    out_image = image.crop(box).resize((image_size, image_size), Image.BILINEAR)
    out_mask = mask.crop(box).resize((image_size, image_size), Image.NEAREST)
    return out_image, out_mask


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_NON_GEOMETRIC_APPLIERS = {
    "jpeg": lambda image, spec, rng: apply_jpeg_compression(image, spec.severity),
    "gaussian_blur": lambda image, spec, rng: apply_gaussian_blur(image, spec.severity),
    "gaussian_noise": lambda image, spec, rng: apply_gaussian_noise(image, spec.severity, rng),
    "color_jitter": lambda image, spec, rng: apply_color_jitter(image, spec.severity, rng),
    "identity": lambda image, spec, rng: image,
}

_GEOMETRIC_APPLIERS = {
    "resize_degrade": lambda image, mask, spec, size: apply_resize_degrade(image, mask, spec.severity, size),
    "center_crop": lambda image, mask, spec, size: apply_center_crop(image, mask, spec.severity, size),
}


def apply_transform(
    image: Image.Image,
    mask: Image.Image,
    spec: TransformSpec,
    rng: random.Random,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Tuple[Image.Image, Image.Image, dict]:
    """Apply one :class:`TransformSpec` to an (image, mask) pair.

    Returns ``(transformed_image, transformed_mask, metadata)`` where
    ``metadata`` is exactly ``{"transform_name", "severity", "is_geometric"}``.
    Non-geometric transforms leave the mask byte-for-byte untouched (aside
    from a final size/binarization safety normalization); geometric
    transforms apply identical crop/resize geometry to both.
    """
    if spec.is_geometric:
        applier = _GEOMETRIC_APPLIERS.get(spec.name)
        if applier is None:
            raise ValueError(f"Unknown geometric transform: {spec.name!r}")
        out_image, out_mask = applier(image, mask, spec, image_size)
    else:
        applier = _NON_GEOMETRIC_APPLIERS.get(spec.name)
        if applier is None:
            raise ValueError(f"Unknown non-geometric transform: {spec.name!r}")
        out_image = applier(image, spec, rng)
        out_mask = mask

    if out_image.size != (image_size, image_size):
        out_image = out_image.resize((image_size, image_size), Image.BILINEAR)
    if out_mask.size != (image_size, image_size):
        out_mask = out_mask.resize((image_size, image_size), Image.NEAREST)
    out_mask = binarize_mask(out_mask)

    metadata = {
        "transform_name": spec.name,
        "severity": spec.severity,
        "is_geometric": spec.is_geometric,
    }
    return out_image.convert("RGB"), out_mask, metadata
