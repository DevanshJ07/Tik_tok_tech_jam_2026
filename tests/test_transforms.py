"""Tests for src/data/transforms.py using synthetic images only."""
from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from src.data.transforms import (
    IDENTITY,
    TransformSpec,
    apply_center_crop,
    apply_color_jitter,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_resize_degrade,
    apply_transform,
    binarize_mask,
    official_transforms,
    zero_mask,
)

IMAGE_SIZE = 224


def _checkerboard_image(size=(96, 96)) -> Image.Image:
    """Distinct-per-pixel image so we can detect misalignment after crop/resize."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[: h // 2, : w // 2] = (255, 0, 0)  # top-left quadrant colored
    return Image.fromarray(arr, mode="RGB")


def _quadrant_mask(size=(96, 96)) -> Image.Image:
    w, h = size
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[: h // 2, : w // 2] = 255  # matches the colored quadrant above
    return Image.fromarray(arr, mode="L")


# ---------------------------------------------------------------------------
# Spec coverage
# ---------------------------------------------------------------------------


def test_official_transforms_match_spec():
    specs = official_transforms()
    by_name = {}
    for spec in specs:
        by_name.setdefault(spec.name, []).append(spec.severity)

    assert sorted(by_name["jpeg"]) == [30, 50, 70, 90]
    assert sorted(by_name["gaussian_blur"]) == [0.5, 1.0, 2.0]
    assert sorted(by_name["resize_degrade"]) == [0.25, 0.5]
    assert sorted(by_name["gaussian_noise"]) == [0.02, 0.05, 0.10]
    assert by_name["color_jitter"] == [0.2]
    assert by_name["center_crop"] == [0.8]
    assert len(specs) == 14

    geometric = {s.name for s in specs if s.is_geometric}
    assert geometric == {"resize_degrade", "center_crop"}


# ---------------------------------------------------------------------------
# Output shape / size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", official_transforms() + [IDENTITY])
def test_apply_transform_output_is_224(spec: TransformSpec):
    image = _checkerboard_image()
    mask = _quadrant_mask()
    rng = random.Random(0)

    out_image, out_mask, _ = apply_transform(image, mask, spec, rng, image_size=IMAGE_SIZE)

    assert out_image.size == (IMAGE_SIZE, IMAGE_SIZE)
    assert out_mask.size == (IMAGE_SIZE, IMAGE_SIZE)
    assert out_image.mode == "RGB"


# ---------------------------------------------------------------------------
# Masks stay binary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", official_transforms() + [IDENTITY])
def test_masks_remain_binary(spec: TransformSpec):
    image = _checkerboard_image()
    mask = _quadrant_mask()
    rng = random.Random(1)

    _, out_mask, _ = apply_transform(image, mask, spec, rng, image_size=IMAGE_SIZE)
    values = set(np.unique(np.array(out_mask)).tolist())
    assert values <= {0, 255}


# ---------------------------------------------------------------------------
# Pixel-value transforms never touch the mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [s for s in official_transforms() if not s.is_geometric])
def test_non_geometric_transforms_leave_mask_untouched(spec: TransformSpec):
    image = _checkerboard_image(size=(IMAGE_SIZE, IMAGE_SIZE))
    mask = binarize_mask(_quadrant_mask(size=(IMAGE_SIZE, IMAGE_SIZE)))
    rng = random.Random(2)

    _, out_mask, _ = apply_transform(image, mask, spec, rng, image_size=IMAGE_SIZE)

    assert np.array_equal(np.array(out_mask), np.array(mask))


def test_zero_mask_is_all_zero():
    mask = zero_mask((32, 32))
    assert np.array(mask).sum() == 0
    assert mask.size == (32, 32)


# ---------------------------------------------------------------------------
# Geometric transforms move the mask identically to the image
# ---------------------------------------------------------------------------


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool, b_bool = a.astype(bool), b.astype(bool)
    intersection = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    return intersection / union if union else 1.0


@pytest.mark.parametrize("spec_name,severity", [("center_crop", 0.8), ("resize_degrade", 0.5), ("resize_degrade", 0.25)])
def test_geometric_transform_keeps_image_mask_aligned(spec_name, severity):
    image = _checkerboard_image()
    mask = _quadrant_mask()
    spec = TransformSpec(spec_name, severity, True)
    rng = random.Random(3)

    out_image, out_mask, meta = apply_transform(image, mask, spec, rng, image_size=IMAGE_SIZE)

    assert meta["is_geometric"] is True
    red_region = (np.array(out_image)[:, :, 0] > 128) & (np.array(out_image)[:, :, 1] < 64)
    mask_region = np.array(out_mask) > 0

    assert _iou(red_region, mask_region) > 0.85


# ---------------------------------------------------------------------------
# Transformation metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", official_transforms() + [IDENTITY])
def test_transform_metadata_contents(spec: TransformSpec):
    image = _checkerboard_image()
    mask = _quadrant_mask()
    rng = random.Random(4)

    _, _, metadata = apply_transform(image, mask, spec, rng, image_size=IMAGE_SIZE)

    assert set(metadata.keys()) == {"transform_name", "severity", "is_geometric"}
    assert metadata["transform_name"] == spec.name
    assert metadata["severity"] == spec.severity
    assert metadata["is_geometric"] == spec.is_geometric


# ---------------------------------------------------------------------------
# Deterministic behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [TransformSpec("gaussian_noise", 0.05, False), TransformSpec("color_jitter", 0.2, False)])
def test_randomized_transforms_are_seed_deterministic(spec: TransformSpec):
    image = _checkerboard_image()
    mask = _quadrant_mask()

    out_a, _, _ = apply_transform(image, mask, spec, random.Random(42), image_size=IMAGE_SIZE)
    out_b, _, _ = apply_transform(image, mask, spec, random.Random(42), image_size=IMAGE_SIZE)
    out_c, _, _ = apply_transform(image, mask, spec, random.Random(43), image_size=IMAGE_SIZE)

    assert np.array_equal(np.array(out_a), np.array(out_b))
    assert not np.array_equal(np.array(out_a), np.array(out_c))


def test_jpeg_and_blur_are_deterministic_regardless_of_rng():
    image = _checkerboard_image()
    mask = _quadrant_mask()
    jpeg_spec = TransformSpec("jpeg", 50, False)

    out_a, _, _ = apply_transform(image, mask, jpeg_spec, random.Random(1), image_size=IMAGE_SIZE)
    out_b, _, _ = apply_transform(image, mask, jpeg_spec, random.Random(999), image_size=IMAGE_SIZE)

    assert np.array_equal(np.array(out_a), np.array(out_b))


def test_identity_leaves_image_unchanged_besides_resize():
    image = _checkerboard_image(size=(IMAGE_SIZE, IMAGE_SIZE))
    mask = _quadrant_mask(size=(IMAGE_SIZE, IMAGE_SIZE))

    out_image, out_mask, meta = apply_transform(image, mask, IDENTITY, random.Random(5), image_size=IMAGE_SIZE)

    assert np.array_equal(np.array(out_image), np.array(image))
    assert meta["transform_name"] == "identity"
    assert meta["severity"] is None
    assert meta["is_geometric"] is False


# ---------------------------------------------------------------------------
# Individual transform sanity checks
# ---------------------------------------------------------------------------


def test_apply_gaussian_noise_changes_pixels():
    image = _checkerboard_image()
    noisy = apply_gaussian_noise(image, sigma=0.1, rng=random.Random(7))
    assert not np.array_equal(np.array(noisy), np.array(image.convert("RGB")))


def test_apply_jpeg_compression_lower_quality_more_lossy():
    image = _checkerboard_image()
    high_q = apply_jpeg_compression(image, quality=90)
    low_q = apply_jpeg_compression(image, quality=10)
    orig = np.array(image.convert("RGB"), dtype=np.float64)
    err_high = np.mean((np.array(high_q, dtype=np.float64) - orig) ** 2)
    err_low = np.mean((np.array(low_q, dtype=np.float64) - orig) ** 2)
    assert err_low >= err_high


def test_apply_gaussian_blur_smooths_image():
    image = _checkerboard_image()
    blurred = apply_gaussian_blur(image, sigma=2.0)
    assert not np.array_equal(np.array(blurred), np.array(image))


def test_apply_color_jitter_deterministic_with_seed():
    image = _checkerboard_image()
    a = apply_color_jitter(image, 0.2, random.Random(11))
    b = apply_color_jitter(image, 0.2, random.Random(11))
    assert np.array_equal(np.array(a), np.array(b))


def test_apply_resize_degrade_shapes():
    image = _checkerboard_image()
    mask = _quadrant_mask()
    out_image, out_mask = apply_resize_degrade(image, mask, scale=0.25, image_size=IMAGE_SIZE)
    assert out_image.size == (IMAGE_SIZE, IMAGE_SIZE)
    assert out_mask.size == (IMAGE_SIZE, IMAGE_SIZE)


def test_apply_center_crop_shapes():
    image = _checkerboard_image()
    mask = _quadrant_mask()
    out_image, out_mask = apply_center_crop(image, mask, retain=0.8, image_size=IMAGE_SIZE)
    assert out_image.size == (IMAGE_SIZE, IMAGE_SIZE)
    assert out_mask.size == (IMAGE_SIZE, IMAGE_SIZE)
