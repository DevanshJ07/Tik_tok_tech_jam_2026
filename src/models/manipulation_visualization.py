"""Manipulation heatmap visualization / overlay (Stage 3, Member 4).

Turns one sample's manipulation heatmap (as produced by
:class:`~src.models.manipulation.ManipulationHead`) into a colour overlay
on the original image, for eventual display by Member 5's UI. This module
is display-only: it never feeds back into the model forward pass, training,
AIGC probability, reliability, threshold calibration, or official JSON
output.

Heatmap semantics (read carefully before using this module): the "heatmap"
key in :class:`~src.models.manipulation.ManipulationOutput` is produced by
``patch_logits_to_heatmap``, which bilinearly upsamples the RAW per-patch
logits -- no sigmoid is applied inside the model. So the model's heatmap
output is interpolated logits, not probabilities, and must be converted with
``torch.sigmoid`` before it means anything as a display intensity. Every
public function here defaults to that conversion (``is_logits`` /
``heatmap_is_logits`` default to ``True``) specifically so a caller cannot
accidentally display raw, unbounded logit values as if they were
probabilities. Passing a false ``is_logits=False`` for genuine logits is not
guarded against everywhere (there is no way to detect that from values
alone), but passing ``is_logits=False`` for anything that turns out to fall
outside ``[0, 1]`` is rejected explicitly.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

__all__ = [
    "heatmap_to_probabilities",
    "create_manipulation_overlay",
]

DEFAULT_ALPHA = 0.45


def _to_tensor(heatmap: Tensor | np.ndarray) -> Tensor:
    """Detach/copy into an independent float32 CPU tensor -- never the caller's own storage."""
    if isinstance(heatmap, torch.Tensor):
        return heatmap.detach().to(dtype=torch.float32, device="cpu").clone()
    if isinstance(heatmap, np.ndarray):
        return torch.from_numpy(np.asarray(heatmap, dtype=np.float32)).clone()
    raise TypeError(f"heatmap must be a torch.Tensor or numpy.ndarray, got {type(heatmap).__name__}.")


def _validate_single_sample_heatmap(tensor: Tensor) -> Tensor:
    """Accept [H,W] or [1,H,W] (a single sample); reject everything else, incl. batches."""
    if tensor.numel() == 0:
        raise ValueError(f"heatmap is empty: shape {tuple(tensor.shape)}.")
    if tensor.dim() == 3:
        if tensor.shape[0] != 1:
            raise ValueError(
                "heatmap with 3 dims must be [1,H,W] (a single sample), got shape "
                f"{tuple(tensor.shape)}. Index the batch dimension first, e.g. heatmap[0]."
            )
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise ValueError(
            "heatmap must be [H,W] or [1,H,W] (a single sample, not a batch) -- "
            f"got rank {tensor.dim()}, shape {tuple(tensor.shape)}. For a batched model "
            'output, index one sample first, e.g. model_output["heatmap"][i].'
        )
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"heatmap is empty: shape {tuple(tensor.shape)}.")
    return tensor


def heatmap_to_probabilities(heatmap: Tensor | np.ndarray, *, is_logits: bool = True) -> Tensor:
    """Convert a single-sample manipulation heatmap into [0,1] display probabilities.

    ``heatmap`` is ``[H,W]`` or ``[1,H,W]``, torch or numpy -- e.g. one row of
    a model output's ``"heatmap"[i]``. By default (``is_logits=True``) it is
    treated as the model's actual output contract -- interpolated raw logits
    (see module docstring) -- and converted via ``torch.sigmoid``. Pass
    ``is_logits=False`` only for a heatmap that has already been sigmoided;
    if any resulting value would fall outside ``[0, 1]`` in that case, this
    raises rather than silently displaying an out-of-range value, since that
    is a strong sign the semantics were mislabelled.

    Returns a ``[H, W]`` float32 tensor with values in ``[0, 1]``. Detaches
    from any autograd graph -- visualization never affects training.
    """
    tensor = _to_tensor(heatmap)
    tensor = _validate_single_sample_heatmap(tensor)
    if not torch.isfinite(tensor).all():
        raise ValueError("heatmap contains NaN or Inf values.")

    if is_logits:
        return torch.sigmoid(tensor)

    if torch.any(tensor < 0.0) or torch.any(tensor > 1.0):
        raise ValueError(
            "is_logits=False but heatmap values fall outside [0, 1]; this looks like "
            "raw logits, not probabilities. Pass is_logits=True (the default) instead."
        )
    return tensor


def _hot_colormap(probabilities: Tensor) -> Tensor:
    """Black -> red -> yellow -> white heat ramp, dependency-free (no matplotlib).

    ``probabilities`` is ``[H,W]`` in ``[0,1]``; returns ``[H,W,3]`` in ``[0,1]``.
    Low probability stays dark (unobtrusive); high probability is bright
    red/yellow (clearly visible) -- the standard analytic "hot" colormap.
    """
    t = probabilities.clamp(0.0, 1.0)
    r = (3.0 * t).clamp(0.0, 1.0)
    g = (3.0 * t - 1.0).clamp(0.0, 1.0)
    b = (3.0 * t - 2.0).clamp(0.0, 1.0)
    return torch.stack([r, g, b], dim=-1)


def create_manipulation_overlay(
    image: Image.Image,
    heatmap: Tensor | np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    heatmap_is_logits: bool = True,
) -> Image.Image:
    """Blend a single-sample manipulation heatmap over ``image`` as a colour overlay.

    ``image`` is the original PIL image (any mode; converted to RGB
    internally). ``heatmap`` is one sample's manipulation heatmap -- ``[H,W]``
    or ``[1,H,W]``, torch or numpy -- such as
    ``model_output["heatmap"][i]``. By default it is treated as the model's
    raw interpolated logits and converted to ``[0,1]`` probabilities via
    :func:`heatmap_to_probabilities` (no per-image min-max normalization --
    that would make a uniformly low-confidence image look artificially
    suspicious, so probability values are used as-is and kept meaningful).

    The probability map is resized (bilinear, ``align_corners=False``, same
    convention as the model) to ``image``'s exact pixel size, then blended in
    with a black->red->yellow "hot" colour ramp. The blend strength at each
    pixel is ``alpha * probability``, not a flat ``alpha`` -- so
    low-probability regions stay visually unobtrusive regardless of
    ``alpha``, while high-probability regions become clearly visible; at
    ``alpha=0`` the output is pixel-identical to the original.

    Returns a new RGB ``PIL.Image`` the same size as ``image``. Never
    mutates ``image``. Deterministic, CPU-only, no filesystem writes, no
    autograd interaction. This is explanatory visualization output only --
    it never touches AIGC probability, reliability, official JSON output, or
    training.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha must be a finite number in [0, 1], got {alpha!r}.")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or alpha_value < 0.0 or alpha_value > 1.0:
        raise ValueError(f"alpha must be a finite number in [0, 1], got {alpha_value}.")

    probabilities = heatmap_to_probabilities(heatmap, is_logits=heatmap_is_logits)

    width, height = image.size
    resized = (
        F.interpolate(
            probabilities.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
        .clamp(0.0, 1.0)
    )

    colour = _hot_colormap(resized)  # [H, W, 3] in [0, 1]
    original = torch.from_numpy(
        np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    )  # [H, W, 3], independent copy -- caller's image is never touched.

    blend_strength = (alpha_value * resized).unsqueeze(-1)  # [H, W, 1]
    blended = original * (1.0 - blend_strength) + colour * blend_strength
    blended_uint8 = (blended.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(blended_uint8, mode="RGB")
