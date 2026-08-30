"""Frozen DINOv2 ViT-S/14 backbone for TraceLens-R (Member 2 component).

This module exposes a single, small, reusable feature extractor that
Members 3-5 can depend on without knowing anything about Hugging Face
internals. It wraps ``facebook/dinov2-small`` and:

* accepts a fixed RGB tensor ``[B, 3, 224, 224]`` (ImageNet / DINOv2
  normalized -- the dataset already normalizes, see ``src/data/dataset.py``);
* returns ``cls_features [B, 384]`` and ``patch_features [B, 256, 384]``
  (patch grid ``16 x 16``);
* freezes **every** backbone parameter (``requires_grad = False``) and never
  fine-tunes DINOv2;
* runs on CPU with no extra configuration.

Design notes
------------
The actual Hugging Face model is loaded through :func:`_load_hf_backbone`,
which is a deliberate seam: unit tests monkeypatch it with a tiny stub so the
test suite never has to download a real checkpoint.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

__all__ = [
    "DINOv2Backbone",
    "DEFAULT_BACKBONE_NAME",
    "EMBED_DIM",
    "NUM_PATCHES",
    "PATCH_GRID",
    "IMAGE_SIZE",
]

# ---------------------------------------------------------------------------
# Shared contract constants (match configs/default.yaml)
# ---------------------------------------------------------------------------
DEFAULT_BACKBONE_NAME = "facebook/dinov2-small"
IMAGE_SIZE = 224
PATCH_GRID = 16                       # 16 x 16 patch grid
NUM_PATCHES = PATCH_GRID * PATCH_GRID  # 256
EMBED_DIM = 384                       # DINOv2 ViT-S/14 hidden size


def _load_hf_backbone(model_name: str):
    """Load the Hugging Face DINOv2 model.

    Kept as a module-level function purely so tests can replace it with a
    lightweight stub (avoids downloading a real checkpoint just to check
    shapes / freezing behaviour).
    """
    try:
        from transformers import Dinov2Model
    except ImportError as exc:  # pragma: no cover - dependency hint only
        raise ImportError(
            "transformers is required for the real DINOv2 backbone. "
            "Install it with `pip install transformers` (or use the test stub)."
        ) from exc
    return Dinov2Model.from_pretrained(model_name)


class DINOv2Backbone(nn.Module):
    """Frozen DINOv2 ViT-S/14 feature extractor.

    Parameters
    ----------
    model_name:
        Hugging Face model id. Defaults to ``facebook/dinov2-small``.
    device:
        Torch device string / object. Defaults to ``"cpu"``.
    model:
        Optional pre-constructed backbone module. When given, ``model_name``
        is ignored for loading (still stored for reference). Mainly a test
        hook; production code should rely on the default loader.

    Notes
    -----
    * The wrapped model is put in ``eval()`` mode and *kept* there even if
      someone calls ``.train()`` on this wrapper -- a frozen backbone should
      never toggle dropout / normalization statistics.
    * ``forward`` runs under ``torch.no_grad()``: the returned features carry
      no autograd history, which is exactly what the lightweight trainable
      heads downstream expect.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BACKBONE_NAME,
        device: str | torch.device = "cpu",
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(device)
        self.model = model if model is not None else _load_hf_backbone(model_name)
        self.model.to(self.device)
        self.freeze()
        self.model.eval()

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------
    def freeze(self) -> None:
        """Set ``requires_grad = False`` on every backbone parameter."""
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def is_frozen(self) -> bool:
        """True iff no backbone parameter requires gradients."""
        return all(not p.requires_grad for p in self.model.parameters())

    def train(self, mode: bool = True) -> "DINOv2Backbone":
        """Override: keep the frozen backbone in eval mode regardless.

        The wrapper module still tracks ``self.training`` so external code
        behaves normally, but the DINOv2 submodule is always eval.
        """
        super().train(mode)
        self.model.eval()
        return self

    def parameters(self, recurse: bool = True):  # noqa: D401 - see note
        """Yield parameters (all frozen). Kept explicit for clarity.

        Downstream optimizers should filter on ``requires_grad`` anyway; this
        method is not overridden to hide parameters, only documented here.
        """
        return super().parameters(recurse=recurse)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract CLS and patch features.

        Parameters
        ----------
        pixel_values:
            Float tensor of shape ``[B, 3, 224, 224]`` (ImageNet / DINOv2
            normalized).

        Returns
        -------
        (cls_features, patch_features):
            ``cls_features``  -> ``[B, 384]``
            ``patch_features`` -> ``[B, 256, 384]``
        """
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(f"pixel_values must be a torch.Tensor, got {type(pixel_values)!r}")
        if pixel_values.dim() != 4:
            raise AssertionError(
                f"pixel_values must be 4D [B, 3, {IMAGE_SIZE}, {IMAGE_SIZE}], "
                f"got shape {tuple(pixel_values.shape)}"
            )
        batch_size, channels, height, width = pixel_values.shape
        if (channels, height, width) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise AssertionError(
                f"pixel_values must have shape [B, 3, {IMAGE_SIZE}, {IMAGE_SIZE}], "
                f"got {tuple(pixel_values.shape)}"
            )

        pixel_values = pixel_values.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, return_dict=True)

        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            # Fall back to tuple-style outputs.
            last_hidden = outputs[0] if isinstance(outputs, (tuple, list)) else None
        if last_hidden is None:
            raise RuntimeError(
                "Backbone output did not expose `last_hidden_state`; "
                f"got type {type(outputs)!r}"
            )

        if last_hidden.dim() != 3 or last_hidden.shape[-1] != EMBED_DIM:
            raise AssertionError(
                f"Expected backbone hidden states [B, tokens, {EMBED_DIM}], "
                f"got {tuple(last_hidden.shape)}"
            )
        expected_tokens = 1 + NUM_PATCHES  # 1 CLS token + 256 patch tokens
        if last_hidden.shape[1] != expected_tokens:
            raise AssertionError(
                f"Expected {expected_tokens} tokens (1 CLS + {NUM_PATCHES} patches) "
                f"for a {IMAGE_SIZE}x{IMAGE_SIZE} input, got {last_hidden.shape[1]}. "
                "facebook/dinov2-small (no register tokens) is required."
            )

        cls_features = last_hidden[:, 0, :].contiguous()
        patch_features = last_hidden[:, 1 : 1 + NUM_PATCHES, :].contiguous()

        assert cls_features.shape == (batch_size, EMBED_DIM), (
            f"cls_features shape {tuple(cls_features.shape)} != "
            f"{(batch_size, EMBED_DIM)}"
        )
        assert patch_features.shape == (batch_size, NUM_PATCHES, EMBED_DIM), (
            f"patch_features shape {tuple(patch_features.shape)} != "
            f"{(batch_size, NUM_PATCHES, EMBED_DIM)}"
        )
        return cls_features, patch_features

    @torch.no_grad()
    def extract_features(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for :meth:`forward` with an explicit no-grad guarantee.

        Convenience entry point for feature-caching / inference code that
        wants the intent to be obvious at the call site.
        """
        return self.forward(pixel_values)
