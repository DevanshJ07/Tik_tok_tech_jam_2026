"""Baseline AIGC detector head for TraceLens-R (Member 2 component).

Consumes frozen DINOv2 features (see ``src/models/backbone.py``) and produces
an image-level probability that an image is *fully AI-generated*.

Architecture
------------
::

    cls_features [B, 384] --> GlobalHead (MLP)            --> global_logit     [B]

    patch_features [B, 256, 384]
        --> PatchEvidenceHead (MLP, applied per patch)    --> patch_logits     [B, 256]
        --> mean over the 256 patches                     --> patch_mean_logit [B]

    final_logit = global_weight * global_logit + patch_weight * patch_mean_logit
    aigc_probability = sigmoid(final_logit)

With the default weights (0.5 / 0.5) this is exactly
``0.5 * global_logit + 0.5 * patch_mean_logit``.

Scope
-----
This is the *baseline* only. No reliability weighting, no manipulation
localisation -- those belong to other members.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

__all__ = [
    "BaselineAIGCDetector",
    "GlobalHead",
    "PatchEvidenceHead",
    "OUTPUT_KEYS",
    "EMBED_DIM",
    "NUM_PATCHES",
]

EMBED_DIM = 384
NUM_PATCHES = 256

#: Exact set of keys the model output dict must contain.
OUTPUT_KEYS = (
    "global_logit",
    "patch_logits",
    "patch_mean_logit",
    "final_logit",
    "aigc_probability",
)


class GlobalHead(nn.Module):
    """Lightweight MLP mapping CLS features ``[B, in_dim]`` to a scalar logit ``[B]``."""

    def __init__(self, in_dim: int = EMBED_DIM, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cls_features: torch.Tensor) -> torch.Tensor:
        if cls_features.dim() != 2:
            raise AssertionError(
                f"GlobalHead expects [B, in_dim], got {tuple(cls_features.shape)}"
            )
        return self.net(cls_features).squeeze(-1)  # [B, 1] -> [B]


class PatchEvidenceHead(nn.Module):
    """Lightweight MLP applied **independently to every patch**.

    Input  : ``[B, num_patches, in_dim]``
    Output : ``[B, num_patches]`` (one logit per patch)

    ``nn.Linear`` already acts on the last dimension only, so passing the full
    ``[B, N, D]`` tensor applies identical weights to each of the ``N`` patches
    with no explicit loop.
    """

    def __init__(self, in_dim: int = EMBED_DIM, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        if patch_features.dim() != 3:
            raise AssertionError(
                f"PatchEvidenceHead expects [B, num_patches, in_dim], "
                f"got {tuple(patch_features.shape)}"
            )
        return self.net(patch_features).squeeze(-1)  # [B, N, 1] -> [B, N]


class BaselineAIGCDetector(nn.Module):
    """Baseline image-level AIGC detector built on frozen DINOv2 features.

    Parameters
    ----------
    embed_dim:
        Feature dimension of both CLS and patch features (384 for ViT-S/14).
    num_patches:
        Expected number of patch tokens (256 for a 16x16 grid).
    hidden_dim:
        Hidden width of the two MLP heads.
    dropout:
        Dropout probability inside the heads.
    global_weight, patch_weight:
        Fusion weights for ``final_logit``. Default ``0.5 / 0.5`` reproduces
        ``0.5 * global_logit + 0.5 * patch_mean_logit`` exactly.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_patches: int = NUM_PATCHES,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        global_weight: float = 0.5,
        patch_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_patches = int(num_patches)
        self.global_weight = float(global_weight)
        self.patch_weight = float(patch_weight)

        self.global_head = GlobalHead(self.embed_dim, hidden_dim, dropout)
        self.patch_head = PatchEvidenceHead(self.embed_dim, hidden_dim, dropout)

    # ------------------------------------------------------------------
    def forward(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run the baseline detector.

        Parameters
        ----------
        cls_features:
            ``[B, embed_dim]`` frozen DINOv2 CLS features.
        patch_features:
            ``[B, num_patches, embed_dim]`` frozen DINOv2 patch features.

        Returns
        -------
        dict with exactly the keys in :data:`OUTPUT_KEYS`:
            ``global_logit``      -> ``[B]``
            ``patch_logits``      -> ``[B, num_patches]``
            ``patch_mean_logit``  -> ``[B]``
            ``final_logit``       -> ``[B]``
            ``aigc_probability``  -> ``[B]`` (in ``[0, 1]``)
        """
        self._validate_inputs(cls_features, patch_features)

        batch_size = cls_features.shape[0]

        global_logit = self.global_head(cls_features)          # [B]
        patch_logits = self.patch_head(patch_features)          # [B, num_patches]
        patch_mean_logit = patch_logits.mean(dim=1)             # [B]

        final_logit = (
            self.global_weight * global_logit
            + self.patch_weight * patch_mean_logit
        )                                                       # [B]
        aigc_probability = torch.sigmoid(final_logit)           # [B]

        # --- output shape validation -----------------------------------
        assert global_logit.shape == (batch_size,), global_logit.shape
        assert patch_logits.shape == (batch_size, self.num_patches), patch_logits.shape
        assert patch_mean_logit.shape == (batch_size,), patch_mean_logit.shape
        assert final_logit.shape == (batch_size,), final_logit.shape
        assert aigc_probability.shape == (batch_size,), aigc_probability.shape
        # sigmoid is mathematically in (0, 1); assert defensively against NaNs.
        assert torch.all(aigc_probability >= 0.0) and torch.all(aigc_probability <= 1.0), (
            "aigc_probability escaped [0, 1] -- likely NaN/inf in logits"
        )

        return {
            "global_logit": global_logit,
            "patch_logits": patch_logits,
            "patch_mean_logit": patch_mean_logit,
            "final_logit": final_logit,
            "aigc_probability": aigc_probability,
        }

    # ------------------------------------------------------------------
    def _validate_inputs(self, cls_features: torch.Tensor, patch_features: torch.Tensor) -> None:
        if not isinstance(cls_features, torch.Tensor) or not isinstance(patch_features, torch.Tensor):
            raise TypeError("cls_features and patch_features must be torch.Tensors")
        if cls_features.dim() != 2 or cls_features.shape[1] != self.embed_dim:
            raise AssertionError(
                f"cls_features must be [B, {self.embed_dim}], got {tuple(cls_features.shape)}"
            )
        if patch_features.dim() != 3 or patch_features.shape[2] != self.embed_dim:
            raise AssertionError(
                f"patch_features must be [B, {self.num_patches}, {self.embed_dim}], "
                f"got {tuple(patch_features.shape)}"
            )
        if patch_features.shape[1] != self.num_patches:
            raise AssertionError(
                f"patch_features must have {self.num_patches} patches, "
                f"got {patch_features.shape[1]}"
            )
        if cls_features.shape[0] != patch_features.shape[0]:
            raise AssertionError(
                f"batch size mismatch: cls {cls_features.shape[0]} vs "
                f"patch {patch_features.shape[0]}"
            )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return only ``aigc_probability`` ``[B]`` (eval convenience)."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(cls_features, patch_features)["aigc_probability"]
        finally:
            if was_training:
                self.train()
