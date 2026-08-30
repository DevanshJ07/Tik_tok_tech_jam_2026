"""Local manipulation detection and localisation head (Stage 1, Member 4).

This module owns only the manipulation branch: a per-patch MLP over
pre-extracted patch features, producing patch-level logits, an upsampled
heatmap, and an image-level manipulation probability. It never touches the
official AIGC probability, the official JSON output contract, or reliability
weighting -- those are owned elsewhere and must stay independent so the
manipulation branch can be trained/evaluated in isolation.

DINOv2 is not implemented or fine-tuned here: callers are expected to supply
already-extracted patch features of shape ``[B, 256, 384]``.

Loss functions (BCE + Dice) are Stage 2 and belong in
``src/training/train_manipulation.py``, not here.
"""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "ManipulationOutput",
    "ManipulationHead",
    "patch_logits_to_heatmap",
    "topk_manipulation_probability",
]

DEFAULT_EMBEDDING_DIM = 384
DEFAULT_HIDDEN_DIM = 128
DEFAULT_PATCH_GRID_SIZE = 16
DEFAULT_HEATMAP_SIZE = 224
DEFAULT_TOP_K = 16


class ManipulationOutput(TypedDict):
    """Return contract for :meth:`ManipulationHead.forward`."""

    manipulation_probability: Tensor
    patch_mask_logits: Tensor
    heatmap: Tensor


def patch_logits_to_heatmap(
    patch_mask_logits: Tensor,
    patch_grid_size: int = DEFAULT_PATCH_GRID_SIZE,
    heatmap_size: int = DEFAULT_HEATMAP_SIZE,
) -> Tensor:
    """Upsample flat per-patch logits ``[B, N]`` to a ``[B, 1, H, W]`` heatmap.

    ``N`` must equal ``patch_grid_size ** 2``. Interpolation is bilinear with
    ``align_corners=False`` (matches the fixed Stage 1 architecture).
    """
    batch_size, num_patches = patch_mask_logits.shape
    expected_patches = patch_grid_size * patch_grid_size
    if num_patches != expected_patches:
        raise ValueError(
            f"patch_mask_logits has {num_patches} patches; expected "
            f"{expected_patches} for patch_grid_size={patch_grid_size}."
        )
    grid = patch_mask_logits.reshape(batch_size, 1, patch_grid_size, patch_grid_size)
    return F.interpolate(
        grid,
        size=(heatmap_size, heatmap_size),
        mode="bilinear",
        align_corners=False,
    )


def topk_manipulation_probability(patch_mask_logits: Tensor, top_k: int = DEFAULT_TOP_K) -> Tensor:
    """Image-level manipulation probability: mean of the top-k patch probabilities.

    A plain max is deliberately avoided -- it collapses the score to a single
    patch and is unstable under noisy per-patch logits. Averaging the top-k
    most suspicious patches is more robust while still being localised
    (unlike a global mean over all patches).
    """
    num_patches = patch_mask_logits.shape[-1]
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError(f"top_k must be an int, got {type(top_k).__name__}.")
    if top_k < 1 or top_k > num_patches:
        raise ValueError(f"top_k must be in [1, {num_patches}], got {top_k}.")

    patch_probabilities = torch.sigmoid(patch_mask_logits)
    top_values, _ = torch.topk(patch_probabilities, k=top_k, dim=-1)
    return top_values.mean(dim=-1)


class ManipulationHead(nn.Module):
    """Per-patch MLP producing manipulation logits, heatmap, and image-level score.

    Operates entirely on pre-extracted patch features (``[B, 256, 384]``) --
    it does not contain or fine-tune a backbone. Kept fully separate from
    AIGC probability logic so the two branches can be trained and evaluated
    independently.
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        patch_grid_size: int = DEFAULT_PATCH_GRID_SIZE,
        heatmap_size: int = DEFAULT_HEATMAP_SIZE,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if patch_grid_size < 1:
            raise ValueError(f"patch_grid_size must be positive, got {patch_grid_size}.")
        if heatmap_size < 1:
            raise ValueError(f"heatmap_size must be positive, got {heatmap_size}.")

        num_patches = patch_grid_size * patch_grid_size
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError(f"top_k must be an int, got {type(top_k).__name__}.")
        if top_k < 1 or top_k > num_patches:
            raise ValueError(f"top_k must be in [1, {num_patches}], got {top_k}.")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.patch_grid_size = patch_grid_size
        self.num_patches = num_patches
        self.heatmap_size = heatmap_size
        self.top_k = top_k

        self.patch_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_features: Tensor) -> ManipulationOutput:
        """Run the manipulation branch on pre-extracted patch features.

        ``patch_features`` must be ``[B, num_patches, embedding_dim]`` where
        ``num_patches == patch_grid_size ** 2``.
        """
        if patch_features.dim() != 3:
            raise ValueError(
                f"patch_features must be rank 3 [B, N, D], got rank {patch_features.dim()}."
            )
        _, num_patches, embedding_dim = patch_features.shape
        if num_patches != self.num_patches:
            raise ValueError(
                f"patch_features has {num_patches} patches; expected {self.num_patches}."
            )
        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"patch_features embedding dim is {embedding_dim}; expected {self.embedding_dim}."
            )

        patch_mask_logits = self.patch_mlp(patch_features).squeeze(-1)
        heatmap = patch_logits_to_heatmap(
            patch_mask_logits, self.patch_grid_size, self.heatmap_size
        )
        manipulation_probability = topk_manipulation_probability(patch_mask_logits, self.top_k)

        return ManipulationOutput(
            manipulation_probability=manipulation_probability,
            patch_mask_logits=patch_mask_logits,
            heatmap=heatmap,
        )
