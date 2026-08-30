"""
TraceLens-R Patch Reliability Module
====================================

Member 3 component.

This module:
1. Predicts a reliability score for each of the 256 DINOv2 patches.
2. Aggregates patch logits using those reliability scores.
3. Combines weighted patch evidence with the global baseline logit.
4. Produces the final AIGC probability.
5. Provides utilities for generating detached survival targets.

The DINOv2 backbone and Member 2 baseline are NOT modified here.

Expected DINOv2 features:
    CLS:    [B, 384]
    Patches:[B, 256, 384]

Expected Member 2 baseline outputs:
    global_logit:      [B]
    patch_logits:      [B, 256]
    patch_mean_logit:  [B]
    final_logit:       [B]
    aigc_probability:  [B]
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


EMBED_DIM = 384
NUM_PATCHES = 256
EPSILON = 1e-8


class ReliabilityHead(nn.Module):
    """
    Lightweight MLP that predicts one reliability value per patch.

    Input:
        patch_features: [B, 256, 384]

    Output:
        reliability: [B, 256]

    Reliability is constrained to [0, 1] using sigmoid.
    """

    def __init__(
        self,
        in_dim: int = EMBED_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_features: [B, 256, 384]

        Returns:
            reliability: [B, 256]
        """

        if not isinstance(patch_features, torch.Tensor):
            raise TypeError(
                "patch_features must be a torch.Tensor"
            )

        if patch_features.dim() != 3:
            raise AssertionError(
                "ReliabilityHead expects [B, 256, 384], "
                f"got {tuple(patch_features.shape)}"
            )

        if patch_features.shape[1] != NUM_PATCHES:
            raise AssertionError(
                f"Expected {NUM_PATCHES} patches, "
                f"got {patch_features.shape[1]}"
            )

        if patch_features.shape[2] != self.in_dim:
            raise AssertionError(
                f"Expected feature dimension {self.in_dim}, "
                f"got {patch_features.shape[2]}"
            )

        logits = self.net(patch_features).squeeze(-1)

        # Required reliability range: [0, 1].
        reliability = torch.sigmoid(logits)

        return reliability


def weighted_patch_aggregation(
    patch_logits: torch.Tensor,
    reliability: torch.Tensor,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """
    Compute reliability-weighted patch evidence.

    Required formula:

        weighted_patch_logit =
            sum(reliability * patch_logits)
            /
            (sum(reliability) + epsilon)

    Args:
        patch_logits: [B, 256]
        reliability:  [B, 256]

    Returns:
        weighted_patch_logit: [B]
    """

    if not isinstance(patch_logits, torch.Tensor):
        raise TypeError(
            "patch_logits must be a torch.Tensor"
        )

    if not isinstance(reliability, torch.Tensor):
        raise TypeError(
            "reliability must be a torch.Tensor"
        )

    if patch_logits.shape != reliability.shape:
        raise AssertionError(
            "patch_logits and reliability must have identical shapes. "
            f"Got {tuple(patch_logits.shape)} and "
            f"{tuple(reliability.shape)}"
        )

    if patch_logits.dim() != 2:
        raise AssertionError(
            f"Expected [B, 256], got {tuple(patch_logits.shape)}"
        )

    numerator = (
        reliability * patch_logits
    ).sum(dim=1)

    denominator = (
        reliability.sum(dim=1) + epsilon
    )

    return numerator / denominator


def combine_logits(
    global_logit: torch.Tensor,
    weighted_patch_logit: torch.Tensor,
) -> torch.Tensor:
    """
    Required final AIGC logit:

        final_logit =
            0.5 * global_logit
            + 0.5 * weighted_patch_logit
    """

    global_logit = global_logit.reshape(-1)
    weighted_patch_logit = weighted_patch_logit.reshape(-1)

    if global_logit.shape != weighted_patch_logit.shape:
        raise AssertionError(
            "global_logit and weighted_patch_logit must have "
            "the same shape."
        )

    return (
        0.5 * global_logit
        + 0.5 * weighted_patch_logit
    )


def correct_class_evidence(
    patch_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Convert raw binary patch logits into evidence supporting
    the true class.

    Label convention:
        0 = authentic
        1 = fully synthetic

    For label 1:
        correct evidence = patch_logit

    For label 0:
        correct evidence = -patch_logit

    Label 2 is rejected because locally tampered images must
    not automatically be treated as fully AI-generated.
    """

    if patch_logits.dim() != 2:
        raise AssertionError(
            f"Expected patch_logits [B, 256], "
            f"got {tuple(patch_logits.shape)}"
        )

    labels = labels.reshape(-1).long()

    if patch_logits.shape[0] != labels.shape[0]:
        raise AssertionError(
            "Batch size mismatch between patch_logits and labels."
        )

    valid_labels = (labels == 0) | (labels == 1)

    if not torch.all(valid_labels):
        raise ValueError(
            "AIGC reliability targets require labels 0 or 1. "
            "Label 2 must not be treated as fully synthetic."
        )

    sign = torch.where(
        labels[:, None] == 1,
        torch.ones_like(patch_logits),
        -torch.ones_like(patch_logits),
    )

    return sign * patch_logits


@torch.no_grad()
def compute_survival_target(
    clean_patch_logits: torch.Tensor,
    degraded_patch_logits: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = EPSILON,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate detached reliability survival targets.

    clean_strength =
        sigmoid(correct-class clean evidence)

    degraded_strength =
        sigmoid(correct-class degraded evidence)

    survival_target =
        clamp(
            degraded_strength /
            (clean_strength + epsilon),
            0,
            1
        )

    Patches with weak clean evidence receive less supervision.

    Returns:
        survival_target: [B, 256]
        target_weight:   [B, 256]
    """

    if clean_patch_logits.shape != degraded_patch_logits.shape:
        raise AssertionError(
            "Clean and degraded patch logits must have identical shapes."
        )

    clean_evidence = correct_class_evidence(
        clean_patch_logits,
        labels,
    )

    degraded_evidence = correct_class_evidence(
        degraded_patch_logits,
        labels,
    )

    clean_strength = torch.sigmoid(
        clean_evidence
    )

    degraded_strength = torch.sigmoid(
        degraded_evidence
    )

    survival_target = torch.clamp(
        degraded_strength
        / (clean_strength + epsilon),
        min=0.0,
        max=1.0,
    )

    # Continuous clean-evidence weighting.
    # Strong clean evidence -> stronger supervision.
    target_weight = clean_strength

    return (
        survival_target.detach(),
        target_weight.detach(),
    )


def survival_loss(
    predicted_reliability: torch.Tensor,
    survival_target: torch.Tensor,
    target_weight: torch.Tensor,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """
    Weighted survival-target loss.

    Smooth L1 is used because the target is continuous in [0, 1].

    Patches with weak clean evidence contribute less.
    """

    if predicted_reliability.shape != survival_target.shape:
        raise AssertionError(
            "predicted_reliability and survival_target "
            "must have identical shapes."
        )

    if target_weight.shape != survival_target.shape:
        raise AssertionError(
            "target_weight and survival_target "
            "must have identical shapes."
        )

    loss = F.smooth_l1_loss(
        predicted_reliability,
        survival_target,
        reduction="none",
    )

    weighted_loss = loss * target_weight

    denominator = (
        target_weight.sum()
        + epsilon
    )

    return weighted_loss.sum() / denominator


class TraceLensReliability(nn.Module):
    """
    Complete Member 3 reliability component.

    IMPORTANT:
    This class does not contain or modify DINOv2.

    It also does not contain Member 2's global or patch heads.

    The caller supplies the frozen baseline outputs:

        patch_features
        patch_logits
        global_logit
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        epsilon: float = EPSILON,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.epsilon = epsilon

        self.reliability_head = ReliabilityHead(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        patch_logits: torch.Tensor,
        global_logit: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            patch_features: [B, 256, 384]
            patch_logits:   [B, 256]
            global_logit:   [B]

        Returns exactly:

            reliability
            weighted_patch_logit
            final_logit
            aigc_probability
            mean_reliability
        """

        reliability = self.reliability_head(
            patch_features
        )

        weighted_patch_logit = weighted_patch_aggregation(
            patch_logits=patch_logits,
            reliability=reliability,
            epsilon=self.epsilon,
        )

        final_logit = combine_logits(
            global_logit=global_logit,
            weighted_patch_logit=weighted_patch_logit,
        )

        aigc_probability = torch.sigmoid(
            final_logit
        )

        mean_reliability = reliability.mean(
            dim=1
        )

        return {
            "reliability": reliability,
            "weighted_patch_logit": weighted_patch_logit,
            "final_logit": final_logit,
            "aigc_probability": aigc_probability,
            "mean_reliability": mean_reliability,
        }