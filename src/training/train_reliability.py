"""
TraceLens-R Reliability Training
================================

Member 3 component.

This trainer uses Member 2's trained baseline as a FROZEN teacher.

The reliability head is the only trainable component.

Expected baseline:
    BaselineAIGCDetector

Expected baseline attributes:
    baseline.global_head
    baseline.patch_head

Expected features:
    clean_patch_features    [B, 256, 384]
    degraded_patch_features [B, 256, 384]
    degraded_cls_features   [B, 384]

Expected labels:
    0 = authentic
    1 = fully synthetic

Label 2 is excluded from AIGC reliability training.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.reliability import (
    TraceLensReliability,
    compute_survival_target,
    survival_loss,
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible lightweight training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Freeze Member 2 teacher
# ---------------------------------------------------------------------

def freeze_baseline(baseline: nn.Module) -> None:
    """
    Freeze Member 2's baseline teacher.

    The baseline is used only to generate teacher evidence.
    No baseline parameter should receive gradients.
    """

    baseline.eval()

    for parameter in baseline.parameters():
        parameter.requires_grad_(False)


# ---------------------------------------------------------------------
# Member 2 baseline helpers
# ---------------------------------------------------------------------

@torch.no_grad()
def get_patch_logits(
    baseline: nn.Module,
    patch_features: torch.Tensor,
) -> torch.Tensor:
    """
    Run Member 2's patch evidence head.

    Member 2's actual attribute is:
        baseline.patch_head

    Input:
        [B, 256, 384]

    Output:
        [B, 256]
    """

    logits = baseline.patch_head(patch_features)

    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)

    if logits.ndim != 2:
        raise AssertionError(
            "Member 2 patch_head must return [B, 256]. "
            f"Got {tuple(logits.shape)}"
        )

    if logits.shape[1] != 256:
        raise AssertionError(
            f"Expected 256 patch logits, got {logits.shape[1]}"
        )

    return logits


@torch.no_grad()
def get_global_logit(
    baseline: nn.Module,
    cls_features: torch.Tensor,
) -> torch.Tensor:
    """
    Run Member 2's global classification head.

    Member 2's actual attribute is:
        baseline.global_head

    Input:
        [B, 384]

    Output:
        [B]
    """

    logit = baseline.global_head(cls_features)

    if logit.ndim == 2 and logit.shape[-1] == 1:
        logit = logit.squeeze(-1)

    logit = logit.reshape(-1)

    return logit


# ---------------------------------------------------------------------
# Classification loss
# ---------------------------------------------------------------------

def classification_loss(
    final_logit: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Binary AIGC classification loss.

    Only labels 0 and 1 are valid.

    Label 2 is NOT automatically AI-generated.
    """

    labels = labels.reshape(-1).long()

    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError(
            "AIGC reliability training only accepts labels 0 and 1. "
            "Label 2 must not enter the AIGC classification loss."
        )

    return F.binary_cross_entropy_with_logits(
        final_logit,
        labels.float(),
    )


# ---------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------

def reliability_training_step(
    reliability_model: TraceLensReliability,
    baseline_teacher: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    classification_weight: float = 1.0,
    survival_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Perform one training step.

    The baseline teacher remains frozen.

    Required batch keys:

        clean_patch_features
        degraded_patch_features
        degraded_cls_features
        label
    """

    labels = batch["label"].to(device).long().reshape(-1)

    clean_patch_features = batch[
        "clean_patch_features"
    ].to(device)

    degraded_patch_features = batch[
        "degraded_patch_features"
    ].to(device)

    degraded_cls_features = batch[
        "degraded_cls_features"
    ].to(device)

    # ---------------------------------------------------------------
    # Keep only official AIGC labels.
    #
    # Label 0 = authentic
    # Label 1 = fully synthetic
    #
    # Label 2 = locally tampered and must not become AIGC-positive.
    # ---------------------------------------------------------------

    valid = (labels == 0) | (labels == 1)

    if not torch.all(valid):

        labels = labels[valid]

        clean_patch_features = (
            clean_patch_features[valid]
        )

        degraded_patch_features = (
            degraded_patch_features[valid]
        )

        degraded_cls_features = (
            degraded_cls_features[valid]
        )

    if labels.numel() == 0:
        raise ValueError(
            "Batch contains no label-0 or label-1 samples."
        )

    # ---------------------------------------------------------------
    # Frozen baseline teacher
    # ---------------------------------------------------------------

    with torch.no_grad():

        clean_patch_logits = get_patch_logits(
            baseline_teacher,
            clean_patch_features,
        )

        degraded_patch_logits = get_patch_logits(
            baseline_teacher,
            degraded_patch_features,
        )

        degraded_global_logit = get_global_logit(
            baseline_teacher,
            degraded_cls_features,
        )

        # -----------------------------------------------------------
        # Generate detached survival targets.
        # -----------------------------------------------------------

        survival_target, target_weight = (
            compute_survival_target(
                clean_patch_logits=clean_patch_logits,
                degraded_patch_logits=degraded_patch_logits,
                labels=labels,
            )
        )

    # ---------------------------------------------------------------
    # Reliability model
    #
    # The reliability model receives degraded patch features and
    # frozen baseline patch/global evidence.
    # ---------------------------------------------------------------

    outputs = reliability_model(
        patch_features=degraded_patch_features,
        patch_logits=degraded_patch_logits,
        global_logit=degraded_global_logit,
    )

    # ---------------------------------------------------------------
    # AIGC classification loss
    # ---------------------------------------------------------------

    cls_loss = classification_loss(
        outputs["final_logit"],
        labels,
    )

    # ---------------------------------------------------------------
    # Survival loss
    # ---------------------------------------------------------------

    surv_loss = survival_loss(
        predicted_reliability=outputs["reliability"],
        survival_target=survival_target,
        target_weight=target_weight,
    )

    # ---------------------------------------------------------------
    # Combined objective
    # ---------------------------------------------------------------

    total_loss = (
        classification_weight * cls_loss
        + survival_weight * surv_loss
    )

    return {
        "loss": total_loss,
        "classification_loss": cls_loss.detach(),
        "survival_loss": surv_loss.detach(),
        "aigc_probability": outputs[
            "aigc_probability"
        ].detach(),
        "mean_reliability": outputs[
            "mean_reliability"
        ].detach(),
    }


# ---------------------------------------------------------------------
# Epoch training
# ---------------------------------------------------------------------

def train_one_epoch(
    reliability_model: TraceLensReliability,
    baseline_teacher: nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    classification_weight: float = 1.0,
    survival_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Train the reliability head for one epoch.
    """

    reliability_model.train()
    baseline_teacher.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_survival_loss = 0.0

    num_batches = 0

    for batch in dataloader:

        optimizer.zero_grad(
            set_to_none=True
        )

        result = reliability_training_step(
            reliability_model=reliability_model,
            baseline_teacher=baseline_teacher,
            batch=batch,
            device=device,
            classification_weight=classification_weight,
            survival_weight=survival_weight,
        )

        result["loss"].backward()

        optimizer.step()

        total_loss += result[
            "loss"
        ].item()

        total_cls_loss += result[
            "classification_loss"
        ].item()

        total_survival_loss += result[
            "survival_loss"
        ].item()

        num_batches += 1

    if num_batches == 0:
        raise RuntimeError(
            "Dataloader produced zero batches."
        )

    return {
        "loss": total_loss / num_batches,
        "classification_loss": (
            total_cls_loss / num_batches
        ),
        "survival_loss": (
            total_survival_loss / num_batches
        ),
    }


# ---------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    reliability_model: TraceLensReliability,
    optimizer: Optional[
        torch.optim.Optimizer
    ] = None,
    epoch: int = 0,
    metrics: Optional[
        Dict[str, float]
    ] = None,
) -> None:
    """Save Member 3 reliability checkpoint."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "component": "tracelens_reliability",
        "member": 3,
        "epoch": epoch,
        "model_state_dict": (
            reliability_model.state_dict()
        ),
        "metrics": metrics or {},
    }

    if optimizer is not None:
        checkpoint[
            "optimizer_state_dict"
        ] = optimizer.state_dict()

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path: str | Path,
    reliability_model: TraceLensReliability,
    optimizer: Optional[
        torch.optim.Optimizer
    ] = None,
    map_location: str | torch.device = "cpu",
) -> Dict:
    """Load Member 3 reliability checkpoint."""

    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    reliability_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint


# ---------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------

def build_reliability_model(
    device: torch.device,
    hidden_dim: int = 128,
    dropout: float = 0.1,
) -> TraceLensReliability:
    """Create the trainable Member 3 model."""

    model = TraceLensReliability(
        embed_dim=384,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    return model.to(device)


# ---------------------------------------------------------------------
# Basic command-line entry point
# ---------------------------------------------------------------------

def main() -> None:
    """
    Basic smoke entry point.

    Real training is intentionally not fabricated here because the
    actual SID-Set feature cache and Member 2 trained checkpoint
    must be supplied before real training begins.
    """

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "TraceLens-R Member 3 reliability trainer"
        )
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "reliability.pt"
        ),
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        args.device
    )

    model = build_reliability_model(
        device=device,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    print(
        "TraceLens-R Member 3 reliability "
        "model initialized."
    )

    print(
        f"Device: {device}"
    )

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    )

    print(
        "Waiting for real SID-Set features "
        "and Member 2 trained baseline checkpoint "
        "before real training."
    )


if __name__ == "__main__":
    main()