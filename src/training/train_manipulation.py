"""Manipulation-branch training utilities (Stage 2, Member 4).

Trains only :class:`~src.models.manipulation.ManipulationHead` on
pre-extracted DINOv2 patch features (``[B, 256, 384]``). This module never
implements or fine-tunes DINOv2, never touches AIGC probability logic, and
never applies reliability weighting -- the manipulation branch stays fully
separable from the rest of TraceLens-R.

SID-Set label convention (see ``src/data/manifests.py``): ``0`` = authentic
(all-zero target mask), ``2`` = locally tampered (real mask), ``1`` = fully
synthetic. Label 1 must never contribute to manipulation training -- it is
excluded here defensively (independent of ``TraceLensDataset(task="manipulation")``
already excluding it upstream) so these utilities are testable in isolation
and safe even if fed a raw, unfiltered batch.

No frozen DINOv2 patch-feature extractor/cache exists yet anywhere in this
repository (Member 2/3 dependency). ``train_one_epoch`` therefore consumes
batches that already carry ``"patch_features"`` -- callers supply mock
tensors for testing today, and a real extractor once one exists. This
module never runs a backbone itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from src.data import manifests
from src.models.manipulation import DEFAULT_PATCH_GRID_SIZE, ManipulationHead

__all__ = [
    "MANIPULATION_TRAINING_LABELS",
    "ManipulationLoss",
    "EpochStats",
    "resize_mask_to_patch_grid",
    "bce_mask_loss",
    "dice_loss",
    "manipulation_loss",
    "filter_manipulation_batch",
    "train_one_epoch",
]

DEFAULT_DICE_EPSILON = 1e-6

# Labels 0 (authentic) and 2 (locally tampered) contribute to manipulation
# training; label 1 (fully synthetic) never does (see module docstring).
MANIPULATION_TRAINING_LABELS = (manifests.LABEL_AUTHENTIC, manifests.LABEL_LOCALLY_TAMPERED)


@dataclass
class ManipulationLoss:
    """BCE + Dice manipulation loss, with components kept for logging/inspection."""

    total: Tensor
    bce: Tensor
    dice: Tensor


@dataclass
class EpochStats:
    """Aggregate statistics returned by one :func:`train_one_epoch` call."""

    mean_total_loss: float
    mean_bce_loss: float
    mean_dice_loss: float
    num_batches: int
    num_skipped_batches: int


def resize_mask_to_patch_grid(
    mask: Tensor,
    patch_grid_size: int = DEFAULT_PATCH_GRID_SIZE,
) -> Tensor:
    """Downsample a pixel-resolution binary mask to a flat per-patch target.

    ``mask`` is ``[B, 1, H, W]`` or ``[B, H, W]`` with values in ``{0, 1}``
    (e.g. the ``[B, 1, 224, 224]`` masks ``TraceLensDataset`` returns). A
    patch is positive if *any* of its pixels are manipulated -- computed via
    max pooling over the whole patch region, not a fractional/majority rule.
    This matters for localisation: a small tampered region confined to a
    minority of a patch's area (e.g. 20-40%) must still mark that patch
    positive, since it genuinely contains manipulated pixels. Because the
    input is binary, the max of a patch's pixels is already exactly ``0`` or
    ``1`` -- no separate thresholding step is needed to keep the result
    binary.

    Returns ``[B, patch_grid_size ** 2]`` with values in ``{0.0, 1.0}``.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(
            f"mask must be [B,1,H,W] or [B,H,W], got shape {tuple(mask.shape)}."
        )
    if patch_grid_size < 1:
        raise ValueError(f"patch_grid_size must be positive, got {patch_grid_size}.")

    batch_size = mask.shape[0]
    patch_targets = F.adaptive_max_pool2d(mask.float(), output_size=(patch_grid_size, patch_grid_size))
    return patch_targets.reshape(batch_size, patch_grid_size * patch_grid_size)


def bce_mask_loss(patch_mask_logits: Tensor, target: Tensor) -> Tensor:
    """BCE-with-logits over flat per-patch logits/targets, batched mean.

    Operates on RAW logits (``F.binary_cross_entropy_with_logits`` applies
    its own numerically-stable sigmoid internally) -- ``patch_mask_logits``
    must not be pre-sigmoided.
    """
    if patch_mask_logits.shape != target.shape:
        raise ValueError(
            f"patch_mask_logits and target must share shape, got "
            f"{tuple(patch_mask_logits.shape)} vs {tuple(target.shape)}."
        )
    return F.binary_cross_entropy_with_logits(patch_mask_logits, target.float())


def dice_loss(patch_probabilities: Tensor, target: Tensor, epsilon: float = DEFAULT_DICE_EPSILON) -> Tensor:
    """Soft Dice loss over flat per-patch probabilities/targets, batched mean.

    ``patch_probabilities`` must already be ``torch.sigmoid(logits)`` --
    this function does not apply sigmoid itself. ``epsilon`` smooths both
    the numerator and denominator so an all-zero target (e.g. an authentic
    sample, where ``target.sum() == 0``) yields a finite loss instead of a
    0/0 division, without special-casing the empty-mask case to force the
    loss to exactly zero: once smoothed, a correctly-predicted empty mask
    already lands near zero on its own.
    """
    if patch_probabilities.shape != target.shape:
        raise ValueError(
            f"patch_probabilities and target must share shape, got "
            f"{tuple(patch_probabilities.shape)} vs {tuple(target.shape)}."
        )
    target = target.float()
    intersection = (patch_probabilities * target).sum(dim=-1)
    union = patch_probabilities.sum(dim=-1) + target.sum(dim=-1)
    dice_coefficient = (2.0 * intersection + epsilon) / (union + epsilon)
    return (1.0 - dice_coefficient).mean()


def manipulation_loss(
    patch_mask_logits: Tensor,
    mask: Tensor,
    *,
    patch_grid_size: int = DEFAULT_PATCH_GRID_SIZE,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    dice_epsilon: float = DEFAULT_DICE_EPSILON,
) -> ManipulationLoss:
    """Combined manipulation loss: ``total = bce_weight * BCE + dice_weight * Dice``.

    ``mask`` may already be a flat ``[B, patch_grid_size ** 2]`` patch-level
    target, or a pixel-resolution mask (e.g. ``[B, 1, 224, 224]``) -- it is
    resized via :func:`resize_mask_to_patch_grid` automatically whenever its
    shape doesn't already match ``patch_mask_logits``. Weights default to
    ``1.0`` (i.e. plain ``BCE + Dice``); they exist only so future tuning
    doesn't require an API change.
    """
    if mask.shape == patch_mask_logits.shape:
        target = mask.float()
    else:
        target = resize_mask_to_patch_grid(mask, patch_grid_size=patch_grid_size)

    bce = bce_mask_loss(patch_mask_logits, target)
    dice = dice_loss(torch.sigmoid(patch_mask_logits), target, epsilon=dice_epsilon)
    total = bce_weight * bce + dice_weight * dice
    return ManipulationLoss(total=total, bce=bce, dice=dice)


def filter_manipulation_batch(
    patch_features: Tensor,
    mask: Tensor,
    label: Tensor,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Drop label==1 (fully synthetic) rows from a batch before manipulation training.

    Labels 0 (authentic) and 2 (locally tampered) pass through unchanged;
    label 1 never contributes and is never reinterpreted as label 2. Returns
    ``None`` when no eligible rows remain (e.g. an all-label-1 batch), so
    callers can skip the batch explicitly instead of computing a loss over
    zero eligible samples.
    """
    label_tensor = label if torch.is_tensor(label) else torch.as_tensor(label)
    if patch_features.shape[0] != label_tensor.shape[0] or mask.shape[0] != label_tensor.shape[0]:
        raise ValueError(
            "patch_features, mask, and label must share batch size, got "
            f"{patch_features.shape[0]}, {mask.shape[0]}, {label_tensor.shape[0]}."
        )
    invalid = set(torch.unique(label_tensor).tolist()) - set(manifests.VALID_LABELS)
    if invalid:
        raise ValueError(f"label contains values outside {manifests.VALID_LABELS}: {sorted(invalid)}")

    eligible = label_tensor != manifests.LABEL_FULLY_SYNTHETIC
    if not torch.any(eligible):
        return None
    return patch_features[eligible], mask[eligible], label_tensor[eligible]


def train_one_epoch(
    model: ManipulationHead,
    batches: Iterable[Mapping[str, Tensor]],
    optimizer: Optimizer,
    *,
    patch_grid_size: int = DEFAULT_PATCH_GRID_SIZE,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    dice_epsilon: float = DEFAULT_DICE_EPSILON,
) -> EpochStats:
    """Run one training epoch over pre-extracted patch-feature batches.

    Each batch must be a mapping with ``"patch_features"`` ``[B, 256, 384]``,
    ``"mask"`` (``[B, 1, H, W]`` or already ``[B, 256]``), and ``"label"``
    ``[B]`` (SID-Set 0/1/2 convention). Label-1 rows are dropped per batch
    via :func:`filter_manipulation_batch`; a batch left with no eligible
    rows is skipped rather than back-propagated through a meaningless loss.
    Suitable for CPU execution with a standard PyTorch optimizer -- this is
    intentionally not a general-purpose trainer.

    Raises ``ValueError`` if every batch was skipped (nothing to train on).
    """
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    num_batches = 0
    num_skipped = 0

    for batch in batches:
        filtered = filter_manipulation_batch(batch["patch_features"], batch["mask"], batch["label"])
        if filtered is None:
            num_skipped += 1
            continue
        patch_features, mask, _label = filtered

        optimizer.zero_grad()
        output = model(patch_features)
        loss = manipulation_loss(
            output["patch_mask_logits"],
            mask,
            patch_grid_size=patch_grid_size,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            dice_epsilon=dice_epsilon,
        )
        loss.total.backward()
        optimizer.step()

        total_loss += loss.total.item()
        total_bce += loss.bce.item()
        total_dice += loss.dice.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError(
            f"No eligible batches to train on ({num_skipped} skipped, all label-1 or empty)."
        )

    return EpochStats(
        mean_total_loss=total_loss / num_batches,
        mean_bce_loss=total_bce / num_batches,
        mean_dice_loss=total_dice / num_batches,
        num_batches=num_batches,
        num_skipped_batches=num_skipped,
    )
