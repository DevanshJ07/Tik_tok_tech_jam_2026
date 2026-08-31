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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from src.data import manifests
from src.models.manipulation import DEFAULT_PATCH_GRID_SIZE, ManipulationHead

__all__ = [
    "MANIPULATION_TRAINING_LABELS",
    "CHECKPOINT_FORMAT_VERSION",
    "ManipulationMaskError",
    "ManipulationCheckpointError",
    "ManipulationLoss",
    "EpochStats",
    "resize_mask_to_patch_grid",
    "validate_manipulation_masks",
    "bce_mask_loss",
    "dice_loss",
    "manipulation_loss",
    "filter_manipulation_batch",
    "resolve_manipulation_device",
    "save_manipulation_checkpoint",
    "load_manipulation_checkpoint",
    "train_one_epoch",
    "evaluate_manipulation_epoch",
]

DEFAULT_DICE_EPSILON = 1e-6
CHECKPOINT_FORMAT_VERSION = 1
CPU_DEVICE = "cpu"

# Labels 0 (authentic) and 2 (locally tampered) contribute to manipulation
# training; label 1 (fully synthetic) never does (see module docstring).
MANIPULATION_TRAINING_LABELS = (manifests.LABEL_AUTHENTIC, manifests.LABEL_LOCALLY_TAMPERED)


class ManipulationMaskError(ValueError):
    """Raised when a manipulation mask is missing or contract-invalid."""


class ManipulationCheckpointError(ValueError):
    """Raised when a manipulation checkpoint is missing or incompatible."""


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


def validate_manipulation_masks(mask: Tensor | None, label: Tensor) -> Tensor:
    """Enforce label-specific mask policy and return a normalized mask tensor.

    * Label ``2`` requires a present, non-empty real mask. Missing or all-zero
      masks raise :class:`ManipulationMaskError`.
    * Label ``0`` is normalized to an all-zero mask (any residual positives
      are cleared so authentic samples cannot leak tamper supervision).
    * Label ``1`` is rejected here; it must already have been excluded.
    """
    if mask is None:
        raise ManipulationMaskError(
            "Manipulation mask is missing. Label-2 samples require a real "
            "non-empty mask; label-0 samples require an all-zero mask."
        )
    if not torch.is_tensor(mask):
        raise ManipulationMaskError(
            f"Manipulation mask must be a tensor, got {type(mask).__name__}."
        )
    if mask.numel() == 0:
        raise ManipulationMaskError("Manipulation mask is empty.")

    label_tensor = label if torch.is_tensor(label) else torch.as_tensor(label)
    if mask.shape[0] != label_tensor.shape[0]:
        raise ManipulationMaskError(
            f"mask batch {mask.shape[0]} does not match label batch {label_tensor.shape[0]}."
        )

    unique = set(int(v) for v in torch.unique(label_tensor).tolist())
    if manifests.LABEL_FULLY_SYNTHETIC in unique:
        raise ManipulationMaskError(
            "Label 1 (fully synthetic) is excluded from manipulation training "
            "and must not be validated as a manipulation sample."
        )
    unexpected = unique - set(MANIPULATION_TRAINING_LABELS)
    if unexpected:
        raise ManipulationMaskError(
            f"label contains values outside {MANIPULATION_TRAINING_LABELS}: {sorted(unexpected)}"
        )

    normalized = mask.detach().clone().float()
    flat = normalized.reshape(normalized.shape[0], -1)
    row_sums = flat.sum(dim=1)

    label2 = label_tensor == manifests.LABEL_LOCALLY_TAMPERED
    if bool(label2.any()) and bool(torch.any(row_sums[label2] <= 0)):
        raise ManipulationMaskError(
            "Label 2 (locally tampered) requires a non-empty real manipulation "
            "mask. All-zero or missing masks are rejected."
        )

    label0 = label_tensor == manifests.LABEL_AUTHENTIC
    if bool(label0.any()):
        normalized[label0] = 0
    return normalized


def resolve_manipulation_device(device: str | None) -> torch.device:
    """Return a torch device. CPU is the default. CUDA is never implied."""
    requested = CPU_DEVICE if device is None or str(device).strip() == "" else str(device).strip()
    if requested == CPU_DEVICE:
        return torch.device(CPU_DEVICE)
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device {requested!r} was requested but CUDA is not available. "
                "Use device='cpu' (the safe default)."
            )
        return torch.device(requested)
    raise RuntimeError(f"Unsupported device {requested!r}. Use 'cpu' or 'cuda'.")


def _move_batch_tensors(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def save_manipulation_checkpoint(
    path: str | Path,
    *,
    model: ManipulationHead,
    epoch: int = 0,
    optimizer: Optional[Optimizer] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically write a versioned ManipulationHead checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": "manipulation_head",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "model_hparams": {
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim,
            "patch_grid_size": model.patch_grid_size,
            "heatmap_size": model.heatmap_size,
            "top_k": model.top_k,
        },
        "train_metadata": {
            "training_labels": list(MANIPULATION_TRAINING_LABELS),
            "patch_grid_size": model.patch_grid_size,
        },
        "extra": dict(extra) if extra else {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_manipulation_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = CPU_DEVICE,
    model: Optional[ManipulationHead] = None,
) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_manipulation_checkpoint`."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise ManipulationCheckpointError(
            f"Manipulation checkpoint not found: {checkpoint_path}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise ManipulationCheckpointError(
            f"Invalid or unreadable manipulation checkpoint {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ManipulationCheckpointError(
            f"{checkpoint_path} is not a TraceLens-R manipulation checkpoint "
            "(missing 'model_state_dict')."
        )
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ManipulationCheckpointError(
            f"Incompatible manipulation checkpoint format_version {version!r}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}."
        )
    kind = payload.get("kind")
    if kind not in (None, "manipulation_head"):
        raise ManipulationCheckpointError(
            f"Incompatible checkpoint kind {kind!r}; expected 'manipulation_head'."
        )
    hparams = payload.get("model_hparams") or {}
    expected = {
        "embedding_dim": 384,
        "patch_grid_size": DEFAULT_PATCH_GRID_SIZE,
    }
    embed_dim = int(hparams.get("embedding_dim", expected["embedding_dim"]))
    grid = int(hparams.get("patch_grid_size", expected["patch_grid_size"]))
    if embed_dim != expected["embedding_dim"] or grid != expected["patch_grid_size"]:
        raise ManipulationCheckpointError(
            f"Incompatible manipulation checkpoint {checkpoint_path}: "
            f"embedding_dim={embed_dim}, patch_grid_size={grid}; "
            f"expected {expected['embedding_dim']} and {expected['patch_grid_size']}."
        )
    created = False
    if model is None:
        model = ManipulationHead(
            embedding_dim=embed_dim,
            hidden_dim=int(hparams.get("hidden_dim", 128)),
            patch_grid_size=grid,
            heatmap_size=int(hparams.get("heatmap_size", 224)),
            top_k=int(hparams.get("top_k", 16)),
        )
        created = True
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ManipulationCheckpointError(
            f"Incompatible manipulation checkpoint state dict: {exc}"
        ) from exc
    return {
        "model": model,
        "created_model": created,
        "epoch": int(payload.get("epoch", 0)),
        "model_hparams": hparams,
        "train_metadata": payload.get("train_metadata", {}),
        "extra": payload.get("extra", {}),
    }


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
    features = patch_features[eligible]
    labels = label_tensor[eligible]
    masks = validate_manipulation_masks(mask[eligible], labels)
    return features, masks, labels


def train_one_epoch(
    model: ManipulationHead,
    batches: Iterable[Mapping[str, Tensor]],
    optimizer: Optimizer,
    *,
    patch_grid_size: int = DEFAULT_PATCH_GRID_SIZE,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    dice_epsilon: float = DEFAULT_DICE_EPSILON,
    device: str | None = CPU_DEVICE,
) -> EpochStats:
    """Run one training epoch over pre-extracted patch-feature batches.

    Each batch must be a mapping with ``"patch_features"`` ``[B, 256, 384]``,
    ``"mask"`` (``[B, 1, H, W]`` or already ``[B, 256]``), and ``"label"``
    ``[B]`` (SID-Set 0/1/2 convention). Label-1 rows are dropped per batch
    via :func:`filter_manipulation_batch`; a batch left with no eligible
    rows is skipped rather than back-propagated through a meaningless loss.
    ``device`` defaults to CPU. CUDA is used only when requested and available.

    Raises ``ValueError`` if every batch was skipped (nothing to train on).
    """
    torch_device = resolve_manipulation_device(device)
    model.to(torch_device)
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    num_batches = 0
    num_skipped = 0

    for batch in batches:
        moved = _move_batch_tensors(batch, torch_device)
        filtered = filter_manipulation_batch(
            moved["patch_features"], moved["mask"], moved["label"]
        )
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


def evaluate_manipulation_epoch(
    model: ManipulationHead,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    patch_grid_size: int = DEFAULT_PATCH_GRID_SIZE,
    device: str | None = CPU_DEVICE,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute validation metrics. Does not update parameters or look at test data."""
    import numpy as np
    from sklearn.metrics import (
        balanced_accuracy_score,
        brier_score_loss,
        roc_auc_score,
    )

    torch_device = resolve_manipulation_device(device)
    model.to(torch_device)
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    n_batches = 0
    image_scores: list[float] = []
    image_labels: list[int] = []
    patch_inter = 0.0
    patch_pred_sum = 0.0
    patch_tgt_sum = 0.0
    heat_inter = 0.0
    heat_pred_sum = 0.0
    heat_tgt_sum = 0.0

    with torch.no_grad():
        for batch in batches:
            moved = _move_batch_tensors(batch, torch_device)
            filtered = filter_manipulation_batch(
                moved["patch_features"], moved["mask"], moved["label"]
            )
            if filtered is None:
                continue
            patch_features, mask, labels = filtered
            output = model(patch_features)
            loss = manipulation_loss(
                output["patch_mask_logits"],
                mask,
                patch_grid_size=patch_grid_size,
            )
            total_loss += float(loss.total)
            total_bce += float(loss.bce)
            total_dice += float(loss.dice)
            n_batches += 1

            image_scores.extend(output["manipulation_probability"].detach().cpu().tolist())
            binary_labels = (labels == manifests.LABEL_LOCALLY_TAMPERED).long().cpu().tolist()
            image_labels.extend(binary_labels)

            target_patches = resize_mask_to_patch_grid(mask, patch_grid_size=patch_grid_size)
            pred_patches = (torch.sigmoid(output["patch_mask_logits"]) >= threshold).float()
            patch_inter += float((pred_patches * target_patches).sum())
            patch_pred_sum += float(pred_patches.sum())
            patch_tgt_sum += float(target_patches.sum())

            heatmap_prob = torch.sigmoid(output["heatmap"])
            if mask.dim() == 3:
                gt = mask.unsqueeze(1)
            else:
                gt = mask
            if gt.shape[-2:] != heatmap_prob.shape[-2:]:
                gt = F.interpolate(gt.float(), size=heatmap_prob.shape[-2:], mode="nearest")
            pred_heat = (heatmap_prob >= threshold).float()
            gt_bin = (gt >= 0.5).float()
            heat_inter += float((pred_heat * gt_bin).sum())
            heat_pred_sum += float(pred_heat.sum())
            heat_tgt_sum += float(gt_bin.sum())

    if n_batches == 0:
        raise ValueError("No eligible validation batches.")

    y_true = np.asarray(image_labels, dtype=int)
    y_score = np.asarray(image_scores, dtype=float)
    y_hat = (y_score >= threshold).astype(int)
    auroc = None
    if len(set(y_true.tolist())) > 1:
        auroc = float(roc_auc_score(y_true, y_score))

    def _dice(inter: float, pred_s: float, tgt_s: float) -> float:
        return float((2.0 * inter) / (pred_s + tgt_s + 1e-6))

    def _iou(inter: float, pred_s: float, tgt_s: float) -> float:
        return float(inter / (pred_s + tgt_s - inter + 1e-6))

    return {
        "mean_total_loss": total_loss / n_batches,
        "mean_bce_loss": total_bce / n_batches,
        "mean_dice_loss": total_dice / n_batches,
        "num_batches": float(n_batches),
        "sample_count": float(len(y_true)),
        "image_balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
        "image_auroc": float("nan") if auroc is None else auroc,
        "image_brier": float(brier_score_loss(y_true, y_score)),
        "patch_dice": _dice(patch_inter, patch_pred_sum, patch_tgt_sum),
        "patch_iou": _iou(patch_inter, patch_pred_sum, patch_tgt_sum),
        "heatmap_dice": _dice(heat_inter, heat_pred_sum, heat_tgt_sum),
        "heatmap_iou": _iou(heat_inter, heat_pred_sum, heat_tgt_sum),
        "threshold": threshold,
    }


def _build_train_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the TraceLens-R manipulation head on cached patch features."
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Root with clean/train and clean/val subdirectories "
        "(e.g. data/cache/manipulation).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=CPU_DEVICE)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--run-name", type=str, default="manipulation_real")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=16)
    return parser


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    """Operational CLI. Model/loss semantics stay those of train_one_epoch."""
    import random
    import shutil
    import time

    import numpy as np
    from torch.utils.data import DataLoader

    from src.training.manipulation_cache import CachedManipulationDataset

    args = _build_train_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache_root = Path(args.cache_dir)
    train_ds = CachedManipulationDataset(cache_root / "clean" / "train")
    val_ds = CachedManipulationDataset(cache_root / "clean" / "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ManipulationHead(hidden_dim=args.hidden_dim, top_k=args.top_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_path: Path | None = None
    best_val_loss = float("inf")
    best_epoch = -1
    t0 = time.time()
    print(
        f"[manip] train_n={len(train_ds)} val_n={len(val_ds)} "
        f"epochs={args.epochs} device={args.device} seed={args.seed}",
        flush=True,
    )

    for epoch in range(args.epochs):
        stats = train_one_epoch(model, train_loader, optimizer, device=args.device)
        val_metrics = evaluate_manipulation_epoch(model, val_loader, device=args.device)
        ckpt_path = out_dir / f"{args.run_name}_epoch{epoch}.pt"
        save_manipulation_checkpoint(
            ckpt_path,
            model=model,
            epoch=epoch + 1,
            optimizer=optimizer,
            extra={"val_metrics": val_metrics, "train_stats": vars(stats)},
        )
        row = {
            "epoch": epoch,
            "train_loss": stats.mean_total_loss,
            "val_loss": val_metrics["mean_total_loss"],
            "val_image_auroc": val_metrics["image_auroc"],
            "val_image_balanced_accuracy": val_metrics["image_balanced_accuracy"],
            "val_patch_dice": val_metrics["patch_dice"],
            "checkpoint": str(ckpt_path),
        }
        history.append(row)
        print(
            f"[manip] epoch {epoch} train_loss={stats.mean_total_loss:.4f} "
            f"val_loss={val_metrics['mean_total_loss']:.4f} "
            f"val_auroc={val_metrics['image_auroc']:.4f} "
            f"val_bal_acc={val_metrics['image_balanced_accuracy']:.4f} "
            f"patch_dice={val_metrics['patch_dice']:.4f}",
            flush=True,
        )
        if val_metrics["mean_total_loss"] < best_val_loss:
            best_val_loss = val_metrics["mean_total_loss"]
            best_epoch = epoch
            best_path = ckpt_path

    if best_path is None:
        raise RuntimeError("No manipulation checkpoint was selected.")
    final_path = out_dir / f"{args.run_name}_final.pt"
    shutil.copy2(best_path, final_path)
    elapsed = time.time() - t0
    summary = {
        "selected_epoch": best_epoch,
        "selected_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "selection_metric": "val_mean_total_loss",
        "best_val_loss": best_val_loss,
        "history": history,
        "elapsed_seconds": elapsed,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    }
    print(f"[manip] selected epoch={best_epoch} val_loss={best_val_loss:.4f} -> {final_path}", flush=True)
    print(f"[manip] elapsed={elapsed:.1f}s summary={summary}", flush=True)
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
