"""Baseline AIGC detector training (Member 2 component).

Trains ONLY the two lightweight heads in
:class:`src.models.baseline.BaselineAIGCDetector` on top of the frozen
DINOv2 backbone. DINOv2 is never fine-tuned.

Label policy
------------
Only labels ``0`` (authentic) and ``1`` (fully synthetic) contribute to the
loss. Label ``2`` (locally tampered) is excluded. This is enforced twice:

1. The dataset is built with ``task="aigc"`` (see ``src/data/dataset.py``),
   which already drops label 2.
2. Every batch is re-filtered here with an explicit ``label in {0, 1}`` mask
   before the loss, so a cached-feature file or a hand-built loader that
   slips a 2 through still cannot poison the AIGC loss.

Loss
----
``BCEWithLogitsLoss`` on each of the three predictions, equally weighted::

    loss = BCE(global_logit, y) + BCE(patch_mean_logit, y) + BCE(final_logit, y)

Data sources
------------
* ``--manifest`` + ``--dataset-root``: images streamed through the frozen
  backbone each step.
* ``--cache-dir``: pre-extracted feature ``.pt`` files (see
  ``scripts/cache_features.py``) -- no backbone needed, ideal for CPU.
* ``--smoke`` with neither of the above: a tiny run on randomly generated
  features. The printed numbers are real computations on random inputs and
  are explicitly labelled as such -- they are NOT model results.

Examples
--------
Smoke test (no data required)::

    python -m src.training.train_baseline --smoke

Real run from cached features::

    python -m src.training.train_baseline --cache-dir data/cache/features/clean/train \
        --epochs 5 --batch-size 64 --out-dir checkpoints

    python -m src.training.train_baseline --cache-dir data/cache/features \
        --cache-subset clean --epochs 5 --batch-size 64 --out-dir checkpoints

Real run from a manifest::

    python -m src.training.train_baseline --manifest data/manifests/manifest.csv \
        --dataset-root /path/to/SID-Set --split train --epochs 3
"""
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.baseline import EMBED_DIM, NUM_PATCHES, BaselineAIGCDetector

CHECKPOINT_FORMAT_VERSION = 1
AIGC_LABELS = (0, 1)
CACHE_GROUP_CLEAN = "clean"
CACHE_GROUP_TRANSFORMED = "transformed"
CACHE_GROUPS = (CACHE_GROUP_CLEAN, CACHE_GROUP_TRANSFORMED)


# ===========================================================================
# Determinism
# ===========================================================================
def set_seed(seed: int) -> None:
    """Seed python / numpy / torch RNGs for reproducible CPU training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # harmless on CPU-only machines
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def seed_worker(worker_id: int) -> None:  # pragma: no cover - DataLoader hook
    """Per-worker seeding so multi-worker DataLoaders stay deterministic."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 3
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 128
    dropout: float = 0.1
    global_weight: float = 0.5
    patch_weight: float = 0.5
    num_workers: int = 0
    max_steps_per_epoch: Optional[int] = None
    log_every: int = 10
    device: str = "cpu"
    out_dir: str = "checkpoints"
    run_name: str = "baseline"
    save_every_epoch: bool = True


# ===========================================================================
# Cached-feature dataset
# ===========================================================================
def classify_cache_group(
    path: Path,
    record: Dict[str, Any],
    cache_root: Path,
) -> Optional[str]:
    """Return ``clean``, ``transformed``, or ``None`` if the group is unknown.

    Prefers the ``clean/`` vs ``transformed/`` directory layout written by
    ``scripts/cache_features.py``, then ``transform_name`` in the record.
    Conflicting path / metadata is an error, not a silent guess.
    """
    try:
        rel_parts = Path(path).resolve().relative_to(Path(cache_root).resolve()).parts
    except ValueError:
        rel_parts = Path(path).parts
    dir_parts = set(rel_parts[:-1])
    has_clean = CACHE_GROUP_CLEAN in dir_parts
    has_transformed = CACHE_GROUP_TRANSFORMED in dir_parts
    if has_clean and has_transformed:
        raise ValueError(
            f"{path} sits under both {CACHE_GROUP_CLEAN!r} and "
            f"{CACHE_GROUP_TRANSFORMED!r} directories"
        )
    path_group: Optional[str] = None
    if has_clean:
        path_group = CACHE_GROUP_CLEAN
    elif has_transformed:
        path_group = CACHE_GROUP_TRANSFORMED

    raw_name = record.get("transform_name")
    rec_group: Optional[str] = None
    if raw_name is not None and str(raw_name) != "":
        rec_group = (
            CACHE_GROUP_CLEAN if str(raw_name) == CACHE_GROUP_CLEAN else CACHE_GROUP_TRANSFORMED
        )

    if path_group is not None and rec_group is not None and path_group != rec_group:
        raise ValueError(
            f"{path.name}: path group {path_group!r} conflicts with "
            f"transform_name={raw_name!r}"
        )
    return path_group or rec_group


class MixedCacheGroupsError(ValueError):
    """Raised when a cache-dir mixes clean and transformed features."""


class CachedFeatureDataset(Dataset):
    """Reads ``.pt`` feature files produced by ``scripts/cache_features.py``.

    Each file is a dict with at least ``cls_features [384]``,
    ``patch_features [256, 384]`` and ``label``. Files whose label is not in
    ``{0, 1}`` are dropped at construction time (belt-and-braces; the loss
    filter would drop them anyway).

    Clean and transformed caches MUST NOT be mixed unless the caller names
    exactly one group via ``cache_subset`` (``clean`` or ``transformed``) or
    points ``cache_dir`` at a single-group directory.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        recursive: bool = True,
        cache_subset: Optional[str] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"cache-dir does not exist: {self.cache_dir}")
        if cache_subset is not None:
            cache_subset = str(cache_subset)
            if cache_subset not in CACHE_GROUPS:
                raise ValueError(
                    f"cache_subset must be one of {CACHE_GROUPS}, got {cache_subset!r}"
                )
        self.cache_subset = cache_subset

        pattern = "**/*.pt" if recursive else "*.pt"
        all_files = sorted(self.cache_dir.glob(pattern))
        if not all_files:
            raise FileNotFoundError(f"No .pt feature files found under {self.cache_dir}")

        classified: List[Tuple[Path, Optional[str]]] = []
        skipped = 0
        observed_groups = set()
        for fp in all_files:
            try:
                rec = torch.load(fp, map_location="cpu", weights_only=False)
                label = int(rec["label"])
            except Exception as exc:  # noqa: BLE001 - surface but keep going
                print(f"[cache] WARN: could not read {fp.name}: {exc}")
                skipped += 1
                continue
            group = classify_cache_group(fp, rec, self.cache_dir)
            if label not in AIGC_LABELS:
                skipped += 1
                continue
            if group is not None:
                observed_groups.add(group)
            classified.append((fp, group))

        if (
            cache_subset is None
            and CACHE_GROUP_CLEAN in observed_groups
            and CACHE_GROUP_TRANSFORMED in observed_groups
        ):
            raise MixedCacheGroupsError(
                f"cache-dir {self.cache_dir} mixes {CACHE_GROUP_CLEAN!r} and "
                f"{CACHE_GROUP_TRANSFORMED!r} features. Pass a single-group "
                f"directory (e.g. .../{CACHE_GROUP_CLEAN}/train) or set "
                f"--cache-subset {CACHE_GROUP_CLEAN}|{CACHE_GROUP_TRANSFORMED}."
            )

        self.files: List[Path] = []
        for fp, group in classified:
            if cache_subset is not None and group != cache_subset:
                skipped += 1
                continue
            self.files.append(fp)

        if not self.files:
            raise ValueError(
                f"No usable label-0/1 feature files under {self.cache_dir} "
                f"(subset={cache_subset!r}, {skipped} skipped)."
            )
        if skipped:
            print(f"[cache] {len(self.files)} usable files, {skipped} skipped (label 2 / unreadable / other subset).")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        cls = torch.as_tensor(rec["cls_features"], dtype=torch.float32)
        patch = torch.as_tensor(rec["patch_features"], dtype=torch.float32)
        if cls.shape != (EMBED_DIM,):
            raise ValueError(f"{self.files[idx].name}: cls_features shape {tuple(cls.shape)}")
        if patch.shape != (NUM_PATCHES, EMBED_DIM):
            raise ValueError(f"{self.files[idx].name}: patch_features shape {tuple(patch.shape)}")
        return {
            "cls_features": cls,
            "patch_features": patch,
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
        }


# ===========================================================================
# Batch adapters -- tolerate imperfect upstream batch formats
# ===========================================================================
def _labels_to_long_tensor(raw: Any) -> torch.Tensor:
    """Coerce whatever the loader produced for ``label`` into ``LongTensor[B]``."""
    if isinstance(raw, torch.Tensor):
        return raw.reshape(-1).long()
    if isinstance(raw, (list, tuple)):
        return torch.tensor([int(x) for x in raw], dtype=torch.long)
    if isinstance(raw, np.ndarray):
        return torch.as_tensor(raw.reshape(-1), dtype=torch.long)
    return torch.tensor([int(raw)], dtype=torch.long)


def unpack_image_batch(batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """From a ``TraceLensDataset`` (default-collated) batch -> (images, labels).

    ``images`` : FloatTensor ``[B, 3, 224, 224]``
    ``labels`` : LongTensor  ``[B]``
    """
    if "image" not in batch or "label" not in batch:
        raise KeyError(
            "Expected a dataset batch with 'image' and 'label' keys; "
            f"got keys {sorted(batch.keys())}"
        )
    images = batch["image"]
    if not isinstance(images, torch.Tensor):
        images = torch.as_tensor(np.asarray(images))
    images = images.float()
    if images.dim() != 4 or images.shape[1:] != (3, 224, 224):
        raise ValueError(
            f"Expected images [B, 3, 224, 224], got {tuple(images.shape)}"
        )
    return images, _labels_to_long_tensor(batch["label"])


def unpack_feature_batch(batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """From a ``CachedFeatureDataset`` batch -> (cls, patch, labels)."""
    cls = batch["cls_features"].float()
    patch = batch["patch_features"].float()
    labels = _labels_to_long_tensor(batch["label"])
    return cls, patch, labels


def aigc_label_mask(labels: torch.Tensor) -> torch.Tensor:
    """Boolean mask selecting only label-0 / label-1 rows (excludes label 2)."""
    return (labels == 0) | (labels == 1)


# ===========================================================================
# Loss
# ===========================================================================
def baseline_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    bce: nn.Module,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """``BCE(global) + BCE(patch_mean) + BCE(final)`` with equal weights."""
    y = targets.float()
    l_global = bce(outputs["global_logit"], y)
    l_patch = bce(outputs["patch_mean_logit"], y)
    l_final = bce(outputs["final_logit"], y)
    total = l_global + l_patch + l_final
    return total, {
        "loss": float(total.detach()),
        "loss_global": float(l_global.detach()),
        "loss_patch": float(l_patch.detach()),
        "loss_final": float(l_final.detach()),
    }


# ===========================================================================
# Checkpoints  (robust: atomic write, explicit validation on load)
# ===========================================================================
def save_checkpoint(
    path: str | Path,
    *,
    model: BaselineAIGCDetector,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    global_step: int,
    config: TrainConfig,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically write a checkpoint (temp file + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_hparams": {
            "embed_dim": model.embed_dim,
            "num_patches": model.num_patches,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "global_weight": model.global_weight,
            "patch_weight": model.patch_weight,
        },
        "train_config": vars(config),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def build_model_from_hparams(hparams: Dict[str, Any]) -> BaselineAIGCDetector:
    """Reconstruct a detector from a checkpoint's ``model_hparams`` block."""
    return BaselineAIGCDetector(
        embed_dim=int(hparams.get("embed_dim", EMBED_DIM)),
        num_patches=int(hparams.get("num_patches", NUM_PATCHES)),
        hidden_dim=int(hparams.get("hidden_dim", 128)),
        dropout=float(hparams.get("dropout", 0.1)),
        global_weight=float(hparams.get("global_weight", 0.5)),
        patch_weight=float(hparams.get("patch_weight", 0.5)),
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: Optional[BaselineAIGCDetector] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    If ``model`` is None, one is reconstructed from the stored hparams and
    returned under the ``"model"`` key. Raises informative errors rather than
    silently returning a half-loaded state.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"{path} is not a TraceLens-R baseline checkpoint "
            "(missing 'model_state_dict')."
        )
    version = ckpt.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        print(
            f"[ckpt] WARN: format_version {version!r} != expected "
            f"{CHECKPOINT_FORMAT_VERSION}; attempting to load anyway."
        )

    created_model = False
    if model is None:
        model = build_model_from_hparams(ckpt.get("model_hparams", {}))
        created_model = True

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if (missing or unexpected) and strict:  # pragma: no cover - load_state_dict raises first
        raise RuntimeError(f"State dict mismatch. missing={missing} unexpected={unexpected}")

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return {
        "model": model,
        "created_model": created_model,
        "epoch": int(ckpt.get("epoch", 0)),
        "global_step": int(ckpt.get("global_step", 0)),
        "model_hparams": ckpt.get("model_hparams", {}),
        "train_config": ckpt.get("train_config", {}),
        "extra": ckpt.get("extra", {}),
    }


# ===========================================================================
# Data-source construction
# ===========================================================================
@dataclass
class DataSource:
    """A uniform handle over the two real training data paths."""

    kind: str  # "features" | "images"
    loader: Iterable
    backbone: Optional[nn.Module] = None
    size: Optional[int] = None


def _make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def build_feature_source(
    cache_dir: str,
    cfg: TrainConfig,
    cache_subset: Optional[str] = None,
) -> DataSource:
    ds = CachedFeatureDataset(cache_dir, cache_subset=cache_subset)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=_make_generator(cfg.seed),
        drop_last=False,
    )
    return DataSource(kind="features", loader=loader, size=len(ds))


def build_image_source(
    manifest: str,
    dataset_root: str,
    split: str,
    cfg: TrainConfig,
    backbone_name: str,
) -> DataSource:
    # Imported lazily so `--smoke` / `--cache-dir` runs don't need the dataset
    # stack or the transformers dependency.
    from src.data.dataset import TASK_AIGC, TraceLensDataset
    from src.models.backbone import DINOv2Backbone

    ds = TraceLensDataset(
        manifest=manifest,
        split=split,
        dataset_root=dataset_root,
        task=TASK_AIGC,          # labels 0/1 only
        seed=cfg.seed,
        normalize=True,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=_make_generator(cfg.seed),
        drop_last=False,
    )
    backbone = DINOv2Backbone(model_name=backbone_name, device=cfg.device)
    return DataSource(kind="images", loader=loader, backbone=backbone, size=len(ds))


def synthetic_smoke_batches(
    cfg: TrainConfig, n_batches: int = 4, batch_size: int = 8
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Random CLS/patch features with 0/1 labels -- smoke plumbing only.

    NOT training data. Metrics printed from this path are real computations
    on random noise and are labelled ``SMOKE`` at the call site.
    """
    gen = torch.Generator().manual_seed(cfg.seed)
    for _ in range(n_batches):
        cls = torch.randn(batch_size, EMBED_DIM, generator=gen)
        patch = torch.randn(batch_size, NUM_PATCHES, EMBED_DIM, generator=gen)
        labels = torch.randint(0, 2, (batch_size,), generator=gen)
        yield cls, patch, labels


# ===========================================================================
# Training loop
# ===========================================================================
def _forward_batch(
    batch: Any,
    source_kind: str,
    model: BaselineAIGCDetector,
    backbone: Optional[nn.Module],
    device: torch.device,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor], int]:
    """Return (outputs, valid_targets, n_dropped_label2). outputs is None if
    the batch has no usable label-0/1 rows."""
    if source_kind == "synthetic":
        cls, patch, labels = batch
    elif source_kind == "features":
        cls, patch, labels = unpack_feature_batch(batch)
    elif source_kind == "images":
        images, labels = unpack_image_batch(batch)
        mask = aigc_label_mask(labels)
        n_dropped = int((~mask).sum())
        if not bool(mask.any()):
            return None, None, n_dropped
        images = images[mask].to(device)
        labels = labels[mask]
        assert backbone is not None
        cls, patch = backbone.extract_features(images)
        outputs = model(cls, patch)
        return outputs, labels.to(device), n_dropped
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"unknown source kind {source_kind!r}")

    # features / synthetic paths share this tail
    mask = aigc_label_mask(labels)
    n_dropped = int((~mask).sum())
    if not bool(mask.any()):
        return None, None, n_dropped
    cls = cls[mask].to(device)
    patch = patch[mask].to(device)
    labels = labels[mask].to(device)
    outputs = model(cls, patch)
    return outputs, labels, n_dropped


def train(
    cfg: TrainConfig,
    *,
    source_kind: str,
    batches_per_epoch: Iterable,
    n_batches_hint: Optional[int],
    backbone: Optional[nn.Module],
    resume_from: Optional[str] = None,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Run training. Returns a summary dict of *actual* observed numbers."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    model = BaselineAIGCDetector(
        embed_dim=EMBED_DIM,
        num_patches=NUM_PATCHES,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        global_weight=cfg.global_weight,
        patch_weight=cfg.patch_weight,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    start_epoch = 0
    global_step = 0
    if resume_from:
        info = load_checkpoint(resume_from, model=model, optimizer=optimizer, map_location=cfg.device)
        start_epoch = info["epoch"]
        global_step = info["global_step"]
        print(f"[resume] loaded {resume_from} (epoch={start_epoch}, step={global_step})")

    tag = "SMOKE" if smoke else "train"
    n_params = sum(p.numel() for p in trainable)
    print(
        f"[{tag}] source={source_kind} device={device} "
        f"trainable_params={n_params} epochs={cfg.epochs} batch_size={cfg.batch_size}"
    )
    if n_batches_hint is not None:
        print(f"[{tag}] ~{n_batches_hint} batches/epoch")

    history: List[Dict[str, Any]] = []
    saved_paths: List[str] = []
    total_seen = 0
    total_dropped_label2 = 0
    t0 = time.time()

    for epoch in range(start_epoch, start_epoch + cfg.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_examples = 0
        step_in_epoch = 0

        # `batches_per_epoch` may be a DataLoader (re-iterable) or a generator
        # factory result. For generators the caller passes a callable; here we
        # accept either an iterable or a zero-arg callable returning one.
        iterator = batches_per_epoch() if callable(batches_per_epoch) else batches_per_epoch

        for batch in iterator:
            if cfg.max_steps_per_epoch is not None and step_in_epoch >= cfg.max_steps_per_epoch:
                break

            outputs, targets, n_dropped = _forward_batch(
                batch, source_kind, model, backbone, device
            )
            total_dropped_label2 += n_dropped
            if outputs is None:
                continue

            loss, parts = baseline_loss(outputs, targets, bce)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {global_step}: {parts}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = int(targets.shape[0])
            epoch_loss_sum += parts["loss"] * bs
            epoch_examples += bs
            total_seen += bs
            step_in_epoch += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                print(
                    f"[{tag}] epoch {epoch} step {global_step} "
                    f"loss={parts['loss']:.4f} "
                    f"(g={parts['loss_global']:.4f} p={parts['loss_patch']:.4f} "
                    f"f={parts['loss_final']:.4f}) n={bs}"
                )

        mean_loss = epoch_loss_sum / epoch_examples if epoch_examples else float("nan")
        elapsed = time.time() - t0
        print(
            f"[{tag}] epoch {epoch} done: mean_loss={mean_loss:.4f} "
            f"examples={epoch_examples} elapsed={elapsed:.1f}s"
        )
        history.append(
            {"epoch": epoch, "mean_loss": mean_loss, "examples": epoch_examples}
        )

        if cfg.save_every_epoch and not smoke:
            ckpt_path = Path(cfg.out_dir) / f"{cfg.run_name}_epoch{epoch}.pt"
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                config=cfg,
                extra={"epoch_history": history},
            )
            saved_paths.append(str(ckpt_path))
            print(f"[{tag}] saved checkpoint -> {ckpt_path}")

    # Always write a final checkpoint (including smoke, so loading is exercised).
    final_path = Path(cfg.out_dir) / f"{cfg.run_name}_final.pt"
    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        epoch=start_epoch + cfg.epochs,
        global_step=global_step,
        config=cfg,
        extra={"epoch_history": history, "smoke": smoke},
    )
    saved_paths.append(str(final_path))
    print(f"[{tag}] saved final checkpoint -> {final_path}")

    if total_dropped_label2:
        print(f"[{tag}] excluded {total_dropped_label2} label-2 rows from the AIGC loss")

    summary = {
        "epochs_run": cfg.epochs,
        "global_step": global_step,
        "examples_seen": total_seen,
        "label2_excluded": total_dropped_label2,
        "epoch_history": history,
        "checkpoints": saved_paths,
        "final_checkpoint": str(final_path),
        "smoke": smoke,
    }
    return summary


# ===========================================================================
# CLI
# ===========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the TraceLens-R baseline AIGC detector.")
    src = p.add_argument_group("data source (choose one; --smoke works with none)")
    src.add_argument("--cache-dir", type=str, default=None,
                     help="Directory of cached feature .pt files (see scripts/cache_features.py). "
                          "Must be a single group (clean or transformed) or used with --cache-subset.")
    src.add_argument("--cache-subset", type=str, default=None, choices=list(CACHE_GROUPS),
                     help="Required when --cache-dir contains both clean/ and transformed/ features.")
    src.add_argument("--manifest", type=str, default=None, help="Path to a manifest CSV.")
    src.add_argument("--dataset-root", type=str, default=None,
                     help="Root dir that manifest image_path entries resolve against.")
    src.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    src.add_argument("--backbone-name", type=str, default="facebook/dinov2-small")

    hp = p.add_argument_group("hyper-parameters")
    hp.add_argument("--seed", type=int, default=42)
    hp.add_argument("--epochs", type=int, default=3)
    hp.add_argument("--batch-size", type=int, default=32)
    hp.add_argument("--lr", type=float, default=1e-3)
    hp.add_argument("--weight-decay", type=float, default=0.0)
    hp.add_argument("--hidden-dim", type=int, default=128)
    hp.add_argument("--dropout", type=float, default=0.1)
    hp.add_argument("--global-weight", type=float, default=0.5)
    hp.add_argument("--patch-weight", type=float, default=0.5)
    hp.add_argument("--num-workers", type=int, default=0)
    hp.add_argument("--max-steps-per-epoch", type=int, default=None)
    hp.add_argument("--log-every", type=int, default=10)

    io = p.add_argument_group("io / misc")
    io.add_argument("--device", type=str, default="cpu")
    io.add_argument("--out-dir", type=str, default="checkpoints")
    io.add_argument("--run-name", type=str, default="baseline")
    io.add_argument("--resume-from", type=str, default=None)
    io.add_argument("--no-save-every-epoch", action="store_true")
    io.add_argument("--smoke", action="store_true",
                    help="Tiny run for plumbing checks. Uses random features if no data source given.")
    return p


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        global_weight=args.global_weight,
        patch_weight=args.patch_weight,
        num_workers=args.num_workers,
        max_steps_per_epoch=args.max_steps_per_epoch,
        log_every=args.log_every,
        device=args.device,
        out_dir=args.out_dir,
        run_name=args.run_name,
        save_every_epoch=not args.no_save_every_epoch,
    )
    if args.smoke:
        # Keep smoke genuinely small but still exercise >1 epoch + save/load.
        cfg.epochs = min(cfg.epochs, 2) or 1
        cfg.batch_size = min(cfg.batch_size, 8)
        cfg.max_steps_per_epoch = cfg.max_steps_per_epoch or 3
        cfg.log_every = 1
    return cfg


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_args(args)
    set_seed(cfg.seed)

    if args.cache_dir:
        source = build_feature_source(args.cache_dir, cfg, cache_subset=args.cache_subset)
        summary = train(
            cfg,
            source_kind="features",
            batches_per_epoch=source.loader,
            n_batches_hint=(source.size // cfg.batch_size + 1) if source.size else None,
            backbone=None,
            resume_from=args.resume_from,
            smoke=args.smoke,
        )
    elif args.manifest:
        if not args.dataset_root:
            raise SystemExit("--manifest also requires --dataset-root")
        source = build_image_source(
            args.manifest, args.dataset_root, args.split, cfg, args.backbone_name
        )
        summary = train(
            cfg,
            source_kind="images",
            batches_per_epoch=source.loader,
            n_batches_hint=(source.size // cfg.batch_size + 1) if source.size else None,
            backbone=source.backbone,
            resume_from=args.resume_from,
            smoke=args.smoke,
        )
    elif args.smoke:
        print("[SMOKE] No data source given -> using RANDOM features. "
              "Printed losses are real computations on noise, NOT model results.")
        summary = train(
            cfg,
            source_kind="synthetic",
            batches_per_epoch=lambda: synthetic_smoke_batches(cfg, n_batches=4, batch_size=cfg.batch_size),
            n_batches_hint=4,
            backbone=None,
            resume_from=args.resume_from,
            smoke=True,
        )
    else:
        raise SystemExit(
            "No data source. Pass --cache-dir, or --manifest + --dataset-root, "
            "or --smoke for a random-feature plumbing test."
        )

    # Exercise checkpoint loading for real, on what we just wrote.
    reloaded = load_checkpoint(summary["final_checkpoint"], map_location=cfg.device)
    print(
        f"[verify] reloaded final checkpoint OK "
        f"(epoch={reloaded['epoch']}, step={reloaded['global_step']})"
    )
    print(f"[done] {summary}")
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
