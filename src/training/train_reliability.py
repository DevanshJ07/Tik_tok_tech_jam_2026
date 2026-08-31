"""Member 3 reliability training for TraceLens-R."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.cache_features import load_cached_feature
from src.models.baseline import BaselineAIGCDetector
from src.models.reliability import (
    TraceLensReliability,
    compute_survival_target,
    survival_loss,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PairedFeatureDataset(Dataset):
    """Clean/transformed AIGC pairs linked by image_id."""

    def __init__(self, cache_root: str, split: str):
        root = Path(cache_root)
        clean_dir = root / "clean" / split
        transformed_dir = root / "transformed" / split

        if not clean_dir.exists():
            raise FileNotFoundError(clean_dir)
        if not transformed_dir.exists():
            raise FileNotFoundError(transformed_dir)

        pairs = []

        for path in sorted(transformed_dir.glob("*.pt")):
            clean_path = clean_dir / path.name
            if not clean_path.exists():
                continue

            clean = load_cached_feature(clean_path)
            transformed = load_cached_feature(path)

            label = int(clean["label"])
            if label not in (0, 1):
                continue

            if int(transformed["label"]) != label:
                raise ValueError(f"Label mismatch for {path.name}")

            pairs.append((clean_path, path))

        if not pairs:
            raise RuntimeError(f"No paired features found in {split}")

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_path, transformed_path = self.pairs[idx]
        clean = load_cached_feature(clean_path)
        transformed = load_cached_feature(transformed_path)

        return {
            "image_id": clean["image_id"],
            "label": int(clean["label"]),
            "clean_cls": torch.as_tensor(clean["cls_features"]).float(),
            "clean_patch": torch.as_tensor(clean["patch_features"]).float(),
            "degraded_cls": torch.as_tensor(transformed["cls_features"]).float(),
            "degraded_patch": torch.as_tensor(transformed["patch_features"]).float(),
        }


def collate(batch):
    return {
        "image_id": [x["image_id"] for x in batch],
        "label": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        "clean_cls": torch.stack([x["clean_cls"] for x in batch]),
        "clean_patch": torch.stack([x["clean_patch"] for x in batch]),
        "degraded_cls": torch.stack([x["degraded_cls"] for x in batch]),
        "degraded_patch": torch.stack([x["degraded_patch"] for x in batch]),
    }


def load_baseline(path: str, device: torch.device):
    """Load frozen Member 2 baseline heads."""
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Baseline checkpoint must contain 'model_state_dict'")

    hparams = checkpoint.get("model_hparams", {})

    model = BaselineAIGCDetector(**hparams)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


@torch.no_grad()
def baseline_outputs(model, cls_features, patch_features):
    return model(cls_features, patch_features)


def run_epoch(
    reliability,
    baseline,
    loader,
    optimizer,
    device,
    training: bool,
):
    reliability.train(training)

    total_loss = 0.0
    total_survival = 0.0
    total_cls = 0.0
    total_correct = 0
    total_n = 0

    for batch in loader:
        label = batch["label"].to(device)

        clean_patch = batch["clean_patch"].to(device)
        clean_cls = batch["clean_cls"].to(device)
        degraded_patch = batch["degraded_patch"].to(device)
        degraded_cls = batch["degraded_cls"].to(device)

        # Baseline is frozen. Its original aigc_probability is never modified.
        with torch.no_grad():
            clean_base = baseline_outputs(
                baseline, clean_cls, clean_patch
            )
            degraded_base = baseline_outputs(
                baseline, degraded_cls, degraded_patch
            )

            survival_target, target_weight = compute_survival_target(
                clean_patch_logits=clean_base["patch_logits"],
                degraded_patch_logits=degraded_base["patch_logits"],
                labels=label,
            )

        if training:
            optimizer.zero_grad(set_to_none=True)

        output = reliability(
            patch_features=degraded_patch,
            patch_logits=degraded_base["patch_logits"].detach(),
            global_logit=degraded_base["global_logit"].detach(),
        )

        loss_survival = survival_loss(
            predicted_reliability=output["reliability"],
            survival_target=survival_target,
            target_weight=target_weight,
        )

        # Train the fused AIGC prediction as well.
        cls_loss = F.binary_cross_entropy_with_logits(
            output["final_logit"],
            label.float(),
        )

        loss = loss_survival + cls_loss

        if training:
            loss.backward()
            optimizer.step()

        batch_n = label.numel()
        predictions = (output["aigc_probability"] >= 0.5).long()

        total_loss += loss.item() * batch_n
        total_survival += loss_survival.item() * batch_n
        total_cls += cls_loss.item() * batch_n
        total_correct += (predictions == label).sum().item()
        total_n += batch_n

    return {
        "loss": total_loss / total_n,
        "survival_loss": total_survival / total_n,
        "classification_loss": total_cls / total_n,
        "accuracy": total_correct / total_n,
    }


@torch.no_grad()
def save_validation_predictions(
    reliability,
    baseline,
    loader,
    device,
    path: Path,
):
    reliability.eval()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "label",
                "baseline_aigc_probability",
                "reliability_aigc_probability",
                "mean_reliability",
            ],
        )
        writer.writeheader()

        for batch in loader:
            label = batch["label"].to(device)
            patch = batch["degraded_patch"].to(device)
            cls = batch["degraded_cls"].to(device)

            base = baseline_outputs(baseline, cls, patch)

            out = reliability(
                patch_features=patch,
                patch_logits=base["patch_logits"],
                global_logit=base["global_logit"],
            )

            for i, image_id in enumerate(batch["image_id"]):
                writer.writerow(
                    {
                        "image_id": image_id,
                        "label": int(label[i].item()),
                        "baseline_aigc_probability": float(
                            base["aigc_probability"][i].item()
                        ),
                        "reliability_aigc_probability": float(
                            out["aigc_probability"][i].item()
                        ),
                        "mean_reliability": float(
                            out["mean_reliability"][i].item()
                        ),
                    }
                )


def main():
    parser = argparse.ArgumentParser(
        description="TraceLens-R Member 3 reliability trainer"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/baseline_real_final.pt",
    )
    parser.add_argument(
        "--cache-root",
        default=r"E:\TikTok_cache\features",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output",
        default="checkpoints/reliability_final.pt",
    )

    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    print(f"[reliability] device={device}")
    print(f"[reliability] cache_root={args.cache_root}")

    baseline = load_baseline(args.checkpoint, device)

    train_ds = PairedFeatureDataset(args.cache_root, "train")
    val_ds = PairedFeatureDataset(args.cache_root, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    reliability = TraceLensReliability(
        embed_dim=384,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        reliability.parameters(),
        lr=args.lr,
    )

    best_val = float("inf")
    best_state = None
    history = []

    print(
        f"[reliability] train_pairs={len(train_ds)} "
        f"val_pairs={len(val_ds)}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            reliability,
            baseline,
            train_loader,
            optimizer,
            device,
            training=True,
        )

        val_metrics = run_epoch(
            reliability,
            baseline,
            val_loader,
            optimizer=None,
            device=device,
            training=False,
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in reliability.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No best reliability checkpoint was selected")

    reliability.load_state_dict(best_state)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": reliability.cpu(),
            "config": vars(args),
            "best_val_loss": best_val,
            "history": history,
        },
        output_path,
    )

    save_validation_predictions(
        reliability.to(device),
        baseline,
        val_loader,
        device,
        Path("outputs/reliability_val_predictions.csv"),
    )

    Path("outputs/reliability_training_log.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print(f"[reliability] saved: {output_path}")
    print(f"[reliability] best_val_loss={best_val:.6f}")
    print("[reliability] test set was NOT used")


if __name__ == "__main__":
    main()