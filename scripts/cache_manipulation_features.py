#!/usr/bin/env python3
"""Cache frozen DINOv2 patch features + aligned masks for manipulation training.

Stores labels 0/2 only, under a path separate from AIGC caches. Test is
refused so it cannot leak into training or model selection.

Example::

    python scripts/cache_manipulation_features.py \\
        --manifest data/manifests/manifest_operational_1000.csv \\
        --dataset-root data/raw/sid_set_operational_1000 \\
        --out-dir data/cache/manipulation \\
        --splits train val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.backbone import DINOv2Backbone  # noqa: E402
from src.training.manipulation_cache import cache_manipulation_split  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache frozen DINOv2 patch features and masks for manipulation training."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", default="data/cache/manipulation")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Train and/or val by default. Test requires --allow-test after model selection.",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Permit caching the test split after checkpoint selection. Never used for training.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone-name", default="facebook/dinov2-small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if "test" in args.splits and not args.allow_test:
        raise SystemExit(
            "Refusing to cache the test split. Pass --allow-test only after "
            "model selection, and never use test caches for training."
        )
    backbone = DINOv2Backbone(model_name=args.backbone_name, device=args.device)
    print(
        f"[manip-cache] backbone={args.backbone_name} device={args.device} "
        f"frozen={backbone.is_frozen}",
        flush=True,
    )
    results: Dict[str, Any] = {}
    for split in args.splits:
        out_dir = Path(args.out_dir) / "clean" / split
        print(f"[manip-cache] === split={split} -> {out_dir} ===", flush=True)
        results[split] = cache_manipulation_split(
            manifest=args.manifest,
            dataset_root=args.dataset_root,
            split=split,
            out_dir=out_dir,
            backbone=backbone,
            seed=args.seed,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            allow_test=args.allow_test,
        )
    print(f"[manip-cache] summary: {results}", flush=True)
    return results


if __name__ == "__main__":
    main()
