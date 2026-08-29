#!/usr/bin/env python3
"""Write official AIGC JSON for every image under a directory.

Until the real model is wired, this script refuses to run without ``--mock``.
``--mock`` selects the testing-only MockPredictor. There is no silent fallback.
Mock results are not model predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.inference.predictor import MockPredictor, emit_mock_warning, run_directory_inference

REAL_MODEL_UNAVAILABLE = (
    "The real TraceLens-R model is not available yet. "
    "Refusing to invent predictions. Re-run with --mock for testing-only "
    "MockPredictor output (not model predictions)."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Directory-to-JSON AIGC inference. Official records contain "
            "exactly image_path and pred."
        )
    )
    parser.add_argument("--input_dir", required=True, help="Directory of images (searched recursively).")
    parser.add_argument("--output_json", required=True, help="Destination official JSON list.")
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="TESTING ONLY. Use MockPredictor. Never implied; must be passed explicitly.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.mock:
        print(REAL_MODEL_UNAVAILABLE, file=sys.stderr)
        return 2

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"error: input path is not a directory: {input_dir}", file=sys.stderr)
        return 2

    load_config()
    emit_mock_warning()
    run_directory_inference(input_dir, Path(args.output_json), MockPredictor())
    return 0


if __name__ == "__main__":
    sys.exit(main())
