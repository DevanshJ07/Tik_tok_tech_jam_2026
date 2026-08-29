#!/usr/bin/env python3
"""Compute AIGC evaluation metrics from labelled prediction files.

This CLI scores internal evaluation records only. Official inference JSON
(image_path and pred) is not sufficient. Mock-identified records are rejected
and must not be reported as model performance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.error_analysis import error_table
from src.evaluation.metrics import DEFAULT_THRESHOLD
from src.evaluation.robustness import RobustnessSummary, summarize_robustness
from src.evaluation.schemas import EvaluationError, load_evaluation_records

OUTPUT_FILES = ("summary.json", "by_condition.csv", "by_family.csv", "errors.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate labelled AIGC predictions. Not for official inference JSON "
            "and not for mock predictor output."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="CSV or JSON file of internal labelled evaluation records.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for metric artefacts.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Fixed decision threshold in [0, 1] (default {DEFAULT_THRESHOLD}).",
    )
    return parser


def write_evaluation_outputs(
    summary: RobustnessSummary,
    errors: pd.DataFrame,
    output_dir: Path,
    *,
    input_path: Path,
    threshold: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / name for name in OUTPUT_FILES}
    _reject_input_overwrite(input_path, paths.values())

    payload = {
        "input_filename": input_path.name,
        "threshold": threshold,
        **summary.to_summary_dict(),
    }
    paths["summary.json"].write_text(
        json.dumps(_json_safe(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([row.to_row() for row in summary.by_condition]).to_csv(
        paths["by_condition.csv"], index=False
    )
    pd.DataFrame([row.to_row() for row in summary.by_family]).to_csv(
        paths["by_family.csv"], index=False
    )
    errors.to_csv(paths["errors.csv"], index=False)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = Path(args.predictions)
    output_dir = Path(args.output_dir)

    try:
        frame = load_evaluation_records(input_path)
        summary = summarize_robustness(frame, args.threshold, warn_stream=sys.stderr)
        errors = error_table(frame, args.threshold)
        paths = write_evaluation_outputs(
            summary,
            errors,
            output_dir,
            input_path=input_path,
            threshold=args.threshold,
        )
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for path in paths.values():
        print(f"Wrote {path}")
    return 0


def _reject_input_overwrite(input_path: Path, output_paths: Any) -> None:
    resolved_input = input_path.resolve()
    for output_path in output_paths:
        if Path(output_path).resolve() == resolved_input:
            raise EvaluationError("Refusing to overwrite the input prediction file.")


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _json_safe(value.item())
    except ImportError:
        pass
    return value


if __name__ == "__main__":
    sys.exit(main())
