"""False-positive and false-negative tables for AIGC evaluation."""

from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import DEFAULT_THRESHOLD, validate_threshold

ERROR_COLUMNS = (
    "image_path",
    "true_label",
    "prediction",
    "predicted_label",
    "error_type",
    "model_name",
    "dataset",
    "transform_name",
    "severity",
    "confidence_distance_from_threshold",
)


def error_table(
    frame: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Return FP and FN rows sorted by greatest confidence distance first.

    False positive: authentic (0) predicted as AI-generated.
    False negative: fully synthetic (1) predicted as authentic.
    """
    threshold = validate_threshold(threshold)
    if frame.empty:
        return pd.DataFrame(columns=list(ERROR_COLUMNS))

    working = frame.copy()
    working["true_label"] = working["label"].astype(int)
    working["prediction"] = working["pred"].astype(float)
    working["predicted_label"] = (working["prediction"] >= threshold).astype(int)
    working["confidence_distance_from_threshold"] = (
        working["prediction"] - threshold
    ).abs()

    false_positive = (working["true_label"] == 0) & (working["predicted_label"] == 1)
    false_negative = (working["true_label"] == 1) & (working["predicted_label"] == 0)
    errors = working.loc[false_positive | false_negative].copy()
    errors["error_type"] = "false_negative"
    errors.loc[errors["true_label"] == 0, "error_type"] = "false_positive"

    table = errors.reindex(columns=list(ERROR_COLUMNS))
    return table.sort_values(
        by=["confidence_distance_from_threshold", "image_path"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
