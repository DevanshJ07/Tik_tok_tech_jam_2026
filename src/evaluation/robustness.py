"""Robustness summaries over clean and transformed evaluation records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TextIO

import pandas as pd

from src.evaluation.metrics import (
    DEFAULT_THRESHOLD,
    ClassificationMetrics,
    metrics_from_frame,
    validate_threshold,
)
from src.evaluation.schemas import EvaluationError


class RobustnessError(EvaluationError):
    """Raised when a robustness summary cannot be computed."""


@dataclass(frozen=True)
class ConditionMetrics:
    transform_name: str
    severity: str
    metrics: ClassificationMetrics

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "transform_name": self.transform_name,
            "severity": self.severity,
        }
        row.update(self.metrics.to_dict())
        return row


@dataclass(frozen=True)
class FamilyMetrics:
    transform_name: str
    metrics: ClassificationMetrics
    severity_count: int

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "transform_name": self.transform_name,
            "severity_count": self.severity_count,
        }
        row.update(self.metrics.to_dict())
        return row


@dataclass(frozen=True)
class RobustnessSummary:
    overall: ClassificationMetrics
    clean: ClassificationMetrics
    transformed_combined: ClassificationMetrics
    by_condition: list[ConditionMetrics]
    by_family: list[FamilyMetrics]
    clean_balanced_accuracy: float
    mean_transformed_balanced_accuracy: float
    robustness_drop: float
    worst_condition: ConditionMetrics
    worst_condition_balanced_accuracy: float

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "clean_balanced_accuracy": self.clean_balanced_accuracy,
            "mean_transformed_balanced_accuracy": self.mean_transformed_balanced_accuracy,
            "robustness_drop": self.robustness_drop,
            "worst_condition": {
                "transform_name": self.worst_condition.transform_name,
                "severity": self.worst_condition.severity,
            },
            "worst_condition_balanced_accuracy": self.worst_condition_balanced_accuracy,
            "overall": self.overall.to_dict(),
            "clean": self.clean.to_dict(),
            "transformed_combined": self.transformed_combined.to_dict(),
        }


def summarize_robustness(
    frame: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    warn_stream: TextIO | None = None,
) -> RobustnessSummary:
    """Compute clean, per-condition, family-average, and pooled transformed metrics.

    Formulas (threshold is fixed; no search):

    - Per-condition balanced accuracy: metrics on that ``(transform_name, severity)``
      group only.
    - ``mean_transformed_balanced_accuracy``: arithmetic mean of those per-condition
      balanced accuracies for ``transform_name != "clean"``. Each condition has
      weight ``1 / N_conditions``. Records are not pooled and sample counts do
      not reweight conditions.
    - ``transformed_combined``: metrics on all non-clean records pooled together.
      Kept separate from the macro condition mean.
    - Family row in ``by_family``: arithmetic mean of that family's per-severity
      metric dicts (equal weight per severity, not pooled across severities).
    - ``robustness_drop`` = ``clean_balanced_accuracy - mean_transformed_balanced_accuracy``
      (signed; not clipped).
    - ``worst_condition``: individual ``(transform_name, severity)`` group with
      the lowest balanced accuracy among transformed conditions, never a family
      average and never the pooled transformed set.
    """
    threshold = validate_threshold(threshold)
    if frame.empty:
        raise RobustnessError("Cannot summarise robustness on an empty table.")

    overall = metrics_from_frame(frame, threshold, warn_stream=warn_stream)
    clean_frame = frame.loc[frame["transform_name"] == "clean"]
    transformed_frame = frame.loc[frame["transform_name"] != "clean"]
    if clean_frame.empty:
        raise RobustnessError("No clean records; cannot compute robustness drop.")
    if transformed_frame.empty:
        raise RobustnessError("No transformed records; cannot compute robustness drop.")

    clean = metrics_from_frame(clean_frame, threshold, warn_stream=warn_stream)
    transformed_combined = metrics_from_frame(
        transformed_frame, threshold, warn_stream=warn_stream
    )
    by_condition = _metrics_by_condition(frame, threshold, warn_stream)
    by_family = _metrics_by_family(by_condition)
    transformed_conditions = [
        condition
        for condition in by_condition
        if condition.transform_name != "clean"
    ]
    if not transformed_conditions:
        raise RobustnessError("No transformed conditions; cannot compute robustness drop.")

    mean_transformed = _macro_mean_balanced_accuracy(transformed_conditions)
    worst = min(transformed_conditions, key=lambda item: item.metrics.balanced_accuracy)
    drop = clean.balanced_accuracy - mean_transformed

    return RobustnessSummary(
        overall=overall,
        clean=clean,
        transformed_combined=transformed_combined,
        by_condition=by_condition,
        by_family=by_family,
        clean_balanced_accuracy=clean.balanced_accuracy,
        mean_transformed_balanced_accuracy=mean_transformed,
        robustness_drop=drop,
        worst_condition=worst,
        worst_condition_balanced_accuracy=worst.metrics.balanced_accuracy,
    )


def _metrics_by_condition(
    frame: pd.DataFrame,
    threshold: float,
    warn_stream: TextIO | None,
) -> list[ConditionMetrics]:
    grouped = frame.groupby(["transform_name", "severity"], dropna=False, sort=True)
    conditions: list[ConditionMetrics] = []
    for (transform_name, severity), group in grouped:
        conditions.append(
            ConditionMetrics(
                transform_name=str(transform_name),
                severity=str(severity),
                metrics=metrics_from_frame(group, threshold, warn_stream=warn_stream),
            )
        )
    return conditions


def _macro_mean_balanced_accuracy(conditions: list[ConditionMetrics]) -> float:
    """Equal-weight arithmetic mean of per-condition balanced accuracies."""
    if not conditions:
        raise RobustnessError("No conditions available for a macro mean.")
    return float(
        sum(condition.metrics.balanced_accuracy for condition in conditions)
        / len(conditions)
    )


def _metrics_by_family(conditions: list[ConditionMetrics]) -> list[FamilyMetrics]:
    """Average each family's per-severity metrics with equal weight per severity."""
    families: dict[str, list[ClassificationMetrics]] = {}
    for condition in conditions:
        families.setdefault(condition.transform_name, []).append(condition.metrics)
    rows: list[FamilyMetrics] = []
    for transform_name in sorted(families):
        members = families[transform_name]
        rows.append(
            FamilyMetrics(
                transform_name=transform_name,
                severity_count=len(members),
                metrics=_mean_metrics(members),
            )
        )
    return rows


def _mean_metrics(members: list[ClassificationMetrics]) -> ClassificationMetrics:
    aurocs = [item.auroc for item in members if item.auroc is not None]
    return ClassificationMetrics(
        sample_count=int(round(_mean([item.sample_count for item in members]))),
        real_count=int(round(_mean([item.real_count for item in members]))),
        synthetic_count=int(round(_mean([item.synthetic_count for item in members]))),
        balanced_accuracy=_mean([item.balanced_accuracy for item in members]),
        auroc=_mean(aurocs) if aurocs else None,
        precision=_mean([item.precision for item in members]),
        recall=_mean([item.recall for item in members]),
        f1=_mean([item.f1 for item in members]),
        false_positive_rate=_mean([item.false_positive_rate for item in members]),
        false_negative_rate=_mean([item.false_negative_rate for item in members]),
        brier_score=_mean([item.brier_score for item in members]),
    )


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))
