"""AIGC classification metrics from labelled evaluation records.

All numbers are computed from inputs. This module does not search for a
threshold and does not invent scores.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from typing import Any, Mapping, TextIO

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.schemas import EvaluationError

DEFAULT_THRESHOLD = 0.5


class MetricsError(EvaluationError):
    """Raised when metrics cannot be computed from the given inputs."""


@dataclass(frozen=True)
class ClassificationMetrics:
    sample_count: int
    real_count: int
    synthetic_count: int
    balanced_accuracy: float
    auroc: float | None
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    brier_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise MetricsError("threshold must be a finite number in [0, 1].")
    value = float(threshold)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise MetricsError("threshold must be a finite number in [0, 1].")
    return value


def compute_metrics(
    labels: Any,
    predictions: Any,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    warn_stream: TextIO | None = None,
) -> ClassificationMetrics:
    """Compute official-task AIGC metrics at a fixed threshold.

    Positive class is label 1 (fully synthetic). False positive: authentic
    predicted as AI-generated. False negative: fully synthetic predicted as
    authentic. ``zero_division=0``. AUROC is ``None`` when only one class is
    present. The threshold is not optimised.
    """
    threshold = validate_threshold(threshold)
    y_true = np.asarray(labels, dtype=float)
    y_score = np.asarray(predictions, dtype=float)
    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.size != y_score.size:
        raise MetricsError("labels and predictions must be 1-D arrays of equal length.")
    if y_true.size == 0:
        raise MetricsError("Cannot compute metrics on an empty prediction set.")
    if not np.all(np.isfinite(y_score)) or np.any(y_score < 0.0) or np.any(y_score > 1.0):
        raise MetricsError("predictions must be finite numbers in [0, 1].")
    if not np.all(np.isin(y_true, (0.0, 1.0))):
        raise MetricsError("labels must be 0 or 1.")

    y_true_int = y_true.astype(int)
    y_hat = (y_score >= threshold).astype(int)
    real_count = int(np.sum(y_true_int == 0))
    synthetic_count = int(np.sum(y_true_int == 1))

    true_positive = int(np.sum((y_true_int == 1) & (y_hat == 1)))
    false_positive = int(np.sum((y_true_int == 0) & (y_hat == 1)))
    false_negative = int(np.sum((y_true_int == 1) & (y_hat == 0)))
    true_negative = int(np.sum((y_true_int == 0) & (y_hat == 0)))

    fpr = _ratio(false_positive, false_positive + true_negative)
    fnr = _ratio(false_negative, false_negative + true_positive)

    return ClassificationMetrics(
        sample_count=int(y_true_int.size),
        real_count=real_count,
        synthetic_count=synthetic_count,
        balanced_accuracy=float(
            balanced_accuracy_score(y_true_int, y_hat)
        ),
        auroc=_auroc(y_true_int, y_score, warn_stream),
        precision=float(
            precision_score(y_true_int, y_hat, pos_label=1, zero_division=0)
        ),
        recall=float(recall_score(y_true_int, y_hat, pos_label=1, zero_division=0)),
        f1=float(f1_score(y_true_int, y_hat, pos_label=1, zero_division=0)),
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        brier_score=float(brier_score_loss(y_true_int, y_score)),
    )


def metrics_from_frame(
    frame: Mapping[str, Any] | Any,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    warn_stream: TextIO | None = None,
) -> ClassificationMetrics:
    return compute_metrics(
        frame["label"],
        frame["pred"],
        threshold,
        warn_stream=warn_stream,
    )


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _auroc(y_true: np.ndarray, y_score: np.ndarray, warn_stream: TextIO | None) -> float | None:
    if np.unique(y_true).size < 2:
        stream = sys.stderr if warn_stream is None else warn_stream
        print(
            "warning: AUROC is undefined because only one class is present; reporting null.",
            file=stream,
        )
        return None
    return float(roc_auc_score(y_true, y_score))
