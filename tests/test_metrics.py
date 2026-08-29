from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import MetricsError, compute_metrics


def test_perfect_predictions() -> None:
    metrics = compute_metrics([0, 1], [0.1, 0.9], threshold=0.5)
    assert metrics.sample_count == 2
    assert metrics.real_count == 1
    assert metrics.synthetic_count == 1
    assert metrics.balanced_accuracy == pytest.approx(1.0)
    assert metrics.auroc == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.false_positive_rate == pytest.approx(0.0)
    assert metrics.false_negative_rate == pytest.approx(0.0)
    assert metrics.brier_score == pytest.approx(0.01)


def test_known_false_positive_and_false_negative() -> None:
    # TN 0.1, FP 0.9, FN 0.1, TP 0.9
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.9, 0.1, 0.9], threshold=0.5)
    assert metrics.false_positive_rate == pytest.approx(0.5)
    assert metrics.false_negative_rate == pytest.approx(0.5)
    assert metrics.balanced_accuracy == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.brier_score == pytest.approx(0.41)


def test_brier_score_hand_calculation() -> None:
    metrics = compute_metrics([0, 1], [0.25, 0.75], threshold=0.5)
    expected = ((0.25 - 0) ** 2 + (0.75 - 1) ** 2) / 2
    assert metrics.brier_score == pytest.approx(expected)


def test_one_class_auroc_is_null(capsys: pytest.CaptureFixture[str]) -> None:
    metrics = compute_metrics([0, 0], [0.1, 0.2], threshold=0.5)
    assert metrics.auroc is None
    assert "AUROC" in capsys.readouterr().err


def test_invalid_probabilities_rejected_by_metrics() -> None:
    with pytest.raises(MetricsError, match="predictions"):
        compute_metrics([0, 1], [0.1, 1.5], threshold=0.5)
    with pytest.raises(MetricsError, match="predictions"):
        compute_metrics([0, 1], [0.1, math.nan], threshold=0.5)


def test_threshold_must_be_in_unit_interval() -> None:
    with pytest.raises(MetricsError, match="threshold"):
        compute_metrics([0, 1], [0.1, 0.9], threshold=1.5)
    with pytest.raises(MetricsError, match="threshold"):
        compute_metrics([0, 1], [0.1, 0.9], threshold=-0.1)


def test_zero_division_precision() -> None:
    metrics = compute_metrics([0, 0], [0.1, 0.2], threshold=0.5)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.false_positive_rate == 0.0
