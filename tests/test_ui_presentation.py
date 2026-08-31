from __future__ import annotations

from pathlib import Path

import pytest

from src.inference.contracts import PredictionResult
from src.ui.presentation import (
    CATEGORY_AUTHENTIC,
    CATEGORY_FULLY_AI,
    CATEGORY_LOCAL,
    HEATMAP_UNAVAILABLE,
    NOT_AVAILABLE,
    PresentationError,
    as_percentage,
    capability_status,
    format_optional_probability,
    heatmap_status,
    provisional_category,
    reliability_wording,
    validate_threshold,
)


def test_probability_formatting() -> None:
    assert as_percentage(0.5) == "50.0%"
    assert as_percentage(0.1234, digits=1) == "12.3%"


def test_category_decision_order() -> None:
    assert provisional_category(0.8, 0.99, 0.5) == CATEGORY_FULLY_AI
    assert provisional_category(0.2, 0.8, 0.5) == CATEGORY_LOCAL
    assert provisional_category(0.2, 0.2, 0.5) == CATEGORY_AUTHENTIC
    assert provisional_category(0.2, None, 0.5) == CATEGORY_AUTHENTIC


def test_missing_optional_values() -> None:
    assert format_optional_probability(None) == NOT_AVAILABLE
    assert NOT_AVAILABLE in reliability_wording(None)
    available, message = heatmap_status(None)
    assert available is False
    assert message == HEATMAP_UNAVAILABLE


def test_heatmap_requires_existing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.png"
    available, message = heatmap_status(str(missing))
    assert available is False
    assert message == HEATMAP_UNAVAILABLE
    real = tmp_path / "map.png"
    real.write_bytes(b"x")
    available, path = heatmap_status(str(real))
    assert available is True
    assert path == str(real)


def test_invalid_threshold_rejection() -> None:
    with pytest.raises(PresentationError, match="threshold"):
        validate_threshold(1.5)
    with pytest.raises(PresentationError, match="threshold"):
        validate_threshold(-0.01)
    with pytest.raises(PresentationError):
        provisional_category(0.2, None, 2.0)


def test_mock_capabilities_are_testing_only() -> None:
    result = PredictionResult(
        image_path="a.png",
        aigc_probability=0.4,
        manipulation_probability=0.7,
        reliability_score=0.3,
    )
    status = capability_status(mock=True, result=result)
    assert all("testing-only" in value for value in status.values())
    baseline = PredictionResult(image_path="a.png", aigc_probability=0.4)
    real_status = capability_status(mock=False, result=baseline)
    assert real_status["aigc_predictor"] == "connected (baseline)"
    assert real_status["reliability"] == "awaiting Member 3 model"
    assert real_status["manipulation"] == "awaiting Member 4 model"
    assert real_status["heatmap"] == "awaiting Member 4 model"


def test_capabilities_when_manipulation_connected(tmp_path: Path) -> None:
    heat = tmp_path / "map.png"
    heat.write_bytes(b"png")
    result = PredictionResult(
        image_path="a.png",
        aigc_probability=0.2,
        manipulation_probability=0.8,
        heatmap_path=str(heat),
    )
    status = capability_status(mock=False, result=result)
    assert status["aigc_predictor"] == "connected (baseline)"
    assert status["reliability"] == "awaiting Member 3 model"
    assert status["manipulation"] == "connected"
    assert status["heatmap"] == "connected"
