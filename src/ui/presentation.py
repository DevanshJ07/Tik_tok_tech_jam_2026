"""Pure presentation helpers for the TraceLens-R screening UI.

These functions format screening results. They do not claim legal authenticity.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.inference.contracts import PredictionResult

NOT_AVAILABLE = "Not available"
HEATMAP_UNAVAILABLE = "Manipulation localisation is not available"

CATEGORY_FULLY_AI = "Fully AI-generated"
CATEGORY_LOCAL = "Locally manipulated"
CATEGORY_AUTHENTIC = "Authentic"

SCREENING_DISCLAIMER = (
    "Screening result only. Model indication — review recommended. "
    "Not proof of authenticity."
)

MOCK_BANNER = (
    "TESTING ONLY. MockPredictor is enabled. These numbers are not model "
    "predictions and must not be described as detection performance."
)


class PresentationError(ValueError):
    """Raised when a display helper receives an invalid value."""


def validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise PresentationError("threshold must be a finite number in [0, 1].")
    value = float(threshold)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise PresentationError("threshold must be a finite number in [0, 1].")
    return value


def as_percentage(probability: float, *, digits: int = 1) -> str:
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise PresentationError("probability must be a finite number in [0, 1].")
    value = float(probability)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise PresentationError("probability must be a finite number in [0, 1].")
    return f"{value * 100:.{digits}f}%"


def format_optional_probability(value: float | None) -> str:
    if value is None:
        return NOT_AVAILABLE
    return as_percentage(value)


def provisional_category(
    aigc_probability: float,
    manipulation_probability: float | None,
    threshold: float,
) -> str:
    """Decide the displayed screening category.

    Order: fully AI-generated if AIGC >= threshold; else locally manipulated
    if a manipulation score exists and is >= threshold; else authentic.
    Authentic is a model indication, not a legal finding.
    """
    threshold = validate_threshold(threshold)
    if aigc_probability >= threshold:
        return CATEGORY_FULLY_AI
    if manipulation_probability is not None and manipulation_probability >= threshold:
        return CATEGORY_LOCAL
    return CATEGORY_AUTHENTIC


def category_explanation(category: str) -> str:
    if category == CATEGORY_FULLY_AI:
        return (
            "Model indication: fully AI-generated (screening result). "
            "Review recommended. Not proof of authenticity."
        )
    if category == CATEGORY_LOCAL:
        return (
            "Model indication: locally manipulated (screening result). "
            "Review recommended. Not proof of authenticity."
        )
    return (
        "Model indication: authentic (screening result). "
        "Review recommended. Not proof of authenticity."
    )


def reliability_wording(score: float | None) -> str:
    if score is None:
        return (
            f"{NOT_AVAILABLE}. Reliability is a screening indicator only, "
            "not a certainty score."
        )
    return (
        f"Screening reliability indicator {as_percentage(score)}. "
        "Review recommended. Not proof of authenticity."
    )


def heatmap_status(heatmap_path: str | None) -> tuple[bool, str]:
    """Return (should_display, message_or_path). Never invents a heatmap."""
    if heatmap_path is None or not str(heatmap_path).strip():
        return False, HEATMAP_UNAVAILABLE
    path = Path(heatmap_path)
    if not path.is_file():
        return False, HEATMAP_UNAVAILABLE
    return True, str(path)


def capability_status(*, mock: bool, result: PredictionResult | None = None) -> dict[str, str]:
    """Report which model capabilities are connected.

    Mock mode labels every model capability as unavailable/testing-only even if
    the mock stub filled optional numeric fields.
    """
    if mock:
        testing = "unavailable / testing-only"
        return {
            "aigc_predictor": testing,
            "reliability": testing,
            "manipulation": testing,
            "heatmap": testing,
        }
    heatmap_ok = False
    if result is not None:
        heatmap_ok, _ = heatmap_status(result.heatmap_path)
    return {
        "aigc_predictor": "not connected",
        "reliability": "not connected",
        "manipulation": "not connected",
        "heatmap": "connected" if heatmap_ok else "not connected",
    }


def result_view(
    result: PredictionResult,
    threshold: float,
    *,
    mock: bool,
) -> dict[str, Any]:
    category = provisional_category(
        result.aigc_probability,
        result.manipulation_probability,
        threshold,
    )
    show_heatmap, heatmap_detail = heatmap_status(result.heatmap_path)
    return {
        "aigc_probability_label": as_percentage(result.aigc_probability),
        "manipulation_probability_label": format_optional_probability(
            result.manipulation_probability
        ),
        "reliability_score_label": format_optional_probability(result.reliability_score),
        "reliability_wording": reliability_wording(result.reliability_score),
        "category": category,
        "category_explanation": category_explanation(category),
        "disclaimer": SCREENING_DISCLAIMER,
        "heatmap_available": show_heatmap,
        "heatmap_detail": heatmap_detail,
        "capabilities": capability_status(mock=mock, result=result),
        "testing_banner": MOCK_BANNER if mock else None,
    }
