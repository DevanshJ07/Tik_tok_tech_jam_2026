"""Shared prediction-result contract for TraceLens-R inference.

Official JSON may contain only ``image_path`` and ``pred`` (SPEC §11).
Optional manipulation and reliability fields belong in detailed records only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


class PredictionContractError(ValueError):
    """Raised when a prediction result violates the shared contract."""


@dataclass
class PredictionResult:
    image_path: str
    aigc_probability: float
    manipulation_probability: float | None = None
    reliability_score: float | None = None
    heatmap_path: str | None = None

    def __post_init__(self) -> None:
        self.aigc_probability = _validated_probability(
            "aigc_probability", self.aigc_probability
        )
        if self.manipulation_probability is not None:
            self.manipulation_probability = _validated_probability(
                "manipulation_probability", self.manipulation_probability
            )
        if self.reliability_score is not None:
            self.reliability_score = _validated_probability(
                "reliability_score", self.reliability_score
            )

    def to_official_record(self) -> dict[str, str | float]:
        """Return the official AIGC JSON object: exactly image_path and pred."""
        return {
            "image_path": self.image_path,
            "pred": self.aigc_probability,
        }

    def to_detailed_record(self) -> dict[str, Any]:
        """Return all available fields, omitting unset optional values."""
        record: dict[str, Any] = {
            "image_path": self.image_path,
            "aigc_probability": self.aigc_probability,
        }
        if self.manipulation_probability is not None:
            record["manipulation_probability"] = self.manipulation_probability
        if self.reliability_score is not None:
            record["reliability_score"] = self.reliability_score
        if self.heatmap_path is not None:
            record["heatmap_path"] = self.heatmap_path
        return record


def official_record_keys(record: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(record.keys())


def _validated_probability(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionContractError(f"{name} must be a finite number in [0, 1].")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise PredictionContractError(f"{name} must be a finite number in [0, 1].")
    return number
