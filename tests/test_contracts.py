from __future__ import annotations

import math
from pathlib import Path

import pytest
from PIL import Image

from src.inference.contracts import PredictionContractError, PredictionResult
from src.inference.predictor import MockPredictor


def test_invalid_probability_is_rejected() -> None:
    with pytest.raises(PredictionContractError):
        PredictionResult(image_path="a.png", aigc_probability=-0.01)
    with pytest.raises(PredictionContractError):
        PredictionResult(image_path="a.png", aigc_probability=1.01)
    with pytest.raises(PredictionContractError):
        PredictionResult(image_path="a.png", aigc_probability=math.nan)
    with pytest.raises(PredictionContractError):
        PredictionResult(image_path="a.png", aigc_probability=math.inf)
    with pytest.raises(PredictionContractError):
        PredictionResult(
            image_path="a.png",
            aigc_probability=0.2,
            manipulation_probability=-0.5,
        )
    with pytest.raises(PredictionContractError):
        PredictionResult(
            image_path="a.png",
            aigc_probability=0.2,
            reliability_score=2.0,
        )


def test_official_record_contains_exactly_image_path_and_pred() -> None:
    result = PredictionResult(
        image_path="nested/sample.png",
        aigc_probability=0.25,
        manipulation_probability=0.8,
        reliability_score=0.4,
        heatmap_path="outputs/heat.png",
    )
    official = result.to_official_record()
    assert official == {"image_path": "nested/sample.png", "pred": 0.25}
    assert set(official.keys()) == {"image_path", "pred"}

    detailed = result.to_detailed_record()
    assert detailed["manipulation_probability"] == 0.8
    assert detailed["reliability_score"] == 0.4
    assert detailed["heatmap_path"] == "outputs/heat.png"


def test_mock_predictions_are_deterministic(tmp_path: Path) -> None:
    image_path = tmp_path / "tiny.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image_path)

    predictor = MockPredictor()
    first = predictor.predict(image_path)
    second = predictor.predict(image_path)
    assert first.aigc_probability == second.aigc_probability
    assert first.manipulation_probability == second.manipulation_probability
    assert first.reliability_score == second.reliability_score
    assert 0.0 <= first.aigc_probability <= 1.0
