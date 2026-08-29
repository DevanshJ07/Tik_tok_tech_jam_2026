from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.error_analysis import error_table


def test_error_sorting_and_known_fp_fn() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_path": "tn.png",
                "label": 0,
                "pred": 0.1,
                "model_name": "baseline",
                "dataset": "unit",
                "split": "test",
                "transform_name": "clean",
                "severity": "none",
            },
            {
                "image_path": "fp.png",
                "label": 0,
                "pred": 0.95,
                "model_name": "baseline",
                "dataset": "unit",
                "split": "test",
                "transform_name": "jpeg",
                "severity": "30",
            },
            {
                "image_path": "fn.png",
                "label": 1,
                "pred": 0.2,
                "model_name": "baseline",
                "dataset": "unit",
                "split": "test",
                "transform_name": "clean",
                "severity": "none",
            },
            {
                "image_path": "tp.png",
                "label": 1,
                "pred": 0.8,
                "model_name": "baseline",
                "dataset": "unit",
                "split": "test",
                "transform_name": "clean",
                "severity": "none",
            },
        ]
    )
    errors = error_table(frame, threshold=0.5)
    assert list(errors["error_type"]) == ["false_positive", "false_negative"]
    assert list(errors["image_path"]) == ["fp.png", "fn.png"]
    assert errors.iloc[0]["true_label"] == 0
    assert errors.iloc[0]["predicted_label"] == 1
    assert errors.iloc[0]["prediction"] == 0.95
    assert errors.iloc[0]["confidence_distance_from_threshold"] == pytest.approx(0.45)
    assert errors.iloc[1]["true_label"] == 1
    assert errors.iloc[1]["predicted_label"] == 0
    assert errors.iloc[1]["confidence_distance_from_threshold"] == pytest.approx(0.3)
    assert errors.iloc[0]["confidence_distance_from_threshold"] > errors.iloc[1][
        "confidence_distance_from_threshold"
    ]
