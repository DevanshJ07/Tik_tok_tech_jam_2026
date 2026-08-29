from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.robustness import summarize_robustness


def _row(
    image_path: str,
    label: int,
    pred: float,
    transform_name: str,
    severity: str,
) -> dict:
    return {
        "image_path": image_path,
        "label": label,
        "pred": pred,
        "model_name": "baseline",
        "dataset": "unit",
        "split": "test",
        "transform_name": transform_name,
        "severity": severity,
    }


def test_clean_versus_transformed_robustness_drop() -> None:
    frame = pd.DataFrame(
        [
            _row("c0.png", 0, 0.1, "clean", "none"),
            _row("c1.png", 1, 0.9, "clean", "none"),
            _row("j0.png", 0, 0.9, "jpeg", "30"),
            _row("j1.png", 1, 0.9, "jpeg", "30"),
            _row("b0.png", 0, 0.1, "gaussian_blur", "1.0"),
            _row("b1.png", 1, 0.9, "gaussian_blur", "1.0"),
        ]
    )
    summary = summarize_robustness(frame, threshold=0.5)
    assert summary.clean_balanced_accuracy == pytest.approx(1.0)
    # jpeg/30 BA = 0.5; blur/1.0 BA = 1.0; mean = 0.75
    assert summary.mean_transformed_balanced_accuracy == pytest.approx(0.75)
    assert summary.robustness_drop == pytest.approx(0.25)
    assert summary.worst_condition.transform_name == "jpeg"
    assert summary.worst_condition.severity == "30"
    assert summary.worst_condition_balanced_accuracy == pytest.approx(0.5)


def test_negative_robustness_drop_is_preserved() -> None:
    frame = pd.DataFrame(
        [
            _row("c0.png", 0, 0.9, "clean", "none"),
            _row("c1.png", 1, 0.9, "clean", "none"),
            _row("j0.png", 0, 0.1, "jpeg", "90"),
            _row("j1.png", 1, 0.9, "jpeg", "90"),
        ]
    )
    summary = summarize_robustness(frame, threshold=0.5)
    assert summary.clean_balanced_accuracy == pytest.approx(0.5)
    assert summary.mean_transformed_balanced_accuracy == pytest.approx(1.0)
    assert summary.robustness_drop == pytest.approx(-0.5)
    assert summary.robustness_drop < 0


def test_worst_condition_uses_individual_groups_not_family_average() -> None:
    frame = pd.DataFrame(
        [
            _row("c0.png", 0, 0.1, "clean", "none"),
            _row("c1.png", 1, 0.9, "clean", "none"),
            _row("j30_0.png", 0, 0.9, "jpeg", "30"),
            _row("j30_1.png", 1, 0.1, "jpeg", "30"),
            _row("j90_0.png", 0, 0.1, "jpeg", "90"),
            _row("j90_1.png", 1, 0.9, "jpeg", "90"),
            _row("b0.png", 0, 0.6, "gaussian_blur", "2.0"),
            _row("b1.png", 1, 0.9, "gaussian_blur", "2.0"),
        ]
    )
    summary = summarize_robustness(frame, threshold=0.5)
    jpeg_family = next(item for item in summary.by_family if item.transform_name == "jpeg")
    # jpeg/30 BA = 0.0, jpeg/90 BA = 1.0, family mean = 0.5; blur BA = 0.5
    assert jpeg_family.metrics.balanced_accuracy == pytest.approx(0.5)
    assert summary.worst_condition.transform_name == "jpeg"
    assert summary.worst_condition.severity == "30"
    assert summary.worst_condition_balanced_accuracy == pytest.approx(0.0)
    assert summary.worst_condition_balanced_accuracy != jpeg_family.metrics.balanced_accuracy


def test_macro_condition_mean_differs_from_pooled_when_sizes_unequal() -> None:
    """jpeg/30 has BA 0.0 on 2 rows; blur/1.0 has BA 1.0 on 20 rows.

    Macro mean = (0 + 1) / 2 = 0.5
    Pooled BA = 10/11 ≈ 0.909
    """
    rows = [
        _row("c0.png", 0, 0.1, "clean", "none"),
        _row("c1.png", 1, 0.9, "clean", "none"),
        _row("j0.png", 0, 0.9, "jpeg", "30"),
        _row("j1.png", 1, 0.1, "jpeg", "30"),
    ]
    for index in range(10):
        rows.append(_row(f"b_real_{index}.png", 0, 0.1, "gaussian_blur", "1.0"))
        rows.append(_row(f"b_synth_{index}.png", 1, 0.9, "gaussian_blur", "1.0"))
    summary = summarize_robustness(pd.DataFrame(rows), threshold=0.5)

    pooled_ba = summary.transformed_combined.balanced_accuracy
    assert summary.mean_transformed_balanced_accuracy == pytest.approx(0.5)
    assert pooled_ba == pytest.approx(10 / 11)
    assert summary.mean_transformed_balanced_accuracy != pytest.approx(pooled_ba)
    assert summary.robustness_drop == pytest.approx(0.5)
    assert summary.worst_condition.transform_name == "jpeg"
    assert summary.worst_condition.severity == "30"


def test_family_average_is_macro_severity_mean_not_pooled() -> None:
    """jpeg/30 BA 0.0 on 2 rows; jpeg/90 BA 1.0 on 20 rows.

    Family BA must be (0 + 1) / 2 = 0.5, not the pooled jpeg BA of 10/11.
    """
    rows = [
        _row("c0.png", 0, 0.1, "clean", "none"),
        _row("c1.png", 1, 0.9, "clean", "none"),
        _row("j30_0.png", 0, 0.9, "jpeg", "30"),
        _row("j30_1.png", 1, 0.1, "jpeg", "30"),
    ]
    for index in range(10):
        rows.append(_row(f"j90_real_{index}.png", 0, 0.1, "jpeg", "90"))
        rows.append(_row(f"j90_synth_{index}.png", 1, 0.9, "jpeg", "90"))
    summary = summarize_robustness(pd.DataFrame(rows), threshold=0.5)
    jpeg_family = next(item for item in summary.by_family if item.transform_name == "jpeg")
    jpeg_rows = [item for item in summary.by_condition if item.transform_name == "jpeg"]
    by_severity = {item.severity: item.metrics.balanced_accuracy for item in jpeg_rows}
    assert by_severity["30"] == pytest.approx(0.0)
    assert by_severity["90"] == pytest.approx(1.0)
    assert jpeg_family.severity_count == 2
    assert jpeg_family.metrics.balanced_accuracy == pytest.approx(0.5)
    assert jpeg_family.metrics.balanced_accuracy != pytest.approx(10 / 11)
    assert summary.mean_transformed_balanced_accuracy == pytest.approx(0.5)
    assert summary.transformed_combined.balanced_accuracy == pytest.approx(10 / 11)
