from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.schemas import EvaluationSchemaError, load_evaluation_records

BASE = {
    "image_path": "a.png",
    "label": 0,
    "pred": 0.1,
    "model_name": "baseline",
    "dataset": "unit",
    "split": "test",
    "transform_name": "clean",
    "severity": "none",
}


def _write_json(path: Path, records: list[dict]) -> Path:
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _write_csv(path: Path, records: list[dict]) -> Path:
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_json_and_csv_load_equivalent(tmp_path: Path) -> None:
    records = [
        dict(BASE),
        dict(BASE, image_path="b.png", label=1, pred=0.9, transform_name="jpeg", severity=70),
    ]
    json_frame = load_evaluation_records(_write_json(tmp_path / "a.json", records))
    csv_frame = load_evaluation_records(_write_csv(tmp_path / "a.csv", records))
    assert list(json_frame["image_path"]) == ["a.png", "b.png"]
    assert list(csv_frame["label"]) == [0, 1]
    assert list(csv_frame["severity"]) == ["none", "70"]


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    empty_json = tmp_path / "empty.json"
    empty_json.write_text("[]", encoding="utf-8")
    with pytest.raises(EvaluationSchemaError, match="empty"):
        load_evaluation_records(empty_json)

    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame(columns=list(BASE)).to_csv(empty_csv, index=False)
    with pytest.raises(EvaluationSchemaError, match="empty"):
        load_evaluation_records(empty_csv)


def test_invalid_probabilities_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvaluationSchemaError, match="pred"):
        load_evaluation_records(
            _write_json(tmp_path / "hi.json", [dict(BASE, pred=1.2)])
        )
    with pytest.raises(EvaluationSchemaError, match="pred"):
        load_evaluation_records(
            _write_json(tmp_path / "nan.json", [dict(BASE, pred=float("nan"))])
        )
    with pytest.raises(EvaluationSchemaError, match="pred"):
        load_evaluation_records(
            _write_json(tmp_path / "inf.json", [dict(BASE, pred=float("inf"))])
        )


def test_label_must_be_zero_or_one(tmp_path: Path) -> None:
    with pytest.raises(EvaluationSchemaError, match="label"):
        load_evaluation_records(_write_json(tmp_path / "l2.json", [dict(BASE, label=2)]))


def test_unknown_transformations_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvaluationSchemaError, match="unknown transform"):
        load_evaluation_records(
            _write_json(tmp_path / "t.json", [dict(BASE, transform_name="elastic")])
        )


def test_duplicate_records_are_rejected(tmp_path: Path) -> None:
    records = [dict(BASE), dict(BASE)]
    with pytest.raises(EvaluationSchemaError, match="Duplicate"):
        load_evaluation_records(_write_json(tmp_path / "dup.json", records))


def test_empty_model_dataset_or_split_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvaluationSchemaError, match="model_name"):
        load_evaluation_records(_write_json(tmp_path / "m.json", [dict(BASE, model_name=" ")]))
    with pytest.raises(EvaluationSchemaError, match="dataset"):
        load_evaluation_records(_write_json(tmp_path / "d.json", [dict(BASE, dataset="")]))
    with pytest.raises(EvaluationSchemaError, match="split"):
        load_evaluation_records(_write_json(tmp_path / "s.json", [dict(BASE, split="")]))


def test_missing_required_fields_rejected(tmp_path: Path) -> None:
    record = dict(BASE)
    del record["label"]
    with pytest.raises(EvaluationSchemaError, match="missing required"):
        load_evaluation_records(_write_json(tmp_path / "miss.json", [record]))


def test_official_inference_json_is_rejected(tmp_path: Path) -> None:
    official = [{"image_path": "x.png", "pred": 0.4}]
    with pytest.raises(EvaluationSchemaError, match="Official inference JSON"):
        load_evaluation_records(_write_json(tmp_path / "official.json", official))


def test_mock_identified_records_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvaluationSchemaError, match="Mock"):
        load_evaluation_records(
            _write_json(
                tmp_path / "mock.json",
                [dict(BASE, model_name="MockPredictor")],
            )
        )
    with pytest.raises(EvaluationSchemaError, match="Mock"):
        load_evaluation_records(
            _write_json(
                tmp_path / "flag.json",
                [dict(BASE, is_mock=True)],
            )
        )
