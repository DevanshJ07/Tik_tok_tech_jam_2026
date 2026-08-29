from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "evaluate.py"


def _record(**overrides: object) -> dict:
    row = {
        "image_path": "a.png",
        "label": 0,
        "pred": 0.1,
        "model_name": "baseline",
        "dataset": "unit",
        "split": "test",
        "transform_name": "clean",
        "severity": "none",
    }
    row.update(overrides)
    return row


def _fixture_records() -> list[dict]:
    return [
        _record(image_path="c0.png", label=0, pred=0.1, transform_name="clean", severity="none"),
        _record(image_path="c1.png", label=1, pred=0.9, transform_name="clean", severity="none"),
        _record(image_path="j0.png", label=0, pred=0.9, transform_name="jpeg", severity="30"),
        _record(image_path="j1.png", label=1, pred=0.2, transform_name="jpeg", severity="30"),
    ]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_csv_input_writes_all_outputs(tmp_path: Path) -> None:
    predictions = tmp_path / "preds.csv"
    pd.DataFrame(_fixture_records()).to_csv(predictions, index=False)
    output_dir = tmp_path / "eval_out"
    completed = _run_cli(
        "--predictions",
        str(predictions),
        "--output_dir",
        str(output_dir),
        "--threshold",
        "0.5",
    )
    assert completed.returncode == 0, completed.stderr
    summary_path = output_dir / "summary.json"
    by_condition = output_dir / "by_condition.csv"
    by_family = output_dir / "by_family.csv"
    errors = output_dir / "errors.csv"
    for path in (summary_path, by_condition, by_family, errors):
        assert path.is_file()
        assert path.name in completed.stdout
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["input_filename"] == "preds.csv"
    assert summary["threshold"] == 0.5
    assert "robustness_drop" in summary
    assert set(pd.read_csv(errors).columns) >= {
        "image_path",
        "error_type",
        "confidence_distance_from_threshold",
    }
    assert predictions.read_text(encoding="utf-8")
    assert not (output_dir / "preds.csv").exists()


def test_cli_json_input(tmp_path: Path) -> None:
    predictions = tmp_path / "preds.json"
    predictions.write_text(json.dumps(_fixture_records()), encoding="utf-8")
    output_dir = tmp_path / "out"
    completed = _run_cli(
        "--predictions",
        str(predictions),
        "--output_dir",
        str(output_dir),
        "--threshold",
        "0.5",
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["input_filename"] == "preds.json"


def test_cli_rejects_official_inference_json(tmp_path: Path) -> None:
    predictions = tmp_path / "official.json"
    predictions.write_text(
        json.dumps([{"image_path": "x.png", "pred": 0.3}]),
        encoding="utf-8",
    )
    completed = _run_cli(
        "--predictions",
        str(predictions),
        "--output_dir",
        str(tmp_path / "out"),
    )
    assert completed.returncode != 0
    assert "Official inference JSON" in completed.stderr


def test_cli_rejects_mock_data(tmp_path: Path) -> None:
    records = _fixture_records()
    records[0]["model_name"] = "MockPredictor"
    predictions = tmp_path / "mock.json"
    predictions.write_text(json.dumps(records), encoding="utf-8")
    completed = _run_cli(
        "--predictions",
        str(predictions),
        "--output_dir",
        str(tmp_path / "out"),
    )
    assert completed.returncode != 0
    assert "Mock" in completed.stderr
    assert "model performance" in completed.stderr.lower()
    assert not (tmp_path / "out" / "summary.json").exists()
