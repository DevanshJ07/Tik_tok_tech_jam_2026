from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from src.inference.predictor import MockPredictor, discover_image_paths, run_directory_inference

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "predict_directory.py"


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (6, 6), color=color).save(path)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_image_discovery_is_recursive_and_sorted(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "z_top.jpg", (1, 2, 3))
    _write_rgb(tmp_path / "nested" / "a.png", (4, 5, 6))
    _write_rgb(tmp_path / "nested" / "deep" / "b.bmp", (7, 8, 9))
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    discovered = discover_image_paths(tmp_path)
    relative = [path.relative_to(tmp_path).as_posix() for path in discovered]
    assert relative == sorted(relative)
    assert relative == ["nested/a.png", "nested/deep/b.bmp", "z_top.jpg"]


def test_valid_images_produce_official_json(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "one.png", (255, 0, 0))
    _write_rgb(tmp_path / "two.jpeg", (0, 255, 0))
    output_json = tmp_path / "out" / "preds.json"

    records = run_directory_inference(tmp_path, output_json, MockPredictor())
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded == records
    assert len(loaded) == 2
    for record in loaded:
        assert set(record.keys()) == {"image_path", "pred"}
        assert "/" in record["image_path"] or record["image_path"] in {"one.png", "two.jpeg"}
        assert 0.0 <= record["pred"] <= 1.0
        assert "\\" not in record["image_path"]


def test_corrupt_images_are_skipped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_rgb(tmp_path / "good.png", (0, 0, 255))
    (tmp_path / "bad.png").write_bytes(b"this is not an image")
    output_json = tmp_path / "preds.json"

    records = run_directory_inference(tmp_path, output_json, MockPredictor())
    captured = capsys.readouterr()
    assert len(records) == 1
    assert records[0]["image_path"] == "good.png"
    assert "skipping corrupt" in captured.err
    assert "bad.png" in captured.err


def test_missing_mock_fails_clearly(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "x.png", (9, 9, 9))
    output_json = tmp_path / "preds.json"
    completed = _run_cli("--input_dir", str(tmp_path), "--output_json", str(output_json))
    assert completed.returncode != 0
    combined = completed.stderr + completed.stdout
    assert "--mock" in combined
    assert not output_json.exists()


def test_missing_input_directory_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    output_json = tmp_path / "preds.json"
    completed = _run_cli(
        "--input_dir",
        str(missing),
        "--output_json",
        str(output_json),
        "--mock",
    )
    assert completed.returncode != 0
    combined = completed.stderr + completed.stdout
    assert "does not exist" in combined.lower() or "not a directory" in combined.lower()
    assert not output_json.exists()


def test_official_json_contains_no_additional_fields(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "sample.png", (12, 34, 56))
    output_json = tmp_path / "official.json"
    completed = _run_cli(
        "--input_dir",
        str(tmp_path),
        "--output_json",
        str(output_json),
        "--mock",
    )
    assert completed.returncode == 0, completed.stderr
    assert "NOT model predictions" in completed.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload
    for record in payload:
        assert set(record.keys()) == {"image_path", "pred"}
        assert "mock" not in record
        assert "manipulation_probability" not in record
        assert "reliability_score" not in record
        assert "heatmap_path" not in record
        assert "aigc_probability" not in record
