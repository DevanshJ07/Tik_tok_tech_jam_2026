"""Predictor interface and testing-only mock implementation.

``MockPredictor`` is not an AI detector. It must be constructed only when the
caller explicitly requested ``--mock``. Inference code must never fall back to
it silently.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Protocol, TextIO, runtime_checkable

from PIL import Image, UnidentifiedImageError

from src.inference.contracts import PredictionResult

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

MOCK_WARNING = (
    "WARNING: MockPredictor is testing-only. These results are NOT model "
    "predictions and MUST NOT be reported as AIGC detector output."
)


@runtime_checkable
class Predictor(Protocol):
    def predict(self, image_path: Path) -> PredictionResult:
        """Return a contract-valid prediction for one image path."""


class MockPredictor:
    """Testing-only predictor. Not an AI detector. Do not use in official runs.

    Probabilities are derived from a SHA-256 hash of the image file bytes so the
    same file always yields the same numbers. This is a deterministic stub for
    wiring tests, not a forensic or AIGC model.
    """

    def predict(self, image_path: Path) -> PredictionResult:
        path = Path(image_path)
        digest = hashlib.sha256(path.read_bytes()).digest()
        aigc_probability = _unit_interval(digest[0:8])
        manipulation_probability = _unit_interval(digest[8:16])
        reliability_score = _unit_interval(digest[16:24])
        return PredictionResult(
            image_path=path.as_posix(),
            aigc_probability=aigc_probability,
            manipulation_probability=manipulation_probability,
            reliability_score=reliability_score,
            heatmap_path=None,
        )


def discover_image_paths(input_dir: Path) -> list[Path]:
    """Recursively find supported images and return them in sorted order."""
    root = Path(input_dir)
    found = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    found.sort(key=lambda path: path.as_posix().replace("\\", "/"))
    return found


def is_openable_image(path: Path) -> bool:
    """Return True if Pillow can identify and load the file as an image."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
        return False


def relative_posix_path(path: Path, root: Path) -> str:
    """Prefer a forward-slash path relative to ``root``."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        try:
            return resolved.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()


def run_directory_inference(
    input_dir: Path,
    output_json: Path,
    predictor: Predictor,
    *,
    warn_stream: TextIO | None = None,
) -> list[dict[str, str | float]]:
    """Predict every valid image under ``input_dir`` and write official JSON."""
    stream = sys.stderr if warn_stream is None else warn_stream
    records: list[dict[str, str | float]] = []

    for image_path in discover_image_paths(input_dir):
        if not is_openable_image(image_path):
            print(f"warning: skipping corrupt or unreadable image: {image_path}", file=stream)
            continue
        result = predictor.predict(image_path)
        result.image_path = relative_posix_path(image_path, input_dir)
        records.append(result.to_official_record())

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def emit_mock_warning(stream: TextIO | None = None) -> None:
    target = sys.stderr if stream is None else stream
    print(MOCK_WARNING, file=target)


def _unit_interval(chunk: bytes) -> float:
    width = 8 * len(chunk)
    maximum = (1 << width) - 1
    return int.from_bytes(chunk, "big") / maximum
