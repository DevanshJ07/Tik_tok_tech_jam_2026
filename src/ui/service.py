"""Framework-independent upload, prediction, and download helpers."""

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.inference.contracts import PredictionResult
from src.inference.factory import create_predictor
from src.inference.predictor import IMAGE_EXTENSIONS, Predictor

SUPPORTED_UPLOAD_EXTENSIONS = IMAGE_EXTENSIONS
SUPPORTED_UPLOAD_LABEL = "JPG, JPEG, PNG, WEBP, BMP"


class UploadError(ValueError):
    """Raised when an upload cannot be accepted."""


class MockConfirmationRequiredError(ValueError):
    """Raised when mock mode is on but the user has not confirmed testing-only use."""


@dataclass(frozen=True)
class SavedUpload:
    path: Path
    display_name: str


def resolve_predictor(*, mock_enabled: bool, mock_acknowledged: bool) -> Predictor:
    """Build a predictor from explicit UI flags. Never implies mock=True."""
    if mock_enabled:
        if not mock_acknowledged:
            raise MockConfirmationRequiredError(
                "Mock mode is testing-only. Enable mock mode and confirm that "
                "results are not model predictions before analysing."
            )
        return create_predictor(mock=True)
    return create_predictor(mock=False)


def validate_image_bytes(data: bytes, filename: str | None = None) -> str:
    """Validate uploaded bytes. Return the normalised suffix including the dot."""
    if data is None or len(data) == 0:
        raise UploadError("Upload is empty.")
    suffix = _normalised_suffix(filename)
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise UploadError(
            f"Unsupported upload type {suffix or '(missing extension)'}. "
            f"Supported: {SUPPORTED_UPLOAD_LABEL}."
        )
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image.load()
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as exc:
        raise UploadError("Upload is corrupt or is not a readable image.") from exc
    return suffix


def display_filename(original_name: str | None) -> str:
    """Keep the original basename for display only."""
    if not original_name or not str(original_name).strip():
        return "upload"
    name = Path(str(original_name).replace("\\", "/")).name.strip()
    return name or "upload"


def safe_disk_filename(original_name: str | None, suffix: str) -> str:
    """Return a generated filename; do not use the original name on disk."""
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise UploadError(f"Unsupported suffix for disk save: {suffix}")
    return f"{uuid.uuid4().hex}{suffix}"


def save_upload(
    data: bytes,
    original_name: str | None,
    *,
    directory: Path | None = None,
) -> SavedUpload:
    """Validate bytes and write them under a generated name in a temp directory."""
    suffix = validate_image_bytes(data, original_name)
    target_dir = Path(directory) if directory is not None else Path(tempfile.mkdtemp(prefix="tracelens_r_"))
    target_dir.mkdir(parents=True, exist_ok=True)
    disk_name = safe_disk_filename(original_name, suffix)
    path = target_dir / disk_name
    path.write_bytes(data)
    return SavedUpload(path=path, display_name=display_filename(original_name))


def predict_upload(predictor: Predictor, saved: SavedUpload) -> PredictionResult:
    """Call any Predictor and rewrite image_path to the display filename."""
    result = predictor.predict(saved.path)
    result.image_path = saved.display_name
    return result


def official_json_text(result: PredictionResult) -> str:
    record = result.to_official_record()
    if set(record.keys()) != {"image_path", "pred"}:
        raise UploadError("Official JSON must contain exactly image_path and pred.")
    return json.dumps(record, indent=2) + "\n"


def detailed_json_text(result: PredictionResult) -> str:
    record = result.to_detailed_record()
    record["record_kind"] = "optional_internal"
    return json.dumps(record, indent=2) + "\n"


def cleanup_path(path: Path | None) -> None:
    """Delete a temporary file if it exists. Missing files are ignored."""
    if path is None:
        return
    target = Path(path)
    try:
        if target.is_file():
            target.unlink()
        if target.parent.is_dir() and target.parent.name.startswith("tracelens_r_"):
            remaining = list(target.parent.iterdir())
            if not remaining:
                target.parent.rmdir()
    except OSError:
        return


def analyse_bytes(
    data: bytes,
    original_name: str | None,
    predictor: Predictor,
) -> tuple[PredictionResult, str, str]:
    """Validate, predict, and always clean the temporary file."""
    saved = save_upload(data, original_name)
    try:
        result = predict_upload(predictor, saved)
        return result, official_json_text(result), detailed_json_text(result)
    finally:
        cleanup_path(saved.path)


def _normalised_suffix(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(str(filename).replace("\\", "/")).suffix.lower()
