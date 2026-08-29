from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.inference.factory import RealModelUnavailableError, create_predictor
from src.ui.service import (
    UploadError,
    analyse_bytes,
    cleanup_path,
    display_filename,
    official_json_text,
    resolve_predictor,
    safe_disk_filename,
    save_upload,
    validate_image_bytes,
)


def _png_bytes() -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", (6, 6), (20, 40, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_empty_upload_rejection() -> None:
    with pytest.raises(UploadError, match="empty"):
        validate_image_bytes(b"", "photo.png")


def test_corrupt_upload_rejection() -> None:
    with pytest.raises(UploadError, match="corrupt"):
        validate_image_bytes(b"not-an-image", "photo.png")


def test_unsupported_upload_rejection() -> None:
    with pytest.raises(UploadError, match="Unsupported"):
        validate_image_bytes(_png_bytes(), "notes.gif")


def test_supported_image_validation() -> None:
    suffix = validate_image_bytes(_png_bytes(), "folder/photo.PNG")
    assert suffix == ".png"


def test_safe_temporary_filenames_ignore_original_name() -> None:
    generated = safe_disk_filename("..\\secret.jpg", ".jpg")
    assert generated.endswith(".jpg")
    assert "secret" not in generated
    assert ".." not in generated
    assert display_filename("nested\\holiday.png") == "holiday.png"


def test_official_json_exact_keys(tmp_path: Path) -> None:
    saved = save_upload(_png_bytes(), "demo.png", directory=tmp_path)
    predictor = create_predictor(mock=True)
    from src.ui.service import predict_upload

    result = predict_upload(predictor, saved)
    payload = json.loads(official_json_text(result))
    assert set(payload.keys()) == {"image_path", "pred"}
    assert payload["image_path"] == "demo.png"
    cleanup_path(saved.path)


def test_temporary_file_cleanup(tmp_path: Path) -> None:
    saved = save_upload(_png_bytes(), "keep-name.png", directory=tmp_path)
    assert saved.path.is_file()
    assert saved.path.name != "keep-name.png"
    cleanup_path(saved.path)
    assert not saved.path.exists()


def test_analyse_bytes_cleans_up_and_handles_unavailable_model(tmp_path: Path) -> None:
    result, official, detailed = analyse_bytes(_png_bytes(), "shown.png", create_predictor(mock=True))
    assert result.image_path == "shown.png"
    assert set(json.loads(official).keys()) == {"image_path", "pred"}
    assert "optional_internal" in detailed
    leftover = list(tmp_path.glob("**/*")) if tmp_path.exists() else []
    assert leftover == [] or all(not path.name.endswith(".png") or "shown.png" != path.name for path in leftover)

    with pytest.raises(RealModelUnavailableError):
        resolve_predictor(mock_enabled=False, mock_acknowledged=False)
    with pytest.raises(RealModelUnavailableError):
        analyse_bytes(_png_bytes(), "x.png", create_predictor(mock=False))
