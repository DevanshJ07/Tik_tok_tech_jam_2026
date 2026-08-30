from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.inference.factory import RealModelUnavailableError, create_predictor
from src.inference.predictor import MockPredictor
from src.ui.service import (
    MockConfirmationRequiredError,
    resolve_predictor,
)


def test_factory_refuses_silent_mock_fallback() -> None:
    with pytest.raises(RealModelUnavailableError, match="checkpoint"):
        create_predictor()
    with pytest.raises(RealModelUnavailableError):
        create_predictor(mock=False)
    with pytest.raises(RealModelUnavailableError):
        create_predictor(mock=False, config={"inference": {"checkpoint": "", "device": "cpu"}})
    try:
        create_predictor(mock=False)
    except RealModelUnavailableError:
        pass
    else:  # pragma: no cover
        raise AssertionError("real mode must not return a predictor without a checkpoint")


def test_explicit_mock_creation_works() -> None:
    predictor = create_predictor(mock=True)
    assert isinstance(predictor, MockPredictor)


def test_resolve_predictor_requires_mock_confirmation() -> None:
    with pytest.raises(MockConfirmationRequiredError):
        resolve_predictor(mock_enabled=True, mock_acknowledged=False)


def test_resolve_predictor_real_mode_does_not_fall_back() -> None:
    with pytest.raises(RealModelUnavailableError):
        resolve_predictor(mock_enabled=False, mock_acknowledged=True)


def test_resolve_predictor_explicit_mock(tmp_path: Path) -> None:
    predictor = resolve_predictor(mock_enabled=True, mock_acknowledged=True)
    assert isinstance(predictor, MockPredictor)
    image = tmp_path / "tiny.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(image)
    result = predictor.predict(image)
    assert 0.0 <= result.aigc_probability <= 1.0
