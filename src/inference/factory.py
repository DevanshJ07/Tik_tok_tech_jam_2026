"""Central predictor construction for TraceLens-R.

``create_predictor(mock=False)`` never returns ``MockPredictor``. A missing,
invalid, or incompatible checkpoint raises ``RealModelUnavailableError``.
Callers must not catch that error and silently enable mock mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from src.inference.predictor import MockPredictor, Predictor
from src.inference.tracelens import (
    CheckpointError,
    TraceLensPredictor,
    resolve_inference_settings,
)
from src.models.backbone import DINOv2Backbone

REAL_MODEL_UNAVAILABLE_MESSAGE = (
    "The real TraceLens-R baseline predictor could not be constructed. "
    "Provide a valid Member 2 baseline checkpoint via --checkpoint, "
    "inference.checkpoint in the YAML config, or checkpoint= to "
    "create_predictor. The application will not switch to mock mode "
    "automatically. For UI wiring tests only, pass mock=True after an "
    "explicit on-screen confirmation that results are not model predictions."
)


class RealModelUnavailableError(RuntimeError):
    """Raised when a real predictor is requested but cannot be constructed."""


def create_predictor(
    *,
    mock: bool = False,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    config: Mapping[str, Any] | None = None,
    backbone: Optional[DINOv2Backbone] = None,
) -> Predictor:
    """Return a Predictor. Mock mode is opt-in and never a silent fallback.

    Parameters
    ----------
    mock:
        If True, return the testing-only ``MockPredictor``. If False (default),
        return ``TraceLensPredictor`` or raise ``RealModelUnavailableError``.
    checkpoint:
        Path to a Member 2 baseline ``.pt`` checkpoint. Optional if the YAML
        config sets ``inference.checkpoint``.
    device:
        ``cpu`` (default) or ``cuda``. CUDA is never implied.
    config:
        Optional already-loaded config mapping. Loaded from disk only when
        needed to resolve a missing checkpoint / device.
    backbone:
        Optional pre-built frozen backbone (tests inject a stub so DINOv2 is
        never downloaded).
    """
    if mock:
        return MockPredictor()
    return _build_real_predictor(
        checkpoint=checkpoint,
        device=device,
        config=config,
        backbone=backbone,
    )


def _build_real_predictor(
    *,
    checkpoint: str | Path | None,
    device: str | None,
    config: Mapping[str, Any] | None,
    backbone: Optional[DINOv2Backbone],
) -> Predictor:
    """Construct the production baseline predictor.

    REAL MODEL CONNECTION POINT
    ---------------------------
    Members 3–4 attach optional modules on ``TraceLensPredictor``
    (``reliability_module``, ``manipulation_module``). Do not wrap this
    construction in a try/except that returns MockPredictor.
    Optional reliability/manipulation failures belong inside
    TraceLensPredictor and must not change the required AIGC probability
    (SPEC §14).
    """
    try:
        if config is None and (
            checkpoint is None or not str(checkpoint).strip()
        ):
            from src.config import load_config

            config = load_config()
        settings = resolve_inference_settings(
            checkpoint=checkpoint,
            device=device,
            config=config,
        )
        return TraceLensPredictor(
            settings.checkpoint,
            device=settings.device,
            backbone_name=settings.backbone_name,
            backbone=backbone,
        )
    except CheckpointError as exc:
        raise RealModelUnavailableError(f"{REAL_MODEL_UNAVAILABLE_MESSAGE} ({exc})") from exc
