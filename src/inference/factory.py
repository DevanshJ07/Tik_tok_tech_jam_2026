"""Central predictor construction for TraceLens-R.

``create_predictor(mock=False)`` never returns ``MockPredictor``. If the real
model is not connected, it raises ``RealModelUnavailableError``. Callers must
not catch that error and silently enable mock mode.
"""

from __future__ import annotations

from src.inference.predictor import MockPredictor, Predictor

REAL_MODEL_UNAVAILABLE_MESSAGE = (
    "The real TraceLens-R model checkpoint has not yet been integrated. "
    "Analysis cannot run until TraceLensPredictor is connected in "
    "src/inference/factory.py. The application will not switch to mock "
    "mode automatically. For UI wiring tests only, pass mock=True after "
    "an explicit on-screen confirmation that results are not model predictions."
)


class RealModelUnavailableError(RuntimeError):
    """Raised when a real predictor is requested but not connected."""


def create_predictor(*, mock: bool = False) -> Predictor:
    """Return a Predictor. Mock mode is opt-in and never a silent fallback.

    Parameters
    ----------
    mock:
        If True, return the testing-only ``MockPredictor``. If False (default),
        return the real model or raise ``RealModelUnavailableError``.
    """
    if mock:
        return MockPredictor()
    return _build_real_predictor()


def _build_real_predictor() -> Predictor:
    """Construct the production predictor.

    REAL MODEL CONNECTION POINT
    ---------------------------
    When Members 2–4 deliver the trained modules, replace the raise below with:

        from src.models.tracelens import TraceLensPredictor
        return TraceLensPredictor(...)

    Do not wrap that construction in a try/except that returns MockPredictor.
    Optional reliability/manipulation failures belong inside TraceLensPredictor
    and must not change the required AIGC probability (SPEC §14).
    """
    raise RealModelUnavailableError(REAL_MODEL_UNAVAILABLE_MESSAGE)
