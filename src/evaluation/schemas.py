"""Internal labelled evaluation records.

This schema is not the official inference JSON. Official output remains exactly
``image_path`` and ``pred`` and cannot be scored by this engine.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

ALLOWED_TRANSFORM_NAMES = frozenset(
    {
        "clean",
        "jpeg",
        "gaussian_blur",
        "resize",
        "gaussian_noise",
        "color_jitter",
        "center_crop",
    }
)

REQUIRED_FIELDS = (
    "image_path",
    "label",
    "pred",
    "model_name",
    "dataset",
    "split",
    "transform_name",
    "severity",
)

DUPLICATE_KEY_FIELDS = (
    "model_name",
    "dataset",
    "split",
    "image_path",
    "transform_name",
    "severity",
)

OFFICIAL_INFERENCE_KEYS = frozenset({"image_path", "pred"})


class EvaluationError(ValueError):
    """Base error for internal evaluation."""


class EvaluationSchemaError(EvaluationError):
    """Raised when labelled evaluation records are missing or invalid."""


@dataclass(frozen=True)
class EvaluationRecord:
    image_path: str
    label: int
    pred: float
    model_name: str
    dataset: str
    split: str
    transform_name: str
    severity: str


def load_evaluation_records(path: str | Path) -> pd.DataFrame:
    """Load CSV or JSON evaluation records and return a validated DataFrame."""
    file_path = Path(path)
    if not file_path.is_file():
        raise EvaluationSchemaError(f"Prediction file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        raw_records = _records_from_csv(file_path)
    elif suffix == ".json":
        raw_records = _records_from_json(file_path)
    else:
        raise EvaluationSchemaError(
            f"Unsupported prediction file type {suffix!r}. Use .csv or .json."
        )

    if not raw_records:
        raise EvaluationSchemaError("Prediction file is empty.")

    if _looks_like_official_inference(raw_records):
        raise EvaluationSchemaError(
            "Official inference JSON is insufficient for evaluation. "
            "Internal evaluation records require labels and metadata; "
            "official output may contain only image_path and pred."
        )

    validated = [parse_evaluation_record(record, index=i) for i, record in enumerate(raw_records)]
    _reject_mock_records(validated)
    _reject_duplicates(validated)
    return records_to_frame(validated)


def parse_evaluation_record(record: Mapping[str, Any], *, index: int) -> EvaluationRecord:
    if not isinstance(record, Mapping):
        raise EvaluationSchemaError(f"Record {index} must be an object/mapping.")

    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise EvaluationSchemaError(
            f"Record {index} is missing required fields: {', '.join(missing)}."
        )

    image_path = _require_non_empty_string(record["image_path"], "image_path", index)
    model_name = _require_non_empty_string(record["model_name"], "model_name", index)
    dataset = _require_non_empty_string(record["dataset"], "dataset", index)
    split = _require_non_empty_string(record["split"], "split", index)
    transform_name = _require_non_empty_string(
        record["transform_name"], "transform_name", index
    )
    if transform_name not in ALLOWED_TRANSFORM_NAMES:
        raise EvaluationSchemaError(
            f"Record {index} has unknown transform_name {transform_name!r}."
        )

    label = _parse_label(record["label"], index)
    pred = _parse_pred(record["pred"], index)
    severity = _parse_severity(record["severity"], transform_name, index)

    parsed = EvaluationRecord(
        image_path=image_path,
        label=label,
        pred=pred,
        model_name=model_name,
        dataset=dataset,
        split=split,
        transform_name=transform_name,
        severity=severity,
    )
    if _record_identifies_as_mock(record, parsed):
        raise EvaluationSchemaError(
            "Mock-identified records cannot be evaluated as model performance. "
            f"Record {index} identifies itself as mock data."
        )
    return parsed


def records_to_frame(records: Iterable[EvaluationRecord]) -> pd.DataFrame:
    rows = [record.__dict__ for record in records]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise EvaluationSchemaError("Prediction file is empty.")
    return frame


def _records_from_csv(path: Path) -> list[dict[str, Any]]:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas message forwarded
        raise EvaluationSchemaError(f"Failed to read CSV: {exc}") from exc
    if frame.empty:
        return []
    return frame.to_dict(orient="records")


def _records_from_json(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationSchemaError(f"Failed to read JSON: {exc}") from exc
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    raise EvaluationSchemaError("JSON evaluation file must be a list of records.")


def _looks_like_official_inference(records: list[Any]) -> bool:
    objects = [record for record in records if isinstance(record, Mapping)]
    if not objects or len(objects) != len(records):
        return False
    return all(frozenset(record.keys()) <= OFFICIAL_INFERENCE_KEYS for record in objects)


def _require_non_empty_string(value: Any, name: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationSchemaError(f"Record {index} field {name} must be a non-empty string.")
    return value.strip()


def _parse_label(value: Any, index: int) -> int:
    if isinstance(value, bool):
        raise EvaluationSchemaError(f"Record {index} label must be integer 0 or 1.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationSchemaError(f"Record {index} label must be integer 0 or 1.") from exc
    if not math.isfinite(number) or number not in (0.0, 1.0):
        raise EvaluationSchemaError(
            f"Record {index} label must be integer 0 or 1, not {value!r}."
        )
    return int(number)


def _parse_pred(value: Any, index: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationSchemaError(
            f"Record {index} pred must be a finite number in [0, 1]."
        ) from exc
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise EvaluationSchemaError(
            f"Record {index} pred must be a finite number in [0, 1], not {value!r}."
        )
    return float(number)


def _parse_severity(value: Any, transform_name: str, index: int) -> str:
    del transform_name
    if value is None:
        raise EvaluationSchemaError(f"Record {index} severity must be a string or number.")
    if isinstance(value, bool):
        raise EvaluationSchemaError(f"Record {index} severity must be a string or number.")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise EvaluationSchemaError(f"Record {index} severity must be a string or number.")
        return text
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationSchemaError(
            f"Record {index} severity must be a string or number."
        ) from exc
    if not math.isfinite(number):
        raise EvaluationSchemaError(f"Record {index} severity must be a string or number.")
    if number.is_integer():
        return str(int(number))
    return format(number, "g")


def _record_identifies_as_mock(raw: Mapping[str, Any], parsed: EvaluationRecord) -> bool:
    text_fields = [
        parsed.model_name,
        parsed.dataset,
        parsed.split,
        parsed.image_path,
    ]
    for key in ("source", "predictor", "origin", "generator"):
        value = raw.get(key)
        if isinstance(value, str):
            text_fields.append(value)
    if any("mock" in text.lower() for text in text_fields):
        return True
    for key in ("mock", "is_mock", "testing_only"):
        value = raw.get(key)
        if value is True:
            return True
        if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "mock"}:
            return True
    return False


def _reject_mock_records(records: list[EvaluationRecord]) -> None:
    # parse_evaluation_record already rejects mock rows; this is a final guard.
    for record in records:
        if "mock" in record.model_name.lower() or "mock" in record.dataset.lower():
            raise EvaluationSchemaError(
                "Mock-identified records cannot be evaluated as model performance."
            )


def _reject_duplicates(records: list[EvaluationRecord]) -> None:
    seen: dict[tuple[str, ...], int] = {}
    for index, record in enumerate(records):
        key = tuple(str(getattr(record, field)) for field in DUPLICATE_KEY_FIELDS)
        if key in seen:
            raise EvaluationSchemaError(
                "Duplicate evaluation rows for the same model, dataset, split, "
                f"image_path, transform and severity (records {seen[key]} and {index})."
            )
        seen[key] = index
