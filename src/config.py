"""Load and validate TraceLens-R YAML configuration.

Invalid values are rejected. This module never substitutes defaults for bad fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

EXPECTED_IMAGE_SIZE = 224
EXPECTED_PATCH_GRID_SIZE = 16
EXPECTED_EMBEDDING_DIMENSION = 384
EXPECTED_LABELS = {
    "authentic": 0,
    "fully_synthetic": 1,
    "locally_tampered": 2,
}
WEIGHT_SUM_TOLERANCE = 1e-6


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed, or contract-invalid."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML with ``yaml.safe_load`` and validate the TraceLens-R contract."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping.")

    validate_config(raw)
    return raw


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate contract fields. Does not mutate or repair ``config``."""
    seed = _require_key(config, "seed")
    if not _is_int(seed):
        raise ConfigError("seed must be an integer.")

    model = _require_mapping(config, "model")
    image_size = _require_key(model, "image_size", "model.")
    if image_size != EXPECTED_IMAGE_SIZE:
        raise ConfigError(f"image_size must equal {EXPECTED_IMAGE_SIZE}.")

    patch_grid_size = _require_key(model, "patch_grid_size", "model.")
    if patch_grid_size != EXPECTED_PATCH_GRID_SIZE:
        raise ConfigError(f"patch_grid_size must equal {EXPECTED_PATCH_GRID_SIZE}.")

    embedding_dimension = _require_key(model, "embedding_dimension", "model.")
    if embedding_dimension != EXPECTED_EMBEDDING_DIMENSION:
        raise ConfigError(
            f"embedding_dimension must equal {EXPECTED_EMBEDDING_DIMENSION}."
        )

    backbone_name = _require_key(model, "backbone_name", "model.")
    if not isinstance(backbone_name, str) or not backbone_name.strip():
        raise ConfigError("backbone_name must be a non-empty string.")

    backbone_frozen = _require_key(model, "backbone_frozen", "model.")
    if backbone_frozen is not True:
        raise ConfigError("backbone_frozen must be true.")

    global_weight = _require_key(model, "global_weight", "model.")
    patch_weight = _require_key(model, "patch_weight", "model.")
    if not _is_number(global_weight) or not _is_number(patch_weight):
        raise ConfigError("global_weight and patch_weight must be numeric.")
    if global_weight < 0 or patch_weight < 0:
        raise ConfigError("global_weight and patch_weight must be non-negative.")
    weight_sum = float(global_weight) + float(patch_weight)
    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ConfigError(
            "global_weight and patch_weight must sum to 1 "
            f"within tolerance {WEIGHT_SUM_TOLERANCE}."
        )

    dataset = _require_mapping(config, "dataset")
    labels = _require_mapping(dataset, "labels", "dataset.")
    for name, expected in EXPECTED_LABELS.items():
        if name not in labels:
            raise ConfigError(f"dataset.labels must contain {name}={expected}.")
        if labels[name] != expected:
            raise ConfigError(
                f"dataset.labels.{name} must equal {expected}, got {labels[name]!r}."
            )

    paths = _require_mapping(config, "paths")
    checkpoints_dir = _require_key(paths, "checkpoints_dir", "paths.")
    outputs_dir = _require_key(paths, "outputs_dir", "paths.")
    if not _is_non_empty_path_value(checkpoints_dir):
        raise ConfigError("paths.checkpoints_dir must be a non-empty path.")
    if not _is_non_empty_path_value(outputs_dir):
        raise ConfigError("paths.outputs_dir must be a non-empty path.")

    if "inference" in config:
        inference = _require_mapping(config, "inference")
        device = inference.get("device", "cpu")
        if device is None or not isinstance(device, str) or not device.strip():
            raise ConfigError("inference.device must be a non-empty string.")
        if device.strip() != "cpu" and device.strip() != "cuda" and not device.strip().startswith("cuda:"):
            raise ConfigError("inference.device must be 'cpu' or 'cuda'.")
        checkpoint = inference.get("checkpoint", "")
        if checkpoint is not None and not isinstance(checkpoint, (str, Path)):
            raise ConfigError("inference.checkpoint must be a path string.")


def _require_key(mapping: Mapping[str, Any], key: str, prefix: str = "") -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key: {prefix}{key}")
    return mapping[key]


def _require_mapping(
    mapping: Mapping[str, Any], key: str, prefix: str = ""
) -> Mapping[str, Any]:
    value = _require_key(mapping, key, prefix)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{prefix}{key} must be a mapping.")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_non_empty_path_value(value: Any) -> bool:
    if isinstance(value, Path):
        return bool(str(value).strip())
    return isinstance(value, str) and bool(value.strip())
