from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    load_config,
    validate_config,
)


def test_valid_default_configuration_loads() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config["seed"] == 42
    assert config["model"]["image_size"] == 224
    assert config["model"]["patch_grid_size"] == 16
    assert config["model"]["embedding_dimension"] == 384
    assert config["model"]["backbone_name"]
    assert config["model"]["backbone_frozen"] is True
    assert config["dataset"]["labels"]["authentic"] == 0
    assert config["dataset"]["labels"]["fully_synthetic"] == 1
    assert config["dataset"]["labels"]["locally_tampered"] == 2
    assert config["paths"]["checkpoints_dir"]
    assert config["paths"]["outputs_dir"]
    assert config["inference"]["device"] == "cpu"


def test_invalid_fusion_weights_are_rejected(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    config["model"]["global_weight"] = 0.7
    config["model"]["patch_weight"] = 0.7
    with pytest.raises(ConfigError, match="sum to 1"):
        validate_config(config)

    config["model"]["global_weight"] = 0.5
    config["model"]["patch_weight"] = 0.5
    negative = dict(config)
    negative["model"] = dict(config["model"])
    negative["model"]["global_weight"] = -0.1
    negative["model"]["patch_weight"] = 1.1
    with pytest.raises(ConfigError, match="non-negative"):
        validate_config(negative)

    bad_path = tmp_path / "bad.yaml"
    payload = load_config(DEFAULT_CONFIG_PATH)
    payload["model"]["global_weight"] = 0.9
    payload["model"]["patch_weight"] = 0.2
    bad_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="sum to 1"):
        load_config(bad_path)
