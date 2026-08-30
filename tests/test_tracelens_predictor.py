"""Real baseline inference tests. Stubs only — DINOv2 is never downloaded."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.inference.contracts import PredictionResult  # noqa: E402
from src.inference.factory import RealModelUnavailableError, create_predictor  # noqa: E402
from src.inference.predictor import MockPredictor, run_directory_inference  # noqa: E402
from src.inference.tracelens import (  # noqa: E402
    CheckpointError,
    TraceLensPredictor,
    load_baseline_detector,
    resolve_device,
)
from src.models.backbone import DINOv2Backbone  # noqa: E402
from src.models.baseline import EMBED_DIM, NUM_PATCHES, BaselineAIGCDetector  # noqa: E402
from src.training.train_baseline import TrainConfig, save_checkpoint  # noqa: E402


class _StubOutput:
    def __init__(self, last_hidden_state: "torch.Tensor") -> None:
        self.last_hidden_state = last_hidden_state


class _StubDINOv2(nn.Module):
    def __init__(self, num_patches: int = NUM_PATCHES) -> None:
        super().__init__()
        self.num_tokens = num_patches + 1
        self.proj = nn.Linear(3, EMBED_DIM)
        self.token_bias = nn.Parameter(torch.zeros(self.num_tokens, EMBED_DIM))

    def forward(self, pixel_values=None, return_dict=True, **_kwargs):
        pooled = pixel_values.mean(dim=(2, 3))
        base = self.proj(pooled).unsqueeze(1)
        hidden = base + self.token_bias.unsqueeze(0)
        return _StubOutput(hidden)


def _stub_backbone() -> DINOv2Backbone:
    return DINOv2Backbone(model_name="stub-dinov2-small", device="cpu", model=_StubDINOv2())


def _write_checkpoint(path: Path, **kwargs) -> Path:
    hidden_dim = int(kwargs.get("hidden_dim", 128))
    dropout = float(kwargs.get("dropout", 0.0))
    embed_dim = int(kwargs.get("embed_dim", EMBED_DIM))
    num_patches = int(kwargs.get("num_patches", NUM_PATCHES))
    torch.manual_seed(0)
    model = BaselineAIGCDetector(
        embed_dim=embed_dim,
        num_patches=num_patches,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.eval()
    save_checkpoint(
        path,
        model=model,
        optimizer=None,
        epoch=1,
        global_step=1,
        config=TrainConfig(hidden_dim=hidden_dim, dropout=dropout),
    )
    return path


def _rgb(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (40, 80, 120)).save(path)
    return path


def test_real_baseline_prediction_on_cpu(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "baseline.pt")
    image = _rgb(tmp_path / "sample.png")
    predictor = TraceLensPredictor(ckpt, device="cpu", backbone=_stub_backbone())
    result = predictor.predict(image)
    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.aigc_probability <= 1.0
    assert result.manipulation_probability is None
    assert result.reliability_score is None
    assert result.heatmap_path is None
    official = result.to_official_record()
    assert set(official.keys()) == {"image_path", "pred"}
    assert official["pred"] == result.aigc_probability


def test_factory_builds_real_predictor_and_does_not_return_mock(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "baseline.pt")
    predictor = create_predictor(
        mock=False,
        checkpoint=ckpt,
        device="cpu",
        backbone=_stub_backbone(),
    )
    assert isinstance(predictor, TraceLensPredictor)
    assert not isinstance(predictor, MockPredictor)
    result = predictor.predict(_rgb(tmp_path / "x.jpg"))
    assert 0.0 <= result.aigc_probability <= 1.0
    assert result.reliability_score is None
    assert result.manipulation_probability is None


def test_missing_checkpoint_fails_without_mock_fallback(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.pt"
    with pytest.raises(CheckpointError, match="not found"):
        TraceLensPredictor(missing, device="cpu", backbone=_stub_backbone())
    with pytest.raises(RealModelUnavailableError, match="checkpoint"):
        create_predictor(mock=False, checkpoint=missing, backbone=_stub_backbone())
    with pytest.raises(RealModelUnavailableError):
        create_predictor(mock=False, config={"inference": {"checkpoint": "", "device": "cpu"}})


def test_corrupt_checkpoint_fails_clearly(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"this is not a torch checkpoint")
    with pytest.raises(CheckpointError, match="Invalid or unreadable"):
        load_baseline_detector(corrupt)


def test_incompatible_checkpoint_fails_clearly(tmp_path: Path) -> None:
    wrong_keys = tmp_path / "wrong.pt"
    torch.save({"not_a_model": 1}, wrong_keys)
    with pytest.raises(CheckpointError, match="Invalid or unreadable"):
        load_baseline_detector(wrong_keys)

    incompatible = _write_checkpoint(
        tmp_path / "bad_dims.pt", embed_dim=16, num_patches=4, hidden_dim=8
    )
    with pytest.raises(CheckpointError, match="Incompatible checkpoint"):
        load_baseline_detector(incompatible)


def test_explicit_mock_is_unchanged() -> None:
    predictor = create_predictor(mock=True)
    assert isinstance(predictor, MockPredictor)


def test_resolve_device_defaults_to_cpu() -> None:
    assert resolve_device(None).type == "cpu"
    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(CheckpointError, match="CUDA is not available"):
            resolve_device("cuda")


def test_cli_real_baseline_writes_official_json_only(tmp_path: Path, monkeypatch) -> None:
    from src.models import backbone as backbone_mod

    monkeypatch.setattr(backbone_mod, "_load_hf_backbone", lambda name: _StubDINOv2())
    script = Path(__file__).resolve().parents[1] / "scripts" / "predict_directory.py"
    spec = importlib.util.spec_from_file_location("predict_directory_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ckpt = _write_checkpoint(tmp_path / "baseline.pt")
    images = tmp_path / "images"
    _rgb(images / "one.png")
    output_json = tmp_path / "official.json"
    rc = module.main(
        [
            "--input_dir",
            str(images),
            "--output_json",
            str(output_json),
            "--checkpoint",
            str(ckpt),
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload
    for record in payload:
        assert set(record.keys()) == {"image_path", "pred"}
        assert 0.0 <= record["pred"] <= 1.0


def test_directory_inference_official_json_from_real_predictor(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "baseline.pt")
    _rgb(tmp_path / "images" / "one.png")
    _rgb(tmp_path / "images" / "two.jpeg")
    predictor = TraceLensPredictor(ckpt, device="cpu", backbone=_stub_backbone())
    output_json = tmp_path / "official.json"
    records = run_directory_inference(tmp_path / "images", output_json, predictor)
    assert len(records) == 2
    for record in records:
        assert set(record.keys()) == {"image_path", "pred"}
        assert 0.0 <= record["pred"] <= 1.0
        assert "manipulation_probability" not in record
        assert "reliability_score" not in record
