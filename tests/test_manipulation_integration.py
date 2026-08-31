"""Member 4 integration: masks, checkpoints, CPU device, adapter, heatmaps."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.data import manifests  # noqa: E402
from src.inference.contracts import PredictionResult  # noqa: E402
from src.inference.factory import create_predictor  # noqa: E402
from src.inference.tracelens import TraceLensPredictor  # noqa: E402
from src.models.backbone import DINOv2Backbone  # noqa: E402
from src.models.baseline import EMBED_DIM, NUM_PATCHES, BaselineAIGCDetector  # noqa: E402
from src.models.manipulation import ManipulationHead  # noqa: E402
from src.models.manipulation_adapter import (  # noqa: E402
    ManipulationPredictorAdapter,
    ManipulationVisualizationError,
)
from src.training.train_baseline import TrainConfig, save_checkpoint  # noqa: E402
from src.training.train_manipulation import (  # noqa: E402
    ManipulationCheckpointError,
    ManipulationMaskError,
    filter_manipulation_batch,
    load_manipulation_checkpoint,
    resolve_manipulation_device,
    save_manipulation_checkpoint,
    train_one_epoch,
    validate_manipulation_masks,
)


class _StubOutput:
    def __init__(self, last_hidden_state: "torch.Tensor") -> None:
        self.last_hidden_state = last_hidden_state


class _StubDINOv2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, EMBED_DIM)
        self.token_bias = nn.Parameter(torch.zeros(1 + NUM_PATCHES, EMBED_DIM))

    def forward(self, pixel_values=None, return_dict=True, **_kwargs):
        pooled = pixel_values.mean(dim=(2, 3))
        base = self.proj(pooled).unsqueeze(1)
        return _StubOutput(base + self.token_bias.unsqueeze(0))


def _stub_backbone() -> DINOv2Backbone:
    return DINOv2Backbone(model_name="stub", device="cpu", model=_StubDINOv2())


def _write_baseline_checkpoint(path: Path) -> Path:
    torch.manual_seed(0)
    model = BaselineAIGCDetector(dropout=0.0)
    model.eval()
    save_checkpoint(
        path,
        model=model,
        optimizer=None,
        epoch=1,
        global_step=1,
        config=TrainConfig(dropout=0.0),
    )
    return path


def _rgb(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 40), (30, 80, 120)).save(path)
    return path


def test_label_2_all_zero_mask_is_rejected() -> None:
    mask = torch.zeros(2, 1, 224, 224)
    label = torch.tensor([2, 2])
    with pytest.raises(ManipulationMaskError, match="Label 2"):
        validate_manipulation_masks(mask, label)


def test_missing_mask_is_rejected() -> None:
    with pytest.raises(ManipulationMaskError, match="missing"):
        validate_manipulation_masks(None, torch.tensor([2]))


def test_label_0_mask_is_normalized_to_zero() -> None:
    mask = torch.ones(2, 1, 224, 224)
    label = torch.tensor([0, 0])
    normalized = validate_manipulation_masks(mask, label)
    assert torch.count_nonzero(normalized) == 0


def test_label_1_remains_excluded_from_mask_validation() -> None:
    mask = torch.ones(1, 1, 224, 224)
    with pytest.raises(ManipulationMaskError, match="Label 1"):
        validate_manipulation_masks(mask, torch.tensor([1]))


def test_filter_rejects_label_2_without_real_mask() -> None:
    features = torch.randn(2, NUM_PATCHES, EMBED_DIM)
    mask = torch.zeros(2, 1, 224, 224)
    label = torch.tensor([2, 0])
    with pytest.raises(ManipulationMaskError, match="Label 2"):
        filter_manipulation_batch(features, mask, label)


def test_checkpoint_roundtrip_preserves_outputs_on_cpu(tmp_path: Path) -> None:
    torch.manual_seed(3)
    head = ManipulationHead()
    head.eval()
    features = torch.randn(2, NUM_PATCHES, EMBED_DIM)
    with torch.no_grad():
        before = head(features)
    ckpt = tmp_path / "manipulation.pt"
    save_manipulation_checkpoint(ckpt, model=head, epoch=4, extra={"note": "unit"})
    loaded = load_manipulation_checkpoint(ckpt, map_location="cpu")
    restored = loaded["model"]
    restored.eval()
    assert loaded["model_hparams"]["embedding_dim"] == head.embedding_dim
    assert loaded["model_hparams"]["hidden_dim"] == head.hidden_dim
    assert loaded["model_hparams"]["patch_grid_size"] == head.patch_grid_size
    assert loaded["model_hparams"]["heatmap_size"] == head.heatmap_size
    assert loaded["model_hparams"]["top_k"] == head.top_k
    assert loaded["epoch"] == 4
    with torch.no_grad():
        after = restored(features)
    for key in before:
        assert torch.allclose(before[key], after[key], atol=1e-6)


def test_incompatible_checkpoint_is_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pt"
    with pytest.raises(ManipulationCheckpointError, match="not found"):
        load_manipulation_checkpoint(missing)
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    with pytest.raises(ManipulationCheckpointError, match="Invalid or unreadable"):
        load_manipulation_checkpoint(corrupt)
    wrong = tmp_path / "wrong.pt"
    torch.save({"not": "manipulation"}, wrong)
    with pytest.raises(ManipulationCheckpointError, match="missing"):
        load_manipulation_checkpoint(wrong)
    bad = tmp_path / "bad_version.pt"
    torch.save(
        {
            "format_version": 99,
            "kind": "manipulation_head",
            "model_state_dict": ManipulationHead().state_dict(),
            "model_hparams": {"embedding_dim": 384, "patch_grid_size": 16},
        },
        bad,
    )
    with pytest.raises(ManipulationCheckpointError, match="format_version"):
        load_manipulation_checkpoint(bad)
    dims = tmp_path / "bad_dims.pt"
    tiny = ManipulationHead(embedding_dim=16, hidden_dim=8, patch_grid_size=4, top_k=2)
    # Save raw payload with incompatible hparams so load validation fires.
    torch.save(
        {
            "format_version": 1,
            "kind": "manipulation_head",
            "model_state_dict": tiny.state_dict(),
            "model_hparams": {"embedding_dim": 16, "patch_grid_size": 4, "hidden_dim": 8, "top_k": 2},
        },
        dims,
    )
    with pytest.raises(ManipulationCheckpointError, match="Incompatible"):
        load_manipulation_checkpoint(dims)


def test_resolve_device_defaults_to_cpu() -> None:
    assert resolve_manipulation_device(None).type == "cpu"
    assert resolve_manipulation_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_manipulation_device("cuda")


def test_train_one_epoch_runs_on_cpu() -> None:
    head = ManipulationHead()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
    mask = torch.zeros(2, 1, 224, 224)
    mask[1, 0, 16:48, 16:48] = 1.0
    batches = [
        {
            "patch_features": torch.randn(2, NUM_PATCHES, EMBED_DIM),
            "mask": mask,
            "label": torch.tensor([0, 2]),
        }
    ]
    stats = train_one_epoch(head, batches, optimizer, device="cpu")
    assert stats.num_batches == 1
    assert next(head.parameters()).device.type == "cpu"


def test_adapter_contract_and_heatmap_file(tmp_path: Path) -> None:
    image = _rgb(tmp_path / "scene.png")
    head = ManipulationHead()
    adapter = ManipulationPredictorAdapter(head, heatmap_dir=tmp_path / "heatmaps")
    adapter.bind_source_image(image)
    features = torch.randn(1, NUM_PATCHES, EMBED_DIM)
    aigc = torch.tensor([0.31])
    outputs = {"final_logit": torch.tensor([0.2]), "aigc_probability": aigc.clone()}
    result = adapter(outputs, features)
    assert set(result.keys()) == {"manipulation_probability", "heatmap_path"}
    assert 0.0 <= float(result["manipulation_probability"]) <= 1.0
    heatmap_path = Path(result["heatmap_path"])
    assert heatmap_path.is_file()
    assert heatmap_path.stat().st_size > 0
    assert heatmap_path.parent == (tmp_path / "heatmaps").resolve() or heatmap_path.parent == tmp_path / "heatmaps"
    with Image.open(heatmap_path) as written:
        written.verify()
    assert torch.equal(outputs["aigc_probability"], aigc)
    assert float(outputs["final_logit"][0]) == pytest.approx(0.2)


def test_adapter_refuses_checkpoint_directory(tmp_path: Path) -> None:
    repo_checkpoints = Path(__file__).resolve().parents[1] / "checkpoints"
    head = ManipulationHead()
    adapter = ManipulationPredictorAdapter(head, heatmap_dir=repo_checkpoints)
    adapter.bind_source_image(Image.new("RGB", (16, 16), (1, 2, 3)))
    with pytest.raises(ManipulationVisualizationError, match="checkpoint"):
        adapter({}, torch.randn(1, NUM_PATCHES, EMBED_DIM))


def test_adapter_reports_missing_image_instead_of_inventing(tmp_path: Path) -> None:
    adapter = ManipulationPredictorAdapter(ManipulationHead(), heatmap_dir=tmp_path)
    with pytest.raises(ManipulationVisualizationError, match="not be invented"):
        adapter({}, torch.randn(1, NUM_PATCHES, EMBED_DIM))


def test_predictor_aigc_unchanged_when_adapter_attached(tmp_path: Path) -> None:
    ckpt = _write_baseline_checkpoint(tmp_path / "baseline.pt")
    image = _rgb(tmp_path / "photo.png")
    predictor = TraceLensPredictor(ckpt, device="cpu", backbone=_stub_backbone())
    without = predictor.predict(image)
    predictor.manipulation_module = ManipulationPredictorAdapter(
        ManipulationHead(), heatmap_dir=tmp_path / "heatmaps"
    )
    with_mod = predictor.predict(image)
    assert without.aigc_probability == with_mod.aigc_probability
    assert set(without.to_official_record().keys()) == {"image_path", "pred"}
    assert set(with_mod.to_official_record().keys()) == {"image_path", "pred"}
    assert with_mod.manipulation_probability is not None
    assert with_mod.heatmap_path is not None
    assert Path(with_mod.heatmap_path).is_file()
    assert isinstance(with_mod, PredictionResult)
    factory_mock = create_predictor(mock=True)
    assert factory_mock.predict(image).to_official_record().keys() == {"image_path", "pred"}


def _write_manipulation_checkpoint(path: Path) -> Path:
    head = ManipulationHead()
    save_manipulation_checkpoint(path, model=head, epoch=1)
    return path


def test_factory_empty_manipulation_checkpoint_stays_absent(tmp_path: Path) -> None:
    baseline = _write_baseline_checkpoint(tmp_path / "baseline.pt")
    predictor = create_predictor(
        mock=False,
        checkpoint=baseline,
        manipulation_checkpoint="",
        config={
            "inference": {
                "checkpoint": str(baseline),
                "manipulation_checkpoint": str(tmp_path / "should-not-load.pt"),
                "device": "cpu",
            }
        },
        backbone=_stub_backbone(),
    )
    assert isinstance(predictor, TraceLensPredictor)
    assert predictor.manipulation_module is None


def test_factory_attaches_manipulation_checkpoint(tmp_path: Path) -> None:
    from src.inference.predictor import MockPredictor

    baseline = _write_baseline_checkpoint(tmp_path / "baseline.pt")
    manip = _write_manipulation_checkpoint(tmp_path / "manip.pt")
    image = _rgb(tmp_path / "photo.png")
    backbone = _stub_backbone()
    without = create_predictor(
        mock=False, checkpoint=baseline, backbone=backbone
    )
    assert without.manipulation_module is None
    with_mod = create_predictor(
        mock=False,
        checkpoint=baseline,
        manipulation_checkpoint=manip,
        backbone=backbone,
    )
    assert not isinstance(with_mod, MockPredictor)
    assert isinstance(with_mod, TraceLensPredictor)
    assert with_mod.manipulation_module is not None
    r0 = without.predict(image)
    r1 = with_mod.predict(image)
    assert r0.aigc_probability == r1.aigc_probability
    assert 0.0 <= r1.manipulation_probability <= 1.0
    assert r1.heatmap_path is not None
    assert Path(r1.heatmap_path).is_file()
    Image.open(r1.heatmap_path).verify()
    assert r1.reliability_score is None
    assert set(r1.to_official_record().keys()) == {"image_path", "pred"}
    assert with_mod.device.type == "cpu"


def test_factory_missing_manipulation_checkpoint_raises(tmp_path: Path) -> None:
    from src.inference.factory import RealModelUnavailableError

    baseline = _write_baseline_checkpoint(tmp_path / "baseline.pt")
    with pytest.raises(RealModelUnavailableError, match="Manipulation checkpoint"):
        create_predictor(
            mock=False,
            checkpoint=baseline,
            manipulation_checkpoint=tmp_path / "missing.pt",
            backbone=_stub_backbone(),
        )


def test_factory_corrupt_manipulation_checkpoint_raises(tmp_path: Path) -> None:
    from src.inference.factory import RealModelUnavailableError

    baseline = _write_baseline_checkpoint(tmp_path / "baseline.pt")
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not-a-checkpoint")
    with pytest.raises(RealModelUnavailableError, match="Manipulation checkpoint"):
        create_predictor(
            mock=False,
            checkpoint=baseline,
            manipulation_checkpoint=bad,
            backbone=_stub_backbone(),
        )


def test_factory_incompatible_manipulation_checkpoint_raises(tmp_path: Path) -> None:
    from src.inference.factory import RealModelUnavailableError

    baseline = _write_baseline_checkpoint(tmp_path / "baseline.pt")
    bad = tmp_path / "incompatible.pt"
    torch.save(
        {
            "format_version": 1,
            "kind": "manipulation_head",
            "model_state_dict": ManipulationHead(
                embedding_dim=16, hidden_dim=8, patch_grid_size=4, top_k=2
            ).state_dict(),
            "model_hparams": {
                "embedding_dim": 16,
                "patch_grid_size": 4,
                "hidden_dim": 8,
                "top_k": 2,
            },
        },
        bad,
    )
    with pytest.raises(RealModelUnavailableError, match="Manipulation checkpoint"):
        create_predictor(
            mock=False,
            checkpoint=baseline,
            manipulation_checkpoint=bad,
            backbone=_stub_backbone(),
        )
