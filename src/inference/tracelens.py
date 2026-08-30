"""Real baseline AIGC predictor (Member 5 inference over Member 2 modules).

Loads a frozen ``facebook/dinov2-small`` backbone and a Member 2 baseline
checkpoint. Reliability and manipulation stay unset until Members 3 and 4
attach modules. Those optional paths must never invent values or change
``aigc_probability`` (SPEC §14).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from src.inference.contracts import PredictionResult
from src.inference.preprocess import preprocess_image_path
from src.models.backbone import DEFAULT_BACKBONE_NAME, DINOv2Backbone
from src.models.baseline import EMBED_DIM, NUM_PATCHES, BaselineAIGCDetector
from src.training.train_baseline import load_checkpoint

EXPECTED_EMBED_DIM = EMBED_DIM
EXPECTED_NUM_PATCHES = NUM_PATCHES
CPU_DEVICE = "cpu"


class CheckpointError(RuntimeError):
    """Missing, unreadable, or contract-incompatible baseline checkpoint."""


@dataclass(frozen=True)
class InferenceSettings:
    checkpoint: Path
    device: str
    backbone_name: str


def resolve_device(device: str | None) -> torch.device:
    """Return a torch device. Default is CPU. CUDA is used only when requested."""
    requested = CPU_DEVICE if device is None or str(device).strip() == "" else str(device).strip()
    if requested == CPU_DEVICE:
        return torch.device(CPU_DEVICE)
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise CheckpointError(
                f"Device {requested!r} was requested but CUDA is not available. "
                "Use --device cpu (the safe default)."
            )
        return torch.device(requested)
    raise CheckpointError(
        f"Unsupported device {requested!r}. Use 'cpu' or 'cuda'."
    )


def resolve_checkpoint_path(
    checkpoint: str | Path | None,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Resolve the baseline checkpoint. Never substitutes mock mode."""
    if checkpoint is not None and str(checkpoint).strip():
        return Path(str(checkpoint).strip())
    if config is not None:
        inference = config.get("inference") if isinstance(config, Mapping) else None
        if isinstance(inference, Mapping):
            configured = inference.get("checkpoint")
            if configured is not None and str(configured).strip():
                return Path(str(configured).strip())
    raise CheckpointError(
        "No baseline checkpoint configured. Pass --checkpoint, set "
        "inference.checkpoint in the YAML config, or provide checkpoint= to "
        "create_predictor. Mock mode is not enabled automatically."
    )


def resolve_inference_settings(
    *,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> InferenceSettings:
    resolved_checkpoint = resolve_checkpoint_path(checkpoint, config)
    if device is not None and str(device).strip():
        resolved_device = str(device).strip()
    else:
        inference = config.get("inference") if isinstance(config, Mapping) else None
        raw = inference.get("device") if isinstance(inference, Mapping) else None
        resolved_device = str(raw).strip() if raw is not None and str(raw).strip() else CPU_DEVICE
    backbone_name = DEFAULT_BACKBONE_NAME
    if isinstance(config, Mapping):
        model = config.get("model")
        if isinstance(model, Mapping):
            name = model.get("backbone_name")
            if isinstance(name, str) and name.strip():
                backbone_name = name.strip()
    return InferenceSettings(
        checkpoint=resolved_checkpoint,
        device=resolved_device,
        backbone_name=backbone_name,
    )


def _validate_hparams(hparams: Mapping[str, Any], path: Path) -> None:
    embed_dim = int(hparams.get("embed_dim", EXPECTED_EMBED_DIM))
    num_patches = int(hparams.get("num_patches", EXPECTED_NUM_PATCHES))
    if embed_dim != EXPECTED_EMBED_DIM or num_patches != EXPECTED_NUM_PATCHES:
        raise CheckpointError(
            f"Incompatible checkpoint {path}: embed_dim={embed_dim}, "
            f"num_patches={num_patches}; expected {EXPECTED_EMBED_DIM} and "
            f"{EXPECTED_NUM_PATCHES}."
        )


def load_baseline_detector(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = CPU_DEVICE,
) -> BaselineAIGCDetector:
    """Load and validate a Member 2 baseline checkpoint."""
    path = Path(checkpoint)
    if not path.exists():
        raise CheckpointError(f"Baseline checkpoint not found: {path}")
    if not path.is_file():
        raise CheckpointError(f"Baseline checkpoint path is not a file: {path}")
    try:
        loaded = load_checkpoint(path, map_location=map_location)
    except FileNotFoundError as exc:
        raise CheckpointError(str(exc)) from exc
    except Exception as exc:
        raise CheckpointError(
            f"Invalid or unreadable baseline checkpoint {path}: {exc}"
        ) from exc
    _validate_hparams(loaded.get("model_hparams") or {}, path)
    model = loaded["model"]
    if not isinstance(model, BaselineAIGCDetector):
        raise CheckpointError(
            f"Checkpoint {path} did not reconstruct a BaselineAIGCDetector."
        )
    return model


class TraceLensPredictor:
    """Production Predictor: frozen DINOv2 + baseline heads only.

    Optional modules
    ----------------
    ``reliability_module`` and ``manipulation_module`` are hooks for Members
    3 and 4. They default to ``None``. A failed optional module is isolated
    and does not change ``aigc_probability``.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | None = CPU_DEVICE,
        backbone_name: str = DEFAULT_BACKBONE_NAME,
        backbone: Optional[DINOv2Backbone] = None,
        reliability_module: Any = None,
        manipulation_module: Any = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint)
        self.device = resolve_device(device)
        self.backbone_name = backbone_name
        self.reliability_module = reliability_module
        self.manipulation_module = manipulation_module

        self.detector = load_baseline_detector(
            self.checkpoint_path, map_location=self.device
        )
        self.detector.to(self.device)
        self.detector.eval()

        if backbone is None:
            self.backbone = DINOv2Backbone(
                model_name=self.backbone_name, device=self.device
            )
        else:
            self.backbone = backbone
        if not getattr(self.backbone, "is_frozen", False):
            raise CheckpointError("DINOv2 backbone must remain frozen at inference.")

    def predict(self, image_path: Path) -> PredictionResult:
        path = Path(image_path)
        pixel_values = preprocess_image_path(path).unsqueeze(0).to(
            device=self.device, dtype=torch.float32
        )
        self.detector.eval()
        with torch.no_grad():
            cls_features, patch_features = self.backbone.extract_features(pixel_values)
            outputs = self.detector(cls_features, patch_features)
        probability = float(outputs["aigc_probability"][0].detach().cpu())

        reliability_score = self._optional_reliability(outputs, patch_features)
        manipulation_probability, heatmap_path = self._optional_manipulation(
            outputs, patch_features
        )
        return PredictionResult(
            image_path=path.as_posix(),
            aigc_probability=probability,
            manipulation_probability=manipulation_probability,
            reliability_score=reliability_score,
            heatmap_path=heatmap_path,
        )

    def _optional_reliability(
        self,
        outputs: Mapping[str, torch.Tensor],
        patch_features: torch.Tensor,
    ) -> float | None:
        if self.reliability_module is None:
            return None
        try:
            result = self.reliability_module(outputs, patch_features)
        except Exception:
            return None
        if result is None:
            return None
        score = result.get("mean_reliability") if isinstance(result, Mapping) else result
        if score is None:
            return None
        value = float(score[0] if hasattr(score, "__getitem__") else score)
        return value

    def _optional_manipulation(
        self,
        outputs: Mapping[str, torch.Tensor],
        patch_features: torch.Tensor,
    ) -> tuple[float | None, str | None]:
        if self.manipulation_module is None:
            return None, None
        try:
            result = self.manipulation_module(outputs, patch_features)
        except Exception:
            return None, None
        if not isinstance(result, Mapping):
            return None, None
        probability = result.get("manipulation_probability")
        heatmap_path = result.get("heatmap_path")
        parsed = None if probability is None else float(
            probability[0] if hasattr(probability, "__getitem__") else probability
        )
        path = None if heatmap_path is None else str(heatmap_path)
        return parsed, path
