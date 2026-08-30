"""Thin TraceLensPredictor hook for the Member 4 manipulation head.

``ManipulationPredictorAdapter.__call__(outputs, patch_features)`` matches
the Member 5 optional-module contract. It never reads, writes, or replaces
``final_logit`` / ``aigc_probability``. Official JSON stays exactly
``image_path`` and ``pred``.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image, UnidentifiedImageError

from src.models.manipulation import ManipulationHead
from src.models.manipulation_visualization import create_manipulation_overlay

__all__ = [
    "DEFAULT_HEATMAP_DIR",
    "ManipulationVisualizationError",
    "ManipulationPredictorAdapter",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEATMAP_DIR = REPO_ROOT / "outputs" / "heatmaps"


class ManipulationVisualizationError(RuntimeError):
    """Raised when a genuine heatmap overlay cannot be written."""


def _assert_heatmap_dir_allowed(heatmap_dir: Path) -> None:
    resolved = heatmap_dir.resolve()
    data_root = (REPO_ROOT / "data").resolve()
    ckpt_root = (REPO_ROOT / "checkpoints").resolve()
    if resolved == ckpt_root or ckpt_root in resolved.parents:
        raise ManipulationVisualizationError(
            f"Refusing to write heatmaps into the checkpoint directory: {resolved}"
        )
    if resolved == data_root or data_root in resolved.parents:
        raise ManipulationVisualizationError(
            f"Refusing to write heatmaps into the dataset directory: {resolved}"
        )


class ManipulationPredictorAdapter:
    """Adapter: ManipulationHead → ``{manipulation_probability, heatmap_path}``.

    Bind the analysed image with :meth:`bind_source_image` before each call
    (``TraceLensPredictor.predict`` does this automatically when the method
    exists). Visualization failures raise :class:`ManipulationVisualizationError`
    instead of inventing a heatmap or localisation.
    """

    def __init__(
        self,
        head: ManipulationHead,
        *,
        heatmap_dir: str | Path | None = None,
        source_image: Image.Image | None = None,
        source_image_path: str | Path | None = None,
    ) -> None:
        self.head = head
        self.heatmap_dir = Path(heatmap_dir) if heatmap_dir is not None else DEFAULT_HEATMAP_DIR
        self.last_error: str | None = None
        self._source_image = source_image
        self._source_image_path = Path(source_image_path) if source_image_path is not None else None

    def bind_source_image(self, image: str | Path | Image.Image) -> None:
        """Record the image currently being analysed (for a genuine overlay)."""
        if isinstance(image, Image.Image):
            self._source_image = image
            self._source_image_path = None
            return
        self._source_image = None
        self._source_image_path = Path(image)

    def __call__(
        self,
        outputs: Mapping[str, Any],
        patch_features: torch.Tensor,
    ) -> dict[str, Any]:
        del outputs  # required hook argument; AIGC tensors are never inspected
        self.last_error = None
        was_training = self.head.training
        self.head.eval()
        try:
            with torch.no_grad():
                result = self.head(patch_features)
            probability = result["manipulation_probability"]
            heatmap = result["heatmap"]
            if probability.numel() < 1:
                raise ManipulationVisualizationError("manipulation_probability is empty.")
            score = float(probability.reshape(-1)[0].detach().cpu())
            heatmap_path = self._write_overlay(heatmap)
            return {
                "manipulation_probability": score,
                "heatmap_path": heatmap_path,
            }
        except ManipulationVisualizationError as exc:
            self.last_error = str(exc)
            raise
        except Exception as exc:
            self.last_error = str(exc)
            raise ManipulationVisualizationError(
                f"Manipulation visualization failed: {exc}"
            ) from exc
        finally:
            if was_training:
                self.head.train()

    def _load_source_image(self) -> Image.Image:
        if self._source_image is not None:
            return self._source_image.convert("RGB")
        if self._source_image_path is None:
            raise ManipulationVisualizationError(
                "No analysed image is bound. Call bind_source_image() before "
                "generating a heatmap. A heatmap will not be invented."
            )
        path = self._source_image_path
        if not path.is_file():
            raise ManipulationVisualizationError(
                f"Analysed image not found: {path}. Heatmap not invented."
            )
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                rgb.load()
                return rgb
        except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as exc:
            raise ManipulationVisualizationError(
                f"Unreadable analysed image {path}: {exc}. Heatmap not invented."
            ) from exc

    def _write_overlay(self, heatmap: torch.Tensor) -> str:
        _assert_heatmap_dir_allowed(self.heatmap_dir)
        image = self._load_source_image()
        sample = heatmap[0] if heatmap.dim() == 4 else heatmap
        overlay = create_manipulation_overlay(
            image, sample, heatmap_is_logits=True
        )
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)
        dest = self.heatmap_dir / f"{uuid.uuid4().hex}.png"
        overlay.save(dest, format="PNG")
        if not dest.is_file() or dest.stat().st_size <= 0:
            raise ManipulationVisualizationError(
                f"Heatmap file was not written: {dest}"
            )
        return str(dest)
