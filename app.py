"""TraceLens-R Streamlit screening shell.

Mock mode is off by default. Real baseline inference uses a configured
Member 2 checkpoint. This page must not fall back to MockPredictor automatically.
"""

from __future__ import annotations

import streamlit as st

from src.config import ConfigError, load_config
from src.inference.factory import RealModelUnavailableError
from src.ui.presentation import (
    MOCK_BANNER,
    SCREENING_DISCLAIMER,
    PresentationError,
    capability_status,
    result_view,
    validate_threshold,
)
from src.ui.service import (
    MockConfirmationRequiredError,
    UploadError,
    analyse_bytes,
    resolve_predictor,
)

SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]


def main() -> None:
    st.set_page_config(
        page_title="TraceLens-R",
        page_icon="◇",
        layout="wide",
    )
    _inject_styles()
    _render_header()

    config = _safe_config()
    mock_enabled, mock_acknowledged, threshold, checkpoint, device = _render_sidebar(config)

    if mock_enabled:
        st.markdown(
            f'<div class="mock-banner">{MOCK_BANNER}</div>',
            unsafe_allow_html=True,
        )

    uploaded = st.file_uploader(
        "Upload an image",
        type=SUPPORTED_TYPES,
        help="JPG, JPEG, PNG, WEBP, or BMP. Analysis does not run until you click Analyse image.",
    )
    if uploaded is not None:
        st.image(uploaded.getvalue(), caption=uploaded.name, use_container_width=True)

    analyse = st.button("Analyse image", type="primary")
    if analyse:
        _run_analysis(
            uploaded,
            mock_enabled=mock_enabled,
            mock_acknowledged=mock_acknowledged,
            threshold=threshold,
            checkpoint=checkpoint,
            device=device,
            config=config,
        )

    result_state = st.session_state.get("analysis")
    if result_state is not None:
        _render_results(result_state, mock_enabled=result_state["mock"])
    else:
        st.info("Upload an image, then click **Analyse image**. Nothing runs automatically.")

    _render_capabilities(mock_enabled, result_state)


def _safe_config() -> dict:
    try:
        return load_config()
    except ConfigError as exc:
        st.warning(f"Configuration could not be loaded: {exc}")
        return {
            "model": {
                "backbone_name": "facebook/dinov2-small",
                "image_size": 224,
            }
        }


def _render_header() -> None:
    st.title("TraceLens-R")
    st.subheader("Robust AI-image and local-manipulation screening")
    st.caption(
        "Prototype disclaimer: this is a research screening tool, not a legal "
        "authenticity authority. Outputs are experimental. Not proof of authenticity."
    )


def _render_sidebar(config: dict) -> tuple[bool, bool, float, str, str]:
    model = config.get("model", {})
    inference = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}
    configured_checkpoint = str(inference.get("checkpoint") or "")
    configured_device = str(inference.get("device") or "cpu")
    if configured_device not in {"cpu", "cuda"} and not configured_device.startswith("cuda:"):
        configured_device = "cpu"

    st.sidebar.header("Configuration")
    st.sidebar.write(f"**Backbone:** `{model.get('backbone_name', 'unknown')}`")
    st.sidebar.write(f"**Image size:** `{model.get('image_size', 'unknown')}`")
    checkpoint = st.sidebar.text_input(
        "Baseline checkpoint",
        value=configured_checkpoint,
        help="Path to a Member 2 baseline .pt file. Required for real inference.",
    )
    device_options = ["cpu", "cuda"]
    device_index = 0 if configured_device == "cpu" else 1
    device = st.sidebar.selectbox(
        "Device",
        device_options,
        index=device_index,
        help="Safe default is CPU. CUDA is used only when selected here.",
    )
    threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    st.sidebar.caption("Display threshold for the provisional model indication. Not a tuned operating point.")
    st.sidebar.caption(
        "Reliability and manipulation await Member 3 and Member 4 models. "
        "They are never invented from the baseline checkpoint."
    )

    st.sidebar.header("Testing-only mock")
    mock_enabled = st.sidebar.toggle("Enable mock mode", value=False)
    mock_acknowledged = False
    if mock_enabled:
        st.sidebar.warning(
            "Mock mode is testing-only. Scores are not model predictions and "
            "must not be reported as detection performance."
        )
        mock_acknowledged = st.sidebar.checkbox(
            "I acknowledge that mock results are not model predictions",
            value=False,
        )
    return mock_enabled, mock_acknowledged, float(threshold), checkpoint, str(device)


def _run_analysis(
    uploaded,
    *,
    mock_enabled: bool,
    mock_acknowledged: bool,
    threshold: float,
    checkpoint: str,
    device: str,
    config: dict,
) -> None:
    st.session_state.pop("analysis", None)
    if uploaded is None:
        st.error("Upload an image before analysing.")
        return
    try:
        validate_threshold(threshold)
        predictor = resolve_predictor(
            mock_enabled=mock_enabled,
            mock_acknowledged=mock_acknowledged,
            checkpoint=checkpoint,
            device=device,
            config=config,
        )
        result, official_json, detailed_json = analyse_bytes(
            uploaded.getvalue(),
            uploaded.name,
            predictor,
        )
        st.session_state["analysis"] = {
            "result": result,
            "official_json": official_json,
            "detailed_json": detailed_json,
            "threshold": threshold,
            "mock": mock_enabled,
            "view": result_view(result, threshold, mock=mock_enabled),
        }
    except MockConfirmationRequiredError as exc:
        st.error(str(exc))
    except RealModelUnavailableError as exc:
        st.error(str(exc))
        st.info(
            "The page remains usable. Upload still works. Mock mode will not "
            "turn on automatically. Set a valid baseline checkpoint path to "
            "run TraceLensPredictor. Reliability and manipulation stay unavailable "
            "until those models are connected."
        )
    except (UploadError, PresentationError) as exc:
        st.error(str(exc))


def _render_results(state: dict, *, mock_enabled: bool) -> None:
    view = state["view"]
    if mock_enabled:
        st.warning(view["testing_banner"])
        st.caption("Mock scores are UI stubs, not detection performance.")

    st.header("Results")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("AIGC probability", view["aigc_probability_label"])
    col_b.metric("Manipulation probability", view["manipulation_probability_label"])
    col_c.metric("Reliability score", view["reliability_score_label"])
    st.markdown(f"**Provisional model indication:** {view['category']}")
    st.caption(view["category_explanation"])
    st.caption(view["disclaimer"] + " " + SCREENING_DISCLAIMER)

    st.subheader("Heatmap")
    if view["heatmap_available"]:
        st.image(view["heatmap_detail"], caption="Manipulation localisation")
    else:
        st.info(view["heatmap_detail"])

    st.subheader("Downloads")
    st.download_button(
        "Download official JSON",
        data=state["official_json"],
        file_name="tracelens_official.json",
        mime="application/json",
        help="Official schema: exactly image_path and pred.",
    )
    st.download_button(
        "Download detailed JSON (optional / internal)",
        data=state["detailed_json"],
        file_name="tracelens_detailed_internal.json",
        mime="application/json",
        help="Optional internal record. Not the official submission schema.",
    )


def _render_capabilities(mock_enabled: bool, result_state: dict | None) -> None:
    st.header("Technical status")
    result = None if result_state is None else result_state["result"]
    status = capability_status(mock=mock_enabled, result=result)
    st.table(
        {
            "Capability": ["AIGC predictor", "Reliability", "Manipulation", "Heatmap"],
            "Status": [
                status["aigc_predictor"],
                status["reliability"],
                status["manipulation"],
                status["heatmap"],
            ],
        }
    )
    st.caption(
        "AIGC uses the Member 2 baseline through src/inference/factory.py. "
        "Reliability (Member 3) and manipulation / heatmap (Member 4) are awaiting "
        "their models. Optional modules must not change AIGC probability."
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .mock-banner {
            background: linear-gradient(90deg, #9a3412, #c2410c);
            color: #fff7ed;
            padding: 0.9rem 1rem;
            border-radius: 0.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            border: 1px solid #fdba74;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
