# TraceLens-R

TraceLens-R is a hackathon-scale image-forensics prototype for TikTok TechJam Problem 5. It predicts whether an image is fully AI-generated under common degradations, and secondarily detects and localises manipulation in otherwise authentic images.

## Current development status

Baseline AIGC inference is connected through `TraceLensPredictor`. A valid Member 2 baseline checkpoint is required. This repository does **not** include a trained checkpoint or any performance results.

Reliability, manipulation, and heatmap modules are pending Members 3 and 4. Those fields stay unavailable and are never invented from the baseline checkpoint.

## Installation

```
python -m pip install -r requirements.txt
```

## Checkpoint configuration

Real inference needs a Member 2 baseline `.pt` file. Provide it in one of these ways:

- Streamlit sidebar field **Baseline checkpoint**
- `inference.checkpoint` in `configs/default.yaml`
- `--checkpoint` on the directory CLI
- `checkpoint=` when calling `create_predictor(mock=False)`

`inference.device` defaults to `cpu`. Use `cuda` only when you explicitly select it (sidebar Device, `--device cuda`, or `device=`). CUDA is never implied. If CUDA is requested but unavailable, inference fails clearly instead of falling back to mock.

A missing, invalid, or incompatible checkpoint raises `RealModelUnavailableError`. Mock mode never turns on automatically.

## Streamlit application

```
python -m streamlit run app.py
```

Set the baseline checkpoint path and leave **Device** on `cpu` unless you intend to use CUDA. Mock mode is **off by default** and is testing-only. It requires both the mock toggle and a confirmation checkbox. Mock scores are not model predictions and must not be reported as detection performance.

### Connected capabilities

| Capability | Status |
| --- | --- |
| AIGC predictor | Connected (Member 2 baseline) when a valid checkpoint is configured |
| Reliability | Pending Member 3 |
| Manipulation | Pending Member 4 |
| Heatmap | Pending Member 4; shown only if a real `heatmap_path` is provided later |

## Real directory inference

```
python scripts/predict_directory.py --input_dir <directory> --output_json <file> --checkpoint <baseline.pt> --device cpu
```

`--device cuda` is optional and must be requested explicitly. Official inference JSON contains only `image_path` and `pred`. It is not a labelled evaluation file.

## Mock inference (testing only)

`--mock` selects testing-only `MockPredictor`. There is no silent fallback. Mock scores are hash-derived stubs, not detector output.

```
python scripts/predict_directory.py --input_dir <directory> --output_json <file> --mock
```

## Internal evaluation

Score labelled prediction CSV/JSON (not official inference JSON, and not mock data):

```
python scripts/evaluate.py --predictions <csv-or-json> --output_dir <directory> --threshold 0.5
```

Writes `summary.json`, `by_condition.csv`, `by_family.csv`, and `errors.csv`. Every reported metric is computed from the input file. This repository currently ships no trained checkpoint and no computed performance numbers.

## Member responsibilities

| Member | Responsibility |
| --- | --- |
| Member 1 | Data loading, labels, preprocessing, and official transformations |
| Member 2 | Frozen DINOv2 backbone, global head, patch evidence head, and baseline AIGC outputs |
| Member 3 | Reliability head and reliability-weighted AIGC fusion |
| Member 4 | Manipulation score and localisation heatmap (must not alter AIGC probability) |
| Member 5 | Integration, official JSON inference, evaluation wiring, and reporting discipline |

## Warning

This project is a prototype, not a legal authenticity authority. Outputs are experimental research scores, not proof that an image is real or fake.

## Specification

See [docs/SPEC.md](docs/SPEC.md).
