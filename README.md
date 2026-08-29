# TraceLens-R

TraceLens-R is a hackathon-scale image-forensics prototype for TikTok TechJam Problem 5. It predicts whether an image is fully AI-generated under common degradations, and secondarily detects and localises manipulation in otherwise authentic images.

## Current development status

Shared contracts, mock directory inference, labelled evaluation, and a Streamlit screening shell. The real model checkpoint is not connected. Reliability, manipulation, and heatmap modules are not connected.

## Installation

```
python -m pip install -r requirements.txt
```

## Streamlit application

```
python -m streamlit run app.py
```

Mock mode is **off by default** and is testing-only. It requires both the mock toggle and a confirmation checkbox. Mock scores are not model predictions and must not be reported as detection performance. The UI never falls back to mock automatically.

### Connected capabilities

| Capability | Status |
| --- | --- |
| AIGC predictor | Not connected. Testing-only `MockPredictor` if mock mode is explicitly enabled and confirmed |
| Reliability | Not connected |
| Manipulation | Not connected |
| Heatmap | Not connected; shown only if a real `heatmap_path` is provided later |

Real inference will be connected in `src/inference/factory.py` after model components are delivered.

## Mock inference (testing only)

The real model is not wired. Directory-to-JSON inference refuses to run unless ``--mock`` is passed explicitly. There is no silent fallback. Mock scores are hash-derived stubs, not detector output.

```
python scripts/predict_directory.py --input_dir <directory> --output_json <file> --mock
```

Official inference JSON contains only `image_path` and `pred`. It is not a labelled evaluation file.

## Internal evaluation

Score labelled prediction CSV/JSON (not official inference JSON, and not mock data):

```
python scripts/evaluate.py --predictions <csv-or-json> --output_dir <directory> --threshold 0.5
```

Writes `summary.json`, `by_condition.csv`, `by_family.csv`, and `errors.csv`. Every reported metric is computed from the input file.

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
