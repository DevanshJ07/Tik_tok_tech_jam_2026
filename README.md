# TraceLens-R

TraceLens-R is a hackathon-scale image-forensics prototype for TikTok TechJam Problem 5. It predicts whether an image is fully AI-generated under common degradations, and secondarily detects and localises manipulation in otherwise authentic images.

## Current development status

Shared repository foundation, configuration loader, prediction-result contract, testing-only mock directory inference, and an internal labelled AIGC evaluation engine. The real model, training, and application UI are not implemented yet.

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
