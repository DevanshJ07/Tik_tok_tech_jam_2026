# TraceLens-R

TraceLens-R is a hackathon-scale image-forensics prototype for TikTok TechJam Problem 5. It predicts whether an image is fully AI-generated under common degradations, and secondarily detects and localises manipulation in otherwise authentic images.

## Current development status

Shared repository foundation and technical contract only. Model, dataset, training, evaluation, inference, and application logic are not implemented yet.

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
