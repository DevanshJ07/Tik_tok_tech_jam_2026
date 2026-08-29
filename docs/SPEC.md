# TraceLens-R Technical Specification

This document is the authoritative technical contract for TraceLens-R. Implementation, evaluation reporting, and member module interfaces MUST conform to this specification. Where code and this document disagree, this document wins until the team explicitly amends it.

## 1. Product objective and scope

TraceLens-R is a hackathon-scale image-forensics prototype for TikTok TechJam Problem 5.

**Mandatory purpose:** predict whether an image is fully AI-generated (AIGC), including after JPEG compression, blur, resizing, Gaussian noise, colour adjustment, and centre cropping.

**Secondary feature:** detect and localise manipulation inside an otherwise authentic image.

**Out of scope for this contract:**

- Any claim of legal authenticity, provenance, or forensic authority.
- Training, calibration, threshold selection, or model selection on the protected WildFake demonstration dataset.
- Treating locally tampered images (label 2) as fully AI-generated.

**Fixed model design (summary):**

1. Input RGB image resized to 224×224.
2. Frozen `facebook/dinov2-small` backbone.
3. CLS features have shape `[B, 384]`.
4. Patch features have shape `[B, 256, 384]`, representing a 16×16 grid.
5. A global head predicts whole-image AIGC evidence.
6. A patch evidence head predicts AIGC evidence for each patch.
7. A reliability head estimates which patch evidence remains reliable after degradation.
8. Reliability-weighted patch evidence is combined with global evidence.
9. A separate manipulation head produces an image-level manipulation score and a 224×224 heatmap.
10. The manipulation module MUST NOT alter the required AIGC probability.

## 2. Mandatory versus secondary outputs

### Mandatory (official AIGC detector)

- `final_logit`
- `aigc_probability` = `sigmoid(final_logit)`
- Official JSON with exactly `image_path` and `pred` (see §11)

These outputs MUST remain available even if optional modules fail (see §14).

### Secondary (manipulation)

- Image-level `manipulation_probability`
- Patch-level `patch_mask_logits`
- Spatial `heatmap` at 224×224

Secondary outputs MUST NOT be required for official AIGC inference and MUST NOT modify `aigc_probability`.

## 3. Dataset labels and task assignment

| Label | Meaning | Official AIGC detector | Manipulation module |
| --- | --- | --- | --- |
| `0` | Authentic | Train | Train |
| `1` | Fully synthetic | Train | Do not use as a manipulation-positive class |
| `2` | Locally tampered | Do not treat as fully AI-generated | Train |

Rules:

- Labels `0` and `1` train the official AIGC detector (authentic vs fully synthetic).
- Labels `0` and `2` train the manipulation module (authentic vs locally tampered).
- Label `2` MUST NOT automatically be treated as fully AI-generated for the official AIGC task.
- AIGC training MUST NOT recode label `2` to `1`.

## 4. Protected-data restrictions

The protected WildFake demonstration dataset MUST NEVER be used for:

- training
- calibration
- threshold selection
- model selection

Demonstration use is permitted only as a frozen, non-tuning showcase after all training, calibration, thresholds, and model choices are fixed on unprotected data.

## 5. Input preprocessing

All model inputs MUST satisfy:

- Colour space: RGB
- Spatial size: 224×224
- Channel layout: `[B, 3, 224, 224]` after tensor conversion
- Normalisation: DINOv2 / ImageNet statistics

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Official robustness transformations in §12 are applied **before** the 224×224 RGB resize-and-normalise pipeline, except where a transform itself includes a resize (resize-down then upscale, and centre-crop-then-resize). After those transforms, the tensor still MUST be RGB 224×224 with the normalisation above.

## 6. Backbone output shapes

Backbone: frozen `facebook/dinov2-small`.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| CLS features | `[B, 384]` | Global image embedding |
| Patch features | `[B, 256, 384]` | 16×16 patch grid, row-major flattened |

`embedding_dimension = 384`  
`patch_grid_size = 16`  
`256 = 16 × 16`

The backbone MUST remain frozen (`backbone_frozen: true`).

## 7. Required baseline output dictionary

The baseline AIGC path (global head + patch evidence head, no reliability module) MUST return at least:

| Key | Shape | Description |
| --- | --- | --- |
| `global_logit` | `[B]` | Whole-image AIGC evidence from the CLS features |
| `patch_logits` | `[B, 256]` | Per-patch AIGC evidence |
| `patch_mean_logit` | `[B]` | Mean of `patch_logits` over the 256 patches |
| `final_logit` | `[B]` | Baseline fusion logit |
| `aigc_probability` | `[B]` | `sigmoid(final_logit)` |

Baseline fusion when reliability is unavailable:

```
final_logit = 0.5 × global_logit + 0.5 × patch_mean_logit
```

This baseline path is the fallback required by §14.

## 8. Required reliability output dictionary

When the reliability module is active, it MUST return at least:

| Key | Shape | Description |
| --- | --- | --- |
| `reliability` | `[B, 256]` | Per-patch reliability in `[0, 1]` |
| `weighted_patch_logit` | `[B]` | Reliability-weighted aggregation of `patch_logits` |
| `final_logit` | `[B]` | Official fused logit (see §10) |
| `aigc_probability` | `[B]` | `sigmoid(final_logit)` |
| `mean_reliability` | `[B]` | Mean of `reliability` over the 256 patches |

`weighted_patch_logit` is the reliability-weighted combination of `patch_logits`. Weights MUST be the per-patch `reliability` values. If all reliabilities in a sample are zero, implementations MUST fall back to uniform weights over the 256 patches so `weighted_patch_logit` remains defined.

## 9. Required manipulation output dictionary

The manipulation module MUST return at least:

| Key | Shape | Description |
| --- | --- | --- |
| `manipulation_probability` | `[B]` | Image-level local-tamper score in `[0, 1]` |
| `patch_mask_logits` | `[B, 256]` | Per-patch manipulation evidence |
| `heatmap` | `[B, 1, 224, 224]` | Spatial localisation map at input resolution |

The manipulation module MUST NOT write to, rescale, or replace `final_logit` or `aigc_probability`.

## 10. Final AIGC calculation

When reliability-weighted patch evidence is available:

```
final_logit = 0.5 × global_logit + 0.5 × weighted_patch_logit
aigc_probability = sigmoid(final_logit)
```

Weights `global_weight = 0.5` and `patch_weight = 0.5` are fixed in `configs/default.yaml` unless the team amends this contract.

`pred` in the official JSON is `aigc_probability` for that image.

## 11. Official JSON schema

Official inference output MUST be JSON containing **exactly** these two keys and no others:

```json
{
  "image_path": "<path or identifier of the input image>",
  "pred": <float aigc_probability>
}
```

- `image_path`: string
- `pred`: floating-point AIGC probability in `[0, 1]`

Manipulation scores, heatmaps, logits, and diagnostics MUST NOT appear in the official JSON. They may be written to separate, non-official artefacts under `outputs/`.

## 12. Official transformation settings

Robustness evaluation MUST use these exact settings. Do not substitute nearby values.

### JPEG compression

Qualities: `90`, `70`, `50`, `30`

### Gaussian blur

Sigma: `0.5`, `1.0`, `2.0`

### Resize then upscale

Downscale factors: `0.5×` and `0.25×`, each followed by upscaling back to the evaluation size used before the standard 224×224 model preprocess.

### Gaussian noise

Sigma: `0.02`, `0.05`, `0.10`  
Noise is applied on images scaled to `[0, 1]` unless a later implementation note records an equivalent integer-scale conversion. Reported protocols MUST state the scale actually used; they MUST still use these sigma values.

### Colour adjustment

Brightness, contrast, and saturation each perturbed by `±20%` (factors `0.80` and `1.20`).

### Centre crop

Retain the central `80%` of each spatial dimension, then resize back to the evaluation size used before the standard 224×224 model preprocess.

## 13. Module ownership (Members 1–5)

| Member | Ownership | Primary paths |
| --- | --- | --- |
| Member 1 | Data: manifests, loaders, label routing (0/1 vs 0/2), RGB 224×224 preprocess, ImageNet/DINOv2 normalisation, official transformations | `src/data/`, `data/manifests/`, `data/samples/` |
| Member 2 | Frozen DINOv2-small backbone; global AIGC head; patch evidence head; baseline output dictionary | `src/models/` (backbone, global head, patch head, baseline forward) |
| Member 3 | Reliability head; reliability-weighted patch aggregation; official fused `final_logit` / `aigc_probability` | `src/models/` (reliability), fusion specified in §8 and §10 |
| Member 4 | Manipulation head: image-level score, patch mask logits, 224×224 heatmap; training on labels 0 and 2 only | `src/models/` (manipulation) |
| Member 5 | Shared contracts, training/evaluation/inference wiring, official JSON, optional-module isolation, reporting discipline | `src/training/`, `src/evaluation/`, `src/inference/`, `scripts/`, `tests/`, `configs/`, `docs/` |

Shared non-negotiables for every member:

- Do not train or tune on protected WildFake demonstration data.
- Do not let the manipulation module alter AIGC probability.
- Do not invent metrics (see §15).

## 14. Optional module failure MUST NOT break baseline inference

Reliability (Member 3) and manipulation (Member 4) are optional at inference time.

If either optional module is missing, throws, or returns invalid tensors:

- Baseline inference (Member 2 path, §7) MUST still produce `final_logit` and `aigc_probability`.
- Official JSON (§11) MUST still be writable.
- Manipulation outputs MUST be omitted rather than crashing the AIGC path.

Callers MUST isolate optional modules behind explicit try/fallback behaviour. A failed optional module is not an official-task failure.

## 15. No reported metric may be manually invented

Every reported number MUST be produced by executable evaluation code from defined inputs.

Forbidden:

- Hand-typed scores, AUC, accuracy, or robustness figures
- Copy-pasted numbers not regenerated from the current checkpoint and split
- Placeholder metrics presented as results

Allowed:

- Metrics computed by `src/evaluation/` (or scripts that call it) and written under `outputs/`
- Explicitly labelled non-results such as “not yet evaluated”

If a metric cannot be computed, omit it or mark it unevaluated. Do not fabricate a value.

## 16. Internal evaluation-record schema

This schema is **not** the official inference JSON. Official inference output remains exactly `image_path` and `pred` (§11) and MUST be rejected as an evaluation input.

Internal labelled evaluation records (CSV or a JSON list of objects) are the only valid input to `src/evaluation/` and `scripts/evaluate.py`. Each record MUST contain:

| Field | Constraint |
| --- | --- |
| `image_path` | Non-empty string |
| `label` | Integer, exactly `0` (authentic) or `1` (fully synthetic) |
| `pred` | Finite number in `[0, 1]` |
| `model_name` | Non-empty string |
| `dataset` | Non-empty string |
| `split` | Non-empty string |
| `transform_name` | One of: `clean`, `jpeg`, `gaussian_blur`, `resize`, `gaussian_noise`, `color_jitter`, `center_crop` |
| `severity` | String or number; clean records may use `none` |

Additional rules:

- Label `2` (locally tampered) is not a valid AIGC-evaluation label in this file.
- Duplicate rows for the same `model_name`, `dataset`, `split`, `image_path`, `transform_name`, and `severity` are rejected.
- An empty prediction file is rejected.
- Records that identify themselves as mock data are rejected and MUST NOT be reported as model performance.
- Metrics written by the evaluator MUST be computed from these records (see §15).
