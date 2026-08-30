# SID-Set operational subset

Member 1's data pipeline (`src/data/`, `scripts/build_sid_subset.py`,
`scripts/prepare_dataset.py`) is built and tested against a deterministic,
balanced 3,000-image subset of **SID-Set (Social media Image Detection
dataSet)**. This document records where that subset comes from, how to
reproduce it, and how the team accesses the actual image/mask files (which
are never committed to this repository).

## Source dataset

- **Dataset:** SID-Set, published at
  [huggingface.co/datasets/saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Label semantics** (identical to TraceLens-R's own):
  - `0` — real/authentic images (sourced from OpenImages V7)
  - `1` — fully synthetic images
  - `2` — locally tampered images, with a binary manipulation mask
- SID-Set incorporates material from COCO, OpenImages V7, and Flickr30k,
  each also under Creative Commons Attribution 4.0 International licensing.
  This project fully complies with those licenses' attribution terms,
  passing through the same acknowledgement SID-Set itself provides.

### Citation

```
@misc{huang2025sidasocialmediaimage,
      title={SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model},
      author={Zhenglin Huang and Jinwei Hu and Xiangtai Li and Yiwei He and Xingyu Zhao and Bei Peng and Baoyuan Wu and Xiaowei Huang and Guangliang Cheng},
      year={2025},
      booktitle={Conference on Computer Vision and Pattern Recognition}
}
```

### Underlying source acknowledgements

- **COCO** — https://cocodataset.org/ (CC BY 4.0)
- **OpenImages V7** — https://storage.googleapis.com/openimages/web/index.html (CC BY 4.0)
- **Flickr30k** — https://arxiv.org/pdf/1505.04870 (CC BY 4.0)

## Explicitly excluded: WildFake

The protected WildFake demonstration dataset referenced in `docs/SPEC.md`
(§4) is **not** part of SID-Set and is not used anywhere in this subset.
`scripts/build_sid_subset.py` enforces this at extraction time: every
selected `img_id` is checked against `src.data.manifests.is_protected_source()`
plus a direct `wildfake` / `wild_fake` / `wild-fake` substring match before
it is written to disk, and the run fails loudly if any match is found. The
manifest's `protected` column is `False` for every row in this subset.

## The `operational_1000` stage

`src/data/manifests.STAGE_COUNTS_PER_CLASS` defines named dataset stages by
per-class image count: `smoke` (100), `operational_1000` (1000), `initial`
(2000), `final` (5000). The delivered subset uses exactly 1,000 images per
label (3,000 total, 1,000 masks for label 2) and is generated under the
**`operational_1000`** stage name — deliberately distinct from the existing
`initial` stage (2,000/class) so the two are never confused. The manifest
lives at `data/manifests/manifest_operational_1000.csv`.

## Reproducing the subset

The seed is never hardcoded — both scripts read it from
`configs/default.yaml`'s top-level `seed` key via `src.config.load_config()`.

```bash
# 1. Deterministically download and export 1,000 images per label (+ 1,000
#    masks for label 2) into the conventional authentic/fully_synthetic/
#    locally_tampered/locally_tampered_masks layout.
python scripts/build_sid_subset.py \
    --output-root data/raw/sid_set_operational_1000 \
    --shard-cache /tmp/sid_shard_cache \
    --repo-root . \
    --target-per-label 1000

# 2. Generate the portable manifest (relative image_path/mask_path values,
#    resolved later against a configurable dataset_root).
cd data/raw/sid_set_operational_1000
python ../../../scripts/prepare_dataset.py \
    --dataset-root . \
    --output-dir ../../../data/manifests \
    --stage operational_1000 \
    --auto-scan \
    --seed 42
```

`build_sid_subset.py` fetches shards from a **pinned commit** of SID-Set's
parquet export (`PARQUET_REVISION_SHA` in that script, resolved from
`refs/convert/parquet`) rather than the mutable "latest conversion"
endpoint, so re-running it reads the same bytes even if the upstream dataset
is later re-converted. Selection within each shard is deterministic
(seed-derived shuffle per label per shard); a label-2 candidate whose mask
has no non-zero pixel is rejected and skipped in favour of the next
candidate, so every exported mask is guaranteed non-empty. Every written
file is re-read from disk and re-hashed immediately after writing to catch
disk-level write corruption.

This is **not** a uniform random sample over SID-Set's full 210,000-row
train split (that would require downloading the ~124GB split in full); it
is a deterministic sample confined to the shards actually needed to fill
the per-label quotas, in a fixed, recorded shard order.

## Getting the actual files

The 3,000 images + 1,000 masks (~1.6GB) are **never committed to Git** —
only the manifest CSV, the extraction/build scripts, and this documentation
are tracked. `.gitignore` excludes `data/raw/`, `data/datasets/`, and all
common image extensions specifically so this can't happen by accident.

The team obtains the actual files from the shared team Drive folder
(`sid_set_subset_3000/`), which mirrors the same conventional layout:

```
sid_set_subset_3000/
    authentic/                  (1000 images)
    fully_synthetic/            (1000 images)
    locally_tampered/           (1000 images)
    locally_tampered_masks/     (1000 masks)
    _checksums.json             (per-file SHA-256, written by build_sid_subset.py)
    _extraction_log.json        (seed, pinned revision, shard log, final counts)
```

After downloading and extracting that folder anywhere locally, point
`TraceLensDataset` (or `scripts/prepare_dataset.py --dataset-root`) at it:

```python
from src.data.dataset import TraceLensDataset, TASK_AIGC

dataset = TraceLensDataset(
    "data/manifests/manifest_operational_1000.csv",
    split="train",
    dataset_root="/path/to/sid_set_subset_3000",  # wherever you extracted the Drive folder
    task=TASK_AIGC,
)
```

`dataset_root` is the only machine-specific value — the manifest itself
contains only paths relative to that root (e.g. `authentic/<id>.jpg`,
`locally_tampered_masks/<id>.png`), so it resolves identically regardless of
where the folder lives on a given machine.

**Do not commit the Drive folder's contents, and do not commit any Drive
API key, service-account credential, or shared-link token to this
repository.** Access to the shared folder is managed entirely in Drive's
own sharing settings; nothing dataset-access-related belongs in Git.
