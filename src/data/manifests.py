"""Manifest schema, validation, and deterministic split assignment for TraceLens-R.

A "manifest" is the single source of truth handed to every downstream member
(2-4): a table with exactly the columns in :data:`MANIFEST_COLUMNS`, one row
per sample. This module builds, validates, and splits that table. It never
touches pixels; see ``transforms.py`` / ``dataset.py`` for that.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

LABEL_AUTHENTIC = 0
LABEL_FULLY_SYNTHETIC = 1
LABEL_LOCALLY_TAMPERED = 2
VALID_LABELS = (LABEL_AUTHENTIC, LABEL_FULLY_SYNTHETIC, LABEL_LOCALLY_TAMPERED)

MANIFEST_COLUMNS = [
    "image_id",
    "image_path",
    "label",
    "mask_path",
    "split",
    "source",
    "protected",
]

VALID_SPLITS = ("train", "val", "test")

DEFAULT_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
DEFAULT_SEED = 42

# Dataset staging sizes: images per class (label), per spec.
STAGE_COUNTS_PER_CLASS = {
    "smoke": 100,
    "initial": 2000,
    "final": 5000,
}

# Any record whose ``source`` field contains one of these (case-insensitive)
# is treated as protected regardless of the caller-supplied ``protected`` flag.
PROTECTED_SOURCE_KEYWORDS = ("wildfake",)


class ProtectedDataError(RuntimeError):
    """Raised when protected data (e.g. WildFake) would enter the pipeline.

    This is intentionally loud: protected samples must never be silently
    dropped *or* silently included. Callers that legitimately need to
    reference protected data (outside of training) must pass
    ``allow_protected=True`` explicitly at the call site that raised this.
    """


class ManifestValidationError(ValueError):
    """Raised when a manifest (or a record destined for one) is malformed."""


def stable_seed(*parts: object) -> int:
    """Deterministic int seed derived from arbitrary hashable-by-str parts.

    ``random.Random`` only accepts None/int/float/str/bytes/bytearray, so
    composite seed "keys" (e.g. (seed, purpose, label)) are hashed down to a
    single int here instead of being passed to it directly.
    """
    digest = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class RawRecord:
    """One row of raw input, prior to manifest validation / split assignment."""

    image_id: str
    image_path: str
    label: int
    source: str
    mask_path: str | None = None
    protected: bool | None = None  # None => auto-detect from `source`.


# ---------------------------------------------------------------------------
# Protected-data detection
# ---------------------------------------------------------------------------


def is_protected_source(source: str) -> bool:
    """Return True if ``source`` names a protected dataset (e.g. WildFake)."""
    lowered = source.lower()
    return any(keyword in lowered for keyword in PROTECTED_SOURCE_KEYWORDS)


def _resolve_protected_flag(record: RawRecord) -> bool:
    detected = is_protected_source(record.source)
    if record.protected is None:
        return detected
    if record.protected is False and detected:
        # Someone tried to mark a WildFake-sourced record as unprotected.
        # Fail loudly rather than trust a possibly-mistaken override.
        raise ProtectedDataError(
            f"Record {record.image_id!r} has source={record.source!r} which matches a "
            "protected-source keyword, but protected=False was explicitly requested. "
            "Refusing to silently override protection."
        )
    return bool(record.protected)


# ---------------------------------------------------------------------------
# Hashing (duplicate detection)
# ---------------------------------------------------------------------------


def compute_file_hash(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hash of a file's raw bytes, used to catch duplicate images
    that were saved under different ``image_id``/``image_path`` values."""
    path = Path(path)
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def build_manifest(
    records: Iterable[RawRecord],
    *,
    allow_protected: bool = False,
    compute_hashes: bool = True,
) -> pd.DataFrame:
    """Validate raw records and assemble them into an (unsplit) manifest.

    Parameters
    ----------
    records:
        Raw rows describing available images.
    allow_protected:
        If False (default), any record identified as protected (WildFake or
        explicitly flagged) raises :class:`ProtectedDataError` immediately.
        This is a fail-fast guard, not a silent filter: a caller must
        explicitly opt in before protected data is allowed anywhere in the
        pipeline (and even then, ``assign_splits`` refuses to place it in
        ``train``).
    compute_hashes:
        If True, compute a content hash per image (used later to keep
        duplicate images out of different splits). Requires that
        ``image_path`` exists on disk.

    Returns
    -------
    pd.DataFrame with columns ``MANIFEST_COLUMNS`` (``split`` left empty,
    ``"" ``) plus an internal ``_hash`` column when ``compute_hashes=True``.
    """
    rows: list[dict] = []
    seen_ids: set[str] = set()

    for record in records:
        if record.label not in VALID_LABELS:
            raise ManifestValidationError(
                f"Invalid label {record.label!r} for image_id={record.image_id!r}; "
                f"expected one of {VALID_LABELS}."
            )
        if record.image_id in seen_ids:
            raise ManifestValidationError(f"Duplicate image_id in input records: {record.image_id!r}")
        seen_ids.add(record.image_id)

        protected = _resolve_protected_flag(record)
        if protected and not allow_protected:
            raise ProtectedDataError(
                f"Record {record.image_id!r} (source={record.source!r}) is protected data "
                "(e.g. WildFake) and allow_protected=False. Refusing to add it to any "
                "manifest. If you have a legitimate non-training use for this data, "
                "call build_manifest(..., allow_protected=True) explicitly."
            )

        if record.label == LABEL_LOCALLY_TAMPERED:
            mask_path = record.mask_path
        else:
            # Labels 0/1 never carry a real mask; force this explicitly so a
            # stray mask_path in the input can't leak into the manifest.
            mask_path = None

        row = {
            "image_id": record.image_id,
            "image_path": record.image_path,
            "label": int(record.label),
            "mask_path": mask_path,
            "split": "",
            "source": record.source,
            "protected": protected,
        }
        if compute_hashes:
            row["_hash"] = compute_file_hash(record.image_path)
        rows.append(row)

    columns = MANIFEST_COLUMNS + (["_hash"] if compute_hashes else [])
    return pd.DataFrame(rows, columns=columns)


def validate_manifest(df: pd.DataFrame, *, require_split: bool = True) -> None:
    """Raise ManifestValidationError if ``df`` doesn't satisfy the shared schema."""
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ManifestValidationError(f"Manifest is missing required columns: {missing}")

    bad_labels = set(df["label"].unique()) - set(VALID_LABELS)
    if bad_labels:
        raise ManifestValidationError(f"Manifest contains invalid labels: {sorted(bad_labels)}")

    if df["image_id"].duplicated().any():
        dupes = df.loc[df["image_id"].duplicated(), "image_id"].tolist()
        raise ManifestValidationError(f"Manifest contains duplicate image_id values: {dupes}")

    if require_split:
        bad_splits = set(df["split"].unique()) - set(VALID_SPLITS)
        if bad_splits:
            raise ManifestValidationError(f"Manifest contains invalid split values: {sorted(bad_splits)}")

    tampered = df[df["label"] == LABEL_LOCALLY_TAMPERED]
    if tampered["mask_path"].isna().any():
        offenders = tampered.loc[tampered["mask_path"].isna(), "image_id"].tolist()
        raise ManifestValidationError(
            f"Locally-tampered (label=2) samples must have a mask_path: {offenders}"
        )

    assert_no_protected_in_train(df)


def assert_no_protected_in_train(df: pd.DataFrame) -> None:
    """Defense-in-depth guard: raise if any protected row is assigned to train.

    Called both at split-assignment time and again at dataset load time, so
    that a hand-edited manifest CSV can't silently smuggle protected data
    into training.
    """
    if "split" not in df.columns or "protected" not in df.columns:
        return
    offenders = df[(df["split"] == "train") & (df["protected"].astype(bool))]
    if not offenders.empty:
        raise ProtectedDataError(
            "Protected data found in the train split (this must never happen): "
            f"{offenders['image_id'].tolist()}"
        )


# ---------------------------------------------------------------------------
# Deterministic, group-aware split assignment
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, keys: Iterable[str]):
        self._parent = {k: k for k in keys}

    def find(self, k: str) -> str:
        root = k
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[k] != root:
            self._parent[k], k = root, self._parent[k]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _duplicate_hash_groups(df: pd.DataFrame) -> Mapping[str, list[str]]:
    """Map each hash value that appears more than once to its image_ids."""
    if "_hash" not in df.columns:
        return {}
    counts = df["_hash"].value_counts()
    dup_hashes = set(counts[counts > 1].index)
    groups: dict[str, list[str]] = {}
    for h, sub in df[df["_hash"].isin(dup_hashes)].groupby("_hash"):
        groups[h] = sub["image_id"].tolist()
    return groups


def assign_splits(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    stratify_by_label: bool = True,
) -> pd.DataFrame:
    """Deterministically assign each row to train/val/test.

    Guarantees:
      * Same seed -> same split assignment, every time (pure function of
        ``df`` contents + ``seed``; no wall-clock or OS randomness).
      * Every row sharing an ``image_id`` is trivially in one group (one row
        per id is expected upstream) and never crosses a split.
      * Rows whose *content hash* matches another row (duplicate images
        saved under different ids) are unioned into one group and placed in
        the same split together, *even if those rows carry different
        labels* -- a duplicate-hash group is always assigned to exactly one
        split as an atomic unit, never split apart by a per-label pass.
      * Protected rows are only ever assigned to "val" or "test", never
        "train" (checked again by :func:`assert_no_protected_in_train`).

    The 80/10/10 target ratio is approximate: whole duplicate-groups are
    never split apart, so exact ratios aren't always achievable, but a
    deterministic greedy-largest-deficit bin-packing keeps it close. Targets
    are still tracked per label (when ``stratify_by_label=True``) so overall
    per-class ratios stay close to 80/10/10; a group spanning multiple
    labels is assigned to whichever split most reduces the *combined*
    deficit across every label it touches.
    """
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    df = df.copy()
    uf = _UnionFind(df["image_id"].tolist())
    for ids in _duplicate_hash_groups(df).values():
        for other in ids[1:]:
            uf.union(ids[0], other)

    df["_group"] = df["image_id"].map(uf.find)

    # Duplicate-hash groups must be assigned as a single atomic unit, so
    # stratification cannot split the dataframe by label *before* grouping
    # (a group spanning two labels would then be assigned independently,
    # and inconsistently, in each label's stratum). Instead, groups are
    # formed once over the whole dataframe, and label targets/assigned
    # counts are tracked in parallel per label for approximate 80/10/10
    # stratification.
    strata_keys = df["label"].unique().tolist() if stratify_by_label else [None]
    label_of = (lambda label: label) if stratify_by_label else (lambda label: None)

    targets = {
        key: {name: ratio * int((df["label"] == key).sum() if stratify_by_label else len(df)) for name, ratio in ratios.items()}
        for key in strata_keys
    }
    assigned_counts = {key: {name: 0.0 for name in ratios} for key in strata_keys}

    groups = list(df.groupby("_group"))
    rng = random.Random(stable_seed(seed, "assign_splits"))
    rng.shuffle(groups)

    split_col = pd.Series(index=df.index, dtype=object)

    for _group_key, group_rows in groups:
        protected = bool(group_rows["protected"].any())
        candidates = [s for s in ratios if not (protected and s == "train")]

        group_label_counts: dict = {}
        for label, count in group_rows["label"].value_counts().items():
            group_label_counts[label_of(label)] = group_label_counts.get(label_of(label), 0) + count

        # Greedily assign to whichever eligible split most reduces the
        # combined deficit (target - assigned so far) summed across every
        # label present in this group -- for a single-label group this is
        # exactly the original per-label deficit rule.
        def combined_deficit(split: str, counts: dict = group_label_counts) -> float:
            return sum(targets[key][split] - assigned_counts[key][split] for key in counts)

        best_split = max(candidates, key=combined_deficit)
        split_col.loc[group_rows.index] = best_split
        for key, count in group_label_counts.items():
            assigned_counts[key][best_split] += count

    df["split"] = split_col
    df = df.drop(columns=["_group"])
    assert_no_protected_in_train(df)
    return df


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    """Write only the shared, public columns (drops internal ``_hash``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[MANIFEST_COLUMNS].to_csv(path, index=False)


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"image_id": str})
    df["protected"] = df["protected"].astype(bool)
    df["mask_path"] = df["mask_path"].where(df["mask_path"].notna(), None)
    validate_manifest(df)
    return df
