"""Read synchronized WESAD files and form a common, auditable window table."""

from __future__ import annotations

import codecs
from collections.abc import Sequence
from pathlib import Path
import pickle
import re
import sys

import numpy as np
import pandas as pd

from .features import BVP_FEATURES, EDA_FEATURES, extract_features

try:  # NumPy moved its implementation module in version 2.
    from numpy._core.multiarray import _reconstruct
except ImportError:  # pragma: no cover - exercised in NumPy 1.x environments.
    from numpy.core.multiarray import _reconstruct

EXPECTED_SUBJECTS = tuple(f"S{i}" for i in (*range(2, 12), *range(13, 18)))
WINDOW_SECONDS = 60
LABEL_HZ = 700
EDA_HZ = 4
BVP_HZ = 64
METADATA_COLUMNS = (
    "subject", "window_id", "start_s", "end_s", "original_condition", "y_true"
)
MANIFEST_COLUMNS = (*METADATA_COLUMNS, "included", "exclusion_reason")
FEATURE_COLUMNS = (*METADATA_COLUMNS, *EDA_FEATURES, *BVP_FEATURES)


class _WesadUnpickler(pickle.Unpickler):
    """Allow only the NumPy objects used by the original WESAD pickles."""

    def find_class(self, module: str, name: str):
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("_codecs", "encode"): codecs.encode,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")
        return allowed[module, name]


def discover_subjects(root: Path) -> list[str]:
    """Find numerically sorted subject directories, requiring their pickles.

    The full-experiment command separately checks against ``EXPECTED_SUBJECTS``;
    this function also supports explicit, clearly identified pilot subsets.
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"WESAD directory not found: {root}")
    subjects = sorted(
        (p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(r"S\d+", p.name)),
        key=lambda subject: int(subject[1:]),
    )
    if not subjects:
        raise ValueError(f"No WESAD participant directories found in {root}")
    for subject in subjects:
        path = root / subject / f"{subject}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Participant pickle not found: {path}")
    return subjects


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional or have shape (n, 1)")
    if not np.issubdtype(result.dtype, np.number) or np.iscomplexobj(result):
        raise ValueError(f"{name} must contain real numeric samples")
    return result


def load_subject(root: Path, subject: str) -> dict:
    """Load one original Python-2 pickle and retain only project inputs.

    The chest data and unused wrist modalities are released when this function
    returns. The source file is opened in read-only mode and is never changed.
    """
    if not re.fullmatch(r"S\d+", subject):
        raise ValueError(f"Invalid participant ID: {subject!r}")
    path = Path(root).expanduser() / subject / f"{subject}.pkl"
    with path.open("rb") as handle:
        source = _WesadUnpickler(handle, encoding="latin1").load()
    try:
        source_subject = source["subject"]
        wrist = source["signal"]["wrist"]
        eda = _vector(wrist["EDA"], "wrist EDA")
        bvp = _vector(wrist["BVP"], "wrist BVP")
        labels = _vector(source["label"], "labels")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Required WESAD fields missing from {path}") from error
    if source_subject != subject:
        raise ValueError(
            f"Participant mismatch in {path}: expected {subject}, got {source_subject!r}"
        )
    return {
        "subject": subject,
        "signal": {"wrist": {"EDA": eda, "BVP": bvp}},
        "label": labels,
    }


def segment_subject(
    subject: str, eda: np.ndarray, bvp: np.ndarray, labels: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a common 60-second grid and features for its valid windows.

    The manifest spans the longest supplied signal, including its final partial
    window. A candidate is included only when all three streams are complete,
    the *original* condition is constant, and that condition is 1, 2, or 3.
    Nonfinite sensor values become missing features, not dropped windows.
    Labels must be finite integers; malformed labels fail explicitly.
    """
    eda = _vector(eda, "wrist EDA")
    bvp = _vector(bvp, "wrist BVP")
    labels = _vector(labels, "labels")
    if not np.isfinite(labels).all():
        raise ValueError("Labels must contain only finite integers")
    if not np.issubdtype(labels.dtype, np.integer) and not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Labels must contain only finite integers")
    streams = (("labels", labels, LABEL_HZ), ("eda", eda, EDA_HZ), ("bvp", bvp, BVP_HZ))
    candidates = max(
        ((len(values) + rate * WINDOW_SECONDS - 1) // (rate * WINDOW_SECONDS)
         for _, values, rate in streams),
        default=0,
    )
    manifest_rows, feature_rows = [], []
    for index in range(candidates):
        start_s, end_s = index * WINDOW_SECONDS, (index + 1) * WINDOW_SECONDS
        pieces = {name: values[start_s * rate:end_s * rate] for name, values, rate in streams}
        reasons = [
            f"incomplete_{name}"
            for name, _, rate in streams
            if pieces[name].size != rate * WINDOW_SECONDS
        ]
        label_piece = pieces["labels"]
        original_condition = None
        y_true = None
        if label_piece.size:
            if np.equal(label_piece, label_piece[0]).all():
                original_condition = int(label_piece[0])
                if original_condition in (1, 2, 3):
                    y_true = int(original_condition == 2)
                else:
                    reasons.append("excluded_condition")
            else:
                reasons.append("mixed_original_labels")
        metadata = {
            "subject": subject,
            "window_id": f"{subject}:{index:05d}",
            "start_s": start_s,
            "end_s": end_s,
            "original_condition": original_condition,
            "y_true": y_true,
        }
        included = not reasons and original_condition in (1, 2, 3)
        manifest_rows.append({
            **metadata, "included": included, "exclusion_reason": "|".join(reasons)
        })
        if included:
            eda_features = extract_features(pieces["eda"], EDA_HZ)
            bvp_features = extract_features(pieces["bvp"], BVP_HZ)
            feature_rows.append({
                **metadata,
                **{f"eda_{key}": value for key, value in eda_features.items()},
                **{f"bvp_{key}": value for key, value in bvp_features.items()},
            })
    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS).astype({
        "original_condition": "Int64", "y_true": "Int64", "included": bool
    })
    features = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    if not features.empty:
        features = features.astype({"original_condition": "int64", "y_true": "int64"})
    return manifest, features


def build_feature_tables(
    root: Path, subjects: Sequence[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one participant at a time and concatenate only small output tables."""
    root = Path(root).expanduser()
    selected = discover_subjects(root) if subjects is None else list(subjects)
    if not selected:
        raise ValueError("At least one participant is required")
    if len(selected) != len(set(selected)):
        raise ValueError("Participant IDs must be unique")
    manifests, feature_tables = [], []
    for subject in selected:
        print(f"Extracting {subject} ...", file=sys.stderr, flush=True)
        data = load_subject(root, subject)
        wrist = data["signal"]["wrist"]
        manifest, features = segment_subject(subject, wrist["EDA"], wrist["BVP"], data["label"])
        manifests.append(manifest)
        feature_tables.append(features)
        # Explicitly release raw arrays before the next large pickle is loaded.
        del wrist, data
    return (
        pd.concat(manifests, ignore_index=True),
        pd.concat(feature_tables, ignore_index=True),
    )
