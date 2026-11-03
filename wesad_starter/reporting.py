"""Save portable reports and verify them against the window predictions."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

from . import __version__
from .data import EXPECTED_SUBJECTS
from .evaluation import (
    BASELINE_PARAMETERS, INPUT_MODES, META_COLUMNS, THRESHOLD,
    subject_order, summarize_predictions, validate_feature_table,
)
from .features import BVP_FEATURES, EDA_FEATURES

REPORT_TABLES = ("predictions", "participant_metrics", "summary", "fusion_deltas",
                 "condition_errors", "error_overlap", "preprocessing")
PATH_OPTIONS = {"--data-root": "$WESAD_ROOT", "--features": "$FEATURES", "--output-dir": "$OUTPUT_DIR"}


def _absolute_path(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def portable_command(command: list[str] | None) -> list[str]:
    """Keep replayable option names without persisting local directory values."""
    result = []
    source = command or []
    index = 0
    while index < len(source):
        token = str(source[index])
        option, separator, value = token.partition("=")
        if option in PATH_OPTIONS:
            if separator:
                result.append(f"{option}={PATH_OPTIONS[option]}")
            else:
                result.append(option)
                if index + 1 < len(source) and not str(source[index + 1]).startswith("--"):
                    result.append(PATH_OPTIONS[option])
                    index += 1
        elif separator and _absolute_path(value):
            result.append(f"{option}=$PATH")
        elif _absolute_path(token):
            result.append("python" if index == 0 else "$PATH")
        else:
            result.append(token)
        index += 1
    return result


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sanitize_extraction_metadata(root: Path) -> bool:
    """Remove machine-specific paths from a validated extraction record.

    Input-table checksums and relative participant file references are retained.
    Missing optional path fields are left absent, preserving portable records.
    """
    path = root / "extraction_metadata.json"
    if not path.is_file():
        return False
    metadata = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    if "data_root" in metadata and metadata["data_root"] != "$WESAD_ROOT":
        metadata["data_root"] = "$WESAD_ROOT"
        changed = True
    for entry in metadata.get("source_files", []):
        value = str(entry.get("file", ""))
        if _absolute_path(value):
            filename = PureWindowsPath(value).name
            subject = str(entry.get("subject", ""))
            entry["file"] = f"{subject}/{filename}" if subject else filename
            changed = True
    if changed:
        write_json(path, metadata)
    return changed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_record() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "nbformat", "nbclient"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def validate_extraction_provenance(root: Path, features: pd.DataFrame) -> bool:
    """Reject stale extraction records without requiring the raw dataset locally."""
    path = root / "extraction_metadata.json"
    if not path.is_file():
        return False
    provenance = json.loads(path.read_text(encoding="utf-8"))
    for name in ("features.csv", "window_manifest.csv"):
        expected = provenance.get("files_sha256", {}).get(name)
        if not expected or not (root / name).is_file() or file_sha256(root / name) != expected:
            raise ValueError(f"Extraction provenance is stale for {name}; extract again or use a separate experiment directory.")
    subjects = subject_order(features["subject"].tolist())
    scope = "full" if set(subjects) == set(EXPECTED_SUBJECTS) else "pilot"
    counts = {str(k): int(v) for k, v in features["original_condition"].value_counts().items()}
    if (subject_order(provenance.get("subjects", [])) != subjects or provenance.get("scope") != scope
            or provenance.get("included_windows") != len(features)
            or provenance.get("original_condition_counts") != counts):
        raise ValueError("Extraction provenance participant/window counts do not match features.")
    return True


def save_evaluation(outputs: dict, output_dir: str | Path, *, command: list[str] | None = None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "extraction_metadata.json").is_file():
        validate_extraction_provenance(output_dir, pd.read_csv(output_dir / "features.csv"))
        sanitize_extraction_metadata(output_dir)
    for name in REPORT_TABLES:
        outputs[name].to_csv(output_dir / f"{name}.csv", index=False, float_format="%.17g", na_rep="")
    write_json(output_dir / "splits.json", outputs["splits"])
    metadata = {
        "starter_version": __version__, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": outputs["splits"]["scope"], "command": portable_command(command),
        "environment": environment_record(),
        "baseline": BASELINE_PARAMETERS | {"regularization": "L2", "threshold": THRESHOLD},
        "preprocessing": {"imputation": "training median", "all_missing_fill": 0,
                          "keep_empty_features": True, "scaling": "training StandardScaler"},
        "features": {"eda": list(EDA_FEATURES), "bvp": list(BVP_FEATURES)},
        "evaluation": {"outer": "participant LOSO", "macro_f1_labels": [0, 1],
                       "zero_division": 0, "summary_unit": "participant", "std_ddof": 1,
                       "prediction_fusion_weights": [0.5, 0.5]},
        "files_sha256": {f"{name}.csv": file_sha256(output_dir / f"{name}.csv") for name in REPORT_TABLES},
    }
    for name in ("features.csv", "window_manifest.csv", "extraction_metadata.json", "splits.json"):
        if (output_dir / name).is_file():
            metadata["files_sha256"][name] = file_sha256(output_dir / name)
    write_json(output_dir / "run_metadata.json", metadata)


def _same_table(actual: pd.DataFrame, expected: pd.DataFrame, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)
    except AssertionError as exc:
        raise ValueError(f"Saved {name} does not match recomputed values: {exc}") from exc


def verify_results(output_dir: str | Path) -> dict:
    """Check coverage, lineage, train statistics, and reports without refitting LR."""
    root = Path(output_dir)
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    for name, expected_hash in metadata["files_sha256"].items():
        if not (root / name).is_file() or file_sha256(root / name) != expected_hash:
            raise ValueError(f"Missing or changed output: {name}")
    f = validate_feature_table(pd.read_csv(root / "features.csv"))
    extraction_verified = validate_extraction_provenance(root, f)
    p = pd.read_csv(root / "predictions.csv")
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    subjects = subject_order(f["subject"].tolist())
    full = set(subjects) == set(EXPECTED_SUBJECTS)
    if splits["subjects"] != subjects or splits["scope"] != ("full" if full else "pilot"):
        raise ValueError("Split participants/scope do not match feature table.")
    if metadata["scope"] != splits["scope"]:
        raise ValueError("Run and split scope disagree.")
    if len(splits["folds"]) != len(subjects) or {r["outer_test_subject"] for r in splits["folds"]} != set(subjects):
        raise ValueError("Every participant must be evaluated exactly once.")
    for fold in splits["folds"]:
        test = fold["outer_test_subject"]
        train = f[f["subject"] != test]
        held_out = f[f["subject"] == test]
        if fold["test_subjects"] != [test] or fold["train_subjects"] != subject_order(train["subject"].tolist()):
            raise ValueError(f"{test}: train/test participants overlap or are missing.")
        if fold["n_train_windows"] != len(train) or fold["n_test_windows"] != len(held_out):
            raise ValueError(f"{test}: split counts are inconsistent.")
        counts = {str(k): int(v) for k, v in train["y_true"].value_counts().items()}
        if fold["train_class_counts"] != counts:
            raise ValueError(f"{test}: training class counts are inconsistent.")
    if set(p["model_family"]) != {"baseline_lr"}:
        raise ValueError("This verifier expects the baseline_lr reference outputs.")
    if "decision_threshold" in p and not p["decision_threshold"].eq(THRESHOLD).all():
        raise ValueError("The fixed baseline must use decision threshold 0.5.")
    expected_meta = f[list(META_COLUMNS)].sort_values("window_id").reset_index(drop=True)
    for mode in INPUT_MODES:
        rows = p[p["input_mode"] == mode].sort_values("window_id")
        _same_table(rows[list(META_COLUMNS)], expected_meta, f"{mode} window coverage")
    recomputed = summarize_predictions(p)
    for name, expected in recomputed.items():
        _same_table(pd.read_csv(root / f"{name}.csv"), expected, name)
    probabilities = p.pivot(index="window_id", columns="input_mode", values="p_stress")
    if not np.allclose(probabilities["prediction_fusion"], (probabilities["eda"] + probabilities["bvp"]) / 2,
                       rtol=0, atol=1e-14):
        raise ValueError("Prediction fusion is not the mean of the saved EDA/BVP probabilities.")

    # Independently reconstruct fitted imputation/scaling statistics from each
    # outer-training feature subset. Test extremes cannot affect these numbers.
    prep = pd.read_csv(root / "preprocessing.csv")
    expected_prep = []
    column_sets = {"eda": list(EDA_FEATURES), "bvp": list(BVP_FEATURES),
                   "feature_fusion": list(EDA_FEATURES) + list(BVP_FEATURES)}
    for subject in subjects:
        train = f[f["subject"] != subject]
        for mode, columns in column_sets.items():
            for column in columns:
                a = train[column].to_numpy(dtype=float)
                all_missing = bool(np.isnan(a).all())
                median = 0.0 if all_missing else float(np.nanmedian(a))
                filled = np.where(np.isnan(a), median, a)
                std = float(np.std(filled, ddof=0))
                expected_prep.append({"outer_test_subject": subject, "input_mode": mode, "feature": column,
                                      "train_all_missing": all_missing, "imputation_value": median,
                                      "scaler_mean": float(np.mean(filled)), "scaler_scale": std if std > 0 else 1.0})
    _same_table(prep, pd.DataFrame(expected_prep), "training preprocessing")
    manifest_path = root / "window_manifest.csv"
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        if manifest["window_id"].duplicated().any() or not manifest["included"].isin([True, False]).all():
            raise ValueError("Invalid window manifest.")
        kept = manifest[manifest["included"]].sort_values("window_id")
        _same_table(kept[list(META_COLUMNS)], expected_meta, "manifest inclusion")
    if full:
        counts = f["original_condition"].value_counts().to_dict()
        if len(f) != 513 or counts != {1: 282, 2: 152, 3: 79}:
            raise ValueError(f"Full dataset window counts differ from verified WESAD reference: {counts}")
    checks = ["file hashes", "LOSO coverage", "identical windows", "training-only statistics",
              "probability fusion", "recomputed metrics/deltas/errors"]
    if extraction_verified:
        checks.append("extraction provenance")
    if manifest_path.is_file():
        checks.append("manifest inclusion")
    return {"status": "passed", "scope": splits["scope"], "participants": len(subjects),
            "included_windows": len(f), "prediction_rows": len(p),
            "participant_metric_rows": len(recomputed["participant_metrics"]),
            "checks": checks}
