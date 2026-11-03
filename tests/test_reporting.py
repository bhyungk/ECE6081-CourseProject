"""Independent report checks using small synthetic tables, never WESAD results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wesad_starter.cli import _evaluate
from wesad_starter.evaluation import META_COLUMNS, evaluate_loso, summarize_predictions
from wesad_starter.features import BVP_FEATURES, EDA_FEATURES
from wesad_starter.reporting import file_sha256, save_evaluation, verify_results, write_json


@pytest.fixture
def feature_table():
    rows = []
    for subject_index, subject in enumerate(("S2", "S3", "S4")):
        for index, original_condition in enumerate((1, 2, 3, 2)):
            target = int(original_condition == 2)
            row = {
                "subject": subject, "window_id": f"{subject}:{index:05d}",
                "start_s": index * 60, "end_s": (index + 1) * 60,
                "original_condition": original_condition, "y_true": target,
            }
            for j, column in enumerate((*EDA_FEATURES, *BVP_FEATURES)):
                row[column] = 2.0 * target + subject_index * 0.2 + index * 0.03 + j * 0.001
            # Held-out S2 sees a feature entirely missing in its training set.
            row["eda_slope"] = float(index) if subject == "S2" else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def save_inputs(root: Path, features: pd.DataFrame, *, provenance=False):
    root.mkdir(parents=True, exist_ok=True)
    features.to_csv(root / "features.csv", index=False, float_format="%.17g")
    manifest = features[list(META_COLUMNS)].copy()
    manifest["included"] = True
    manifest["exclusion_reason"] = ""
    manifest.to_csv(root / "window_manifest.csv", index=False)
    if provenance:
        subjects = ["S2", "S3", "S4"]
        write_json(root / "extraction_metadata.json", {
            "source_type": "synthetic unit-test fixture; not actual WESAD results",
            "subjects": subjects, "scope": "pilot", "included_windows": len(features),
            "original_condition_counts": {str(k): int(v) for k, v in features.original_condition.value_counts().items()},
            "feature_columns": list(EDA_FEATURES) + list(BVP_FEATURES),
            "window_seconds": 60, "step_seconds": 60, "grid_origin_seconds": 0,
            "sample_rates_hz": {"eda": 4, "bvp": 64, "labels": 700},
            "label_mapping": {"1": 0, "2": 1, "3": 0},
            "source_files": [{"subject": s, "file": f"{s}/{s}.pkl", "size_bytes": 1, "mtime_ns": 1} for s in subjects],
            "files_sha256": {name: file_sha256(root / name) for name in ("features.csv", "window_manifest.csv")},
        })


def refresh_output_hash(root: Path, name: str):
    metadata_path = root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["files_sha256"][name] = file_sha256(root / name)
    write_json(metadata_path, metadata)


@pytest.fixture
def saved_reports(tmp_path, feature_table):
    root = tmp_path / "reports"
    save_inputs(root, feature_table)
    save_evaluation(evaluate_loso(feature_table), root, command=["synthetic-test"])
    return root


def test_pilot_reports_recompute_and_preserve_all_columns(saved_reports):
    result = verify_results(saved_reports)
    assert result["status"] == "passed"
    assert result["scope"] == "pilot"
    assert result["participants"] == 3
    assert result["included_windows"] == 12
    assert result["prediction_rows"] == 48
    assert result["participant_metric_rows"] == 12
    assert pd.read_csv(saved_reports / "features.csv").shape == (12, 20)
    metadata = json.loads((saved_reports / "run_metadata.json").read_text())
    assert metadata["scope"] == "pilot"
    assert metadata["command"] == ["synthetic-test"]
    assert metadata["evaluation"]["std_ddof"] == 1
    prep = pd.read_csv(saved_reports / "preprocessing.csv")
    row = prep[(prep.outer_test_subject == "S2") & (prep.input_mode == "eda") & (prep.feature == "eda_slope")].iloc[0]
    assert bool(row.train_all_missing)
    assert row.imputation_value == 0


def test_modified_output_is_detected_by_its_hash(saved_reports):
    with (saved_reports / "predictions.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Missing or changed output: predictions.csv"):
        verify_results(saved_reports)


def test_summary_is_recomputed_even_if_its_hash_was_updated(saved_reports):
    path = saved_reports / "summary.csv"
    summary = pd.read_csv(path)
    summary["mean_macro_f1"] = summary["mean_macro_f1"].astype(float)
    summary.loc[0, "mean_macro_f1"] += 0.1
    summary.to_csv(path, index=False)
    refresh_output_hash(saved_reports, "summary.csv")
    with pytest.raises(ValueError, match="summary does not match recomputed"):
        verify_results(saved_reports)


def test_probability_fusion_is_checked_beyond_recomputed_scores(tmp_path, feature_table):
    root = tmp_path / "reports"
    save_inputs(root, feature_table)
    outputs = evaluate_loso(feature_table)
    predictions = outputs["predictions"].copy()
    index = predictions.index[predictions.input_mode == "prediction_fusion"][0]
    predictions.loc[index, "p_stress"] = 0.01
    predictions.loc[index, "y_pred"] = 0
    outputs.update(summarize_predictions(predictions))
    outputs["predictions"] = predictions
    save_evaluation(outputs, root)
    with pytest.raises(ValueError, match="Prediction fusion is not the mean"):
        verify_results(root)


def test_manifest_inclusion_is_checked_beyond_hashes(saved_reports):
    path = saved_reports / "window_manifest.csv"
    manifest = pd.read_csv(path)
    manifest["exclusion_reason"] = manifest["exclusion_reason"].astype("string")
    manifest.loc[0, "included"] = False
    manifest.loc[0, "exclusion_reason"] = "incorrectly_removed"
    manifest.to_csv(path, index=False)
    refresh_output_hash(saved_reports, "window_manifest.csv")
    with pytest.raises(ValueError, match="manifest inclusion"):
        verify_results(saved_reports)


def test_train_test_overlap_is_rejected_even_if_split_hash_updated(saved_reports):
    path = saved_reports / "splits.json"
    splits = json.loads(path.read_text())
    splits["folds"][0]["train_subjects"].append("S2")
    write_json(path, splits)
    refresh_output_hash(saved_reports, "splits.json")
    with pytest.raises(ValueError, match="train/test participants overlap"):
        verify_results(saved_reports)


def test_evaluate_copies_matching_source_provenance(tmp_path, feature_table):
    source, destination = tmp_path / "source", tmp_path / "destination"
    save_inputs(source, feature_table, provenance=True)
    _evaluate(source / "features.csv", destination, ["synthetic-test"])
    assert (destination / "extraction_metadata.json").read_bytes() == (source / "extraction_metadata.json").read_bytes()
    assert (destination / "window_manifest.csv").read_bytes() == (source / "window_manifest.csv").read_bytes()
    assert verify_results(destination)["scope"] == "pilot"


def test_evaluate_removes_unrelated_destination_provenance(tmp_path, feature_table):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    feature_table.to_csv(source / "new_features.csv", index=False)
    destination.mkdir()
    (destination / "extraction_metadata.json").write_text('{"unrelated": true}')
    (destination / "window_manifest.csv").write_text("unrelated manifest")
    _evaluate(source / "new_features.csv", destination, ["synthetic-test"])
    assert not (destination / "extraction_metadata.json").exists()
    assert not (destination / "window_manifest.csv").exists()
    result = verify_results(destination)
    assert result["scope"] == "pilot"
    assert "manifest inclusion" not in result["checks"]


def test_stale_extraction_hashes_are_rejected(tmp_path, feature_table):
    root = tmp_path / "reports"
    save_inputs(root, feature_table, provenance=True)
    save_evaluation(evaluate_loso(feature_table), root)
    # Keep window metadata unchanged but alter a genuine feature value. Saving
    # reports must not silently endorse the old extraction provenance.
    changed = feature_table.copy()
    changed.loc[0, "eda_mean"] += 1
    changed.to_csv(root / "features.csv", index=False, float_format="%.17g")
    refresh_output_hash(root, "features.csv")
    with pytest.raises(ValueError, match="(?i)provenance|extraction|source"):
        verify_results(root)
    with pytest.raises(ValueError, match="(?i)provenance|extraction|source"):
        save_evaluation(evaluate_loso(changed), root)


def test_extraction_scope_must_match_participants(tmp_path, feature_table):
    root = tmp_path / "reports"
    save_inputs(root, feature_table, provenance=True)
    save_evaluation(evaluate_loso(feature_table), root)
    path = root / "extraction_metadata.json"
    provenance = json.loads(path.read_text())
    provenance["scope"] = "full"
    write_json(path, provenance)
    refresh_output_hash(root, "extraction_metadata.json")
    with pytest.raises(ValueError, match="(?i)scope|provenance|extraction"):
        verify_results(root)
