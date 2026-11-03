"""Fairness and reproducibility checks using small, labeled synthetic tables.

These fixtures test evaluation rules only; their scores are not WESAD results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.metrics import f1_score

from wesad_starter.evaluation import (
    INPUT_MODES,
    evaluate_loso,
    iter_inner_subject_splits,
    make_baseline,
    summarize_predictions,
)
from wesad_starter.features import BVP_FEATURES, EDA_FEATURES


@pytest.fixture
def feature_table() -> pd.DataFrame:
    rows = []
    for subject_index, subject in enumerate(("S2", "S3", "S4")):
        for window_index, condition in enumerate((1, 2, 3, 2)):
            target = int(condition == 2)
            row = {
                "subject": subject,
                "window_id": f"{subject}:{window_index:05d}",
                "start_s": window_index * 60,
                "end_s": (window_index + 1) * 60,
                "original_condition": condition,
                "y_true": target,
            }
            for feature_index, feature in enumerate(EDA_FEATURES + BVP_FEATURES):
                row[feature] = (
                    target * 2.0 + subject_index * 0.1
                    + window_index * 0.07 + feature_index * 0.02
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _ordered_probabilities(predictions: pd.DataFrame) -> np.ndarray:
    return predictions.sort_values(["subject", "window_id", "input_mode"])[
        "p_stress"
    ].to_numpy()


def test_preprocessing_statistics_are_fitted_on_training_participants_only(feature_table):
    frame = feature_table.copy()
    # Include a missing training value so the expected scaler statistics also
    # depend on median imputation, rather than merely on raw feature means.
    frame.loc[(frame.subject == "S3") & (frame.start_s == 0), "eda_mean"] = np.nan
    original = evaluate_loso(frame)
    changed = frame.copy()
    changed.loc[changed.subject == "S2", "eda_mean"] = 1e9
    changed.loc[changed.subject == "S2", "bvp_mean"] = -1e9
    perturbed = evaluate_loso(changed)

    def held_out_statistics(result):
        return result["preprocessing"].query(
            "outer_test_subject == 'S2'"
        ).reset_index(drop=True)

    assert_frame_equal(held_out_statistics(original), held_out_statistics(perturbed))
    train = frame.loc[frame.subject != "S2", "eda_mean"].to_numpy()
    median = np.nanmedian(train)
    imputed = np.where(np.isnan(train), median, train)
    for mode in ("eda", "feature_fusion"):
        actual = held_out_statistics(original).query(
            "input_mode == @mode and feature == 'eda_mean'"
        ).iloc[0]
        assert actual.imputation_value == pytest.approx(median)
        assert actual.scaler_mean == pytest.approx(np.mean(imputed))
        assert actual.scaler_scale == pytest.approx(np.std(imputed, ddof=0))


def test_all_missing_training_column_preserves_width_and_uses_zero(feature_table):
    frame = feature_table.copy()
    frame.loc[frame.subject != "S2", "eda_slope"] = np.nan
    result = evaluate_loso(frame)
    audit = result["preprocessing"].query(
        "outer_test_subject == 'S2' and feature == 'eda_slope'"
    )
    assert set(audit.input_mode) == {"eda", "feature_fusion"}
    assert audit.train_all_missing.all()
    np.testing.assert_array_equal(audit.imputation_value, [0.0, 0.0])
    np.testing.assert_array_equal(audit.scaler_mean, [0.0, 0.0])
    np.testing.assert_array_equal(audit.scaler_scale, [1.0, 1.0])
    assert len(result["predictions"]) == len(frame) * 4
    assert np.isfinite(result["predictions"].p_stress).all()

    pipeline = make_baseline()
    pipeline.fit([[0.0, np.nan], [1.0, np.nan], [2.0, np.nan], [3.0, np.nan]], [0, 0, 1, 1])
    transformed = pipeline.named_steps["imputer"].transform([[np.nan, np.nan]])
    assert transformed.shape == (1, 2)
    np.testing.assert_array_equal(transformed, [[1.5, 0.0]])


def test_metadata_and_unlisted_target_copies_cannot_become_model_inputs(feature_table):
    original = evaluate_loso(feature_table)
    altered = feature_table.copy()
    altered["leaked_target_copy"] = altered.y_true * 1e8
    altered["filename"] = altered.subject + ".pkl"
    # Required timing metadata still obeys the grid but has radically changed
    # magnitudes. It must not affect the actual model predictions.
    altered["start_s"] += 600_000
    altered["end_s"] += 600_000
    modified = evaluate_loso(altered)
    np.testing.assert_array_equal(
        _ordered_probabilities(original["predictions"]),
        _ordered_probabilities(modified["predictions"]),
    )
    assert set(original["preprocessing"].feature) == set(EDA_FEATURES + BVP_FEATURES)
    widths = original["preprocessing"].groupby(["outer_test_subject", "input_mode"]).size()
    assert set(widths.xs("eda", level="input_mode")) == {7}
    assert set(widths.xs("bvp", level="input_mode")) == {7}
    assert set(widths.xs("feature_fusion", level="input_mode")) == {14}


def test_loso_holds_out_every_subject_once_and_reuses_single_signal_probabilities(feature_table):
    result = evaluate_loso(feature_table.sample(frac=1, random_state=7))
    folds = result["splits"]["folds"]
    assert [fold["outer_test_subject"] for fold in folds] == ["S2", "S3", "S4"]
    assert result["splits"]["scope"] == "pilot"
    for fold in folds:
        held_out = fold["outer_test_subject"]
        assert held_out not in fold["train_subjects"]
        assert fold["test_subjects"] == [held_out]
        assert fold["n_train_windows"] == 8
        assert fold["n_test_windows"] == 4

    predictions = result["predictions"]
    assert len(predictions) == len(feature_table) * len(INPUT_MODES)
    assert predictions.groupby("window_id").input_mode.nunique().eq(4).all()
    assert predictions.subject.eq(predictions.outer_test_subject).all()
    probabilities = predictions.pivot(index="window_id", columns="input_mode", values="p_stress")
    np.testing.assert_array_equal(
        probabilities.prediction_fusion,
        (probabilities.eda + probabilities.bvp) / 2.0,
    )
    assert set(predictions.y_pred) <= {0, 1}
    recomputed = summarize_predictions(predictions)
    for name, table in recomputed.items():
        assert_frame_equal(result[name], table)


def test_inner_splits_remain_grouped_and_reject_outer_test_rows():
    groups = np.array(["S3", "S3", "S4", "S4", "S5", "S5"])
    splits = list(iter_inner_subject_splits(groups, outer_test_subject="S2"))
    assert len(splits) == 3
    validation_subjects = []
    for train, validation in splits:
        assert not (set(groups[train]) & set(groups[validation]))
        assert set(train) | set(validation) == set(range(len(groups)))
        assert not (set(train) & set(validation))
        assert len(set(groups[validation])) == 1
        validation_subjects.extend(set(groups[validation]))
    assert sorted(validation_subjects) == ["S3", "S4", "S5"]
    with pytest.raises(ValueError, match="outer-training"):
        list(iter_inner_subject_splits(["S2", "S3", "S4"], outer_test_subject="S2"))
    with pytest.raises(ValueError, match="At least two"):
        list(iter_inner_subject_splits(["S3", "S3"], outer_test_subject="S2"))


@pytest.fixture
def saved_predictions() -> pd.DataFrame:
    rows = []
    cases = {
        "S2": ([1, 2], [0, 1]),
        "S3": ([1, 2, 3, 2], [0, 0, 0, 0]),
        "S4": ([1, 2, 3, 2, 1, 2], [1, 1, 1, 1, 1, 1]),
    }
    for mode in INPUT_MODES:
        for subject, (conditions, predictions) in cases.items():
            for index, (condition, predicted) in enumerate(zip(conditions, predictions, strict=True)):
                rows.append({
                    "model_family": "synthetic_example",
                    "input_mode": mode,
                    "subject": subject,
                    "outer_test_subject": subject,
                    "window_id": f"{subject}:{index:05d}",
                    "start_s": index * 60,
                    "end_s": (index + 1) * 60,
                    "original_condition": condition,
                    "y_true": int(condition == 2),
                    "p_stress": 0.8 if predicted else 0.2,
                    "y_pred": predicted,
                })
    return pd.DataFrame(rows)


def test_summary_averages_participants_and_uses_sample_sd_and_error_denominators(saved_predictions):
    result = summarize_predictions(saved_predictions)
    participant_scores = np.array([1.0, 1.0 / 3, 1.0 / 3])
    for row in result["summary"].itertuples():
        assert row.n_participants == 3
        assert row.mean_macro_f1 == pytest.approx(participant_scores.mean())
        assert row.std_macro_f1 == pytest.approx(participant_scores.std(ddof=1))
    pooled = saved_predictions.query("input_mode == 'eda'")
    pooled_score = f1_score(pooled.y_true, pooled.y_pred, labels=[0, 1], average="macro")
    assert pooled_score != pytest.approx(participant_scores.mean())

    errors = result["condition_errors"]
    absent = errors.query("subject == 'S2' and original_condition == 3")
    assert absent.denominator.eq(0).all()
    assert absent.numerator.eq(0).all()
    assert absent.error_rate.isna().all()
    missed_stress = errors.query("subject == 'S3' and original_condition == 2")
    assert missed_stress.numerator.eq(2).all()
    assert missed_stress.denominator.eq(2).all()
    assert missed_stress.error_rate.eq(1).all()
    assert result["fusion_deltas"].delta_macro_f1.eq(0).all()
    overlap = result["error_overlap"].set_index("subject")
    assert overlap.loc["S2", "both_correct"] == 2
    assert np.isnan(overlap.loc["S2", "error_jaccard"])
    assert overlap.loc["S3", "both_wrong"] == 2
    assert overlap.loc["S3", "error_jaccard"] == 1


def test_summarizer_rejects_different_windows_between_input_modes(saved_predictions):
    broken = saved_predictions.drop(saved_predictions.query("input_mode == 'bvp'").index[0])
    with pytest.raises(ValueError, match="identical windows"):
        summarize_predictions(broken)


def test_summarizer_rejects_different_windows_between_model_families(saved_predictions):
    advanced = saved_predictions.copy()
    advanced["model_family"] = "advanced_example"
    # Each family remains internally consistent, but the advanced model has
    # omitted the same window in every mode, violating paired comparison.
    advanced = advanced[advanced.window_id != "S2:00000"]
    combined = pd.concat([saved_predictions, advanced], ignore_index=True)
    with pytest.raises(ValueError, match="identical windows|same windows"):
        summarize_predictions(combined)


def test_summarizer_rejects_missing_family_metadata_instead_of_dropping_rows(saved_predictions):
    broken = saved_predictions.copy()
    broken["model_family"] = None
    with pytest.raises(ValueError, match="missing|Missing|metadata"):
        summarize_predictions(broken)


def test_student_thresholds_can_be_recorded_per_fold_without_changing_baseline(saved_predictions):
    advanced = saved_predictions.copy()
    advanced["decision_threshold"] = np.where(advanced.subject == "S2", 0.9, 0.5)
    advanced["y_pred"] = (advanced.p_stress >= advanced.decision_threshold).astype(int)
    reports = summarize_predictions(advanced)
    assert reports["participant_metrics"].query("subject == 'S2'").macro_f1.lt(1).all()
    broken = advanced.copy()
    broken["decision_threshold"] = np.nan
    with pytest.raises(ValueError, match="decision_threshold"):
        summarize_predictions(broken)

