"""Participant-independent evaluation and reusable analysis of saved predictions.

Learned preprocessing belongs inside every training split. Feature extraction is
fixed and window-local, so its table can be cached before splitting.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import EXPECTED_SUBJECTS
from .features import BVP_FEATURES, EDA_FEATURES

INPUT_MODES = ("eda", "bvp", "feature_fusion", "prediction_fusion")
META_COLUMNS = ("subject", "window_id", "start_s", "end_s", "original_condition", "y_true")
BASELINE_PARAMETERS = {
    "C": 1.0, "class_weight": "balanced", "max_iter": 2000,
    "solver": "lbfgs", "tol": 1e-4, "random_state": 42,
}
THRESHOLD = 0.5


def subject_order(subjects: Sequence[str]) -> list[str]:
    return sorted(set(subjects), key=lambda s: (int(s[1:]) if s.startswith("S") and s[1:].isdigit() else float("inf"), s))


def make_baseline() -> Pipeline:
    """A fresh training-only median/scaler/L2 logistic-regression pipeline.

    keep_empty_features retains a training-all-missing column and imputes NaNs
    with zero. It never estimates a replacement from the held-out participant.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(**BASELINE_PARAMETERS)),
    ])


def validate_feature_table(features: pd.DataFrame) -> pd.DataFrame:
    required = list(META_COLUMNS) + list(EDA_FEATURES) + list(BVP_FEATURES)
    missing = sorted(set(required) - set(features.columns))
    if missing:
        raise ValueError(f"Missing feature-table columns: {missing}")
    if features.empty:
        raise ValueError("No included windows to evaluate.")
    f = features.loc[:, required].copy()
    if f.loc[:, list(META_COLUMNS)].isna().any().any():
        raise ValueError("Feature-table metadata must not contain missing values.")
    if f["window_id"].duplicated().any():
        raise ValueError("window_id must be unique across all participants.")
    if not f["subject"].map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError("subject must be a nonempty string.")
    for c in ("start_s", "end_s", "original_condition", "y_true"):
        f[c] = pd.to_numeric(f[c], errors="raise")
    if not np.isfinite(f[["start_s", "end_s"]].to_numpy()).all():
        raise ValueError("Window times must be finite.")
    if (f["start_s"] < 0).any() or not np.allclose(f["end_s"] - f["start_s"], 60, rtol=0, atol=1e-9):
        raise ValueError("Every included window must span exactly 60 seconds.")
    if not np.allclose(f["start_s"] % 60, 0, rtol=0, atol=1e-9):
        raise ValueError("Window starts must follow the recording-origin 60-second grid.")
    if not f["original_condition"].isin([1, 2, 3]).all():
        raise ValueError("Only original conditions 1, 2, and 3 are eligible.")
    if not f["y_true"].isin([0, 1]).all() or not np.array_equal(
        f["y_true"].to_numpy(), (f["original_condition"] == 2).astype(int).to_numpy()
    ):
        raise ValueError("Binary targets do not match original conditions.")
    if f.duplicated(["subject", "start_s"]).any():
        raise ValueError("Duplicate participant/time windows are not allowed.")
    for c in list(EDA_FEATURES) + list(BVP_FEATURES):
        f[c] = pd.to_numeric(f[c], errors="raise")
        if np.isinf(f[c].to_numpy(dtype=float)).any():
            raise ValueError(f"Feature {c} contains infinity; use NaN for extraction failures.")
    f["original_condition"] = f["original_condition"].astype(int)
    f["y_true"] = f["y_true"].astype(int)
    order = {s: i for i, s in enumerate(subject_order(f["subject"].tolist()))}
    return f.assign(_order=f["subject"].map(order)).sort_values(["_order", "start_s"]).drop(columns="_order").reset_index(drop=True)


def iter_inner_subject_splits(
    groups: Sequence[str], outer_test_subject: str,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Return positional inner-LOSO indexes into the supplied outer-training data.

    This helper only makes splits. Students choose the search space and fit a
    fresh preprocessing/model pipeline inside each inner-training split.
    Passing an outer-test row is an error, rather than silently discarding it.
    """
    groups = np.asarray(groups, dtype=str)
    if groups.ndim != 1 or outer_test_subject in groups:
        raise ValueError("Inner groups must be 1-D outer-training participants only.")
    subjects = subject_order(groups.tolist())
    if len(subjects) < 2:
        raise ValueError("At least two outer-training participants are needed for inner LOSO.")
    for validation_subject in subjects:
        yield np.flatnonzero(groups != validation_subject), np.flatnonzero(groups == validation_subject)


def _macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0))


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Recompute reports from window predictions; no fitting or test-set selection.

    The function can also summarize a student's model family, provided each
    family supplies the same four input_mode values and window schema.
    """
    required = set(META_COLUMNS) | {"model_family", "input_mode", "outer_test_subject", "p_stress", "y_pred"}
    if not required.issubset(predictions.columns) or predictions.empty:
        raise ValueError("Missing prediction columns or empty prediction table.")
    p = predictions.copy()
    if p[list(required - {"p_stress"})].isna().any().any():
        raise ValueError("Prediction metadata and targets must not contain missing values.")
    keys = ["model_family", "input_mode", "subject", "window_id"]
    if p.duplicated(keys).any():
        raise ValueError("Duplicate model/window predictions.")
    prob = p["p_stress"].to_numpy(dtype=float)
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("p_stress must contain finite probabilities in [0, 1].")
    # The baseline uses 0.5. A student's internally selected threshold may be
    # supplied per outer fold/model in decision_threshold; selection is their
    # experiment, while this helper only checks consistent saved decisions.
    thresholds = (p["decision_threshold"].to_numpy(dtype=float)
                  if "decision_threshold" in p else np.full(len(p), THRESHOLD))
    if not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("decision_threshold must contain finite values in [0, 1].")
    if not np.array_equal(p["y_pred"].to_numpy(), (prob >= thresholds).astype(int)):
        raise ValueError("Predicted labels must use the recorded decision threshold (default 0.5).")
    if not p["subject"].eq(p["outer_test_subject"]).all():
        raise ValueError("Prediction participant must equal the held-out participant.")
    if not p["original_condition"].isin([1, 2, 3]).all() or not np.array_equal(
        p["y_true"].to_numpy(), (p["original_condition"] == 2).astype(int).to_numpy()
    ):
        raise ValueError("Prediction original/binary labels are inconsistent.")
    reference = None
    for family, family_rows in p.groupby("model_family", sort=False):
        if set(family_rows["input_mode"]) != set(INPUT_MODES):
            raise ValueError(f"{family}: all four input modes are required.")
        for mode in INPUT_MODES:
            rows = family_rows[family_rows["input_mode"] == mode].sort_values("window_id")
            metadata = rows[list(META_COLUMNS)].reset_index(drop=True)
            if reference is not None and not metadata.equals(reference):
                raise ValueError(f"{family}: input modes do not use identical windows/labels.")
            reference = metadata

    metrics, errors = [], []
    for (family, mode, subject), rows in p.groupby(["model_family", "input_mode", "subject"], sort=False):
        metrics.append({"model_family": family, "input_mode": mode, "subject": subject,
                        "n_windows": len(rows), "macro_f1": _macro_f1(rows["y_true"], rows["y_pred"])})
        for condition, name, event in ((1, "baseline", "false_positive"), (2, "stress", "false_negative"), (3, "amusement", "false_positive")):
            subset = rows[rows["original_condition"] == condition]
            denominator = len(subset)
            numerator = int((subset["y_true"] != subset["y_pred"]).sum())
            errors.append({"model_family": family, "input_mode": mode, "subject": subject,
                           "original_condition": condition, "condition_name": name, "error_type": event,
                           "numerator": numerator, "denominator": denominator,
                           "error_rate": numerator / denominator if denominator else np.nan})
    participant = pd.DataFrame(metrics)
    summary = participant.groupby(["model_family", "input_mode"], sort=False).agg(
        n_participants=("subject", "size"), mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std")
    ).reset_index()
    deltas, overlaps = [], []
    for (family, subject), rows in participant.groupby(["model_family", "subject"], sort=False):
        score = rows.set_index("input_mode")["macro_f1"]
        for fusion in ("feature_fusion", "prediction_fusion"):
            for reference in ("eda", "bvp"):
                deltas.append({"model_family": family, "subject": subject, "fusion_mode": fusion,
                               "reference_mode": reference, "delta_macro_f1": float(score[fusion] - score[reference])})
        group = p[(p["model_family"] == family) & (p["subject"] == subject)]
        a = group[group["input_mode"] == "eda"].set_index("window_id")
        b = group[group["input_mode"] == "bvp"].set_index("window_id").loc[a.index]
        wrong_a = (a["y_true"] != a["y_pred"]).to_numpy()
        wrong_b = (b["y_true"] != b["y_pred"]).to_numpy()
        union = int((wrong_a | wrong_b).sum())
        intersection = int((wrong_a & wrong_b).sum())
        overlaps.append({"model_family": family, "subject": subject, "n_windows": len(a),
                         "both_correct": int((~wrong_a & ~wrong_b).sum()),
                         "eda_only_wrong": int((wrong_a & ~wrong_b).sum()),
                         "bvp_only_wrong": int((~wrong_a & wrong_b).sum()),
                         "both_wrong": intersection, "either_wrong": union,
                         "error_jaccard": intersection / union if union else np.nan})
    return {"participant_metrics": participant, "summary": summary,
            "fusion_deltas": pd.DataFrame(deltas), "condition_errors": pd.DataFrame(errors),
            "error_overlap": pd.DataFrame(overlaps)}


def evaluate_loso(features: pd.DataFrame) -> dict:
    """Run the fixed baseline in every participant LOSO fold (3 fits per fold)."""
    f = validate_feature_table(features)
    subjects = subject_order(f["subject"].tolist())
    if len(subjects) < 2:
        raise ValueError("LOSO needs at least two participants; a one-participant extract is inspection only.")
    predictions, folds, preprocessing = [], [], []
    column_sets = {"eda": list(EDA_FEATURES), "bvp": list(BVP_FEATURES),
                   "feature_fusion": list(EDA_FEATURES) + list(BVP_FEATURES)}
    for test_subject in subjects:
        train = f[f["subject"] != test_subject]
        test = f[f["subject"] == test_subject]
        if set(train["y_true"]) != {0, 1}:
            raise ValueError(f"{test_subject}: training fold must contain both binary classes.")
        folds.append({"outer_test_subject": test_subject,
                      "train_subjects": subject_order(train["subject"].tolist()),
                      "test_subjects": [test_subject], "n_train_windows": len(train), "n_test_windows": len(test),
                      "train_class_counts": {str(k): int(v) for k, v in train["y_true"].value_counts().items()}})
        probabilities = {}
        for mode, columns in column_sets.items():
            pipeline = make_baseline()
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                pipeline.fit(train[columns], train["y_true"])
            positive = int(np.flatnonzero(pipeline.classes_ == 1)[0])
            probabilities[mode] = pipeline.predict_proba(test[columns])[:, positive]
            # Save fitted statistics as an audit trail, not as model input.
            imputer = pipeline.named_steps["imputer"]
            scaler = pipeline.named_steps["scaler"]
            for i, column in enumerate(columns):
                preprocessing.append({"outer_test_subject": test_subject, "input_mode": mode,
                                      "feature": column, "train_all_missing": bool(train[column].isna().all()),
                                      "imputation_value": float(imputer.statistics_[i]),
                                      "scaler_mean": float(scaler.mean_[i]), "scaler_scale": float(scaler.scale_[i])})
        probabilities["prediction_fusion"] = (probabilities["eda"] + probabilities["bvp"]) / 2.0
        for mode in INPUT_MODES:
            rows = test[list(META_COLUMNS)].copy()
            rows["model_family"] = "baseline_lr"
            rows["input_mode"] = mode
            rows["outer_test_subject"] = test_subject
            rows["p_stress"] = probabilities[mode]
            rows["decision_threshold"] = THRESHOLD
            rows["y_pred"] = (probabilities[mode] >= THRESHOLD).astype(int)
            predictions.append(rows)
    predictions = pd.concat(predictions, ignore_index=True)
    outputs = summarize_predictions(predictions)
    outputs.update({"predictions": predictions, "preprocessing": pd.DataFrame(preprocessing),
                    "splits": {"protocol": "LOSO", "scope": "full" if set(subjects) == set(EXPECTED_SUBJECTS) else "pilot",
                               "subjects": subjects, "folds": folds}})
    return outputs
