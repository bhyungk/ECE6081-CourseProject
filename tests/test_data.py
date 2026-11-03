import pickle
from pathlib import Path

import numpy as np
import pytest

from wesad_starter.data import discover_subjects, load_subject, segment_subject
from wesad_starter.features import BVP_FEATURES, EDA_FEATURES


def signals(seconds):
    return np.arange(seconds * 4, dtype=float), np.arange(seconds * 64, dtype=float)


def test_original_condition_transition_is_rejected_before_binary_mapping():
    eda, bvp = signals(120)
    labels = np.concatenate([np.full(30 * 700, 1), np.full(30 * 700, 3), np.full(60 * 700, 2)])
    manifest, features = segment_subject("S2", eda, bvp, labels)
    assert manifest["included"].tolist() == [False, True]
    assert manifest.iloc[0]["exclusion_reason"] == "mixed_original_labels"
    assert features["window_id"].tolist() == ["S2:00001"]
    assert features["y_true"].tolist() == [1]
    assert features["start_s"].tolist() == [60]
    assert features.iloc[0]["eda_min"] == 240
    assert features.iloc[0]["bvp_min"] == 3840


def test_partial_and_excluded_conditions_are_audited():
    eda, bvp = signals(121)
    labels = np.concatenate([np.full(60 * 700, 4), np.full(61 * 700, 1)])
    manifest, features = segment_subject("S3", eda, bvp, labels)
    assert manifest["included"].tolist() == [False, True, False]
    assert manifest.iloc[0]["exclusion_reason"] == "excluded_condition"
    assert manifest.iloc[2]["exclusion_reason"] == "incomplete_labels|incomplete_eda|incomplete_bvp"
    assert manifest.iloc[2]["start_s"] == 120
    assert features["window_id"].tolist() == ["S3:00001"]


def test_a_short_signal_does_not_shift_or_silently_truncate_the_grid():
    eda, bvp = signals(120)
    labels = np.full(120 * 700, 1)
    manifest, features = segment_subject("S4", eda[:-1], bvp, labels)
    assert len(manifest) == 2
    assert manifest.iloc[1]["exclusion_reason"] == "incomplete_eda"
    assert features["window_id"].tolist() == ["S4:00000"]


def test_nonfinite_sensor_samples_keep_the_window_and_other_modality():
    eda, bvp = signals(60)
    eda[20] = np.nan
    manifest, features = segment_subject("S5", eda, bvp, np.full(60 * 700, 3))
    assert manifest["included"].tolist() == [True]
    assert features.loc[:, list(EDA_FEATURES)].isna().all().all()
    assert np.isfinite(features.loc[:, list(BVP_FEATURES)].to_numpy()).all()
    assert features["original_condition"].tolist() == [3]
    assert features["y_true"].tolist() == [0]


@pytest.mark.parametrize("value", [1.5, np.nan, np.inf])
def test_malformed_labels_fail_instead_of_being_rounded(value):
    eda, bvp = signals(60)
    labels = np.ones(60 * 700)
    labels[0] = value
    with pytest.raises(ValueError, match="finite integers"):
        segment_subject("S2", eda, bvp, labels)


def test_discovery_is_numeric_and_requires_pickles(tmp_path: Path):
    for subject in ["S10", "S2"]:
        (tmp_path / subject).mkdir()
        (tmp_path / subject / f"{subject}.pkl").write_bytes(b"")
    assert discover_subjects(tmp_path) == ["S2", "S10"]
    (tmp_path / "S3").mkdir()
    with pytest.raises(FileNotFoundError, match="S3.pkl"):
        discover_subjects(tmp_path)


def test_loader_rejects_unexpected_pickle_globals(tmp_path: Path):
    (tmp_path / "S2").mkdir()
    (tmp_path / "S2" / "S2.pkl").write_bytes(pickle.dumps(Path("unwanted")))
    with pytest.raises(pickle.UnpicklingError, match="Unsupported pickle global"):
        load_subject(tmp_path, "S2")
