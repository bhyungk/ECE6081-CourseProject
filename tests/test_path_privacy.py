"""Prevent local directory information from leaking into portable outputs."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from wesad_starter import cli
from wesad_starter.evaluation import META_COLUMNS, evaluate_loso
from wesad_starter.features import BVP_FEATURES, EDA_FEATURES
from wesad_starter.reporting import (
    file_sha256, portable_command, save_evaluation, verify_results, write_json,
)


@pytest.fixture
def feature_table():
    rows = []
    for subject_index, subject in enumerate(("S2", "S3")):
        for index, condition in enumerate((1, 2, 3)):
            target = int(condition == 2)
            rows.append({
                "subject": subject, "window_id": f"{subject}:{index:05d}",
                "start_s": index * 60, "end_s": (index + 1) * 60,
                "original_condition": condition, "y_true": target,
                **{name: float(target + subject_index / 10 + j / 100)
                   for j, name in enumerate((*EDA_FEATURES, *BVP_FEATURES))},
            })
    return pd.DataFrame(rows)


def save_feature_inputs(root, features, private_root):
    root.mkdir(parents=True, exist_ok=True)
    features.to_csv(root / "features.csv", index=False, float_format="%.17g")
    manifest = features[list(META_COLUMNS)].assign(included=True, exclusion_reason="")
    manifest.to_csv(root / "window_manifest.csv", index=False)
    subjects = features.subject.unique().tolist()
    write_json(root / "extraction_metadata.json", {
        "data_root": str(private_root), "subjects": subjects, "scope": "pilot",
        "included_windows": len(features),
        "original_condition_counts": {str(k): int(v) for k, v in features.original_condition.value_counts().items()},
        "source_files": [{"subject": s, "file": str(private_root / s / f"{s}.pkl")} for s in subjects],
        "files_sha256": {n: file_sha256(root / n) for n in ("features.csv", "window_manifest.csv")},
    })


@pytest.mark.parametrize("equals", [False, True])
def test_saved_command_uses_placeholders_for_local_paths(tmp_path, feature_table, equals):
    private_root = tmp_path / "personal-dataset"
    output = tmp_path / "personal-results"
    save_feature_inputs(output, feature_table, private_root)
    arguments = [str(tmp_path / "private-interpreter"), "-m", "wesad_starter", "run"]
    for option, value in (("--data-root", private_root), ("--features", output / "features.csv"), ("--output-dir", output)):
        arguments.extend([f"{option}={value}"] if equals else [option, str(value)])
    arguments.extend(["--subjects", "S2", "S3"])
    before = {n: file_sha256(output / n) for n in ("features.csv", "window_manifest.csv")}
    save_evaluation(evaluate_loso(feature_table), output, command=arguments)
    metadata = json.loads((output / "run_metadata.json").read_text())
    extraction = json.loads((output / "extraction_metadata.json").read_text())
    serialized = json.dumps([metadata, extraction])
    assert str(tmp_path) not in serialized
    assert extraction["data_root"] == "$WESAD_ROOT"
    assert extraction["source_files"] == [
        {"subject": s, "file": f"{s}/{s}.pkl"} for s in ("S2", "S3")
    ]
    assert metadata["command"][0] == "python"
    assert metadata["command"][-3:] == ["--subjects", "S2", "S3"]
    assert "$WESAD_ROOT" in serialized and "$FEATURES" in serialized and "$OUTPUT_DIR" in serialized
    assert before == {n: file_sha256(output / n) for n in before}
    assert verify_results(output)["status"] == "passed"


def test_command_redaction_also_handles_windows_and_relative_paths():
    drive = "C:" + "\\" + "private-user" + "\\" + "dataset"
    original = ["python", "--data-root=" + drive, "--features", "local-features.csv", "--output-dir", "local-results"]
    assert portable_command(original) == [
        "python", "--data-root=$WESAD_ROOT", "--features", "$FEATURES", "--output-dir", "$OUTPUT_DIR"
    ]
    assert original[1].endswith(drive)


def test_extraction_does_not_persist_or_print_local_roots(tmp_path, feature_table, monkeypatch, capsys):
    data_root = tmp_path / "personal-dataset"
    output = tmp_path / "personal-results"
    subjects = ["S2", "S3"]
    for subject in subjects:
        folder = data_root / subject
        folder.mkdir(parents=True)
        (folder / f"{subject}.pkl").write_bytes(b"unused synthetic fixture")
    manifest = feature_table[list(META_COLUMNS)].assign(included=True, exclusion_reason="")
    monkeypatch.setattr(cli, "build_feature_tables", lambda root, subjects: (manifest, feature_table))
    args = argparse.Namespace(data_root=data_root, output_dir=output, subjects=subjects)
    cli._extract(args)
    record = json.loads((output / "extraction_metadata.json").read_text())
    assert record["data_root"] == "$WESAD_ROOT"
    assert str(tmp_path) not in (output / "extraction_metadata.json").read_text()
    assert str(tmp_path) not in capsys.readouterr().out


def test_cli_missing_file_errors_do_not_echo_personal_paths(tmp_path, capsys):
    private = tmp_path / "personal-results"
    status = cli.main(["evaluate", "--features", str(private / "missing.csv"), "--output-dir", str(private)])
    output = capsys.readouterr()
    assert status == 1
    assert str(tmp_path) not in output.err
    assert "Cannot access" in output.err


@pytest.mark.parametrize("invalid_action", [False, True])
def test_cli_argument_errors_do_not_echo_personal_paths(tmp_path, capsys, invalid_action):
    private = str(tmp_path / "personal-folder")
    arguments = ([private] if invalid_action else
                 ["run", "--data-root", private, "--unknown-option", private])
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2
    output = capsys.readouterr()
    assert private not in output.err
    assert "personal-folder" not in output.err
    assert "Invalid command-line arguments" in output.err


def test_evaluate_sanitizes_legacy_source_provenance_on_copy(tmp_path, feature_table, capsys):
    source, destination = tmp_path / "personal-source", tmp_path / "personal-destination"
    private = tmp_path / "personal-dataset"
    save_feature_inputs(source, feature_table, private)
    cli._evaluate(source / "features.csv", destination, ["python", "--features", str(source / "features.csv")])
    assert str(tmp_path) not in (destination / "extraction_metadata.json").read_text()
    assert str(tmp_path) not in (destination / "run_metadata.json").read_text()
    assert verify_results(destination)["status"] == "passed"
    assert str(tmp_path) not in capsys.readouterr().out
