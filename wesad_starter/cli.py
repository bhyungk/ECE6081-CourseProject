"""Extract, run, evaluate, and verify the fixed WESAD classroom baseline."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import EXPECTED_SUBJECTS, build_feature_tables, discover_subjects
from .evaluation import evaluate_loso
from .features import BVP_FEATURES, EDA_FEATURES
from .reporting import (
    environment_record, file_sha256, save_evaluation, validate_extraction_provenance,
    verify_results, write_json,
)


def _extract(args: argparse.Namespace) -> Path:
    root = args.data_root.expanduser().resolve()
    discovered = discover_subjects(root)
    subjects = args.subjects or list(EXPECTED_SUBJECTS)
    if len(subjects) != len(set(subjects)):
        raise ValueError("--subjects contains duplicates.")
    missing = sorted(set(subjects) - set(discovered))
    if missing:
        raise ValueError(f"Missing requested participants: {missing}")
    if not args.subjects and set(discovered) != set(EXPECTED_SUBJECTS):
        raise ValueError("Default full run requires exactly the 15 documented participants. Use --subjects for an explicit pilot.")
    manifest, features = build_feature_tables(root, subjects=subjects)
    if features.empty:
        raise ValueError("No valid windows were extracted.")
    if set(features["subject"]) != set(subjects):
        raise ValueError("Some requested participants have no eligible windows; inspect their recordings before evaluation.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "window_manifest.csv", index=False, float_format="%.17g", na_rep="")
    features.to_csv(output_dir / "features.csv", index=False, float_format="%.17g", na_rep="")
    write_json(output_dir / "extraction_metadata.json", {
        "source_type": "WESAD synchronized participant pickle", "data_root": "$WESAD_ROOT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "environment": environment_record(),
        "subjects": subjects, "scope": "full" if set(subjects) == set(EXPECTED_SUBJECTS) else "pilot",
        "window_seconds": 60, "step_seconds": 60, "grid_origin_seconds": 0,
        "sample_rates_hz": {"eda": 4, "bvp": 64, "labels": 700},
        "purity_rule": "all original condition samples identical, before binary mapping",
        "label_mapping": {"1": 0, "2": 1, "3": 0}, "included_windows": len(features),
        "original_condition_counts": {str(k): int(v) for k, v in features["original_condition"].value_counts().items()},
        "feature_columns": list(EDA_FEATURES) + list(BVP_FEATURES),
        "source_files": [{"subject": s, "file": f"{s}/{s}.pkl", "size_bytes": (root / s / f"{s}.pkl").stat().st_size,
                          "mtime_ns": (root / s / f"{s}.pkl").stat().st_mtime_ns} for s in subjects],
        "files_sha256": {n: file_sha256(output_dir / n) for n in ("features.csv", "window_manifest.csv")},
    })
    print(f"Extracted {len(features)} windows from {len(subjects)} participants; results saved.")
    return output_dir / "features.csv"


def _evaluate(feature_path: Path, output_dir: Path, command: list[str]) -> None:
    feature_path = feature_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    features = pd.read_csv(feature_path)
    if feature_path == output_dir / "features.csv":
        validate_extraction_provenance(output_dir, features)
    outputs = evaluate_loso(features)
    output_dir.mkdir(parents=True, exist_ok=True)
    if feature_path != output_dir / "features.csv":
        shutil.copy2(feature_path, output_dir / "features.csv")
        # Copy provenance only when it belongs to the supplied features.
        origin = feature_path.parent / "extraction_metadata.json"
        for name in ("extraction_metadata.json", "window_manifest.csv"):
            (output_dir / name).unlink(missing_ok=True)
        if origin.is_file():
            provenance = json.loads(origin.read_text(encoding="utf-8"))
            if provenance.get("files_sha256", {}).get("features.csv") == file_sha256(feature_path):
                shutil.copy2(origin, output_dir / "extraction_metadata.json")
                manifest = feature_path.parent / "window_manifest.csv"
                if manifest.is_file() and provenance.get("files_sha256", {}).get("window_manifest.csv") == file_sha256(manifest):
                    shutil.copy2(manifest, output_dir / "window_manifest.csv")
    save_evaluation(outputs, output_dir, command=command)
    print(outputs["summary"].to_string(index=False))
    print(json.dumps(verify_results(output_dir), ensure_ascii=False, indent=2))


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse otherwise echoes invalid arguments, including personal paths.
        self.print_usage(sys.stderr)
        self.exit(2, "ERROR: Invalid command-line arguments; use --help for options.\n")


def parser() -> argparse.ArgumentParser:
    p = _ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("extract", "run"):
        q = sub.add_parser(name, help="Extract features only" if name == "extract" else "Extract, evaluate, and verify")
        q.add_argument("--data-root", type=Path, required=True, help="Folder containing S2/, ..., S17/")
        q.add_argument("--output-dir", type=Path, default=Path("results/baseline"))
        q.add_argument("--subjects", nargs="+", help="Explicit participant subset (pilot); omit for all 15")
    q = sub.add_parser("evaluate", help="Evaluate previously extracted features")
    q.add_argument("--features", type=Path, required=True)
    q.add_argument("--output-dir", type=Path, default=Path("results/baseline"))
    q = sub.add_parser("verify", help="Verify saved results without fitting models")
    q.add_argument("--output-dir", type=Path, default=Path("results/baseline"))
    return p


def _safe_error_message(error: Exception, args: argparse.Namespace) -> str:
    """Keep useful diagnostics without echoing personal command-line paths."""
    if isinstance(error, OSError):
        return "Cannot access a required file or directory; check the input and output options."
    message = str(error)
    placeholders = {"data_root": "$WESAD_ROOT", "features": "$FEATURES", "output_dir": "$OUTPUT_DIR"}
    replacements = []
    for field, placeholder in placeholders.items():
        value = getattr(args, field, None)
        if value is not None:
            path = Path(value)
            replacements.extend((str(candidate), placeholder) for candidate in (path, path.expanduser(), path.expanduser().resolve()))
    for value, placeholder in sorted(replacements, key=lambda pair: len(pair[0]), reverse=True):
        if value:
            message = message.replace(value, placeholder)
    return re.sub(r"(?<![A-Za-z0-9_$])(?:[A-Za-z]:[\\/]|/)[^\s'\"<>]*", "$PATH", message)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    command = ["python", "-m", "wesad_starter"] + (argv if argv is not None else sys.argv[1:])
    try:
        if args.action in ("extract", "run"):
            feature_path = _extract(args)
            if args.action == "run":
                _evaluate(feature_path, args.output_dir, command)
        elif args.action == "evaluate":
            _evaluate(args.features, args.output_dir, command)
        else:
            print(json.dumps(verify_results(args.output_dir), ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {_safe_error_message(exc, args)}", file=sys.stderr)
        return 1
    return 0
