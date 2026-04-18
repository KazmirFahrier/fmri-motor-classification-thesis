#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Notebook execution fallback (__file__ is undefined in cells).
    ROOT = Path.cwd().resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fmri_pipeline.data.manifest import read_manifest
from fmri_pipeline.data.qc import check_nifti_integrity, run_manifest_checks
from fmri_pipeline.utils.io import read_json, write_json


def _parse_class_names(raw: str) -> list[str]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    if not names:
        raise ValueError("--class-names must contain at least one class")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manifest integrity and class/run consistency.")
    parser.add_argument("--index", required=True, help="Parquet manifest path.")
    parser.add_argument("--class-names", required=True, help="Comma-separated expected class names.")
    parser.add_argument("--manifest-qc", default=None, help="QC JSON from build_index, for parse failure info.")
    parser.add_argument("--check-all-files", action="store_true", help="Run nibabel load check for all files.")
    parser.add_argument("--max-integrity-files", type=int, default=500, help="When not checking all files, number of files to sample.")
    parser.add_argument("--out-report", required=True, help="Output QC report JSON.")
    parser.add_argument("--strict", action="store_true", help="Fail on any check failure.")
    args = parser.parse_args()

    expected_classes = _parse_class_names(args.class_names)
    df = read_manifest(args.index)

    parse_failure_count = 0
    if args.manifest_qc:
        qc_data = read_json(args.manifest_qc)
        parse_failure_count = int(qc_data.get("parse_failure_count", 0))

    report, failures = run_manifest_checks(
        df,
        expected_class_names=expected_classes,
        parse_failure_count=parse_failure_count,
    )

    filepaths = df["filepath"].tolist()
    integrity = check_nifti_integrity(
        filepaths,
        max_files=None if args.check_all_files else args.max_integrity_files,
    )
    report["integrity"] = integrity

    if int(integrity["corrupted_count"]) > 0:
        failures.append(f"Corrupted NIfTI files detected: {integrity['corrupted_count']}")

    report["failure_count"] = len(failures)
    report["failures"] = failures

    write_json(report, args.out_report)

    print(f"QC report written: {args.out_report}")
    print(f"Failure count: {len(failures)}")
    if failures:
        for f in failures:
            print(f"- {f}")

    if args.strict and failures:
        raise SystemExit("Data QC failed in strict mode")


if __name__ == "__main__":
    main()
