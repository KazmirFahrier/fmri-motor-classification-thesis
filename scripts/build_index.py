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

from fmri_pipeline.data.manifest import build_manifest, save_manifest_parquet
from fmri_pipeline.utils.io import write_json


def _parse_class_names(raw: str) -> list[str]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    if not names:
        raise ValueError("--class-names must contain at least one class")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified parquet manifest for fMRI batch data.")
    parser.add_argument("--data-roots", nargs="+", required=True, help="One or more batch root directories.")
    parser.add_argument("--out-index", required=True, help="Output parquet index path.")
    parser.add_argument("--class-names", required=True, help="Comma-separated class names.")
    parser.add_argument("--out-qc", default=None, help="Optional QC JSON path.")
    parser.add_argument("--strict", action="store_true", help="Fail if parse failures occur.")
    args = parser.parse_args()

    class_names = _parse_class_names(args.class_names)
    df, qc = build_manifest(args.data_roots, class_names)

    save_manifest_parquet(df, args.out_index)

    out_qc = args.out_qc
    if out_qc is None:
        p = Path(args.out_index)
        out_qc = str(p.with_name(f"{p.stem}_qc.json"))

    write_json(qc, out_qc)

    print(f"Manifest written: {args.out_index}")
    print(f"QC written: {out_qc}")
    print(f"Samples: {qc['num_samples']}, Subjects: {qc['num_subjects']}, Runs: {qc['num_runs']}")

    if args.strict and int(qc.get("parse_failure_count", 0)) > 0:
        raise SystemExit("Parse failures detected in strict mode")


if __name__ == "__main__":
    main()
