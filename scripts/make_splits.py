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
from fmri_pipeline.data.splits import assert_no_subject_leakage, create_subjectwise_splits
from fmri_pipeline.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic subject-wise CV and holdout splits.")
    parser.add_argument("--index", required=True, help="Parquet manifest path.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for split generation.")
    parser.add_argument("--holdout-subject-count", type=int, required=True, help="Number of subjects in holdout set.")
    parser.add_argument("--cv-folds", type=int, required=True, help="Number of CV folds.")
    parser.add_argument("--out-splits", required=True, help="Output splits JSON path.")
    args = parser.parse_args()

    df = read_manifest(args.index)

    split_data = create_subjectwise_splits(
        df,
        seed=args.seed,
        holdout_subject_count=args.holdout_subject_count,
        cv_folds=args.cv_folds,
    )
    split_data["class_names"] = sorted(df["class_name"].unique().tolist())
    assert_no_subject_leakage(split_data)

    out_path = Path(args.out_splits)
    write_json(split_data, out_path)

    print(f"Splits written: {out_path}")
    print(
        f"Subjects total={split_data['num_subjects']} holdout={len(split_data['holdout_subjects'])} cv={len(split_data['cv_subjects'])}"
    )


if __name__ == "__main__":
    main()
