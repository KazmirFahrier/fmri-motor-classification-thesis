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

from fmri_pipeline.report import aggregate_cv, export_publication_bundle, summaries_to_dataframe
from fmri_pipeline.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication report tables/plots from run outputs.")
    parser.add_argument("--run-dir", required=True, help="Run root directory (contains cv_fold_summaries.json).")
    parser.add_argument("--out-dir", required=True, help="Output report bundle directory.")
    parser.add_argument("--holdout-summary", default=None, help="Optional holdout summary JSON path.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fold_summaries = read_json(run_dir / "cv_fold_summaries.json")

    holdout_summary = None
    if args.holdout_summary:
        holdout_summary = read_json(args.holdout_summary)
    else:
        inferred = run_dir / "holdout_eval" / "holdout_summary.json"
        if inferred.exists():
            holdout_summary = read_json(inferred)

    fold_df = summaries_to_dataframe(fold_summaries)
    cv_stats = aggregate_cv(fold_df)

    export_publication_bundle(
        fold_df=fold_df,
        cv_stats_df=cv_stats,
        holdout_summary=holdout_summary,
        out_dir=args.out_dir,
    )

    print(f"Report written to: {args.out_dir}")


if __name__ == "__main__":
    main()
