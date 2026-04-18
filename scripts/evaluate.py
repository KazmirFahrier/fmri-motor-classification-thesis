#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
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
from fmri_pipeline.training import evaluate_checkpoint
from fmri_pipeline.utils.io import read_json, write_json


FOLD_SELECTOR = re.compile(r"^fold:(?P<fold>\d+):(?P<split>train|val)$")


def resolve_sample_ids(split_arg: str, splits_json_path: str | None):
    p = Path(split_arg)
    if p.exists() and p.is_file():
        data = read_json(p)
        if isinstance(data, dict) and "sample_ids" in data:
            return [int(x) for x in data["sample_ids"]], data.get("name", p.stem)
        if isinstance(data, list):
            return [int(x) for x in data], p.stem
        raise ValueError("Split file must be either {'sample_ids': [...]} or a list of sample IDs")

    if not splits_json_path:
        raise ValueError("Named split selectors require --splits")

    splits = read_json(splits_json_path)
    if split_arg == "holdout":
        return [int(x) for x in splits["holdout_sample_ids"]], "holdout"

    m = FOLD_SELECTOR.match(split_arg)
    if m:
        fold_idx = int(m.group("fold"))
        split_name = m.group("split")
        for fold in splits["folds"]:
            if int(fold["fold"]) == fold_idx:
                key = "train_sample_ids" if split_name == "train" else "val_sample_ids"
                return [int(x) for x in fold[key]], f"fold_{fold_idx:02d}_{split_name}"
        raise ValueError(f"Fold {fold_idx} not found")

    raise ValueError(f"Unrecognized split selector: {split_arg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on a chosen split.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path from train/run_cv.")
    parser.add_argument("--index", required=True, help="Parquet manifest path.")
    parser.add_argument("--split", required=True, help="Split selector (holdout, fold:<k>:val) or split JSON file path.")
    parser.add_argument("--out-dir", required=True, help="Output directory for metrics and plots.")
    parser.add_argument("--splits", default=None, help="Main splits JSON (required for named selectors).")
    args = parser.parse_args()

    manifest_df = read_manifest(args.index)
    sample_ids, split_name = resolve_sample_ids(args.split, args.splits)

    summary = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        manifest_df=manifest_df,
        sample_ids=sample_ids,
        out_dir=args.out_dir,
        split_name=split_name,
    )

    write_json(summary, Path(args.out_dir) / "summary.json")
    print(f"Evaluation complete: {Path(args.out_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
