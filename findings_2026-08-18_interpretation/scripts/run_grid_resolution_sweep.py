#!/usr/bin/env python3
"""Sweep the spatial feature grid, the last unquantified disclosed design choice.

The manuscript discloses three choices made with all 62 subjects visible: the temporal
averaging window, the covariance caps, and the `24^3` feature grid. The first has since
been quantified — nesting removed its entire apparent gain — and ANOVA feature selection
lost more than half of its. The grid never has, because unlike those it cannot be tested
without re-extracting the cohort, which is why it stayed open.

The `24^3` arm of this extraction reproduces the frozen checkpoints **bit-identically**,
so any difference between grids is the grid and nothing else.

Both the fixed and the nested estimate are reported. A fixed-grid number chosen by
looking across the whole cohort would be exactly the kind of cohort-visible choice this
analysis exists to measure.

No smoothing is applied. A `3x3x3` box filter spans a different physical distance on each
grid, so applying it would confound resolution with smoothing extent; the comparison here
is of resolution alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "findings_2026-08-12" / "scripts",
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from run_balanced_event_assignment import apply_balanced_assignment, metrics  # noqa: E402
from run_detrended_pair_feature_selection import (  # noqa: E402
    inner_splits,
    load_checkpoints,
    outer_splits,
)
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402
from run_standard_mvpa_baseline import dual_basis, fit_projected, standardize  # noqa: E402


def scores_for(block, y, train_idx, eval_idx, c_value, model):
    mean, scale = standardize(block, train_idx)
    z_train, z_eval = dual_basis(
        (block[train_idx] - mean) / scale, (block[eval_idx] - mean) / scale
    )
    return fit_projected(model, z_train, y[train_idx], z_eval,
                         z_train @ z_train.T, z_eval @ z_train.T, c_value, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial grid resolution sweep.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--grids", nargs="+", type=int, default=[16, 24, 32])
    parser.add_argument("--model", default="linear_svm")
    parser.add_argument("--fixed-c", type=float, default=0.0001)
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument("--split-limit", type=int)
    args = parser.parse_args()

    keys = [f"grid{g}_offset_3_length_8_sequence" for g in args.grids]
    feature_dict, y, records = load_checkpoints(Path(args.checkpoint_dir), keys)

    blocks = {}
    for grid, key in zip(args.grids, keys):
        sequence, _ = preprocess_sequence(feature_dict.pop(key), records)
        blocks[grid] = sequence.mean(axis=1, dtype=np.float32)
        del sequence
        print(f"grid {grid}: {blocks[grid].shape[1]} features", flush=True)

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    fixed = {g: {"independent": [], "balanced": []} for g in args.grids}
    nested = {"independent": [], "balanced": []}
    selections = []

    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        for grid in args.grids:
            s = scores_for(blocks[grid], y, train_idx, val_idx, args.fixed_c, args.model)
            for rule, prediction in (
                ("independent", s.argmax(axis=1).astype(np.int64)),
                ("balanced", apply_balanced_assignment(s, val_idx, records)),
            ):
                fixed[grid][rule].append(
                    float(metrics(y[val_idx], prediction)["balanced_accuracy"]))

        inner = inner_splits(records, train_idx, "subject", args.inner_subject_fold_count)
        inner_means = {}
        for grid in args.grids:
            values = []
            for inner_split in inner:
                s = scores_for(blocks[grid], y, inner_split["train_idx"],
                               inner_split["val_idx"], args.fixed_c, args.model)
                values.append(metrics(y[inner_split["val_idx"]],
                                      s.argmax(axis=1))["balanced_accuracy"])
            inner_means[grid] = float(np.mean(values))
        best = max(args.grids, key=lambda g: inner_means[g])

        s = scores_for(blocks[best], y, train_idx, val_idx, args.fixed_c, args.model)
        for rule, prediction in (
            ("independent", s.argmax(axis=1).astype(np.int64)),
            ("balanced", apply_balanced_assignment(s, val_idx, records)),
        ):
            nested[rule].append(
                float(metrics(y[val_idx], prediction)["balanced_accuracy"]))
        selections.append({"split": split["split"], "grid": best,
                           "inner_means": inner_means})
        print(f"{split['split']} selected {best} nested={nested['independent'][-1]:.4f}",
              flush=True)

    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "model": args.model,
        "grids": args.grids,
        "outer_split_count": len(splits),
        "fixed": {str(g): {r: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                           for r, v in fixed[g].items()} for g in args.grids},
        "nested": {r: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                   for r, v in nested.items()},
        "selected_grid_counts": {str(k): v for k, v in
                                 Counter(s["grid"] for s in selections).items()},
        "selections": selections,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print("\n  grid    features   independent   balanced")
    for g in args.grids:
        print(f"  {g:<7} {g**3:<10} {np.mean(fixed[g]['independent']):.4f}        "
              f"{np.mean(fixed[g]['balanced']):.4f}")
    print(f"  nested             {np.mean(nested['independent']):.4f}        "
          f"{np.mean(nested['balanced']):.4f}")
    print(f"\nselected: {dict(Counter(s['grid'] for s in selections))}")


if __name__ == "__main__":
    main()
