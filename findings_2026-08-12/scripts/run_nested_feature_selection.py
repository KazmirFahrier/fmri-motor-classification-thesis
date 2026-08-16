#!/usr/bin/env python3
"""ANOVA feature selection with the threshold chosen inside the folds.

Univariate ANOVA feature selection improves the linear decoder, but the improvement
measured so far is **optimistic**: the retention threshold was chosen by looking at
results across the whole cohort, which is exactly the design-search problem this
project already discloses for the `3:8` window and the covariance caps. A threshold
picked that way is a hyperparameter fitted on the test set.

Here the threshold is selected jointly with `C` on the inner subject folds of each
outer fold, so nothing about the choice sees held-out subjects. That makes the result
quotable rather than indicative, and it also tests something the fixed-threshold run
cannot: whether the *best* threshold is stable across folds, or whether the apparent
gain came from a threshold that happens to suit the cohort as a whole.

The ANOVA statistic is computed on training rows with training labels only. Univariate
selection inside the training fold is standard MVPA practice; computing it on all
events would leak.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (REPO_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from run_balanced_event_assignment import apply_balanced_assignment, metrics  # noqa: E402
from run_detrended_pair_feature_selection import (  # noqa: E402
    inner_splits,
    load_checkpoints,
    outer_splits,
)
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402
from run_standard_mvpa_baseline import (  # noqa: E402
    dual_basis,
    fit_projected,
    standardize,
)


def anova_statistic(block: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    grand = block.mean(axis=0)
    between = np.zeros(block.shape[1], dtype=np.float64)
    within = np.zeros(block.shape[1], dtype=np.float64)
    for class_id in range(class_count):
        rows = block[labels == class_id]
        if len(rows) < 2:
            continue
        centre = rows.mean(axis=0)
        between += len(rows) * (centre - grand) ** 2
        within += ((rows - centre) ** 2).sum(axis=0)
    return between / np.maximum(within, 1e-12)


def evaluate(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    quantile: float,
    c_value: float,
    class_count: int,
    model: str,
) -> np.ndarray:
    statistic = anova_statistic(x[train_idx], y[train_idx], class_count)
    keep = statistic > np.quantile(statistic, quantile)
    if keep.sum() < 20:
        keep = statistic >= np.sort(statistic)[-20]
    block = x[:, keep]
    mean, scale = standardize(block, train_idx)
    z_train, z_eval = dual_basis(
        (block[train_idx] - mean) / scale, (block[eval_idx] - mean) / scale
    )
    return fit_projected(
        model, z_train, y[train_idx], z_eval,
        z_train @ z_train.T, z_eval @ z_train.T, c_value, 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested ANOVA feature selection.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--model", default="linear_svm")
    parser.add_argument("--quantiles", nargs="+", type=float,
                        default=[0.0, 0.5, 0.8, 0.9, 0.95])
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.0001, 0.001, 0.01])
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument("--split-limit", type=int)
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    x = sequence.mean(axis=1, dtype=np.float32)
    del sequence
    class_count = int(y.max()) + 1

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    rows, selections = [], []
    for split in splits:
        inner = inner_splits(
            records, split["train_idx"], "subject", args.inner_subject_fold_count
        )
        best, best_score = None, -1.0
        inner_table = {}
        for quantile in args.quantiles:
            for c_value in args.c_grid:
                total = []
                for inner_split in inner:
                    scores = evaluate(
                        x, y, inner_split["train_idx"], inner_split["val_idx"],
                        quantile, c_value, class_count, args.model,
                    )
                    total.append(
                        metrics(
                            y[inner_split["val_idx"]], scores.argmax(axis=1)
                        )["balanced_accuracy"]
                    )
                mean_inner = float(np.mean(total))
                inner_table[f"q{quantile}_c{c_value}"] = round(mean_inner, 6)
                if mean_inner > best_score:
                    best_score, best = mean_inner, (quantile, c_value)

        quantile, c_value = best
        scores = evaluate(
            x, y, split["train_idx"], split["val_idx"],
            quantile, c_value, class_count, args.model,
        )
        for rule, prediction in (
            ("independent", scores.argmax(axis=1).astype(np.int64)),
            ("balanced", apply_balanced_assignment(scores, split["val_idx"], records)),
        ):
            rows.append(
                {
                    "split": split["split"],
                    "selected_quantile": quantile,
                    "selected_c": c_value,
                    "prediction_rule": rule,
                    "balanced_accuracy": float(
                        metrics(y[split["val_idx"]], prediction)["balanced_accuracy"]
                    ),
                }
            )
        selections.append(
            {"split": split["split"], "quantile": quantile, "c": c_value,
             "inner_best": round(best_score, 6), "inner_table": inner_table}
        )
        print(f"{split['split']} selected q={quantile} C={c_value} "
              f"indep={rows[-2]['balanced_accuracy']:.4f}", flush=True)

    summary = {}
    for rule in ("independent", "balanced"):
        values = [r["balanced_accuracy"] for r in rows if r["prediction_rule"] == rule]
        summary[rule] = {"mean": float(np.mean(values)), "sd": float(np.std(values))}

    Path(args.out_json).write_text(
        json.dumps(
            {
                "checkpoint_dir": args.checkpoint_dir,
                "model": args.model,
                "quantiles": args.quantiles,
                "c_grid": args.c_grid,
                "outer_split_count": len(splits),
                "selected_quantile_counts": dict(
                    Counter(s["quantile"] for s in selections)
                ),
                "summary": summary,
                "selections": selections,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nselected quantiles: {dict(Counter(s['quantile'] for s in selections))}")
    for rule, row in summary.items():
        print(f"{rule:14s} {row['mean']:.4f} (sd {row['sd']:.4f})")


if __name__ == "__main__":
    main()
