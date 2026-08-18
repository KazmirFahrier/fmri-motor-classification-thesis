#!/usr/bin/env python3
"""Nested selection of the temporal averaging window.

The `3:8` window is one of the choices the manuscript discloses as having been made
with all 62 subjects visible. Testing other *offsets* would need re-extraction, but
the frozen checkpoints retain all eight lags, so the choice of **which lags to
average** can be nested inside the folds and its cost measured.

This is the second measured instance of the design-search effect. ANOVA feature
selection went from `+0.0143` with the threshold fixed across the cohort to `+0.0060`
with it selected on inner folds — nesting removed more than half. If the temporal
window behaves similarly, the manuscript can say something quantitative about what
its disclosed search is worth rather than only flagging that one happened.

Candidate windows are contiguous lag ranges, since a haemodynamic response is smooth
and non-contiguous subsets would be fitting noise. The full `0:8` range is the frozen
pipeline's own choice and is included so the comparison is against what the project
actually does.
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


def contiguous_windows(lag_count: int, minimum: int = 2) -> list[tuple[int, int]]:
    return [
        (start, stop)
        for start in range(lag_count)
        for stop in range(start + minimum, lag_count + 1)
    ]


def scores_for(
    sequence: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    window: tuple[int, int],
    c_value: float,
    model: str,
) -> np.ndarray:
    block = sequence[:, window[0] : window[1], :].mean(axis=1, dtype=np.float32)
    mean, scale = standardize(block, train_idx)
    z_train, z_eval = dual_basis(
        (block[train_idx] - mean) / scale, (block[eval_idx] - mean) / scale
    )
    return fit_projected(
        model, z_train, y[train_idx], z_eval,
        z_train @ z_train.T, z_eval @ z_train.T, c_value, 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested temporal window selection.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--model", default="linear_svm")
    parser.add_argument("--fixed-c", type=float, default=0.0001)
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument("--split-limit", type=int)
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    lag_count = sequence.shape[1]
    windows = contiguous_windows(lag_count)
    print(f"{lag_count} lags, {len(windows)} contiguous windows", flush=True)

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    rows, selections = [], []
    fixed_full = []
    for split in splits:
        inner = inner_splits(
            records, split["train_idx"], "subject", args.inner_subject_fold_count
        )
        inner_means = {}
        for window in windows:
            values = []
            for inner_split in inner:
                s = scores_for(
                    sequence, y, inner_split["train_idx"], inner_split["val_idx"],
                    window, args.fixed_c, args.model,
                )
                values.append(
                    metrics(y[inner_split["val_idx"]], s.argmax(axis=1))["balanced_accuracy"]
                )
            inner_means[f"{window[0]}:{window[1]}"] = float(np.mean(values))
        best = max(windows, key=lambda w: inner_means[f"{w[0]}:{w[1]}"])

        s = scores_for(
            sequence, y, split["train_idx"], split["val_idx"], best, args.fixed_c, args.model
        )
        for rule, prediction in (
            ("independent", s.argmax(axis=1).astype(np.int64)),
            ("balanced", apply_balanced_assignment(s, split["val_idx"], records)),
        ):
            rows.append(
                {
                    "split": split["split"],
                    "selected_window": f"{best[0]}:{best[1]}",
                    "prediction_rule": rule,
                    "balanced_accuracy": float(
                        metrics(y[split["val_idx"]], prediction)["balanced_accuracy"]
                    ),
                }
            )
        # The frozen pipeline's own choice, for reference on the same fold.
        s_full = scores_for(
            sequence, y, split["train_idx"], split["val_idx"], (0, lag_count),
            args.fixed_c, args.model,
        )
        fixed_full.append(
            metrics(y[split["val_idx"]], s_full.argmax(axis=1))["balanced_accuracy"]
        )
        selections.append(
            {"split": split["split"], "window": f"{best[0]}:{best[1]}",
             "inner_means": inner_means}
        )
        print(
            f"{split['split']} selected {best[0]}:{best[1]} "
            f"indep={rows[-2]['balanced_accuracy']:.4f} full={fixed_full[-1]:.4f}",
            flush=True,
        )

    # Oracle: the single window that is best on average across outer folds, which is
    # what choosing a window by looking at cohort-wide results amounts to.
    oracle = {}
    for window in windows:
        values = []
        for split in splits:
            s = scores_for(
                sequence, y, split["train_idx"], split["val_idx"], window,
                args.fixed_c, args.model,
            )
            values.append(
                metrics(y[split["val_idx"]], s.argmax(axis=1))["balanced_accuracy"]
            )
        oracle[f"{window[0]}:{window[1]}"] = float(np.mean(values))
    best_oracle = max(oracle, key=oracle.get)

    summary = {}
    for rule in ("independent", "balanced"):
        values = [r["balanced_accuracy"] for r in rows if r["prediction_rule"] == rule]
        summary[rule] = {"mean": float(np.mean(values)), "sd": float(np.std(values))}

    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "model": args.model,
        "outer_split_count": len(splits),
        "nested": summary,
        "frozen_full_window": float(np.mean(fixed_full)),
        "oracle_best_window": {"window": best_oracle, "accuracy": oracle[best_oracle]},
        "oracle_all_windows": oracle,
        "selected_window_counts": dict(Counter(s["window"] for s in selections)),
        "selections": selections,
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"\nnested         {summary['independent']['mean']:.4f}")
    print(f"frozen 0:{lag_count} window {np.mean(fixed_full):.4f}")
    print(f"oracle {best_oracle}    {oracle[best_oracle]:.4f}")
    print(f"selected windows: {dict(Counter(s['window'] for s in selections))}")


if __name__ == "__main__":
    main()
