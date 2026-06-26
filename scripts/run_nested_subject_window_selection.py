#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    apply_balanced_assignment,
    center_by_subject_run,
    centroid_matrix,
    metrics,
    score_with_centroids,
)
from run_detrended_pair_feature_selection import inner_splits, load_checkpoints, outer_splits
from run_subject_window_heterogeneity import DEFAULT_WINDOWS
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def balanced_accuracy_for_split(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> float:
    centroids = centroid_matrix(x[train_idx], y[train_idx])
    scores = score_with_centroids(x[val_idx], centroids)
    pred = apply_balanced_assignment(scores, val_idx, records)
    return float(metrics(y[val_idx], pred)["balanced_accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a global continuous-BOLD window inside each outer subject training split."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--windows", nargs="*", default=DEFAULT_WINDOWS)
    parser.add_argument("--fixed-window", default="offset_3_length_8")
    parser.add_argument("--subject-fold-count", type=int, default=6)
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument("--subject-seeds", nargs="*", type=int, default=[11, 23, 37, 53, 71])
    args = parser.parse_args()

    records = None
    labels = None
    splits = None
    window_rows: dict[str, dict[str, dict]] = {}
    for window in args.windows:
        print(f"evaluating nested scores for {window}", flush=True)
        features, loaded_labels, loaded_records = load_checkpoints(
            Path(args.checkpoint_dir),
            [window],
        )
        if records is None:
            records = loaded_records
            labels = loaded_labels
            splits = outer_splits(
                records,
                "subject",
                args.subject_fold_count,
                args.subject_seeds,
            )
        elif records != loaded_records or not np.array_equal(labels, loaded_labels):
            raise ValueError(f"Checkpoint label/record order changed while loading {window}.")

        centered = center_by_subject_run(features[window], records)
        detrended, _ = temporal_detrend_by_subject_run(centered, records, degree=1)
        per_split = {}
        for split in splits:
            inner = inner_splits(
                records,
                split["train_idx"],
                "subject",
                args.inner_subject_fold_count,
            )
            inner_scores = [
                balanced_accuracy_for_split(
                    detrended,
                    labels,
                    records,
                    inner_split["train_idx"],
                    inner_split["val_idx"],
                )
                for inner_split in inner
            ]
            per_split[split["split"]] = {
                "inner_scores": inner_scores,
                "mean_inner_balanced_accuracy": float(np.mean(inner_scores)),
                "outer_balanced_accuracy": balanced_accuracy_for_split(
                    detrended,
                    labels,
                    records,
                    split["train_idx"],
                    split["val_idx"],
                ),
            }
        window_rows[window] = per_split
        del features, centered, detrended

    selected_rows = []
    for split in splits:
        split_name = split["split"]
        selected_window = min(
            args.windows,
            key=lambda window: (
                -window_rows[window][split_name]["mean_inner_balanced_accuracy"],
                args.windows.index(window),
            ),
        )
        selected_rows.append(
            {
                "split": split_name,
                "selected_window": selected_window,
                "selected_inner_balanced_accuracy": window_rows[selected_window][split_name][
                    "mean_inner_balanced_accuracy"
                ],
                "selected_outer_balanced_accuracy": window_rows[selected_window][split_name][
                    "outer_balanced_accuracy"
                ],
                "fixed_window_outer_balanced_accuracy": window_rows[args.fixed_window][split_name][
                    "outer_balanced_accuracy"
                ],
                "window_diagnostics": {
                    window: window_rows[window][split_name]
                    for window in args.windows
                },
            }
        )

    selected_values = np.asarray(
        [row["selected_outer_balanced_accuracy"] for row in selected_rows]
    )
    fixed_values = np.asarray(
        [row["fixed_window_outer_balanced_accuracy"] for row in selected_rows]
    )
    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "windows": args.windows,
        "fixed_window": args.fixed_window,
        "subject_seeds": args.subject_seeds,
        "rows": selected_rows,
        "summary": {
            "outer_split_count": len(selected_rows),
            "selected_window_counts": dict(Counter(row["selected_window"] for row in selected_rows)),
            "mean_nested_selected_balanced_accuracy": float(np.mean(selected_values)),
            "mean_fixed_window_balanced_accuracy": float(np.mean(fixed_values)),
            "mean_selected_minus_fixed": float(np.mean(selected_values - fixed_values)),
            "selected_better_splits": int(np.sum(selected_values > fixed_values)),
            "selected_tied_splits": int(np.sum(selected_values == fixed_values)),
            "selected_worse_splits": int(np.sum(selected_values < fixed_values)),
        },
        "note": (
            "Each outer split chooses one global window using only inner subject folds from its "
            "training subjects. This tests window-selection optimism but does not personalize the "
            "window to individual held-out subjects."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(json.dumps({"out_json": args.out_json, "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
