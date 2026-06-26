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
from run_clip_offset_event_sweep import coarse_metrics
from run_detrended_pair_feature_selection import load_checkpoints
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


DEFAULT_WINDOWS = [
    "offset_2_length_6",
    "offset_3_length_6",
    "offset_3_length_8",
    "offset_4_length_2",
    "offset_5_length_4",
    "offset_6_length_2",
]


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def evaluate_window(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
) -> dict:
    centered = center_by_subject_run(x, records)
    detrended, group_rows = temporal_detrend_by_subject_run(centered, records, degree=1)
    subjects = np.asarray([str(record["subject_id"]) for record in records])
    rows = []
    for subject in sorted(set(subjects.tolist())):
        val_idx = np.flatnonzero(subjects == subject)
        train_idx = np.flatnonzero(subjects != subject)
        centroids = centroid_matrix(detrended[train_idx], y[train_idx])
        scores = score_with_centroids(detrended[val_idx], centroids)
        predictions = [
            ("independent_argmax", scores.argmax(axis=1).astype(np.int64)),
            ("balanced_subject_run_assignment", apply_balanced_assignment(scores, val_idx, records)),
        ]
        for rule, pred in predictions:
            rows.append(
                {
                    "subject": subject,
                    "prediction_rule": rule,
                    "metrics": metrics(y[val_idx], pred),
                    "coarse_metrics": coarse_metrics(y[val_idx], pred),
                }
            )

    summary = []
    for rule in ["independent_argmax", "balanced_subject_run_assignment"]:
        group = [row for row in rows if row["prediction_rule"] == rule]
        summary.append(
            {
                "prediction_rule": rule,
                "subject_count": len(group),
                "mean_accuracy": float(np.mean([row["metrics"]["accuracy"] for row in group])),
                "mean_balanced_accuracy": float(
                    np.mean([row["metrics"]["balanced_accuracy"] for row in group])
                ),
                "mean_macro_f1": float(np.mean([row["metrics"]["macro_f1"] for row in group])),
                "mean_leg_vs_arm_accuracy": float(
                    np.mean([row["coarse_metrics"]["leg_vs_arm_accuracy"] for row in group])
                ),
            }
        )
    return {
        "mean_linear_time_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in group_rows])
        ),
        "rows": rows,
        "summary": summary,
    }


def subject_profiles(
    variants: dict[str, dict],
    canonical_window: str,
) -> dict:
    balanced = {}
    for window, variant in variants.items():
        balanced[window] = {
            row["subject"]: row["metrics"]["balanced_accuracy"]
            for row in variant["rows"]
            if row["prediction_rule"] == "balanced_subject_run_assignment"
        }
    global_window = max(
        variants,
        key=lambda window: next(
            row["mean_balanced_accuracy"]
            for row in variants[window]["summary"]
            if row["prediction_rule"] == "balanced_subject_run_assignment"
        ),
    )
    subjects = sorted(next(iter(balanced.values())))
    rows = []
    for subject in subjects:
        scores = {window: values[subject] for window, values in balanced.items()}
        best_window = max(scores, key=lambda window: (scores[window], window))
        rows.append(
            {
                "subject": subject,
                "window_balanced_accuracy": scores,
                "best_window_oracle": best_window,
                "best_window_oracle_accuracy": scores[best_window],
                "global_window_accuracy": scores[global_window],
                "canonical_window_accuracy": scores[canonical_window],
                "global_minus_canonical": scores[global_window] - scores[canonical_window],
                "oracle_minus_global": scores[best_window] - scores[global_window],
            }
        )
    deltas = np.asarray([row["global_minus_canonical"] for row in rows])
    oracle_gains = np.asarray([row["oracle_minus_global"] for row in rows])
    return {
        "global_window": global_window,
        "canonical_window": canonical_window,
        "global_vs_canonical": {
            "improved_subjects": int(np.sum(deltas > 0)),
            "tied_subjects": int(np.sum(deltas == 0)),
            "harmed_subjects": int(np.sum(deltas < 0)),
            "mean_delta": float(np.mean(deltas)),
            "min_delta": float(np.min(deltas)),
            "max_delta": float(np.max(deltas)),
        },
        "oracle_window_heterogeneity": {
            "best_window_counts": dict(Counter(row["best_window_oracle"] for row in rows)),
            "mean_oracle_gain": float(np.mean(oracle_gains)),
            "median_oracle_gain": float(np.median(oracle_gains)),
            "subjects_with_positive_oracle_gain": int(np.sum(oracle_gains > 0)),
        },
        "subjects": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure leave-one-subject temporal-window heterogeneity after linear detrending."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--windows", nargs="*", default=DEFAULT_WINDOWS)
    parser.add_argument("--canonical-window", default="offset_2_length_6")
    args = parser.parse_args()

    variants = {}
    labels = None
    records = None
    for window in args.windows:
        print(f"loading and evaluating {window}", flush=True)
        features, loaded_labels, loaded_records = load_checkpoints(
            Path(args.checkpoint_dir),
            [window],
        )
        if labels is None:
            labels = loaded_labels
            records = loaded_records
        elif not np.array_equal(labels, loaded_labels) or records != loaded_records:
            raise ValueError(f"Checkpoint label/record order changed while loading {window}.")
        variants[window] = evaluate_window(features[window], labels, records)
        del features

    profiles = subject_profiles(variants, args.canonical_window)
    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "windows": args.windows,
        "subject_count": len(profiles["subjects"]),
        "variants": variants,
        "profiles": profiles,
        "note": (
            "Per-subject best windows use held-out subject labels and are oracle diagnostics only. "
            "The globally best window is selected from this same cohort and should be confirmed in "
            "a nested or external window-selection protocol before final performance reporting."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "variant_summaries": {
                    window: variant["summary"]
                    for window, variant in variants.items()
                },
                "profiles": profiles,
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
