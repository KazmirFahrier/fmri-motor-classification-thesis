#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    apply_balanced_assignment,
    apply_imbalance_gated_balanced_assignment,
    center_by_subject_run,
    centroid_matrix,
    metrics,
    score_with_centroids,
    split_indices,
)
from run_clip_offset_event_sweep import aggregate_events_for_offset, coarse_metrics


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run_keys(records: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            f'{record["subject_id"]}|run-{int(record["run_id"])}'
            for record in records
        ]
    )


def permute_event_times_within_runs(
    records: list[dict],
    rng: np.random.Generator,
) -> list[dict]:
    permuted = [dict(record) for record in records]
    keys = run_keys(records)
    for key in sorted(set(keys.tolist())):
        indices = np.flatnonzero(keys == key)
        shuffled_times = rng.permutation(
            [int(records[idx]["event_start"]) for idx in indices]
        )
        for idx, event_start in zip(indices, shuffled_times):
            permuted[int(idx)]["event_start"] = int(event_start)
    return permuted


def temporal_detrend_by_subject_run(
    x_centered: np.ndarray,
    records: list[dict],
    degree: int,
) -> tuple[np.ndarray, list[dict]]:
    if degree < 1:
        return x_centered.copy(), []

    keys = run_keys(records)
    out = x_centered.astype(np.float64)
    group_rows = []
    for key in sorted(set(keys.tolist())):
        indices = np.flatnonzero(keys == key)
        times = np.asarray(
            [float(records[idx]["event_start"]) for idx in indices],
            dtype=np.float64,
        )
        times -= times.mean()
        scale = max(float(np.std(times)), 1e-8)
        times /= scale
        design = np.stack([times ** power for power in range(1, degree + 1)], axis=1)
        design -= design.mean(axis=0, keepdims=True)
        q, _ = np.linalg.qr(design)
        group_x = out[indices]
        fitted = q @ (q.T @ group_x)
        total_energy = float(np.sum(group_x ** 2))
        fitted_energy = float(np.sum(fitted ** 2))
        out[indices] = group_x - fitted
        group_rows.append(
            {
                "group": key,
                "event_count": int(len(indices)),
                "degree": int(degree),
                "temporal_variance_fraction": fitted_energy / max(total_energy, 1e-8),
            }
        )
    return out.astype(np.float32), group_rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["family"], row["prediction_rule"])].append(row)
    summary = []
    for (variant, family, rule), group in sorted(grouped.items()):
        summary.append(
            {
                "variant": variant,
                "family": family,
                "prediction_rule": rule,
                "count": len(group),
                "mean_accuracy": float(
                    np.mean([row["metrics"]["accuracy"] for row in group])
                ),
                "mean_balanced_accuracy": float(
                    np.mean([row["metrics"]["balanced_accuracy"] for row in group])
                ),
                "mean_macro_f1": float(
                    np.mean([row["metrics"]["macro_f1"] for row in group])
                ),
                "mean_leg_vs_arm_accuracy": float(
                    np.mean(
                        [
                            row["coarse_metrics"]["leg_vs_arm_accuracy"]
                            for row in group
                        ]
                    )
                ),
                "min_accuracy": float(
                    np.min([row["metrics"]["accuracy"] for row in group])
                ),
                "max_accuracy": float(
                    np.max([row["metrics"]["accuracy"] for row in group])
                ),
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            row["family"],
            row["prediction_rule"],
            -row["mean_accuracy"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate unlabeled within-run temporal detrending on offset-specific event features."
        )
    )
    parser.add_argument(
        "--feature-dir",
        required=True,
        help="Directory containing features.npy, labels.npy, records.json.",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--clip-offset", type=int, default=2)
    parser.add_argument("--degrees", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--time-permutation-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--split-family", choices=["all", "run", "subject"], default="all")
    parser.add_argument("--subject-fold-count", type=int, default=6)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    clip_x = np.load(feature_dir / "features.npy").astype(np.float32)
    clip_y = np.load(feature_dir / "labels.npy").astype(np.int64)
    clip_records = json.loads((feature_dir / "records.json").read_text())
    event_x, event_y, event_records = aggregate_events_for_offset(
        clip_x,
        clip_y,
        clip_records,
        args.clip_offset,
    )
    centered = center_by_subject_run(event_x, event_records)

    variants = {}
    rows = []
    subject_predictions: dict[tuple[str, str, str], dict[str, list[int]]] = defaultdict(
        lambda: {"y_true": [], "y_pred": []}
    )
    variant_specs = [
        (
            "run_centered" if degree == 0 else f"run_centered_temporal_degree_{degree}",
            degree,
            event_records,
            False,
        )
        for degree in args.degrees
    ]
    rng = np.random.default_rng(args.seed)
    for permutation_idx in range(args.time_permutation_count):
        variant_specs.append(
            (
                f"run_centered_temporal_degree_1_permuted_time_{permutation_idx}",
                1,
                permute_event_times_within_runs(event_records, rng),
                True,
            )
        )

    for variant, degree, detrend_records, is_permuted_control in variant_specs:
        transformed, group_rows = temporal_detrend_by_subject_run(
            centered,
            detrend_records,
            degree,
        )
        variants[variant] = {
            "degree": int(degree),
            "is_permuted_time_control": is_permuted_control,
            "mean_temporal_variance_fraction": (
                float(
                    np.mean(
                        [row["temporal_variance_fraction"] for row in group_rows]
                    )
                )
                if group_rows
                else 0.0
            ),
            "median_temporal_variance_fraction": (
                float(
                    np.median(
                        [row["temporal_variance_fraction"] for row in group_rows]
                    )
                )
                if group_rows
                else 0.0
            ),
            "group_rows": group_rows,
        }

        for split in split_indices(
            event_records,
            args.split_family,
            args.subject_fold_count,
        ):
            train_idx = split["train_idx"]
            val_idx = split["val_idx"]
            centroids = centroid_matrix(transformed[train_idx], event_y[train_idx])
            scores = score_with_centroids(transformed[val_idx], centroids)
            predictions = [
                ("independent_argmax", scores.argmax(axis=1).astype(np.int64)),
                (
                    "balanced_subject_run_assignment",
                    apply_balanced_assignment(scores, val_idx, event_records),
                ),
                (
                    "gated_balanced_imbalance_l1_4",
                    apply_imbalance_gated_balanced_assignment(
                        scores,
                        val_idx,
                        event_records,
                        4.0,
                    ),
                ),
            ]
            for rule, pred in predictions:
                rows.append(
                    {
                        "variant": variant,
                        "split": split["split"],
                        "family": split["family"],
                        "prediction_rule": rule,
                        "train_count": int(len(train_idx)),
                        "val_count": int(len(val_idx)),
                        "metrics": metrics(event_y[val_idx], pred),
                        "coarse_metrics": coarse_metrics(event_y[val_idx], pred),
                    }
                )
                if split["family"] == "subject":
                    for local_pos, record_idx in enumerate(val_idx):
                        subject = str(event_records[int(record_idx)]["subject_id"])
                        key = (variant, rule, subject)
                        subject_predictions[key]["y_true"].append(
                            int(event_y[int(record_idx)])
                        )
                        subject_predictions[key]["y_pred"].append(int(pred[local_pos]))

    subject_rows = []
    for (variant, rule, subject), values in sorted(subject_predictions.items()):
        y_true = np.asarray(values["y_true"], dtype=np.int64)
        y_pred = np.asarray(values["y_pred"], dtype=np.int64)
        subject_rows.append(
            {
                "variant": variant,
                "prediction_rule": rule,
                "subject": subject,
                "event_count": int(len(y_true)),
                "metrics": metrics(y_true, y_pred),
                "coarse_metrics": coarse_metrics(y_true, y_pred),
            }
        )

    result = {
        "feature_dir": str(feature_dir),
        "clip_offset": int(args.clip_offset),
        "event_feature_shape": list(event_x.shape),
        "variants": variants,
        "rows": rows,
        "summary": summarize(rows),
        "subject_rows": subject_rows,
        "time_permutation_count": int(args.time_permutation_count),
        "seed": int(args.seed),
        "note": (
            "Temporal detrending is fitted independently within each subject-run from unlabeled event "
            "features and event timestamps. It is a test-time adaptation diagnostic, not a training-only "
            "supervised preprocessing result."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "event_feature_shape": result["event_feature_shape"],
                "summary": result["summary"],
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
