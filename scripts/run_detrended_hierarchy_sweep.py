#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    apply_balanced_assignment,
    center_by_subject_run,
    l2_normalize,
    metrics,
    split_indices,
)
from run_clip_offset_event_sweep import aggregate_events_for_offset, coarse_metrics
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def centroid_matrix_for_classes(
    x_train: np.ndarray,
    y_train: np.ndarray,
    classes: list[int],
) -> np.ndarray:
    x_train = l2_normalize(x_train.astype(np.float32))
    centroids = []
    for class_idx in classes:
        mask = y_train == class_idx
        if not np.any(mask):
            raise ValueError(f"Missing class {class_idx}.")
        centroids.append(x_train[mask].mean(axis=0))
    return l2_normalize(np.stack(centroids, axis=0))


def score(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return l2_normalize(x.astype(np.float32)).astype(np.float64) @ centroids.astype(np.float64).T


def exact_scores_from_hierarchy(
    coarse_scores: np.ndarray,
    leg_scores: np.ndarray,
    arm_scores: np.ndarray,
    coarse_weight: float,
) -> np.ndarray:
    exact = np.zeros((coarse_scores.shape[0], 4), dtype=np.float64)
    exact[:, 0] = leg_scores[:, 0] + coarse_weight * coarse_scores[:, 0]
    exact[:, 1] = leg_scores[:, 1] + coarse_weight * coarse_scores[:, 0]
    exact[:, 2] = arm_scores[:, 0] + coarse_weight * coarse_scores[:, 1]
    exact[:, 3] = arm_scores[:, 1] + coarse_weight * coarse_scores[:, 1]
    return exact


def evaluate_split(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    coarse_weights: list[float],
) -> list[tuple[str, np.ndarray]]:
    coarse_y = (y >= 2).astype(np.int64)
    flat_centroids = centroid_matrix_for_classes(x[train_idx], y[train_idx], [0, 1, 2, 3])
    coarse_centroids = centroid_matrix_for_classes(
        x[train_idx],
        coarse_y[train_idx],
        [0, 1],
    )
    leg_train = train_idx[np.isin(y[train_idx], [0, 1])]
    arm_train = train_idx[np.isin(y[train_idx], [2, 3])]
    leg_centroids = centroid_matrix_for_classes(x[leg_train], y[leg_train], [0, 1])
    arm_centroids = centroid_matrix_for_classes(x[arm_train], y[arm_train] - 2, [0, 1])

    flat_scores = score(x[val_idx], flat_centroids)
    coarse_scores = score(x[val_idx], coarse_centroids)
    leg_scores = score(x[val_idx], leg_centroids)
    arm_scores = score(x[val_idx], arm_centroids)

    coarse_pred = coarse_scores.argmax(axis=1).astype(np.int64)
    leg_pred = leg_scores.argmax(axis=1).astype(np.int64)
    arm_pred = arm_scores.argmax(axis=1).astype(np.int64)
    true_coarse = coarse_y[val_idx]

    predictions = [
        ("flat_independent", flat_scores.argmax(axis=1).astype(np.int64)),
        ("flat_balanced_assignment", apply_balanced_assignment(flat_scores, val_idx, records)),
        ("predicted_coarse_pairwise", np.where(coarse_pred == 0, leg_pred, arm_pred + 2)),
        ("oracle_coarse_pairwise", np.where(true_coarse == 0, leg_pred, arm_pred + 2)),
    ]
    for coarse_weight in coarse_weights:
        exact_scores = exact_scores_from_hierarchy(
            coarse_scores,
            leg_scores,
            arm_scores,
            coarse_weight,
        )
        name = f"fused_weight_{coarse_weight:g}"
        predictions.extend(
            [
                (f"{name}_independent", exact_scores.argmax(axis=1).astype(np.int64)),
                (
                    f"{name}_balanced_assignment",
                    apply_balanced_assignment(exact_scores, val_idx, records),
                ),
            ]
        )
    return predictions


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["prediction_rule"])].append(row)
    summary = []
    for (family, rule), group in sorted(grouped.items()):
        summary.append(
            {
                "family": family,
                "prediction_rule": rule,
                "split_count": len(group),
                "mean_accuracy": float(np.mean([row["metrics"]["accuracy"] for row in group])),
                "mean_balanced_accuracy": float(
                    np.mean([row["metrics"]["balanced_accuracy"] for row in group])
                ),
                "mean_macro_f1": float(np.mean([row["metrics"]["macro_f1"] for row in group])),
                "mean_leg_vs_arm_accuracy": float(
                    np.mean(
                        [
                            row["coarse_metrics"]["leg_vs_arm_accuracy"]
                            for row in group
                        ]
                    )
                ),
            }
        )
    return sorted(summary, key=lambda row: (row["family"], -row["mean_accuracy"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate coarse-to-fine hierarchy on detrended offset event features."
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--clip-offset", type=int, default=2)
    parser.add_argument("--coarse-weights", nargs="*", type=float, default=[0.0, 0.25, 0.5, 1.0])
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
    detrended, group_rows = temporal_detrend_by_subject_run(centered, event_records, degree=1)

    rows = []
    for split in split_indices(event_records, args.split_family, args.subject_fold_count):
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        predictions = evaluate_split(
            detrended,
            event_y,
            event_records,
            train_idx,
            val_idx,
            args.coarse_weights,
        )
        for rule, pred in predictions:
            rows.append(
                {
                    "split": split["split"],
                    "family": split["family"],
                    "prediction_rule": rule,
                    "train_count": int(len(train_idx)),
                    "val_count": int(len(val_idx)),
                    "metrics": metrics(event_y[val_idx], pred),
                    "coarse_metrics": coarse_metrics(event_y[val_idx], pred),
                }
            )

    result = {
        "feature_dir": str(feature_dir),
        "clip_offset": int(args.clip_offset),
        "event_feature_shape": list(event_x.shape),
        "coarse_weights": args.coarse_weights,
        "mean_linear_time_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in group_rows])
        ),
        "rows": rows,
        "summary": summarize(rows),
        "note": (
            "Oracle-coarse pairwise uses true leg-vs-arm labels for held-out events and is an upper bound. "
            "Predicted-coarse and fused variants are deployable diagnostics using only train labels and held-out features."
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
