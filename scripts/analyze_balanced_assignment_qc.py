#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    CLASS_NAMES,
    aggregate_events,
    apply_balanced_assignment,
    center_by_subject_run,
    centroid_matrix,
    metrics,
    score_with_centroids,
    split_indices,
)


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)


def pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    return float(a_centered @ b_centered / denom) if denom else 0.0


def group_quality_row(
    split_name: str,
    scores: np.ndarray,
    y_true: np.ndarray,
    independent_pred: np.ndarray,
    balanced_pred: np.ndarray,
    positions: list[int],
    val_idx: np.ndarray,
    records: list[dict],
) -> dict:
    positions = sorted(
        positions,
        key=lambda pos: (
            str(records[int(val_idx[pos])]["subject_id"]),
            int(records[int(val_idx[pos])]["run_id"]),
            int(records[int(val_idx[pos])]["event_start"]),
        ),
    )
    group_scores = scores[positions]
    group_y = y_true[positions]
    group_independent = independent_pred[positions]
    group_balanced = balanced_pred[positions]
    row_ids = np.arange(group_scores.shape[0])
    sorted_scores = np.sort(group_scores, axis=1)
    probabilities = softmax(group_scores)
    independent_score = float(group_scores[row_ids, group_independent].sum())
    balanced_score = float(group_scores[row_ids, group_balanced].sum())
    independent_counts = np.bincount(group_independent, minlength=len(CLASS_NAMES))
    expected_count = group_scores.shape[0] // len(CLASS_NAMES)
    record0 = records[int(val_idx[positions[0]])]
    return {
        "split": split_name,
        "subject": str(record0["subject_id"]),
        "run_id": int(record0["run_id"]),
        "event_count": int(group_scores.shape[0]),
        "independent_accuracy": float(np.mean(group_y == group_independent)),
        "balanced_accuracy": float(np.mean(group_y == group_balanced)),
        "balanced_delta": float(np.mean(group_y == group_balanced) - np.mean(group_y == group_independent)),
        "penalty_per_event": (independent_score - balanced_score) / float(group_scores.shape[0]),
        "mean_top1_margin": float(np.mean(sorted_scores[:, -1] - sorted_scores[:, -2])),
        "min_top1_margin": float(np.min(sorted_scores[:, -1] - sorted_scores[:, -2])),
        "mean_entropy": float(np.mean(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1))),
        "score_std": float(np.std(group_scores)),
        "imbalance_l1": float(np.sum(np.abs(independent_counts - expected_count))),
        "unique_predicted_classes": int(np.count_nonzero(independent_counts)),
        "independent_counts": independent_counts.tolist(),
    }


def threshold_gate_summary(group_rows: list[dict], metric_name: str) -> list[dict]:
    values = np.asarray([row[metric_name] for row in group_rows], dtype=np.float64)
    thresholds = sorted(set(values.tolist()))
    candidates = []
    for direction in ("<=", ">="):
        for threshold in thresholds:
            y_true_all = []
            y_pred_all = []
            used_balanced = 0
            for row in group_rows:
                use_balanced = row[metric_name] <= threshold if direction == "<=" else row[metric_name] >= threshold
                used_balanced += int(use_balanced)
                y_true_all.extend(row["y_true"])
                y_pred_all.extend(row["balanced_pred"] if use_balanced else row["independent_pred"])
            metric_row = metrics(np.asarray(y_true_all, dtype=np.int64), np.asarray(y_pred_all, dtype=np.int64))
            candidates.append(
                {
                    "metric": metric_name,
                    "direction": direction,
                    "threshold": float(threshold),
                    "balanced_group_count": int(used_balanced),
                    "accuracy": metric_row["accuracy"],
                    "macro_f1": metric_row["macro_f1"],
                }
            )
    return sorted(candidates, key=lambda row: (row["accuracy"], row["macro_f1"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze whether unlabeled subject-run score-quality metrics predict balanced-assignment gains."
    )
    parser.add_argument("--feature-dir", required=True, help="Directory containing features.npy, labels.npy, records.json.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--subject-fold-count", type=int, default=6)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    clip_x = np.load(feature_dir / "features.npy").astype(np.float32)
    clip_y = np.load(feature_dir / "labels.npy").astype(np.int64)
    clip_records = json.loads((feature_dir / "records.json").read_text())
    x, y, records = aggregate_events(clip_x, clip_y, clip_records)
    x_centered = center_by_subject_run(x, records)

    all_true = []
    independent_all = []
    balanced_all = []
    group_rows = []
    for split in split_indices(records, "subject", args.subject_fold_count):
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        centroids = centroid_matrix(x_centered[train_idx], y[train_idx])
        scores = score_with_centroids(x_centered[val_idx], centroids)
        independent_pred = scores.argmax(axis=1).astype(np.int64)
        balanced_pred = apply_balanced_assignment(scores, val_idx, records)

        grouped: dict[str, list[int]] = defaultdict(list)
        for local_pos, record_idx in enumerate(val_idx):
            record = records[int(record_idx)]
            grouped[f'{record["subject_id"]}|run-{int(record["run_id"])}'].append(local_pos)

        all_true.extend(y[val_idx].tolist())
        independent_all.extend(independent_pred.tolist())
        balanced_all.extend(balanced_pred.tolist())
        for positions in grouped.values():
            row = group_quality_row(
                split_name=split["split"],
                scores=scores,
                y_true=y[val_idx],
                independent_pred=independent_pred,
                balanced_pred=balanced_pred,
                positions=positions,
                val_idx=val_idx,
                records=records,
            )
            row["y_true"] = y[val_idx][positions].astype(np.int64).tolist()
            row["independent_pred"] = independent_pred[positions].astype(np.int64).tolist()
            row["balanced_pred"] = balanced_pred[positions].astype(np.int64).tolist()
            group_rows.append(row)

    metric_names = [
        "penalty_per_event",
        "mean_top1_margin",
        "min_top1_margin",
        "mean_entropy",
        "score_std",
        "imbalance_l1",
        "unique_predicted_classes",
    ]
    deltas = np.asarray([row["balanced_delta"] for row in group_rows], dtype=np.float64)
    correlations = {
        metric_name: pearsonr(np.asarray([row[metric_name] for row in group_rows], dtype=np.float64), deltas)
        for metric_name in metric_names
    }
    best_threshold_gates = {
        metric_name: threshold_gate_summary(group_rows, metric_name)[:10]
        for metric_name in metric_names
    }
    independent_metrics = metrics(np.asarray(all_true, dtype=np.int64), np.asarray(independent_all, dtype=np.int64))
    balanced_metrics = metrics(np.asarray(all_true, dtype=np.int64), np.asarray(balanced_all, dtype=np.int64))
    best_single_metric_gate = max(
        (candidate for candidates in best_threshold_gates.values() for candidate in candidates),
        key=lambda row: (row["accuracy"], row["macro_f1"]),
    )

    compact_rows = [
        {key: value for key, value in row.items() if key not in {"y_true", "independent_pred", "balanced_pred"}}
        for row in group_rows
    ]
    result = {
        "feature_dir": str(feature_dir),
        "event_feature_shape": list(x.shape),
        "group_count": len(group_rows),
        "independent_metrics": independent_metrics,
        "balanced_metrics": balanced_metrics,
        "balanced_delta_summary": {
            "mean": float(np.mean(deltas)),
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
            "helped_group_count": int(np.sum(deltas > 0.0)),
            "tied_group_count": int(np.sum(deltas == 0.0)),
            "hurt_group_count": int(np.sum(deltas < 0.0)),
        },
        "correlation_with_balanced_delta": correlations,
        "best_single_metric_gate": best_single_metric_gate,
        "best_threshold_gates": best_threshold_gates,
        "group_rows": compact_rows,
        "note": (
            "Threshold gates are diagnostic/oracle model-selection sweeps over unlabeled score-quality metrics; "
            "they should not be reported as validated classifier performance without a separate validation protocol."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "group_count": result["group_count"],
                "independent_accuracy": independent_metrics["accuracy"],
                "balanced_accuracy": balanced_metrics["accuracy"],
                "balanced_delta_summary": result["balanced_delta_summary"],
                "correlation_with_balanced_delta": correlations,
                "best_single_metric_gate": best_single_metric_gate,
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
