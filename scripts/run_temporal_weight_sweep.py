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
from run_clip_offset_event_sweep import clip_offset, coarse_metrics, event_start


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def simplex_weights(offsets: list[int], step: float) -> list[dict[int, float]]:
    if len(offsets) != 3:
        raise ValueError("This sweep currently expects exactly three clip offsets.")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("--grid-step must evenly divide 1.0.")
    weights = []
    for first in range(units + 1):
        for second in range(units - first + 1):
            third = units - first - second
            values = [first * step, second * step, third * step]
            weights.append({offset: float(value) for offset, value in zip(offsets, values)})
    return weights


def aggregate_events_with_weights(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    weights: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    grouped: dict[tuple[str, int, int], dict[int, int]] = defaultdict(dict)
    for idx, record in enumerate(records):
        key = (str(record["subject_id"]), int(record["run_id"]), event_start(record))
        grouped[key][clip_offset(record)] = idx

    event_x = []
    event_y = []
    event_records = []
    expected_offsets = sorted(weights)
    for (subject, run_id, start), offset_to_idx in sorted(grouped.items()):
        if sorted(offset_to_idx) != expected_offsets:
            raise ValueError(
                f"Event {(subject, run_id, start)} has offsets {sorted(offset_to_idx)}, expected {expected_offsets}."
            )
        labels = sorted(set(int(y[idx]) for idx in offset_to_idx.values()))
        if len(labels) != 1:
            raise ValueError(f"Malformed event group: {(subject, run_id, start)} labels={labels}")
        weighted = np.zeros_like(x[next(iter(offset_to_idx.values()))], dtype=np.float32)
        for offset, weight in weights.items():
            weighted += float(weight) * x[offset_to_idx[offset]]
        event_x.append(weighted)
        event_y.append(labels[0])
        event_records.append(
            {
                "subject_id": subject,
                "run_id": run_id,
                "event_start": start,
                "class_id": labels[0],
                "clip_weights": {str(offset): float(weights[offset]) for offset in expected_offsets},
            }
        )
    return np.asarray(event_x, dtype=np.float32), np.asarray(event_y, dtype=np.int64), event_records


def weight_name(weights: dict[int, float]) -> str:
    return "w_" + "_".join(f"o{offset}_{weight:g}" for offset, weight in sorted(weights.items()))


def evaluate_weight(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    split_family: str,
    subject_fold_count: int,
) -> list[dict]:
    x_centered = center_by_subject_run(x, records)
    rows = []
    for split in split_indices(records, split_family, subject_fold_count):
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        centroids = centroid_matrix(x_centered[train_idx], y[train_idx])
        scores = score_with_centroids(x_centered[val_idx], centroids)
        prediction_rows = [
            ("independent_argmax", scores.argmax(axis=1).astype(np.int64)),
            ("balanced_subject_run_assignment", apply_balanced_assignment(scores, val_idx, records)),
            (
                "gated_balanced_imbalance_l1_4",
                apply_imbalance_gated_balanced_assignment(scores, val_idx, records, 4.0),
            ),
        ]
        for prediction_rule, pred in prediction_rows:
            rows.append(
                {
                    "split": split["split"],
                    "family": split["family"],
                    "prediction_rule": prediction_rule,
                    "train_count": int(len(train_idx)),
                    "val_count": int(len(val_idx)),
                    "metrics": metrics(y[val_idx], pred),
                    "coarse_metrics": coarse_metrics(y[val_idx], pred),
                }
            )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for family in sorted(set(row["family"] for row in rows)):
        for prediction_rule in sorted(set(row["prediction_rule"] for row in rows if row["family"] == family)):
            group = [
                row
                for row in rows
                if row["family"] == family and row["prediction_rule"] == prediction_rule
            ]
            summary.append(
                {
                    "family": family,
                    "prediction_rule": prediction_rule,
                    "count": len(group),
                    "mean_accuracy": float(np.mean([row["metrics"]["accuracy"] for row in group])),
                    "mean_macro_f1": float(np.mean([row["metrics"]["macro_f1"] for row in group])),
                    "mean_leg_vs_arm_accuracy": float(
                        np.mean([row["coarse_metrics"]["leg_vs_arm_accuracy"] for row in group])
                    ),
                    "min_accuracy": float(np.min([row["metrics"]["accuracy"] for row in group])),
                    "max_accuracy": float(np.max([row["metrics"]["accuracy"] for row in group])),
                }
            )
    return sorted(summary, key=lambda row: (row["family"], row["prediction_rule"]))


def best_rows(variants: dict[str, dict], family: str, prediction_rule: str, limit: int) -> list[dict]:
    rows = []
    for name, variant in variants.items():
        for summary_row in variant["summary"]:
            if summary_row["family"] == family and summary_row["prediction_rule"] == prediction_rule:
                rows.append(
                    {
                        "variant": name,
                        "weights": variant["weights"],
                        **summary_row,
                    }
                )
    return sorted(rows, key=lambda row: (row["mean_accuracy"], row["mean_macro_f1"]), reverse=True)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search temporal weights over overlapping clip offsets for event-level features."
    )
    parser.add_argument("--feature-dir", required=True, help="Directory containing features.npy, labels.npy, records.json.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--split-family", choices=["all", "run", "subject"], default="all")
    parser.add_argument("--subject-fold-count", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    clip_x = np.load(feature_dir / "features.npy").astype(np.float32)
    clip_y = np.load(feature_dir / "labels.npy").astype(np.int64)
    clip_records = json.loads((feature_dir / "records.json").read_text())
    offsets = sorted(set(clip_offset(record) for record in clip_records))

    variants = {}
    for weights in simplex_weights(offsets, args.grid_step):
        x, y, records = aggregate_events_with_weights(clip_x, clip_y, clip_records, weights)
        rows = evaluate_weight(x, y, records, args.split_family, args.subject_fold_count)
        name = weight_name(weights)
        variants[name] = {
            "weights": {str(offset): float(weight) for offset, weight in sorted(weights.items())},
            "event_feature_shape": list(x.shape),
            "rows": rows,
            "summary": summarize(rows),
        }

    result = {
        "feature_dir": str(feature_dir),
        "available_clip_offsets": offsets,
        "grid_step": args.grid_step,
        "variant_count": len(variants),
        "variants": variants,
        "best": {
            family: {
                prediction_rule: best_rows(variants, family, prediction_rule, args.top_k)
                for prediction_rule in [
                    "independent_argmax",
                    "balanced_subject_run_assignment",
                    "gated_balanced_imbalance_l1_4",
                ]
            }
            for family in ["run", "subject"]
        },
        "note": (
            "Weights are applied to same-event overlapping clip features before subject-run centering. "
            "This is a preprocessing/window-selection diagnostic, not a nested hyperparameter-validated model."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "grid_step": args.grid_step,
                "variant_count": result["variant_count"],
                "best": result["best"],
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
