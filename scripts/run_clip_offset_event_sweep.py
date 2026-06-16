#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    CLASS_NAMES,
    apply_balanced_assignment,
    apply_imbalance_gated_balanced_assignment,
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


def event_start(record: dict) -> int:
    vol_start = min(int(vol_id) for vol_id in record["vol_ids"])
    return vol_start - (vol_start % 8)


def clip_offset(record: dict) -> int:
    return min(int(vol_id) for vol_id in record["vol_ids"]) - event_start(record)


def aggregate_events_for_offset(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    offset: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    grouped: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        if offset is not None and clip_offset(record) != offset:
            continue
        key = (str(record["subject_id"]), int(record["run_id"]), event_start(record))
        grouped[key].append(idx)

    event_x = []
    event_y = []
    event_records = []
    for (subject, run_id, start), indices in sorted(grouped.items()):
        labels = sorted(set(int(y[idx]) for idx in indices))
        if len(labels) != 1:
            raise ValueError(f"Malformed event group: {(subject, run_id, start)} labels={labels}")
        offsets = sorted(set(clip_offset(records[idx]) for idx in indices))
        event_x.append(x[indices].mean(axis=0))
        event_y.append(labels[0])
        event_records.append(
            {
                "subject_id": subject,
                "run_id": run_id,
                "event_start": start,
                "class_id": labels[0],
                "clip_count": len(indices),
                "clip_offsets": offsets,
            }
        )
    return np.asarray(event_x, dtype=np.float32), np.asarray(event_y, dtype=np.int64), event_records


def coarse_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_group = np.where(y_true <= 1, 0, 1)
    pred_group = np.where(y_pred <= 1, 0, 1)
    return {
        "leg_vs_arm_accuracy": float(np.mean(true_group == pred_group)),
        "leg_vs_arm_balanced_accuracy": float(
            0.5
            * (
                np.mean(pred_group[true_group == 0] == 0)
                + np.mean(pred_group[true_group == 1] == 1)
            )
        ),
    }


def evaluate_variant(x: np.ndarray, y: np.ndarray, records: list[dict], split_family: str, subject_fold_count: int) -> dict:
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
    return {"rows": rows, "summary": sorted(summary, key=lambda row: (row["family"], -row["mean_accuracy"]))}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether individual overlapping clip offsets outperform event-mean features."
    )
    parser.add_argument("--feature-dir", required=True, help="Directory containing features.npy, labels.npy, records.json.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--split-family", choices=["all", "run", "subject"], default="all")
    parser.add_argument("--subject-fold-count", type=int, default=6)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    clip_x = np.load(feature_dir / "features.npy").astype(np.float32)
    clip_y = np.load(feature_dir / "labels.npy").astype(np.int64)
    clip_records = json.loads((feature_dir / "records.json").read_text())
    offsets = sorted(set(clip_offset(record) for record in clip_records))

    variants = {}
    for offset in [None, *offsets]:
        x, y, records = aggregate_events_for_offset(clip_x, clip_y, clip_records, offset)
        name = "event_mean_all_offsets" if offset is None else f"clip_offset_{offset}"
        variants[name] = {
            "offset": offset,
            "event_feature_shape": list(x.shape),
            "clip_count_per_event": sorted(set(int(record["clip_count"]) for record in records)),
            **evaluate_variant(x, y, records, args.split_family, args.subject_fold_count),
        }

    result = {
        "feature_dir": str(feature_dir),
        "available_clip_offsets": offsets,
        "variants": variants,
        "note": (
            "Offset-specific variants use one overlapping clip per event; event_mean_all_offsets averages all "
            "available overlapping clips for the same event."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "available_clip_offsets": offsets,
                "summary": {
                    name: variant["summary"]
                    for name, variant in variants.items()
                },
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
