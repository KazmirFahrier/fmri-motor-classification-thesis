#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    CLASS_NAMES,
    aggregate_events,
    apply_balanced_assignment,
    apply_imbalance_gated_balanced_assignment,
    center_by_subject_run,
    centroid_matrix,
    metrics,
    score_with_centroids,
    split_indices,
)
from run_clip_offset_event_sweep import aggregate_events_for_offset


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def confusion_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    rows = []
    for actual in range(len(CLASS_NAMES)):
        for pred in range(len(CLASS_NAMES)):
            if actual == pred:
                continue
            count = int(np.sum((y_true == actual) & (y_pred == pred)))
            if count:
                rows.append(
                    {
                        "actual_class": CLASS_NAMES[actual],
                        "predicted_class": CLASS_NAMES[pred],
                        "count": count,
                    }
                )
    return sorted(rows, key=lambda row: row["count"], reverse=True)


def hierarchical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_limb_group = np.where(y_true <= 1, 0, 1)
    pred_limb_group = np.where(y_pred <= 1, 0, 1)
    leg_mask = y_true <= 1
    arm_mask = y_true >= 2
    within_limb_error_mask = (y_true != y_pred) & (true_limb_group == pred_limb_group)
    cross_limb_error_mask = true_limb_group != pred_limb_group
    return {
        "leg_vs_arm_accuracy": float(np.mean(true_limb_group == pred_limb_group)),
        "leg_pair_exact_accuracy": float(np.mean(y_true[leg_mask] == y_pred[leg_mask])) if np.any(leg_mask) else 0.0,
        "arm_pair_exact_accuracy": float(np.mean(y_true[arm_mask] == y_pred[arm_mask])) if np.any(arm_mask) else 0.0,
        "within_limb_error_count": int(np.sum(within_limb_error_mask)),
        "cross_limb_error_count": int(np.sum(cross_limb_error_mask)),
        "within_limb_error_fraction_of_all_errors": (
            float(np.sum(within_limb_error_mask) / np.sum(y_true != y_pred))
            if np.any(y_true != y_pred)
            else 0.0
        ),
    }


def summarize_group(rows: list[dict], key: str, prediction_rule: str) -> list[dict]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for row in rows:
        grouped[str(row[key])]["y_true"].append(int(row["class_id"]))
        grouped[str(row[key])]["y_pred"].append(int(row[prediction_rule]))
    out = []
    for value, values in grouped.items():
        y_true = np.asarray(values["y_true"], dtype=np.int64)
        y_pred = np.asarray(values["y_pred"], dtype=np.int64)
        out.append({"group": value, "event_count": int(len(y_true)), "metrics": metrics(y_true, y_pred)})
    return sorted(out, key=lambda row: row["metrics"]["accuracy"])


def summarize_subject_run(rows: list[dict], prediction_rule: str) -> list[dict]:
    grouped: dict[tuple[str, int], dict[str, list[int]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for row in rows:
        grouped[(str(row["subject_id"]), int(row["run_id"]))]["y_true"].append(int(row["class_id"]))
        grouped[(str(row["subject_id"]), int(row["run_id"]))]["y_pred"].append(int(row[prediction_rule]))
    out = []
    for (subject, run_id), values in grouped.items():
        y_true = np.asarray(values["y_true"], dtype=np.int64)
        y_pred = np.asarray(values["y_pred"], dtype=np.int64)
        out.append(
            {
                "subject": subject,
                "run_id": run_id,
                "event_count": int(len(y_true)),
                "metrics": metrics(y_true, y_pred),
            }
        )
    return sorted(out, key=lambda row: row["metrics"]["accuracy"])


def summarize_pair_group(rows: list[dict], keys: tuple[str, ...], prediction_rule: str) -> list[dict]:
    grouped: dict[tuple[str, ...], dict[str, list[int]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        grouped[key]["y_true"].append(int(row["class_id"]))
        grouped[key]["y_pred"].append(int(row[prediction_rule]))
    out = []
    for key, values in grouped.items():
        y_true = np.asarray(values["y_true"], dtype=np.int64)
        y_pred = np.asarray(values["y_pred"], dtype=np.int64)
        out.append(
            {
                "group": dict(zip(keys, key)),
                "event_count": int(len(y_true)),
                "metrics": metrics(y_true, y_pred),
            }
        )
    return sorted(out, key=lambda row: row["metrics"]["accuracy"])


def class_sequence_summary(rows: list[dict]) -> dict:
    sequences = Counter()
    for subject in sorted({row["subject_id"] for row in rows}):
        for run_id in sorted({int(row["run_id"]) for row in rows if row["subject_id"] == subject}):
            run_rows = sorted(
                [
                    row
                    for row in rows
                    if row["subject_id"] == subject and int(row["run_id"]) == run_id
                ],
                key=lambda row: int(row["event_start"]),
            )
            sequence = tuple(int(row["class_id"]) for row in run_rows)
            sequences[sequence] += 1
    return {
        "unique_sequence_count": len(sequences),
        "most_common_sequences": [
            {
                "sequence_class_ids": list(sequence),
                "sequence_class_names": [CLASS_NAMES[class_id] for class_id in sequence],
                "count": count,
            }
            for sequence, count in sequences.most_common(10)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize event-level error anatomy for dense feature predictions.")
    parser.add_argument("--feature-dir", required=True, help="Directory containing features.npy, labels.npy, records.json.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--subject-fold-count", type=int, default=6)
    parser.add_argument("--imbalance-threshold", type=float, default=4.0)
    parser.add_argument(
        "--clip-offset",
        type=int,
        default=None,
        help="If set, use only one overlapping clip offset per event instead of averaging all offsets.",
    )
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    clip_x = np.load(feature_dir / "features.npy").astype(np.float32)
    clip_y = np.load(feature_dir / "labels.npy").astype(np.int64)
    clip_records = json.loads((feature_dir / "records.json").read_text())
    if args.clip_offset is None:
        x, y, records = aggregate_events(clip_x, clip_y, clip_records)
        feature_variant = "event_mean_all_offsets"
    else:
        x, y, records = aggregate_events_for_offset(clip_x, clip_y, clip_records, args.clip_offset)
        feature_variant = f"clip_offset_{args.clip_offset}"
    x_centered = center_by_subject_run(x, records)

    event_rows = []
    for split in split_indices(records, "subject", args.subject_fold_count):
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        centroids = centroid_matrix(x_centered[train_idx], y[train_idx])
        scores = score_with_centroids(x_centered[val_idx], centroids)
        predictions = {
            "independent_argmax": scores.argmax(axis=1).astype(np.int64),
            "balanced_subject_run_assignment": apply_balanced_assignment(scores, val_idx, records),
            "gated_balanced_imbalance_l1_4": apply_imbalance_gated_balanced_assignment(
                scores=scores,
                val_idx=val_idx,
                records=records,
                min_imbalance_l1=args.imbalance_threshold,
            ),
        }

        grouped_positions: dict[tuple[str, int], list[int]] = defaultdict(list)
        for local_pos, record_idx in enumerate(val_idx):
            record = records[int(record_idx)]
            grouped_positions[(str(record["subject_id"]), int(record["run_id"]))].append(local_pos)
        ordinal_by_local_pos = {}
        class_occurrence_by_local_pos = {}
        for positions in grouped_positions.values():
            sorted_positions = sorted(positions, key=lambda pos: int(records[int(val_idx[pos])]["event_start"]))
            class_counts: Counter[int] = Counter()
            for ordinal, local_pos in enumerate(sorted_positions):
                ordinal_by_local_pos[local_pos] = ordinal
                class_id = int(y[int(val_idx[local_pos])])
                class_counts[class_id] += 1
                class_occurrence_by_local_pos[local_pos] = class_counts[class_id]

        for local_pos, record_idx in enumerate(val_idx):
            record = records[int(record_idx)]
            event_rows.append(
                {
                    "split": split["split"],
                    "subject_id": str(record["subject_id"]),
                    "run_id": int(record["run_id"]),
                    "event_start": int(record["event_start"]),
                    "event_ordinal": int(ordinal_by_local_pos[local_pos]),
                    "class_id": int(y[int(record_idx)]),
                    "class_name": CLASS_NAMES[int(y[int(record_idx)])],
                    "class_occurrence_in_run": int(class_occurrence_by_local_pos[local_pos]),
                    **{
                        prediction_rule: int(pred[local_pos])
                        for prediction_rule, pred in predictions.items()
                    },
                }
            )

    result = {
        "feature_dir": str(feature_dir),
        "feature_variant": feature_variant,
        "clip_offset": args.clip_offset,
        "event_count": len(event_rows),
        "class_sequence_summary": class_sequence_summary(event_rows),
        "rules": {},
    }
    for prediction_rule in [
        "independent_argmax",
        "balanced_subject_run_assignment",
        "gated_balanced_imbalance_l1_4",
    ]:
        y_true = np.asarray([row["class_id"] for row in event_rows], dtype=np.int64)
        y_pred = np.asarray([row[prediction_rule] for row in event_rows], dtype=np.int64)
        subject_rows = summarize_group(event_rows, "subject_id", prediction_rule)
        run_rows = summarize_group(event_rows, "run_id", prediction_rule)
        ordinal_rows = summarize_group(event_rows, "event_ordinal", prediction_rule)
        occurrence_rows = summarize_group(event_rows, "class_occurrence_in_run", prediction_rule)
        class_occurrence_rows = summarize_pair_group(
            event_rows,
            ("class_name", "class_occurrence_in_run"),
            prediction_rule,
        )
        class_ordinal_rows = summarize_pair_group(
            event_rows,
            ("class_name", "event_ordinal"),
            prediction_rule,
        )
        subject_run_rows = summarize_subject_run(event_rows, prediction_rule)
        result["rules"][prediction_rule] = {
            "overall": metrics(y_true, y_pred),
            "hierarchical_metrics": hierarchical_metrics(y_true, y_pred),
            "confusion_pairs": confusion_pairs(y_true, y_pred),
            "worst_subjects": subject_rows[:12],
            "best_subjects": subject_rows[-12:][::-1],
            "by_run": run_rows,
            "by_event_ordinal": ordinal_rows,
            "by_class_occurrence_in_run": occurrence_rows,
            "by_class_and_occurrence": class_occurrence_rows,
            "worst_class_ordinals": class_ordinal_rows[:16],
            "worst_subject_runs": subject_run_rows[:20],
            "perfect_subject_run_count": int(
                sum(row["metrics"]["accuracy"] == 1.0 for row in subject_run_rows)
            ),
            "chance_or_worse_subject_run_count": int(
                sum(row["metrics"]["accuracy"] <= 0.25 for row in subject_run_rows)
            ),
        }

    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "event_count": result["event_count"],
                "class_sequence_summary": result["class_sequence_summary"],
                "overall": {
                    rule: result["rules"][rule]["overall"]
                    for rule in result["rules"]
                },
                "hierarchical_metrics": {
                    rule: result["rules"][rule]["hierarchical_metrics"]
                    for rule in result["rules"]
                },
                "worst_subjects_gated": result["rules"]["gated_balanced_imbalance_l1_4"]["worst_subjects"][:8],
                "by_event_ordinal_gated": result["rules"]["gated_balanced_imbalance_l1_4"]["by_event_ordinal"],
                "by_class_occurrence_gated": result["rules"]["gated_balanced_imbalance_l1_4"][
                    "by_class_occurrence_in_run"
                ],
                "worst_class_ordinals_gated": result["rules"]["gated_balanced_imbalance_l1_4"][
                    "worst_class_ordinals"
                ][:8],
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
