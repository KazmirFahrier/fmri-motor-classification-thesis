#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_subject_run_consistency import nearest_centroid_predict
from run_balanced_event_assignment import CLASS_NAMES, center_by_subject_run, l2_normalize, metrics
from run_clip_offset_event_sweep import aggregate_events_for_offset


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def quantile_rank(value: float, population: list[float], higher_is_better: bool) -> float:
    arr = np.asarray(population, dtype=np.float64)
    if len(arr) == 0:
        return float("nan")
    if higher_is_better:
        return float(np.mean(arr <= value))
    return float(np.mean(arr >= value))


def leave_one_event_accuracy(x_centered: np.ndarray, y: np.ndarray, indices: list[int]) -> dict:
    if len(indices) <= len(CLASS_NAMES):
        return {"accuracy": None, "macro_f1": None}
    true = []
    pred = []
    for val_idx in indices:
        train_idx = [idx for idx in indices if idx != val_idx]
        y_train = y[train_idx]
        if len(set(y_train.tolist())) < len(CLASS_NAMES):
            continue
        predicted = nearest_centroid_predict(x_centered[train_idx], y_train, x_centered[[val_idx]])
        true.append(int(y[val_idx]))
        pred.append(int(predicted[0]))
    if not true:
        return {"accuracy": None, "macro_f1": None}
    row = metrics(np.asarray(true, dtype=np.int64), np.asarray(pred, dtype=np.int64))
    return {"accuracy": row["accuracy"], "macro_f1": row["macro_f1"]}


def pairwise_geometry(x_centered: np.ndarray, y: np.ndarray, indices: list[int]) -> dict:
    x_norm = l2_normalize(x_centered[indices])
    y_local = y[indices]
    same = []
    different = []
    same_by_class = defaultdict(list)
    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            sim = float(x_norm[left] @ x_norm[right])
            if int(y_local[left]) == int(y_local[right]):
                same.append(sim)
                same_by_class[int(y_local[left])].append(sim)
            else:
                different.append(sim)
    return {
        "same_class_cosine_mean": float(np.mean(same)) if same else None,
        "same_class_cosine_min": float(np.min(same)) if same else None,
        "different_class_cosine_mean": float(np.mean(different)) if different else None,
        "different_class_cosine_max": float(np.max(different)) if different else None,
        "same_minus_different_mean": (
            float(np.mean(same) - np.mean(different)) if same and different else None
        ),
        "same_class_cosine_by_class": {
            CLASS_NAMES[class_idx]: float(np.mean(values))
            for class_idx, values in sorted(same_by_class.items())
        },
    }


def summarize_run(
    x_raw: np.ndarray,
    x_centered: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    subject: str,
    run_id: int,
) -> dict:
    indices = [
        idx
        for idx, record in enumerate(records)
        if str(record["subject_id"]) == subject and int(record["run_id"]) == run_id
    ]
    raw = x_raw[indices]
    centered = x_centered[indices]
    geometry = pairwise_geometry(x_centered, y, indices)
    loo = leave_one_event_accuracy(x_centered, y, indices)
    return {
        "subject": subject,
        "run_id": int(run_id),
        "event_count": int(len(indices)),
        "class_counts": {
            CLASS_NAMES[class_idx]: int(np.sum(y[indices] == class_idx))
            for class_idx in range(len(CLASS_NAMES))
        },
        "raw_run_mean_norm": float(np.linalg.norm(raw.mean(axis=0))),
        "raw_event_norm_mean": float(np.mean(np.linalg.norm(raw, axis=1))),
        "raw_event_norm_std": float(np.std(np.linalg.norm(raw, axis=1))),
        "raw_voxel_std_mean": float(np.mean(raw.std(axis=0))),
        "centered_event_norm_mean": float(np.mean(np.linalg.norm(centered, axis=1))),
        "centered_event_norm_std": float(np.std(np.linalg.norm(centered, axis=1))),
        "within_run_leave_one_event": loo,
        "pairwise_geometry": geometry,
    }


def summarize_subject(run_rows: list[dict]) -> dict:
    def mean_of(path: tuple[str, ...]) -> float | None:
        values = []
        for row in run_rows:
            value = row
            for key in path:
                value = value[key]
            if value is not None:
                values.append(float(value))
        return float(np.mean(values)) if values else None

    return {
        "subject": run_rows[0]["subject"],
        "run_count": len(run_rows),
        "mean_raw_run_mean_norm": mean_of(("raw_run_mean_norm",)),
        "mean_raw_voxel_std_mean": mean_of(("raw_voxel_std_mean",)),
        "mean_centered_event_norm_mean": mean_of(("centered_event_norm_mean",)),
        "mean_within_run_leave_one_event_accuracy": mean_of(("within_run_leave_one_event", "accuracy")),
        "mean_same_class_cosine": mean_of(("pairwise_geometry", "same_class_cosine_mean")),
        "mean_different_class_cosine": mean_of(("pairwise_geometry", "different_class_cosine_mean")),
        "mean_same_minus_different_cosine": mean_of(("pairwise_geometry", "same_minus_different_mean")),
        "worst_run_same_minus_different_cosine": min(
            row["pairwise_geometry"]["same_minus_different_mean"]
            for row in run_rows
            if row["pairwise_geometry"]["same_minus_different_mean"] is not None
        ),
        "worst_run_leave_one_event_accuracy": min(
            row["within_run_leave_one_event"]["accuracy"]
            for row in run_rows
            if row["within_run_leave_one_event"]["accuracy"] is not None
        ),
    }


def add_percentiles(subject_rows: list[dict]) -> list[dict]:
    metrics_to_rank = {
        "mean_raw_run_mean_norm": False,
        "mean_raw_voxel_std_mean": False,
        "mean_centered_event_norm_mean": False,
        "mean_within_run_leave_one_event_accuracy": True,
        "mean_same_class_cosine": True,
        "mean_same_minus_different_cosine": True,
        "worst_run_same_minus_different_cosine": True,
        "worst_run_leave_one_event_accuracy": True,
    }
    populations = {
        key: [float(row[key]) for row in subject_rows if row[key] is not None]
        for key in metrics_to_rank
    }
    enriched = []
    for row in subject_rows:
        enriched_row = dict(row)
        enriched_row["percentile_ranks"] = {
            key: quantile_rank(float(row[key]), populations[key], higher_is_better)
            for key, higher_is_better in metrics_to_rank.items()
            if row[key] is not None
        }
        enriched.append(enriched_row)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature-space QC audit for subject/run signal consistency from saved event features."
    )
    parser.add_argument("--feature-dir", required=True, help="Directory containing features.npy, labels.npy, records.json.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--clip-offset", type=int, default=2)
    parser.add_argument(
        "--focus-subjects",
        nargs="*",
        default=["sub-52", "sub-42", "sub-17", "sub-20", "sub-26", "sub-27", "sub-54", "sub-63", "sub-30", "sub-62"],
    )
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    clip_x = np.load(feature_dir / "features.npy").astype(np.float32)
    clip_y = np.load(feature_dir / "labels.npy").astype(np.int64)
    clip_records = json.loads((feature_dir / "records.json").read_text())
    event_x, event_y, event_records = aggregate_events_for_offset(clip_x, clip_y, clip_records, args.clip_offset)
    x_centered = center_by_subject_run(event_x, event_records)

    run_rows = []
    for subject in sorted(set(str(record["subject_id"]) for record in event_records)):
        run_ids = sorted(
            set(int(record["run_id"]) for record in event_records if str(record["subject_id"]) == subject)
        )
        for run_id in run_ids:
            run_rows.append(summarize_run(event_x, x_centered, event_y, event_records, subject, run_id))

    subject_to_runs = defaultdict(list)
    for row in run_rows:
        subject_to_runs[row["subject"]].append(row)
    subject_rows = add_percentiles(
        [summarize_subject(rows) for _, rows in sorted(subject_to_runs.items())]
    )

    subject_lookup = {row["subject"]: row for row in subject_rows}
    focus = [subject_lookup[subject] for subject in args.focus_subjects if subject in subject_lookup]
    result = {
        "feature_dir": str(feature_dir),
        "clip_offset": int(args.clip_offset),
        "event_feature_shape": list(event_x.shape),
        "subject_count": len(subject_rows),
        "run_count": len(run_rows),
        "focus_subjects": focus,
        "ranked_worst_same_minus_different": sorted(
            subject_rows,
            key=lambda row: row["mean_same_minus_different_cosine"],
        )[:12],
        "ranked_worst_within_run_leave_one_event": sorted(
            subject_rows,
            key=lambda row: row["mean_within_run_leave_one_event_accuracy"],
        )[:12],
        "ranked_highest_raw_run_mean_norm": sorted(
            subject_rows,
            key=lambda row: row["mean_raw_run_mean_norm"],
            reverse=True,
        )[:12],
        "subjects": subject_rows,
        "runs": run_rows,
        "note": (
            "This is a saved-feature QC audit, not a raw motion/confound audit. "
            "Lower same-minus-different cosine and lower within-run leave-one-event accuracy indicate weaker "
            "internal class geometry after subject-run centering."
        ),
    }

    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "clip_offset": args.clip_offset,
                "focus_subjects": [
                    {
                        "subject": row["subject"],
                        "mean_within_run_loo": row["mean_within_run_leave_one_event_accuracy"],
                        "mean_same_minus_diff": row["mean_same_minus_different_cosine"],
                        "worst_run_same_minus_diff": row["worst_run_same_minus_different_cosine"],
                        "percentiles": row["percentile_ranks"],
                    }
                    for row in focus
                ],
                "worst_same_minus_diff": [
                    {
                        "subject": row["subject"],
                        "mean_same_minus_diff": row["mean_same_minus_different_cosine"],
                        "within_run_loo": row["mean_within_run_leave_one_event_accuracy"],
                    }
                    for row in result["ranked_worst_same_minus_different"][:8]
                ],
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
