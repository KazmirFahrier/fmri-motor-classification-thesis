#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


CLASS_NAMES = [
    "Left leg movements",
    "Right leg movements",
    "Forearm movements",
    "Upper arm movements",
]


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return np.nan_to_num(x / denom, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for actual, pred in zip(y_true, y_pred):
        cm[int(actual), int(pred)] += 1

    recalls = []
    precisions = []
    f1s = []
    for class_idx in range(len(CLASS_NAMES)):
        tp = float(cm[class_idx, class_idx])
        recall_den = float(cm[class_idx].sum())
        precision_den = float(cm[:, class_idx].sum())
        recall = tp / recall_den if recall_den else 0.0
        precision = tp / precision_den if precision_den else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    return {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class_recall": {
            class_name: float(recalls[class_idx])
            for class_idx, class_name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),
    }


def event_start(record: dict) -> int:
    vol_start = min(int(vol_id) for vol_id in record["vol_ids"])
    return vol_start - (vol_start % 8)


def event_key(record: dict) -> tuple[str, int, int]:
    return str(record["subject_id"]), int(record["run_id"]), event_start(record)


def aggregate_events(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[event_key(record)].append(idx)

    event_x = []
    event_y = []
    event_records = []
    malformed = []
    for key, indices in sorted(groups.items()):
        labels = sorted(set(int(y[idx]) for idx in indices))
        if len(labels) != 1:
            malformed.append({"key": key, "labels": labels})
            continue
        subject, run_id, start = key
        event_x.append(x[indices].mean(axis=0))
        event_y.append(labels[0])
        event_records.append(
            {
                "subject_id": subject,
                "run_id": run_id,
                "event_start": start,
                "class_id": labels[0],
                "class_name": CLASS_NAMES[labels[0]],
                "clip_count": len(indices),
            }
        )

    qc = {
        "clip_count": int(len(records)),
        "event_count": int(len(event_records)),
        "malformed_event_groups": malformed,
        "clip_counts_per_event": {
            str(count): int(sum(1 for record in event_records if record["clip_count"] == count))
            for count in sorted(set(record["clip_count"] for record in event_records))
        },
    }
    return (
        np.asarray(event_x, dtype=np.float32),
        np.asarray(event_y, dtype=np.int64),
        event_records,
        qc,
    )


def center_by_subject_run(x: np.ndarray, records: list[dict]) -> np.ndarray:
    keys = np.asarray([f'{record["subject_id"]}|run-{int(record["run_id"])}' for record in records])
    out = x.copy()
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        out[mask] -= out[mask].mean(axis=0)
    return out


def nearest_centroid_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
) -> np.ndarray:
    x_train = l2_normalize(x_train)
    x_val = l2_normalize(x_val)
    centroids = []
    for class_idx in range(len(CLASS_NAMES)):
        mask = y_train == class_idx
        if not np.any(mask):
            raise ValueError(f"Missing class {class_idx} in training split.")
        centroids.append(x_train[mask].mean(axis=0))
    centroid_arr = l2_normalize(np.stack(centroids, axis=0))
    scores = x_val.astype(np.float64) @ centroid_arr.astype(np.float64).T
    return scores.argmax(axis=1).astype(np.int64)


def subject_run_class_centroids(
    x_centered: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    subject: str,
) -> dict[int, dict[int, np.ndarray]]:
    subject_idx = [idx for idx, record in enumerate(records) if record["subject_id"] == subject]
    centroids: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for run_id in sorted(set(int(records[idx]["run_id"]) for idx in subject_idx)):
        run_idx = [idx for idx in subject_idx if int(records[idx]["run_id"]) == run_id]
        for class_idx in range(len(CLASS_NAMES)):
            class_idx_rows = [idx for idx in run_idx if int(y[idx]) == class_idx]
            if class_idx_rows:
                centroids[run_id][class_idx] = x_centered[class_idx_rows].mean(axis=0)
    return centroids


def classify_split(
    x_centered: np.ndarray,
    y: np.ndarray,
    train_idx: Iterable[int],
    val_idx: Iterable[int],
) -> dict:
    train_idx = np.asarray(list(train_idx), dtype=np.int64)
    val_idx = np.asarray(list(val_idx), dtype=np.int64)
    pred = nearest_centroid_predict(x_centered[train_idx], y[train_idx], x_centered[val_idx])
    return metrics(y[val_idx], pred)


def leave_one_run_metrics(x_centered: np.ndarray, y: np.ndarray, records: list[dict], subject: str) -> dict:
    subject_idx = [idx for idx, record in enumerate(records) if record["subject_id"] == subject]
    run_ids = sorted(set(int(records[idx]["run_id"]) for idx in subject_idx))
    rows = []
    all_true = []
    all_pred = []
    for holdout_run in run_ids:
        train_idx = [idx for idx in subject_idx if int(records[idx]["run_id"]) != holdout_run]
        val_idx = [idx for idx in subject_idx if int(records[idx]["run_id"]) == holdout_run]
        pred = nearest_centroid_predict(x_centered[train_idx], y[train_idx], x_centered[val_idx])
        split_metrics = metrics(y[val_idx], pred)
        rows.append({"run_id": holdout_run, **split_metrics})
        all_true.extend(y[val_idx].tolist())
        all_pred.extend(pred.tolist())
    return {"overall": metrics(np.asarray(all_true), np.asarray(all_pred)), "runs": rows}


def within_run_loo_metrics(x_centered: np.ndarray, y: np.ndarray, records: list[dict], subject: str) -> dict:
    subject_idx = [idx for idx, record in enumerate(records) if record["subject_id"] == subject]
    all_true = []
    all_pred = []
    for run_id in sorted(set(int(records[idx]["run_id"]) for idx in subject_idx)):
        run_idx = [idx for idx in subject_idx if int(records[idx]["run_id"]) == run_id]
        for val in run_idx:
            train_idx = [idx for idx in run_idx if idx != val]
            pred = nearest_centroid_predict(x_centered[train_idx], y[train_idx], x_centered[[val]])
            all_true.append(int(y[val]))
            all_pred.append(int(pred[0]))
    return metrics(np.asarray(all_true), np.asarray(all_pred))


def run_pair_matrix(x_centered: np.ndarray, y: np.ndarray, records: list[dict], subject: str) -> list[list[float | None]]:
    subject_idx = [idx for idx, record in enumerate(records) if record["subject_id"] == subject]
    run_ids = sorted(set(int(records[idx]["run_id"]) for idx in subject_idx))
    matrix: list[list[float | None]] = []
    for train_run in run_ids:
        row: list[float | None] = []
        for val_run in run_ids:
            if train_run == val_run:
                row.append(None)
                continue
            train_idx = [idx for idx in subject_idx if int(records[idx]["run_id"]) == train_run]
            val_idx = [idx for idx in subject_idx if int(records[idx]["run_id"]) == val_run]
            row.append(classify_split(x_centered, y, train_idx, val_idx)["accuracy"])
        matrix.append(row)
    return matrix


def centroid_stability(x_centered: np.ndarray, y: np.ndarray, records: list[dict], subject: str) -> dict:
    centroids = subject_run_class_centroids(x_centered, y, records, subject)
    run_ids = sorted(centroids)
    same_class = []
    different_class = []
    correct_vs_next_margin = []
    for left_pos, left_run in enumerate(run_ids):
        for right_run in run_ids[left_pos + 1 :]:
            left = l2_normalize(np.stack([centroids[left_run][c] for c in range(len(CLASS_NAMES))]))
            right = l2_normalize(np.stack([centroids[right_run][c] for c in range(len(CLASS_NAMES))]))
            sim = left @ right.T
            for class_idx in range(len(CLASS_NAMES)):
                same_class.append(float(sim[class_idx, class_idx]))
                wrong = np.delete(sim[class_idx], class_idx)
                different_class.extend(float(v) for v in wrong)
                correct_vs_next_margin.append(float(sim[class_idx, class_idx] - np.max(wrong)))

    return {
        "same_class_cosine_mean": float(np.mean(same_class)),
        "same_class_cosine_p10": float(np.quantile(same_class, 0.10)),
        "different_class_cosine_mean": float(np.mean(different_class)),
        "different_class_cosine_p90": float(np.quantile(different_class, 0.90)),
        "same_minus_different_mean": float(np.mean(same_class) - np.mean(different_class)),
        "correct_vs_nearest_wrong_margin_mean": float(np.mean(correct_vs_next_margin)),
        "correct_vs_nearest_wrong_margin_p10": float(np.quantile(correct_vs_next_margin, 0.10)),
    }


def schedule_qc(records: list[dict], subject: str) -> dict:
    subject_records = [record for record in records if record["subject_id"] == subject]
    by_run_class: dict[tuple[int, int], list[int]] = defaultdict(list)
    for record in subject_records:
        by_run_class[(int(record["run_id"]), int(record["class_id"]))].append(int(record["event_start"]))

    anomalies = []
    for run_id in sorted(set(int(record["run_id"]) for record in subject_records)):
        for class_idx in range(len(CLASS_NAMES)):
            starts = sorted(by_run_class[(run_id, class_idx)])
            if len(starts) != 2:
                anomalies.append(
                    {
                        "run_id": run_id,
                        "class_name": CLASS_NAMES[class_idx],
                        "event_starts": starts,
                    }
                )
    return {"event_count_anomalies": anomalies}


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if float(x_arr.std()) == 0.0 or float(y_arr.std()) == 0.0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit per-subject run consistency from saved corrected-clip feature matrices."
    )
    parser.add_argument("--feature-dir", required=True, help="Directory containing features.npy, labels.npy, records.json.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--focus-subjects",
        nargs="*",
        default=["sub-52", "sub-42", "sub-17", "sub-20", "sub-54", "sub-63", "sub-30", "sub-62", "sub-10", "sub-47"],
    )
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    x = np.load(feature_dir / "features.npy").astype(np.float32)
    y = np.load(feature_dir / "labels.npy").astype(np.int64)
    records = json.loads((feature_dir / "records.json").read_text())

    event_x, event_y, event_records, qc = aggregate_events(x, y, records)
    x_centered = center_by_subject_run(event_x, event_records)
    subjects = sorted(set(record["subject_id"] for record in event_records))

    subject_rows = []
    for subject in subjects:
        subject_idx = [idx for idx, record in enumerate(event_records) if record["subject_id"] == subject]
        other_idx = [idx for idx, record in enumerate(event_records) if record["subject_id"] != subject]
        loso = classify_split(x_centered, event_y, other_idx, subject_idx)
        leave_run = leave_one_run_metrics(x_centered, event_y, event_records, subject)
        within_run = within_run_loo_metrics(x_centered, event_y, event_records, subject)
        stability = centroid_stability(x_centered, event_y, event_records, subject)
        pair_matrix = run_pair_matrix(x_centered, event_y, event_records, subject)
        pair_values = [value for row in pair_matrix for value in row if value is not None]
        subject_rows.append(
            {
                "subject": subject,
                "leave_one_subject_adapted": loso,
                "same_subject_leave_one_run": leave_run,
                "within_run_leave_one_event": within_run,
                "run_pair_accuracy_mean": float(np.mean(pair_values)),
                "run_pair_accuracy_min": float(np.min(pair_values)),
                "run_pair_accuracy_max": float(np.max(pair_values)),
                "centroid_stability": stability,
                "schedule_qc": schedule_qc(event_records, subject),
                "run_pair_accuracy_matrix": pair_matrix if subject in set(args.focus_subjects) else None,
            }
        )

    ranked_by_loso = sorted(
        subject_rows,
        key=lambda row: row["leave_one_subject_adapted"]["accuracy"],
    )
    ranked_by_run = sorted(
        subject_rows,
        key=lambda row: row["same_subject_leave_one_run"]["overall"]["accuracy"],
    )
    ranked_by_margin = sorted(
        subject_rows,
        key=lambda row: row["centroid_stability"]["correct_vs_nearest_wrong_margin_mean"],
    )

    correlations = {
        "loso_acc_vs_same_subject_leave_one_run_acc": pearson(
            [row["leave_one_subject_adapted"]["accuracy"] for row in subject_rows],
            [row["same_subject_leave_one_run"]["overall"]["accuracy"] for row in subject_rows],
        ),
        "loso_acc_vs_centroid_margin": pearson(
            [row["leave_one_subject_adapted"]["accuracy"] for row in subject_rows],
            [row["centroid_stability"]["correct_vs_nearest_wrong_margin_mean"] for row in subject_rows],
        ),
        "same_subject_leave_one_run_acc_vs_centroid_margin": pearson(
            [row["same_subject_leave_one_run"]["overall"]["accuracy"] for row in subject_rows],
            [row["centroid_stability"]["correct_vs_nearest_wrong_margin_mean"] for row in subject_rows],
        ),
    }

    report = {
        "feature_dir": str(feature_dir),
        "qc": qc,
        "event_feature_shape": list(event_x.shape),
        "method": (
            "Overlapping clips are averaged into event-window features, features are centered "
            "within each subject-run using unlabeled run statistics, and cosine nearest-centroid "
            "classifiers/stability metrics are computed on the centered event features."
        ),
        "correlations": correlations,
        "ranked_worst_leave_one_subject_adapted": ranked_by_loso[:12],
        "ranked_worst_same_subject_leave_one_run": ranked_by_run[:12],
        "ranked_worst_centroid_margin": ranked_by_margin[:12],
        "ranked_best_leave_one_subject_adapted": ranked_by_loso[-12:][::-1],
        "focus_subjects": [
            row for row in subject_rows if row["subject"] in set(args.focus_subjects)
        ],
        "subjects": subject_rows,
    }

    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(report, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": str(out_path),
                "event_feature_shape": report["event_feature_shape"],
                "worst_loso": [
                    {
                        "subject": row["subject"],
                        "acc": row["leave_one_subject_adapted"]["accuracy"],
                        "run_acc": row["same_subject_leave_one_run"]["overall"]["accuracy"],
                        "margin": row["centroid_stability"]["correct_vs_nearest_wrong_margin_mean"],
                    }
                    for row in ranked_by_loso[:8]
                ],
                "correlations": correlations,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
