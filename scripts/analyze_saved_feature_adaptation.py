#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


CLASS_NAMES = [
    "Left leg movements",
    "Right leg movements",
    "Forearm movements",
    "Upper arm movements",
]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -1e6, 1e6)
    normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return np.nan_to_num(normalized, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for actual, pred in zip(y_true, y_pred):
        cm[int(actual), int(pred)] += 1

    recalls = []
    precisions = []
    f1s = []
    for class_idx in range(len(CLASS_NAMES)):
        tp = float(cm[class_idx, class_idx])
        recall_den = float(cm[class_idx, :].sum())
        precision_den = float(cm[:, class_idx].sum())
        recall = tp / recall_den if recall_den else 0.0
        precision = tp / precision_den if precision_den else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    return {
        "top1_accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class_recall": {
            class_name: float(recalls[class_idx])
            for class_idx, class_name in enumerate(CLASS_NAMES)
        },
        "per_class_precision": {
            class_name: float(precisions[class_idx])
            for class_idx, class_name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),
    }


def nearest_centroid_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    classifier: str,
) -> np.ndarray:
    if classifier == "cosine":
        x_train = l2_normalize(x_train)
        x_val = l2_normalize(x_val)

    centroids = []
    for class_idx in range(len(CLASS_NAMES)):
        mask = y_train == class_idx
        if not np.any(mask):
            raise ValueError(f"Missing class {class_idx} in training split.")
        centroids.append(x_train[mask].mean(axis=0))
    centroid_arr = np.stack(centroids, axis=0)

    if classifier == "cosine":
        centroid_arr = l2_normalize(centroid_arr)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            scores = x_val.astype(np.float64) @ centroid_arr.astype(np.float64).T
        scores = np.nan_to_num(scores, copy=False, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        return scores.argmax(axis=1).astype(np.int64)
    if classifier == "euclidean":
        dist = ((x_val[:, None, :] - centroid_arr[None, :, :]) ** 2).mean(axis=2)
        return dist.argmin(axis=1).astype(np.int64)
    if classifier == "weighted_euclidean":
        pooled_var = np.maximum(x_train.var(axis=0), 1e-6)
        dist = (((x_val[:, None, :] - centroid_arr[None, :, :]) ** 2) / pooled_var).mean(axis=2)
        return dist.argmin(axis=1).astype(np.int64)
    raise ValueError(f"Unsupported classifier: {classifier}")


def event_window_key(record: dict) -> str:
    vol_start = min(int(vol_id) for vol_id in record["vol_ids"])
    # The extracted class-folder dataset contains 8-volume event windows. With 6-volume
    # clips and stride 1, starts base/base+1/base+2 belong to the same event window.
    event_start = vol_start - (vol_start % 8)
    return f'{record["subject_id"]}|run-{int(record["run_id"])}|event-{event_start}'


def aggregate_event_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    val_idx: np.ndarray,
    records: List[dict],
) -> tuple[np.ndarray, np.ndarray]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for local_pos, record_idx in enumerate(val_idx):
        grouped[event_window_key(records[int(record_idx)])].append(local_pos)

    trial_true = []
    trial_pred = []
    for positions in grouped.values():
        true_labels = y_true[positions]
        pred_labels = y_pred[positions]
        pred_counts = np.bincount(pred_labels, minlength=len(CLASS_NAMES))
        trial_true.append(int(true_labels[0]))
        trial_pred.append(int(pred_counts.argmax()))
    return np.asarray(trial_true, dtype=np.int64), np.asarray(trial_pred, dtype=np.int64)


def center_by_group(x: np.ndarray, keys: Sequence[str]) -> np.ndarray:
    key_arr = np.asarray(keys)
    out = x.copy()
    for key in sorted(set(key_arr.tolist())):
        mask = key_arr == key
        out[mask] -= out[mask].mean(axis=0)
    return out


def standardize_by_group(x: np.ndarray, keys: Sequence[str]) -> np.ndarray:
    key_arr = np.asarray(keys)
    out = x.copy()
    for key in sorted(set(key_arr.tolist())):
        mask = key_arr == key
        mean = out[mask].mean(axis=0)
        std = out[mask].std(axis=0)
        out[mask] = (out[mask] - mean) / np.where(std < 1e-6, 1.0, std)
    out = np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, -1e6, 1e6)


def train_only_center(
    x: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    key_arr = np.asarray(keys)
    fallback = x[train_idx].mean(axis=0)
    means = {}
    for key in sorted(set(key_arr[train_idx].tolist())):
        means[key] = x[train_idx[key_arr[train_idx] == key]].mean(axis=0)

    def transform(indices: np.ndarray) -> np.ndarray:
        return np.asarray([x[idx] - means.get(key_arr[idx], fallback) for idx in indices], dtype=np.float32)

    return transform(train_idx), transform(val_idx)


def split_rows(
    x: np.ndarray,
    y: np.ndarray,
    records: List[dict],
    split_name: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> List[dict]:
    subjects = np.asarray([str(record["subject_id"]) for record in records])
    run_ids = np.asarray([f'run-{int(record["run_id"])}' for record in records])
    subject_runs = np.asarray([f'{record["subject_id"]}|run-{int(record["run_id"])}' for record in records])

    transforms = {
        "raw": lambda: (x[train_idx], x[val_idx]),
        "train_global_center": lambda: train_only_center(x, train_idx, val_idx, ["global"] * len(records)),
        "train_subject_center": lambda: train_only_center(x, train_idx, val_idx, subjects),
        "train_runid_center": lambda: train_only_center(x, train_idx, val_idx, run_ids),
        "train_subject_run_center": lambda: train_only_center(x, train_idx, val_idx, subject_runs),
        # Test-time adaptation probes use unlabeled target-domain statistics.
        "tta_subject_run_center": lambda: (
            center_by_group(x, subject_runs)[train_idx],
            center_by_group(x, subject_runs)[val_idx],
        ),
        "tta_subject_run_standardize": lambda: (
            standardize_by_group(x, subject_runs)[train_idx],
            standardize_by_group(x, subject_runs)[val_idx],
        ),
    }

    rows = []
    for transform_name, transform_fn in transforms.items():
        x_train, x_val = transform_fn()
        for classifier in ("euclidean", "cosine", "weighted_euclidean"):
            pred = nearest_centroid_predict(x_train, y[train_idx], x_val, classifier)
            trial_y, trial_pred = aggregate_event_predictions(y[val_idx], pred, val_idx, records)
            row = {
                "split": split_name,
                "transform": transform_name,
                "classifier": classifier,
                "train_count": int(len(train_idx)),
                "val_count": int(len(val_idx)),
                "metrics": metrics(y[val_idx], pred),
                "trial_count": int(len(trial_y)),
                "trial_metrics": metrics(trial_y, trial_pred),
            }
            rows.append(row)
    return rows


def summarize(rows: List[dict]) -> List[dict]:
    grouped: Dict[tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["transform"], row["classifier"])].append(row)

    summaries = []
    for (split_name, transform, classifier), group_rows in sorted(grouped.items()):
        metric_rows = [row["metrics"] for row in group_rows]
        summaries.append(
            {
                "split": split_name,
                "transform": transform,
                "classifier": classifier,
                "count": int(len(group_rows)),
                "mean_top1_accuracy": float(np.mean([m["top1_accuracy"] for m in metric_rows])),
                "mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in metric_rows])),
                "mean_macro_f1": float(np.mean([m["macro_f1"] for m in metric_rows])),
                "min_top1_accuracy": float(np.min([m["top1_accuracy"] for m in metric_rows])),
                "max_top1_accuracy": float(np.max([m["top1_accuracy"] for m in metric_rows])),
                "mean_trial_top1_accuracy": float(
                    np.mean([row["trial_metrics"]["top1_accuracy"] for row in group_rows])
                ),
                "mean_trial_macro_f1": float(
                    np.mean([row["trial_metrics"]["macro_f1"] for row in group_rows])
                ),
                "min_trial_top1_accuracy": float(
                    np.min([row["trial_metrics"]["top1_accuracy"] for row in group_rows])
                ),
                "max_trial_top1_accuracy": float(
                    np.max([row["trial_metrics"]["top1_accuracy"] for row in group_rows])
                ),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved clip features under domain-adaptation probes.")
    parser.add_argument("--feature-dir", required=True, help="Directory containing features.npy, labels.npy, records.json.")
    parser.add_argument("--subject-fold-count", type=int, default=6)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    x = np.load(feature_dir / "features.npy").astype(np.float32)
    y = np.load(feature_dir / "labels.npy").astype(np.int64)
    records = json.loads((feature_dir / "records.json").read_text())
    all_idx = np.arange(len(records), dtype=np.int64)
    runs = np.asarray([int(record["run_id"]) for record in records])
    subjects = np.asarray([str(record["subject_id"]) for record in records])

    rows: List[dict] = []
    for holdout_run in sorted(set(runs.tolist())):
        rows.extend(
            split_rows(
                x=x,
                y=y,
                records=records,
                split_name=f"run_holdout_{holdout_run}",
                train_idx=all_idx[runs != holdout_run],
                val_idx=all_idx[runs == holdout_run],
            )
        )

    subject_list = sorted(set(subjects.tolist()))
    for fold_idx in range(int(args.subject_fold_count)):
        fold_subjects = subject_list[fold_idx :: int(args.subject_fold_count)]
        val_mask = np.isin(subjects, fold_subjects)
        rows.extend(
            split_rows(
                x=x,
                y=y,
                records=records,
                split_name=f"subject_fold_{fold_idx}",
                train_idx=all_idx[~val_mask],
                val_idx=all_idx[val_mask],
            )
        )

    result = {
        "feature_dir": str(feature_dir),
        "sample_count": int(len(records)),
        "feature_dim": int(x.shape[1]),
        "rows": rows,
        "summary": summarize(rows),
        "note": "Transforms prefixed with tta_ use unlabeled target-domain statistics and are adaptation probes.",
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps({"out_json": args.out_json, "summary_rows": len(result["summary"])}, indent=2))


if __name__ == "__main__":
    main()
