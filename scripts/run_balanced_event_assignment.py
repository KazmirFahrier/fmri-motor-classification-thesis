#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

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
        "per_class_precision": {
            class_name: float(precisions[class_idx])
            for class_idx, class_name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),
    }


def event_start(record: dict) -> int:
    vol_start = min(int(vol_id) for vol_id in record["vol_ids"])
    return vol_start - (vol_start % 8)


def aggregate_events(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    grouped: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        grouped[(str(record["subject_id"]), int(record["run_id"]), event_start(record))].append(idx)

    event_x = []
    event_y = []
    event_records = []
    for (subject, run_id, start), indices in sorted(grouped.items()):
        labels = sorted(set(int(y[idx]) for idx in indices))
        if len(labels) != 1:
            raise ValueError(f"Malformed event group: {(subject, run_id, start)} labels={labels}")
        event_x.append(x[indices].mean(axis=0))
        event_y.append(labels[0])
        event_records.append(
            {
                "subject_id": subject,
                "run_id": run_id,
                "event_start": start,
                "class_id": labels[0],
                "clip_count": len(indices),
            }
        )
    return np.asarray(event_x, dtype=np.float32), np.asarray(event_y, dtype=np.int64), event_records


def center_by_subject_run(x: np.ndarray, records: list[dict]) -> np.ndarray:
    keys = np.asarray([f'{record["subject_id"]}|run-{int(record["run_id"])}' for record in records])
    out = x.copy()
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        out[mask] -= out[mask].mean(axis=0)
    return out


def centroid_matrix(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    x_train = l2_normalize(x_train.astype(np.float32))
    centroids = []
    for class_idx in range(len(CLASS_NAMES)):
        mask = y_train == class_idx
        if not np.any(mask):
            raise ValueError(f"Missing class {class_idx}.")
        centroids.append(x_train[mask].mean(axis=0))
    return l2_normalize(np.stack(centroids, axis=0))


def score_with_centroids(x_val: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    x_val = l2_normalize(x_val.astype(np.float32))
    return x_val.astype(np.float64) @ centroids.astype(np.float64).T


def centroid_scores(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray) -> np.ndarray:
    return score_with_centroids(x_val, centroid_matrix(x_train, y_train))


@lru_cache(maxsize=None)
def balanced_assignment_templates(group_size: int, class_count: int) -> tuple[tuple[int, ...], ...]:
    if group_size % class_count != 0:
        raise ValueError(f"Cannot evenly assign {group_size} events to {class_count} classes.")
    per_class = group_size // class_count
    labels = tuple(class_idx for class_idx in range(class_count) for _ in range(per_class))
    return tuple(sorted(set(itertools.permutations(labels))))


def balanced_assign_group(scores: np.ndarray) -> np.ndarray:
    templates = balanced_assignment_templates(scores.shape[0], scores.shape[1])
    best_score = -np.inf
    best_assignment: tuple[int, ...] | None = None
    row_ids = np.arange(scores.shape[0])
    for assignment in templates:
        assignment_arr = np.asarray(assignment, dtype=np.int64)
        score = float(scores[row_ids, assignment_arr].sum())
        if score > best_score:
            best_score = score
            best_assignment = assignment
    if best_assignment is None:
        raise RuntimeError("No balanced assignment found.")
    return np.asarray(best_assignment, dtype=np.int64)


def apply_balanced_assignment(scores: np.ndarray, val_idx: np.ndarray, records: list[dict]) -> np.ndarray:
    pred = scores.argmax(axis=1).astype(np.int64)
    grouped: dict[str, list[int]] = defaultdict(list)
    for local_pos, record_idx in enumerate(val_idx):
        record = records[int(record_idx)]
        grouped[f'{record["subject_id"]}|run-{int(record["run_id"])}'].append(local_pos)
    for positions in grouped.values():
        positions = sorted(positions, key=lambda pos: records[int(val_idx[pos])]["event_start"])
        group_scores = scores[positions]
        pred[positions] = balanced_assign_group(group_scores)
    return pred


def apply_pseudo_centroid_adaptation(
    x_val: np.ndarray,
    source_centroids: np.ndarray,
    val_idx: np.ndarray,
    records: list[dict],
    target_weight: float,
    iterations: int,
) -> np.ndarray:
    x_val_norm = l2_normalize(x_val.astype(np.float32))
    pred = score_with_centroids(x_val_norm, source_centroids).argmax(axis=1).astype(np.int64)
    grouped: dict[str, list[int]] = defaultdict(list)
    for local_pos, record_idx in enumerate(val_idx):
        record = records[int(record_idx)]
        grouped[f'{record["subject_id"]}|run-{int(record["run_id"])}'].append(local_pos)

    for positions in grouped.values():
        positions = sorted(positions, key=lambda pos: records[int(val_idx[pos])]["event_start"])
        group_x = x_val_norm[positions]
        centroids = source_centroids.copy()
        assignment = balanced_assign_group(group_x.astype(np.float64) @ centroids.astype(np.float64).T)
        for _ in range(max(0, iterations)):
            target_centroids = []
            for class_idx in range(len(CLASS_NAMES)):
                mask = assignment == class_idx
                if np.any(mask):
                    target_centroids.append(group_x[mask].mean(axis=0))
                else:
                    target_centroids.append(source_centroids[class_idx])
            target_centroids_arr = l2_normalize(np.stack(target_centroids, axis=0))
            centroids = l2_normalize(
                (1.0 - float(target_weight)) * source_centroids
                + float(target_weight) * target_centroids_arr
            )
            assignment = balanced_assign_group(group_x.astype(np.float64) @ centroids.astype(np.float64).T)
        pred[positions] = assignment
    return pred


def split_indices(records: list[dict], split_family: str, subject_fold_count: int) -> list[dict]:
    all_idx = np.arange(len(records), dtype=np.int64)
    runs = np.asarray([int(record["run_id"]) for record in records])
    subjects = np.asarray([str(record["subject_id"]) for record in records])
    splits = []
    if split_family in ("all", "run"):
        for holdout_run in sorted(set(runs.tolist())):
            splits.append(
                {
                    "split": f"run_holdout_{holdout_run}",
                    "family": "run",
                    "train_idx": all_idx[runs != holdout_run],
                    "val_idx": all_idx[runs == holdout_run],
                }
            )
    if split_family in ("all", "subject"):
        subject_list = sorted(set(subjects.tolist()))
        for fold_idx in range(subject_fold_count):
            fold_subjects = subject_list[fold_idx::subject_fold_count]
            val_mask = np.isin(subjects, fold_subjects)
            splits.append(
                {
                    "split": f"subject_fold_{fold_idx}",
                    "family": "subject",
                    "val_subjects": fold_subjects,
                    "train_idx": all_idx[~val_mask],
                    "val_idx": all_idx[val_mask],
                }
            )
    return splits


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["prediction_rule"])].append(row)
    summary = []
    for (family, rule), group in sorted(grouped.items()):
        accs = [row["metrics"]["accuracy"] for row in group]
        f1s = [row["metrics"]["macro_f1"] for row in group]
        summary.append(
            {
                "family": family,
                "prediction_rule": rule,
                "count": len(group),
                "mean_accuracy": float(np.mean(accs)),
                "mean_macro_f1": float(np.mean(f1s)),
                "min_accuracy": float(np.min(accs)),
                "max_accuracy": float(np.max(accs)),
            }
        )
    return sorted(summary, key=lambda row: (row["family"], -row["mean_accuracy"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate balanced per-subject-run assignment on target-run-centered event features."
    )
    parser.add_argument("--feature-dir", required=True, help="Directory containing features.npy, labels.npy, records.json.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--split-family", choices=["all", "run", "subject"], default="all")
    parser.add_argument("--subject-fold-count", type=int, default=6)
    parser.add_argument("--pseudo-target-weights", nargs="*", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--pseudo-iterations", nargs="*", type=int, default=[1, 2])
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    clip_x = np.load(feature_dir / "features.npy").astype(np.float32)
    clip_y = np.load(feature_dir / "labels.npy").astype(np.int64)
    clip_records = json.loads((feature_dir / "records.json").read_text())
    x, y, records = aggregate_events(clip_x, clip_y, clip_records)
    x_centered = center_by_subject_run(x, records)

    rows = []
    subject_predictions: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: {"y_true": [], "y_pred": []})
    )
    for split in split_indices(records, args.split_family, args.subject_fold_count):
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        source_centroids = centroid_matrix(x_centered[train_idx], y[train_idx])
        scores = score_with_centroids(x_centered[val_idx], source_centroids)
        independent_pred = scores.argmax(axis=1).astype(np.int64)
        balanced_pred = apply_balanced_assignment(scores, val_idx, records)
        prediction_rows = [
            ("independent_argmax", independent_pred),
            ("balanced_subject_run_assignment", balanced_pred),
        ]
        for target_weight in args.pseudo_target_weights:
            for iteration_count in args.pseudo_iterations:
                prediction_rows.append(
                    (
                        f"pseudo_centroid_balanced_w{target_weight:g}_i{iteration_count}",
                        apply_pseudo_centroid_adaptation(
                            x_val=x_centered[val_idx],
                            source_centroids=source_centroids,
                            val_idx=val_idx,
                            records=records,
                            target_weight=target_weight,
                            iterations=iteration_count,
                        ),
                    )
                )
        for rule, pred in prediction_rows:
            rows.append(
                {
                    "split": split["split"],
                    "family": split["family"],
                    "prediction_rule": rule,
                    "train_count": int(len(train_idx)),
                    "val_count": int(len(val_idx)),
                    "metrics": metrics(y[val_idx], pred),
                }
            )
            if split["family"] == "subject":
                for local_pos, record_idx in enumerate(val_idx):
                    subject = str(records[int(record_idx)]["subject_id"])
                    subject_predictions[subject][rule]["y_true"].append(int(y[int(record_idx)]))
                    subject_predictions[subject][rule]["y_pred"].append(int(pred[local_pos]))

    summary = summarize(rows)
    subject_effects = []
    for subject in sorted(subject_predictions):
        for rule in sorted(subject_predictions[subject]):
            y_true = np.asarray(subject_predictions[subject][rule]["y_true"], dtype=np.int64)
            y_pred = np.asarray(subject_predictions[subject][rule]["y_pred"], dtype=np.int64)
            row = {
                "subject": subject,
                "prediction_rule": rule,
                "event_count": int(len(y_true)),
                "metrics": metrics(y_true, y_pred),
            }
            subject_effects.append(row)

    subject_balanced_delta = []
    by_subject_rule = {
        (row["subject"], row["prediction_rule"]): row
        for row in subject_effects
    }
    for subject in sorted(subject_predictions):
        base = by_subject_rule.get((subject, "independent_argmax"))
        balanced = by_subject_rule.get((subject, "balanced_subject_run_assignment"))
        if base and balanced:
            subject_balanced_delta.append(
                {
                    "subject": subject,
                    "independent_accuracy": base["metrics"]["accuracy"],
                    "balanced_accuracy": balanced["metrics"]["accuracy"],
                    "delta": balanced["metrics"]["accuracy"] - base["metrics"]["accuracy"],
                }
            )
    subject_balanced_delta = sorted(subject_balanced_delta, key=lambda row: row["delta"])
    result = {
        "feature_dir": str(feature_dir),
        "event_feature_shape": list(x.shape),
        "rows": rows,
        "summary": summary,
        "subject_effects": subject_effects,
        "subject_balanced_delta": subject_balanced_delta,
        "best_by_family": {
            family: next(row for row in summary if row["family"] == family)
            for family in sorted(set(row["family"] for row in summary))
        },
        "note": (
            "Balanced assignment uses unlabeled target subject-run groups and the known balanced task design "
            "to assign exactly two events per class in each 8-event subject-run."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "event_feature_shape": result["event_feature_shape"],
                "summary": summary,
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
