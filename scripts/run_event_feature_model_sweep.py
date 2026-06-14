#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    grouped: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        grouped[(str(record["subject_id"]), int(record["run_id"]), event_start(record))].append(idx)

    event_x = []
    event_y = []
    event_records = []
    malformed = []
    for (subject, run_id, start), indices in sorted(grouped.items()):
        labels = sorted(set(int(y[idx]) for idx in indices))
        if len(labels) != 1:
            malformed.append({"subject": subject, "run_id": run_id, "event_start": start, "labels": labels})
            continue
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

    qc = {
        "clip_count": int(len(records)),
        "event_count": int(len(event_records)),
        "malformed_event_groups": malformed,
        "clip_counts_per_event": {
            str(count): int(sum(1 for record in event_records if record["clip_count"] == count))
            for count in sorted(set(record["clip_count"] for record in event_records))
        },
    }
    return np.asarray(event_x, dtype=np.float32), np.asarray(event_y, dtype=np.int64), event_records, qc


def center_by_keys(x: np.ndarray, keys: list[str]) -> np.ndarray:
    key_arr = np.asarray(keys)
    out = x.copy()
    for key in sorted(set(keys)):
        mask = key_arr == key
        out[mask] -= out[mask].mean(axis=0)
    return out


def train_global_center(x: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x[train_idx].mean(axis=0)
    return x[train_idx] - mean, x[val_idx] - mean


def nearest_centroid_cosine(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray) -> np.ndarray:
    x_train = l2_normalize(x_train.astype(np.float32))
    x_val = l2_normalize(x_val.astype(np.float32))
    centroids = []
    for class_idx in range(len(CLASS_NAMES)):
        mask = y_train == class_idx
        if not np.any(mask):
            raise ValueError(f"Missing class {class_idx}.")
        centroids.append(x_train[mask].mean(axis=0))
    centroids = l2_normalize(np.stack(centroids, axis=0))
    scores = x_val.astype(np.float64) @ centroids.astype(np.float64).T
    return scores.argmax(axis=1).astype(np.int64)


def dual_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    ridge_lambda: float,
    normalize: bool,
) -> np.ndarray:
    if normalize:
        x_train = l2_normalize(x_train.astype(np.float32))
        x_val = l2_normalize(x_val.astype(np.float32))
    else:
        # A single global scale keeps the Gram matrix numerically tame without using validation labels.
        scale = max(float(np.sqrt(np.mean(x_train.astype(np.float64) ** 2))), 1e-8)
        x_train = (x_train / scale).astype(np.float32)
        x_val = (x_val / scale).astype(np.float32)

    y_onehot = np.eye(len(CLASS_NAMES), dtype=np.float64)[y_train]
    gram = x_train.astype(np.float64) @ x_train.astype(np.float64).T
    gram.flat[:: gram.shape[0] + 1] += float(ridge_lambda)
    alpha = np.linalg.solve(gram, y_onehot)
    scores = x_val.astype(np.float64) @ x_train.astype(np.float64).T @ alpha
    return scores.argmax(axis=1).astype(np.int64)


def primal_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    ridge_lambda: float,
    add_intercept: bool = True,
) -> np.ndarray:
    x_train = x_train.astype(np.float64)
    x_val = x_val.astype(np.float64)
    if add_intercept:
        x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float64)], axis=1)
        x_val = np.concatenate([x_val, np.ones((x_val.shape[0], 1), dtype=np.float64)], axis=1)

    y_onehot = np.eye(len(CLASS_NAMES), dtype=np.float64)[y_train]
    xtx = x_train.T @ x_train
    xtx.flat[:: xtx.shape[0] + 1] += float(ridge_lambda)
    if add_intercept:
        xtx[-1, -1] -= float(ridge_lambda)
    weights = np.linalg.solve(xtx, x_train.T @ y_onehot)
    return (x_val @ weights).argmax(axis=1).astype(np.int64)


def random_projection(
    x_train: np.ndarray,
    x_val: np.ndarray,
    out_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(out_dim), size=(x_train.shape[1], out_dim)).astype(np.float32)
    return x_train @ projection, x_val @ projection


def transform_split(
    x: np.ndarray,
    records: list[dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    if transform == "raw":
        return x[train_idx], x[val_idx]
    if transform == "train_global_center":
        return train_global_center(x, train_idx, val_idx)
    if transform == "tta_subject_run_center":
        keys = [f'{record["subject_id"]}|run-{int(record["run_id"])}' for record in records]
        centered = center_by_keys(x, keys)
        return centered[train_idx], centered[val_idx]
    raise ValueError(f"Unsupported transform: {transform}")


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


def evaluate_split(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    split: dict,
    transforms: list[str],
    ridge_lambdas: list[float],
    random_projection_dims: list[int],
    seed: int,
    include_dual: bool,
) -> list[dict]:
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    rows = []
    for transform in transforms:
        x_train, x_val = transform_split(x, records, train_idx, val_idx, transform)
        classifier_jobs = [("centroid_cosine", None, None)]
        for ridge_lambda in ridge_lambdas:
            if include_dual:
                classifier_jobs.append(("dual_ridge_l2norm", ridge_lambda, None))
                classifier_jobs.append(("dual_ridge_scaled", ridge_lambda, None))
            for dim in random_projection_dims:
                classifier_jobs.append(("rp_primal_ridge_l2norm", ridge_lambda, dim))

        for classifier, ridge_lambda, rp_dim in classifier_jobs:
            if classifier == "centroid_cosine":
                pred = nearest_centroid_cosine(x_train, y[train_idx], x_val)
            elif classifier == "dual_ridge_l2norm":
                pred = dual_ridge_predict(x_train, y[train_idx], x_val, ridge_lambda, normalize=True)
            elif classifier == "dual_ridge_scaled":
                pred = dual_ridge_predict(x_train, y[train_idx], x_val, ridge_lambda, normalize=False)
            elif classifier == "rp_primal_ridge_l2norm":
                rp_train, rp_val = random_projection(
                    l2_normalize(x_train.astype(np.float32)),
                    l2_normalize(x_val.astype(np.float32)),
                    int(rp_dim),
                    seed + int(rp_dim) + int(1000 * float(ridge_lambda)),
                )
                pred = primal_ridge_predict(rp_train, y[train_idx], rp_val, ridge_lambda)
            else:
                raise ValueError(classifier)
            rows.append(
                {
                    "split": split["split"],
                    "family": split["family"],
                    "transform": transform,
                    "classifier": classifier,
                    "ridge_lambda": ridge_lambda,
                    "random_projection_dim": rp_dim,
                    "train_count": int(len(train_idx)),
                    "val_count": int(len(val_idx)),
                    "metrics": metrics(y[val_idx], pred),
                }
            )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["family"],
                row["transform"],
                row["classifier"],
                row["ridge_lambda"],
                row["random_projection_dim"],
            )
        ].append(row)

    summary = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        family, transform, classifier, ridge_lambda, rp_dim = key
        accs = [row["metrics"]["accuracy"] for row in group]
        f1s = [row["metrics"]["macro_f1"] for row in group]
        bals = [row["metrics"]["balanced_accuracy"] for row in group]
        summary.append(
            {
                "family": family,
                "transform": transform,
                "classifier": classifier,
                "ridge_lambda": ridge_lambda,
                "random_projection_dim": rp_dim,
                "count": int(len(group)),
                "mean_accuracy": float(np.mean(accs)),
                "mean_balanced_accuracy": float(np.mean(bals)),
                "mean_macro_f1": float(np.mean(f1s)),
                "min_accuracy": float(np.min(accs)),
                "max_accuracy": float(np.max(accs)),
            }
        )
    return sorted(summary, key=lambda row: (row["family"], -row["mean_accuracy"], -row["mean_macro_f1"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple event-level model sweeps on saved corrected-clip features.")
    parser.add_argument("--feature-dir", required=True, help="Directory containing features.npy, labels.npy, records.json.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--split-family", choices=["all", "run", "subject"], default="all")
    parser.add_argument("--subject-fold-count", type=int, default=6)
    parser.add_argument("--transforms", nargs="+", default=["raw", "train_global_center", "tta_subject_run_center"])
    parser.add_argument("--ridge-lambdas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--random-projection-dims", nargs="*", type=int, default=[64, 128, 256])
    parser.add_argument("--include-dual", action="store_true", help="Also run slower full dual-ridge classifiers.")
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    clip_x = np.load(feature_dir / "features.npy").astype(np.float32)
    clip_y = np.load(feature_dir / "labels.npy").astype(np.int64)
    clip_records = json.loads((feature_dir / "records.json").read_text())
    x, y, records, qc = aggregate_events(clip_x, clip_y, clip_records)
    splits = split_indices(records, args.split_family, args.subject_fold_count)

    rows = []
    for split in splits:
        rows.extend(
            evaluate_split(
                x=x,
                y=y,
                records=records,
                split=split,
                transforms=args.transforms,
                ridge_lambdas=args.ridge_lambdas,
                random_projection_dims=args.random_projection_dims,
                seed=args.seed,
                include_dual=args.include_dual,
            )
        )

    summary = summarize(rows)
    result = {
        "feature_dir": str(feature_dir),
        "event_feature_shape": list(x.shape),
        "qc": qc,
        "split_family": args.split_family,
        "rows": rows,
        "summary": summary,
        "best_by_family": {
            family: next(row for row in summary if row["family"] == family)
            for family in sorted(set(row["family"] for row in summary))
        },
        "note": "Transforms prefixed with tta_ use unlabeled validation subject-run statistics.",
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "event_feature_shape": result["event_feature_shape"],
                "best_by_family": result["best_by_family"],
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
