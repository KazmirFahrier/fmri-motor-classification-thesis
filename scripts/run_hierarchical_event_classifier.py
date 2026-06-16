#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    CLASS_NAMES,
    aggregate_events,
    center_by_subject_run,
    metrics,
    split_indices,
)


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return np.nan_to_num(x / denom, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def centroid_matrix_for_labels(x_train: np.ndarray, y_train: np.ndarray, labels: list[int]) -> np.ndarray:
    x_train = l2_normalize(x_train.astype(np.float32))
    centroids = []
    for label in labels:
        mask = y_train == label
        if not np.any(mask):
            raise ValueError(f"Missing label {label}.")
        centroids.append(x_train[mask].mean(axis=0))
    return l2_normalize(np.stack(centroids, axis=0))


def predict_from_centroids(x_val: np.ndarray, centroids: np.ndarray, labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    scores = l2_normalize(x_val.astype(np.float32)).astype(np.float64) @ centroids.astype(np.float64).T
    label_arr = np.asarray(labels, dtype=np.int64)
    return label_arr[scores.argmax(axis=1)], scores


def coarse_labels(y: np.ndarray) -> np.ndarray:
    return np.where(y <= 1, 0, 1).astype(np.int64)


def hierarchical_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    oracle_coarse: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    y_train_coarse = coarse_labels(y_train)
    coarse_centroids = centroid_matrix_for_labels(x_train, y_train_coarse, [0, 1])
    pred_coarse, coarse_scores = predict_from_centroids(x_val, coarse_centroids, [0, 1])
    if oracle_coarse is not None:
        pred_coarse = oracle_coarse.astype(np.int64)

    leg_centroids = centroid_matrix_for_labels(x_train, y_train, [0, 1])
    arm_centroids = centroid_matrix_for_labels(x_train, y_train, [2, 3])
    pred = np.zeros(x_val.shape[0], dtype=np.int64)
    leg_mask = pred_coarse == 0
    arm_mask = pred_coarse == 1
    if np.any(leg_mask):
        pred[leg_mask], _ = predict_from_centroids(x_val[leg_mask], leg_centroids, [0, 1])
    if np.any(arm_mask):
        pred[arm_mask], _ = predict_from_centroids(x_val[arm_mask], arm_centroids, [2, 3])
    aux = {
        "coarse_prediction": pred_coarse,
        "coarse_scores": coarse_scores,
    }
    return pred, aux


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for family in sorted(set(row["family"] for row in rows)):
        for rule in sorted(set(row["prediction_rule"] for row in rows if row["family"] == family)):
            group = [row for row in rows if row["family"] == family and row["prediction_rule"] == rule]
            out.append(
                {
                    "family": family,
                    "prediction_rule": rule,
                    "count": len(group),
                    "mean_accuracy": float(np.mean([row["metrics"]["accuracy"] for row in group])),
                    "mean_macro_f1": float(np.mean([row["metrics"]["macro_f1"] for row in group])),
                    "min_accuracy": float(np.min([row["metrics"]["accuracy"] for row in group])),
                    "max_accuracy": float(np.max([row["metrics"]["accuracy"] for row in group])),
                }
            )
    return sorted(out, key=lambda row: (row["family"], -row["mean_accuracy"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate flat versus hierarchical leg/arm event classifiers on centered dense features."
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
    x, y, records = aggregate_events(clip_x, clip_y, clip_records)
    x_centered = center_by_subject_run(x, records)

    rows = []
    for split in split_indices(records, args.split_family, args.subject_fold_count):
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        flat_centroids = centroid_matrix_for_labels(x_centered[train_idx], y[train_idx], list(range(len(CLASS_NAMES))))
        flat_pred, _ = predict_from_centroids(x_centered[val_idx], flat_centroids, list(range(len(CLASS_NAMES))))
        hierarchical_pred, hierarchical_aux = hierarchical_predict(
            x_centered[train_idx],
            y[train_idx],
            x_centered[val_idx],
        )
        oracle_hierarchical_pred, _ = hierarchical_predict(
            x_centered[train_idx],
            y[train_idx],
            x_centered[val_idx],
            oracle_coarse=coarse_labels(y[val_idx]),
        )
        coarse_true = coarse_labels(y[val_idx])
        coarse_pred = hierarchical_aux["coarse_prediction"]

        prediction_rows = [
            ("flat_4class_centroid", flat_pred),
            ("hierarchical_centroid", hierarchical_pred),
            ("oracle_coarse_hierarchical_centroid", oracle_hierarchical_pred),
        ]
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
        rows.append(
            {
                "split": split["split"],
                "family": split["family"],
                "prediction_rule": "coarse_leg_vs_arm_centroid",
                "train_count": int(len(train_idx)),
                "val_count": int(len(val_idx)),
                "metrics": {
                    "accuracy": float(np.mean(coarse_true == coarse_pred)),
                    "balanced_accuracy": float(
                        0.5
                        * (
                            np.mean(coarse_pred[coarse_true == 0] == 0)
                            + np.mean(coarse_pred[coarse_true == 1] == 1)
                        )
                    ),
                    "macro_f1": 0.0,
                    "per_class_recall": {
                        "Leg movements": float(np.mean(coarse_pred[coarse_true == 0] == 0)),
                        "Arm movements": float(np.mean(coarse_pred[coarse_true == 1] == 1)),
                    },
                    "confusion_matrix": [
                        [
                            int(np.sum((coarse_true == 0) & (coarse_pred == 0))),
                            int(np.sum((coarse_true == 0) & (coarse_pred == 1))),
                        ],
                        [
                            int(np.sum((coarse_true == 1) & (coarse_pred == 0))),
                            int(np.sum((coarse_true == 1) & (coarse_pred == 1))),
                        ],
                    ],
                },
            }
        )

    summary = summarize(rows)
    result = {
        "feature_dir": str(feature_dir),
        "event_feature_shape": list(x.shape),
        "rows": rows,
        "summary": summary,
        "best_by_family": {
            family: next(row for row in summary if row["family"] == family)
            for family in sorted(set(row["family"] for row in summary))
        },
        "note": (
            "The oracle coarse hierarchy uses the true leg-vs-arm group at test time only to estimate "
            "the upper bound of the within-pair fine classifier. It is diagnostic, not deployable."
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
