#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    apply_balanced_assignment,
    center_by_subject_run,
    l2_normalize,
    metrics,
)
from run_detrended_pair_feature_selection import load_checkpoints
from run_subject_calibration_curve import centroid_matrix_from_normalized, score
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


PERMUTATIONS = list(itertools.permutations(range(4)))


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def best_prediction_permutation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[tuple[int, ...], float]:
    best = max(
        PERMUTATIONS,
        key=lambda permutation: float(
            np.mean(y_true == np.asarray(permutation, dtype=np.int64)[y_pred])
        ),
    )
    accuracy = float(np.mean(y_true == np.asarray(best, dtype=np.int64)[y_pred]))
    return best, accuracy


def run_class_centroids(
    x_norm: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    return centroid_matrix_from_normalized(x_norm, y, indices)


def best_template_permutation(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[tuple[int, ...], float, float]:
    similarity = first.astype(np.float64) @ second.astype(np.float64).T
    best = max(
        PERMUTATIONS,
        key=lambda permutation: float(
            np.mean([similarity[class_idx, permutation[class_idx]] for class_idx in range(4)])
        ),
    )
    best_score = float(
        np.mean([similarity[class_idx, best[class_idx]] for class_idx in range(4)])
    )
    identity_score = float(np.mean(np.diag(similarity)))
    return best, best_score, identity_score


def analyze_subject(
    subject: str,
    x_norm: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    subject_indices: np.ndarray,
) -> dict:
    run_ids = sorted(set(int(records[idx]["run_id"]) for idx in subject_indices))
    true_all = []
    pred_all = []
    run_prediction_rows = []
    run_centroids = {}
    for holdout_run in run_ids:
        val_idx = np.asarray(
            [idx for idx in subject_indices if int(records[idx]["run_id"]) == holdout_run],
            dtype=np.int64,
        )
        train_idx = np.asarray(
            [idx for idx in subject_indices if int(records[idx]["run_id"]) != holdout_run],
            dtype=np.int64,
        )
        centroids = centroid_matrix_from_normalized(x_norm, y, train_idx)
        scores = score(x_norm, val_idx, centroids)
        pred = apply_balanced_assignment(scores, val_idx, records)
        permutation, permuted_accuracy = best_prediction_permutation(y[val_idx], pred)
        true_all.extend(y[val_idx].tolist())
        pred_all.extend(pred.tolist())
        run_prediction_rows.append(
            {
                "run_id": int(holdout_run),
                "identity_accuracy": float(np.mean(y[val_idx] == pred)),
                "best_prediction_permutation": list(permutation),
                "best_permuted_accuracy": permuted_accuracy,
                "confusion_matrix": metrics(y[val_idx], pred)["confusion_matrix"],
            }
        )
        run_centroids[holdout_run] = run_class_centroids(x_norm, y, val_idx)

    y_true = np.asarray(true_all, dtype=np.int64)
    y_pred = np.asarray(pred_all, dtype=np.int64)
    global_permutation, global_permuted_accuracy = best_prediction_permutation(y_true, y_pred)

    pair_rows = []
    for first_run, second_run in itertools.combinations(run_ids, 2):
        permutation, best_score, identity_score = best_template_permutation(
            run_centroids[first_run],
            run_centroids[second_run],
        )
        pair_rows.append(
            {
                "first_run": int(first_run),
                "second_run": int(second_run),
                "best_template_permutation": list(permutation),
                "identity_is_optimal": permutation == (0, 1, 2, 3),
                "identity_class_fraction": float(
                    np.mean(np.asarray(permutation) == np.arange(4))
                ),
                "best_template_similarity": best_score,
                "identity_template_similarity": identity_score,
                "permutation_similarity_gain": best_score - identity_score,
            }
        )
    permutation_counts = Counter(
        tuple(row["best_template_permutation"])
        for row in pair_rows
    )
    most_common_permutation, most_common_count = permutation_counts.most_common(1)[0]
    identity_metrics = metrics(y_true, y_pred)
    return {
        "subject": subject,
        "run_count": len(run_ids),
        "same_subject_leave_one_run_metrics": identity_metrics,
        "global_best_prediction_permutation": list(global_permutation),
        "global_permuted_accuracy": global_permuted_accuracy,
        "global_permutation_gain": global_permuted_accuracy - identity_metrics["accuracy"],
        "mean_run_best_permuted_accuracy": float(
            np.mean([row["best_permuted_accuracy"] for row in run_prediction_rows])
        ),
        "run_pair_identity_optimal_fraction": float(
            np.mean([row["identity_is_optimal"] for row in pair_rows])
        ),
        "run_pair_mean_identity_class_fraction": float(
            np.mean([row["identity_class_fraction"] for row in pair_rows])
        ),
        "run_pair_mean_permutation_similarity_gain": float(
            np.mean([row["permutation_similarity_gain"] for row in pair_rows])
        ),
        "most_common_run_pair_permutation": list(most_common_permutation),
        "most_common_run_pair_permutation_fraction": float(most_common_count / len(pair_rows)),
        "run_prediction_rows": run_prediction_rows,
        "run_pair_rows": pair_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit within-subject run-to-run class-template correspondence and permutations."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--window-name", default="offset_3_length_8")
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--focus-subjects",
        nargs="*",
        default=["sub-42", "sub-52", "sub-68", "sub-17", "sub-20", "sub-30", "sub-62"],
    )
    args = parser.parse_args()

    features, y, records = load_checkpoints(Path(args.checkpoint_dir), [args.window_name])
    centered = center_by_subject_run(features[args.window_name], records)
    detrended, group_rows = temporal_detrend_by_subject_run(centered, records, degree=1)
    x_norm = l2_normalize(detrended.astype(np.float32))
    subjects = np.asarray([str(record["subject_id"]) for record in records])
    rows = []
    for subject in sorted(set(subjects.tolist())):
        subject_indices = np.flatnonzero(subjects == subject)
        rows.append(analyze_subject(subject, x_norm, y, records, subject_indices))

    ranked = sorted(
        rows,
        key=lambda row: (
            row["run_pair_identity_optimal_fraction"],
            row["same_subject_leave_one_run_metrics"]["accuracy"],
        ),
    )
    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "window_name": args.window_name,
        "subject_count": len(rows),
        "mean_linear_time_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in group_rows])
        ),
        "rows": rows,
        "lowest_identity_correspondence": ranked[:15],
        "focus_subjects": [row for row in rows if row["subject"] in set(args.focus_subjects)],
        "note": (
            "Best label/template permutations use true labels and are forensic diagnostics only. "
            "Per-run permutation accuracy is optimistic with eight events; run-pair template "
            "correspondence across all 15 run pairs is the more useful stability measure."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "lowest_identity_correspondence": [
                    {
                        "subject": row["subject"],
                        "loro_accuracy": row["same_subject_leave_one_run_metrics"]["accuracy"],
                        "global_permuted_accuracy": row["global_permuted_accuracy"],
                        "identity_optimal_fraction": row["run_pair_identity_optimal_fraction"],
                        "mean_identity_class_fraction": row["run_pair_mean_identity_class_fraction"],
                        "most_common_permutation": row["most_common_run_pair_permutation"],
                        "most_common_fraction": row["most_common_run_pair_permutation_fraction"],
                    }
                    for row in ranked[:15]
                ],
                "focus_subjects": [
                    {
                        "subject": row["subject"],
                        "loro_accuracy": row["same_subject_leave_one_run_metrics"]["accuracy"],
                        "global_permuted_accuracy": row["global_permuted_accuracy"],
                        "identity_optimal_fraction": row["run_pair_identity_optimal_fraction"],
                        "mean_identity_class_fraction": row["run_pair_mean_identity_class_fraction"],
                        "most_common_permutation": row["most_common_run_pair_permutation"],
                        "most_common_fraction": row["most_common_run_pair_permutation_fraction"],
                    }
                    for row in result["focus_subjects"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
