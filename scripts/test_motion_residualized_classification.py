#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    apply_balanced_assignment,
    center_by_subject_run,
    l2_normalize,
    metrics,
)
from run_subject_calibration_curve import centroid_matrix_from_normalized, score
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


CLASS_NAMES = [
    "Left leg movements",
    "Right leg movements",
    "Forearm movements",
    "Upper arm movements",
]

MOTION_METRICS = [
    "normalized_dvars_percent",
    "phase_framewise_mm",
    "phase_absolute_mm",
    "center_framewise_mm",
    "center_absolute_mm",
]


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def standardize_by_run(values: np.ndarray, records: list[dict]) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    run_ids = np.asarray([int(row["run_id"]) for row in records])
    for run_id in sorted(set(run_ids.tolist())):
        indices = np.flatnonzero(run_ids == run_id)
        group = values[indices]
        result[indices] = (group - group.mean(axis=0, keepdims=True)) / np.maximum(
            group.std(axis=0, keepdims=True), 1e-8
        )
    return result


def motion_lookup(paths: list[Path]) -> dict[tuple[str, int, int, int], dict]:
    result = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for run in payload["runs"]:
            for event in run["event_motion"]["events"]:
                key = (
                    run["subject"],
                    int(run["run_id"]),
                    int(event["window_start"]),
                    CLASS_NAMES.index(event["trial_type"]),
                )
                result[key] = event
    return result


def align_motion(
    subject: str,
    records: list[dict],
    lookup: dict[tuple[str, int, int, int], dict],
    prefixes: tuple[str, ...],
    event_offset: int,
) -> np.ndarray:
    rows = []
    missing = []
    for record in records:
        key = (
            subject,
            int(record["run_id"]),
            int(record["event_start"]) + event_offset,
            int(record["class_id"]),
        )
        event = lookup.get(key)
        if event is None:
            missing.append(key)
            continue
        rows.append(
            [event[f"{prefix}{metric}"] for metric in MOTION_METRICS for prefix in prefixes]
        )
    if missing:
        raise ValueError(f"Missing {len(missing)} motion-event matches; examples: {missing[:3]}")
    return standardize_by_run(np.asarray(rows, dtype=np.float64), records)


def remove_class_means_from_motion(
    values: np.ndarray,
    y: np.ndarray,
    records: list[dict],
) -> np.ndarray:
    result = values.copy()
    run_ids = np.asarray([int(row["run_id"]) for row in records])
    for run_id in sorted(set(run_ids.tolist())):
        for class_id in range(4):
            indices = np.flatnonzero((run_ids == run_id) & (y == class_id))
            result[indices] -= result[indices].mean(axis=0, keepdims=True)
    return standardize_by_run(result, records)


def fit_motion_model(
    z: np.ndarray,
    x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    gram = z.T @ z
    penalty = alpha * np.eye(gram.shape[0], dtype=np.float64)
    return np.linalg.solve(gram + penalty, z.T @ x.astype(np.float64))


def variance_removed(original: np.ndarray, residual: np.ndarray) -> float:
    denominator = float(np.sum(original.astype(np.float64) ** 2))
    return 1.0 - float(np.sum(residual.astype(np.float64) ** 2)) / max(denominator, 1e-8)


def classify_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    val_indices: np.ndarray,
    records: list[dict],
) -> np.ndarray:
    normalized_train = l2_normalize(x_train.astype(np.float32))
    normalized_val = l2_normalize(x_val.astype(np.float32))
    local_train_indices = np.arange(len(y_train), dtype=np.int64)
    centroids = centroid_matrix_from_normalized(
        normalized_train,
        y_train,
        local_train_indices,
    )
    fold_scores = normalized_val.astype(np.float64) @ centroids.astype(np.float64).T
    return apply_balanced_assignment(fold_scores, val_indices, records)


def evaluate_subject(
    subject: str,
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    motion: np.ndarray,
    alphas: list[float],
) -> dict:
    run_ids = np.asarray([int(row["run_id"]) for row in records], dtype=np.int64)
    configurations = {"baseline": {"true": [], "pred": [], "folds": []}}
    for alpha in alphas:
        configurations[f"motion_ridge_{alpha:g}"] = {"true": [], "pred": [], "folds": []}

    for holdout_run in sorted(set(run_ids.tolist())):
        val_idx = np.flatnonzero(run_ids == holdout_run)
        train_idx = np.flatnonzero(run_ids != holdout_run)
        baseline_pred = classify_fold(
            x[train_idx], y[train_idx], x[val_idx], val_idx, records
        )
        baseline = configurations["baseline"]
        baseline["true"].extend(y[val_idx].tolist())
        baseline["pred"].extend(baseline_pred.tolist())
        baseline["folds"].append(
            {
                "holdout_run": holdout_run,
                "accuracy": float(np.mean(y[val_idx] == baseline_pred)),
            }
        )

        for alpha in alphas:
            beta = fit_motion_model(motion[train_idx], x[train_idx], alpha)
            train_residual = x[train_idx] - motion[train_idx] @ beta
            val_residual = x[val_idx] - motion[val_idx] @ beta
            prediction = classify_fold(
                train_residual,
                y[train_idx],
                val_residual,
                val_idx,
                records,
            )
            configuration = configurations[f"motion_ridge_{alpha:g}"]
            configuration["true"].extend(y[val_idx].tolist())
            configuration["pred"].extend(prediction.tolist())
            configuration["folds"].append(
                {
                    "holdout_run": holdout_run,
                    "accuracy": float(np.mean(y[val_idx] == prediction)),
                    "train_variance_removed": variance_removed(
                        x[train_idx], train_residual
                    ),
                    "validation_variance_removed": variance_removed(
                        x[val_idx], val_residual
                    ),
                }
            )

    result = {}
    baseline_accuracy = None
    for name, configuration in configurations.items():
        target = np.asarray(configuration.pop("true"), dtype=np.int64)
        prediction = np.asarray(configuration.pop("pred"), dtype=np.int64)
        summary = metrics(target, prediction)
        if name == "baseline":
            baseline_accuracy = summary["accuracy"]
        configuration.update(summary)
        if name != "baseline":
            configuration["accuracy_change_from_baseline"] = (
                summary["accuracy"] - baseline_accuracy
            )
            configuration["mean_train_variance_removed"] = float(
                np.mean([row["train_variance_removed"] for row in configuration["folds"]])
            )
            configuration["mean_validation_variance_removed"] = float(
                np.mean(
                    [row["validation_variance_removed"] for row in configuration["folds"]]
                )
            )
        result[name] = configuration
    return result


def load_subject_checkpoint(
    path: Path,
    window_name: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    with np.load(path, allow_pickle=False) as payload:
        x = payload[window_name].astype(np.float32)
        y = payload["labels"].astype(np.int64)
        records = json.loads(str(payload["records_json"]))
    centered = center_by_subject_run(x, records)
    detrended, _ = temporal_detrend_by_subject_run(centered, records, degree=1)
    return detrended.astype(np.float32), y, records


def permute_within_run(
    values: np.ndarray,
    records: list[dict],
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.empty_like(values)
    run_ids = np.asarray([int(row["run_id"]) for row in records])
    for run_id in sorted(set(run_ids.tolist())):
        indices = np.flatnonzero(run_ids == run_id)
        result[indices] = values[rng.permutation(indices)]
    return result


def permutation_null(
    subject: str,
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    motion: np.ndarray,
    observed_accuracy: float,
    alpha: float,
    permutations: int,
    rng: np.random.Generator,
) -> dict:
    accuracies = []
    for _ in range(permutations):
        shuffled = permute_within_run(motion, records, rng)
        evaluated = evaluate_subject(subject, x, y, records, shuffled, [alpha])
        accuracies.append(evaluated[f"motion_ridge_{alpha:g}"]["accuracy"])
    values = np.asarray(accuracies, dtype=np.float64)
    return {
        "alpha": alpha,
        "permutations": permutations,
        "observed_accuracy": observed_accuracy,
        "null_mean_accuracy": float(np.mean(values)),
        "null_p05_accuracy": float(np.percentile(values, 5)),
        "null_p95_accuracy": float(np.percentile(values, 95)),
        "lower_tail_p_value": float(
            (1 + np.sum(values <= observed_accuracy)) / (permutations + 1)
        ),
        "null_accuracies": accuracies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test train-only removal of image-derived motion from event patterns."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--motion-json", action="append", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--window-name", default="offset_3_length_8")
    parser.add_argument("--subjects", nargs="*", default=["sub-30", "sub-42", "sub-52", "sub-62"])
    parser.add_argument("--alphas", default="0.1,1,10,100")
    parser.add_argument("--event-offset", type=int, default=3)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--null-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260627)
    args = parser.parse_args()

    alphas = [float(value) for value in args.alphas.split(",")]
    lookup = motion_lookup([Path(value) for value in args.motion_json])
    checkpoint_dir = Path(args.checkpoint_dir)
    results = {}
    null_results = {}
    rng = np.random.default_rng(args.seed)
    for subject in args.subjects:
        print(f"evaluating {subject}", flush=True)
        x, y, records = load_subject_checkpoint(
            checkpoint_dir / f"{subject}.npz", args.window_name
        )
        motion_mean = align_motion(
            subject, records, lookup, ("mean_",), args.event_offset
        )
        motion_mean_max = align_motion(
            subject, records, lookup, ("mean_", "max_"), args.event_offset
        )
        motion_mean_class_residual = remove_class_means_from_motion(
            motion_mean, y, records
        )
        motion_mean_max_class_residual = remove_class_means_from_motion(
            motion_mean_max, y, records
        )
        results[subject] = {
            "mean_motion_features": evaluate_subject(
                subject, x, y, records, motion_mean, alphas
            ),
            "mean_and_max_motion_features": evaluate_subject(
                subject, x, y, records, motion_mean_max, alphas
            ),
            "mean_motion_features_class_residual": evaluate_subject(
                subject, x, y, records, motion_mean_class_residual, alphas
            ),
            "mean_and_max_motion_features_class_residual": evaluate_subject(
                subject, x, y, records, motion_mean_max_class_residual, alphas
            ),
        }
        null_results[subject] = {
            "mean_motion_features": permutation_null(
                subject,
                x,
                y,
                records,
                motion_mean,
                results[subject]["mean_motion_features"][
                    f"motion_ridge_{args.null_alpha:g}"
                ]["accuracy"],
                args.null_alpha,
                args.permutations,
                rng,
            ),
            "mean_and_max_motion_features": permutation_null(
                subject,
                x,
                y,
                records,
                motion_mean_max,
                results[subject]["mean_and_max_motion_features"][
                    f"motion_ridge_{args.null_alpha:g}"
                ]["accuracy"],
                args.null_alpha,
                args.permutations,
                rng,
            ),
        }

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "motion_jsons": args.motion_json,
        "window_name": args.window_name,
        "subjects": args.subjects,
        "alphas": alphas,
        "permutations": args.permutations,
        "null_alpha": args.null_alpha,
        "seed": args.seed,
        "results": results,
        "within_run_shuffled_motion_null": null_results,
        "note": (
            "Motion-to-voxel coefficients are fitted on five training runs and applied to the "
            "unseen sixth run. Motion summaries are standardized within run without labels. "
            "Class-residual variants use true labels only as a forensic control: class means are "
            "removed within each run, leaving repetition-level excess motion that cannot directly "
            "encode motor class. "
            "A loss after residualization can indicate motion-linked signal but cannot by itself "
            "separate motion artifact from neural responses correlated with task execution."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    compact = {
        subject: {
            feature_set: {
                name: {
                    "accuracy": row["accuracy"],
                    "accuracy_change_from_baseline": row.get(
                        "accuracy_change_from_baseline"
                    ),
                    "mean_train_variance_removed": row.get(
                        "mean_train_variance_removed"
                    ),
                    "mean_validation_variance_removed": row.get(
                        "mean_validation_variance_removed"
                    ),
                }
                for name, row in configurations.items()
            }
            for feature_set, configurations in subject_result.items()
        }
        for subject, subject_result in results.items()
    }
    compact_null = {
        subject: {
            feature_set: {
                key: value
                for key, value in row.items()
                if key != "null_accuracies"
            }
            for feature_set, row in subject_result.items()
        }
        for subject, subject_result in null_results.items()
    }
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "results": compact,
                "within_run_shuffled_motion_null": compact_null,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
