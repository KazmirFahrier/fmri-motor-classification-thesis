#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import center_by_subject_run
from run_detrended_pair_feature_selection import load_checkpoints, rank_pair_features
from run_spatial_scale_feature_sweep import transform_scale
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


ARM_CLASSES = (2, 3)


def arm_indices(indices: np.ndarray, y: np.ndarray) -> np.ndarray:
    return indices[np.isin(y[indices], ARM_CLASSES)]


def class_means(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    indices = arm_indices(indices, y)
    return np.stack(
        [x[indices[y[indices] == class_id]][:, selected].mean(axis=0) for class_id in ARM_CLASSES]
    ).astype(np.float64)


def source_inverse_covariance(
    x: np.ndarray,
    y: np.ndarray,
    source_idx: np.ndarray,
    selected: np.ndarray,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_idx = arm_indices(source_idx, y)
    train = x[source_idx][:, selected].astype(np.float64)
    means = class_means(x, y, source_idx, selected)
    target = (y[source_idx] == ARM_CLASSES[1]).astype(np.int64)
    residuals = np.concatenate(
        [train[target == class_id] - means[class_id] for class_id in range(2)]
    )
    covariance = residuals.T @ residuals / max(len(residuals) - 2, 1)
    diagonal = np.diag(covariance)
    positive = diagonal[diagonal > 0]
    variance_target = float(np.median(positive)) if len(positive) else 1.0
    regularized = (1.0 - shrinkage) * covariance
    regularized.flat[:: len(selected) + 1] += shrinkage * variance_target
    regularized.flat[:: len(selected) + 1] += max(variance_target * 1e-5, 1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    inverse = (eigenvectors / np.maximum(eigenvalues, variance_target * 1e-8)) @ eigenvectors.T
    return means, inverse


def lda_scores(
    x: np.ndarray,
    indices: np.ndarray,
    selected: np.ndarray,
    means: np.ndarray,
    inverse_covariance: np.ndarray,
) -> np.ndarray:
    linear = means @ inverse_covariance
    intercept = -0.5 * np.sum(means * linear, axis=1)
    return x[indices][:, selected].astype(np.float64) @ linear.T + intercept


def predictions(scores: np.ndarray) -> dict[str, np.ndarray]:
    independent = scores.argmax(axis=1).astype(np.int64)
    balanced = np.zeros(len(scores), dtype=np.int64)
    difference = scores[:, 1] - scores[:, 0]
    balanced[np.argsort(difference, kind="stable")[len(scores) // 2 :]] = 1
    return {"independent": independent, "balanced": balanced}


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    subject_idx: np.ndarray,
    calibration_runs: tuple[int, ...],
    selected: np.ndarray,
    source_means: np.ndarray,
    inverse_covariance: np.ndarray,
    alphas: list[float],
    fallback: float,
) -> float:
    if len(calibration_runs) < 2:
        return fallback
    rows = []
    for alpha in alphas:
        accuracies = []
        for validation_run in calibration_runs:
            train_runs = set(calibration_runs) - {validation_run}
            train_idx = np.asarray(
                [idx for idx in subject_idx if int(records[idx]["run_id"]) in train_runs]
            )
            val_idx = np.asarray(
                [
                    idx
                    for idx in subject_idx
                    if int(records[idx]["run_id"]) == validation_run
                    and y[idx] in ARM_CLASSES
                ]
            )
            subject_means = class_means(x, y, train_idx, selected)
            means = (1.0 - alpha) * source_means + alpha * subject_means
            score = lda_scores(x, val_idx, selected, means, inverse_covariance)
            pred = predictions(score)["balanced"]
            accuracies.append(float(np.mean(pred == (y[val_idx] == ARM_CLASSES[1]))))
        rows.append((float(np.mean(accuracies)), -alpha, alpha))
    return float(max(rows)[2])


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["protocol"],
                row["prediction_rule"],
                row["calibration_run_count"],
                row["summary_alpha"],
            )
        ].append(row["accuracy"])
    return [
        {
            "protocol": key[0],
            "prediction_rule": key[1],
            "calibration_run_count": key[2],
            "alpha": key[3],
            "split_count": len(values),
            "mean_accuracy": float(np.mean(values)),
        }
        for key, values in sorted(grouped.items())
    ]


def summarize_by_residual_group(rows: list[dict], group_by_subject: dict[str, str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["subject"] not in group_by_subject:
            continue
        grouped[
            (
                group_by_subject[row["subject"]],
                row["protocol"],
                row["prediction_rule"],
                row["calibration_run_count"],
                row["summary_alpha"],
            )
        ].append(row["accuracy"])
    return [
        {
            "residual_type": key[0],
            "protocol": key[1],
            "prediction_rule": key[2],
            "calibration_run_count": key[3],
            "alpha": key[4],
            "split_count": len(values),
            "mean_accuracy": float(np.mean(values)),
        }
        for key, values in sorted(grouped.items())
    ]


def append_cross_subject_fallback(rows: list[dict]) -> None:
    fixed = [
        row
        for row in rows
        if row["protocol"] == "fixed_alpha" and row["calibration_run_count"] == 1
    ]
    subjects = sorted({row["subject"] for row in fixed})
    for prediction_rule in sorted({row["prediction_rule"] for row in fixed}):
        rule_rows = [row for row in fixed if row["prediction_rule"] == prediction_rule]
        alphas = sorted({row["alpha"] for row in rule_rows})
        for subject in subjects:
            source_rows = [row for row in rule_rows if row["subject"] != subject]
            mean_by_alpha = {
                alpha: float(
                    np.mean([row["accuracy"] for row in source_rows if row["alpha"] == alpha])
                )
                for alpha in alphas
            }
            selected_alpha = min(
                alphas, key=lambda alpha: (-mean_by_alpha[alpha], alpha)
            )
            for row in rule_rows:
                if row["subject"] == subject and row["alpha"] == selected_alpha:
                    selected = dict(row)
                    selected["protocol"] = "cross_subject_fallback"
                    selected["summary_alpha"] = -2.0
                    rows.append(selected)


def append_cross_subject_gate(rows: list[dict]) -> None:
    fixed = [
        row
        for row in rows
        if row["protocol"] == "fixed_alpha"
        and row["calibration_run_count"] == 1
        and np.isclose(row["alpha"], 0.1)
    ]
    source = {
        (row["subject"], row["holdout_run"], row["prediction_rule"]): row["accuracy"]
        for row in rows
        if row["protocol"] == "source_only"
    }
    thresholds = [-1.0, 0.5, 0.75, 1.0]
    subjects = sorted({row["subject"] for row in fixed})
    for prediction_rule in sorted({row["prediction_rule"] for row in fixed}):
        rule_rows = [row for row in fixed if row["prediction_rule"] == prediction_rule]
        for subject in subjects:
            candidate_means = {}
            for threshold in thresholds:
                accuracies = []
                for row in rule_rows:
                    if row["subject"] == subject:
                        continue
                    if row["calibration_source_accuracy"] <= threshold:
                        accuracies.append(row["accuracy"])
                    else:
                        accuracies.append(
                            source[(row["subject"], row["holdout_run"], prediction_rule)]
                        )
                candidate_means[threshold] = float(np.mean(accuracies))
            selected_threshold = min(
                thresholds,
                key=lambda threshold: (-candidate_means[threshold], threshold),
            )
            for row in rule_rows:
                if row["subject"] != subject:
                    continue
                selected = dict(row)
                selected["protocol"] = "cross_subject_calibration_gate"
                selected["summary_alpha"] = -3.0
                selected["gate_threshold"] = selected_threshold
                if row["calibration_source_accuracy"] > selected_threshold:
                    selected["accuracy"] = source[
                        (subject, row["holdout_run"], prediction_rule)
                    ]
                    selected["calibration_applied"] = False
                else:
                    selected["calibration_applied"] = True
                rows.append(selected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure labeled target-subject calibration of the arm LDA branch."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--residual-json")
    parser.add_argument("--window-name", default="offset_3_length_8")
    parser.add_argument("--feature-shape", default="24,24,24")
    parser.add_argument("--transform", default="smooth_3")
    parser.add_argument("--feature-count", type=int, default=1024)
    parser.add_argument("--shrinkage", type=float, default=0.75)
    parser.add_argument("--max-calibration-runs", type=int, default=5)
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--fallback-alpha", type=float, default=0.25)
    args = parser.parse_args()

    shape = tuple(int(value) for value in args.feature_shape.split(","))
    features, y, records = load_checkpoints(Path(args.checkpoint_dir), [args.window_name])
    centered = center_by_subject_run(features[args.window_name], records)
    detrended, detrend_rows = temporal_detrend_by_subject_run(centered, records, degree=1)
    x, transformed_shape = transform_scale(detrended, shape, args.transform, 64)
    subjects = sorted({str(record["subject_id"]) for record in records})
    all_idx = np.arange(len(y), dtype=np.int64)
    rows = []

    for subject in subjects:
        print(f"calibrating {subject}", flush=True)
        subject_idx = np.asarray(
            [idx for idx, record in enumerate(records) if str(record["subject_id"]) == subject]
        )
        source_idx = all_idx[~np.isin(all_idx, subject_idx)]
        ranking = rank_pair_features(x, y, source_idx, ARM_CLASSES)
        selected = ranking[: min(args.feature_count, x.shape[1])]
        source_means, inverse_covariance = source_inverse_covariance(
            x, y, source_idx, selected, args.shrinkage
        )
        run_ids = sorted({int(records[idx]["run_id"]) for idx in subject_idx})
        for holdout_run in run_ids:
            val_idx = np.asarray(
                [
                    idx
                    for idx in subject_idx
                    if int(records[idx]["run_id"]) == holdout_run and y[idx] in ARM_CLASSES
                ]
            )
            available = [run_id for run_id in run_ids if run_id != holdout_run]
            for count in range(1, min(args.max_calibration_runs, len(available)) + 1):
                for calibration_runs in itertools.combinations(available, count):
                    calibration_idx = np.asarray(
                        [
                            idx
                            for idx in subject_idx
                            if int(records[idx]["run_id"]) in set(calibration_runs)
                        ]
                    )
                    target_means = class_means(x, y, calibration_idx, selected)
                    calibration_arm_idx = arm_indices(calibration_idx, y)
                    calibration_source_score = lda_scores(
                        x,
                        calibration_arm_idx,
                        selected,
                        source_means,
                        inverse_covariance,
                    )
                    calibration_source_accuracy = {
                        rule: float(
                            np.mean(
                                prediction
                                == (y[calibration_arm_idx] == ARM_CLASSES[1])
                            )
                        )
                        for rule, prediction in predictions(
                            calibration_source_score
                        ).items()
                    }
                    combination_protocols = [
                        ("fixed_alpha", calibration_runs, alpha, alpha)
                        for alpha in args.alphas
                    ]
                    selected_alpha = select_alpha(
                        x,
                        y,
                        records,
                        subject_idx,
                        calibration_runs,
                        selected,
                        source_means,
                        inverse_covariance,
                        args.alphas,
                        args.fallback_alpha,
                    )
                    combination_protocols.append(
                        ("validated_alpha", calibration_runs, selected_alpha, -1.0)
                    )
                    for protocol, runs, alpha, summary_alpha in combination_protocols:
                        means = (1.0 - alpha) * source_means + alpha * target_means
                        score = lda_scores(x, val_idx, selected, means, inverse_covariance)
                        for rule, pred in predictions(score).items():
                            rows.append(
                                {
                                    "subject": subject,
                                    "holdout_run": holdout_run,
                                    "protocol": protocol,
                                    "prediction_rule": rule,
                                    "calibration_runs": list(runs),
                                    "calibration_run_count": len(runs),
                                    "alpha": alpha,
                                    "summary_alpha": summary_alpha,
                                    "calibration_source_accuracy": calibration_source_accuracy[
                                        rule
                                    ],
                                    "accuracy": float(
                                        np.mean(pred == (y[val_idx] == ARM_CLASSES[1]))
                                    ),
                                }
                            )
            source_score = lda_scores(x, val_idx, selected, source_means, inverse_covariance)
            for rule, pred in predictions(source_score).items():
                rows.append(
                    {
                        "subject": subject,
                        "holdout_run": holdout_run,
                        "protocol": "source_only",
                        "prediction_rule": rule,
                        "calibration_runs": [],
                        "calibration_run_count": 0,
                        "alpha": 0.0,
                        "summary_alpha": 0.0,
                        "accuracy": float(np.mean(pred == (y[val_idx] == ARM_CLASSES[1]))),
                    }
                )

    append_cross_subject_fallback(rows)
    append_cross_subject_gate(rows)
    residual_group_summary = []
    if args.residual_json:
        residual = json.loads(Path(args.residual_json).read_text())
        group_by_subject = {
            row["subject"]: row["residual_type"] for row in residual["subjects"]
        }
        residual_group_summary = summarize_by_residual_group(rows, group_by_subject)
    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "window_name": args.window_name,
        "transform": args.transform,
        "transformed_shape": transformed_shape,
        "feature_count": args.feature_count,
        "shrinkage": args.shrinkage,
        "alphas": args.alphas,
        "mean_temporal_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in detrend_rows])
        ),
        "summary": summarize(rows),
        "residual_group_summary": residual_group_summary,
        "rows": rows,
        "note": (
            "Every evaluation run is excluded from target-subject calibration. Feature ranking and "
            "covariance are source-subject only; labeled calibration runs update arm class means."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps({"out_json": args.out_json, "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
