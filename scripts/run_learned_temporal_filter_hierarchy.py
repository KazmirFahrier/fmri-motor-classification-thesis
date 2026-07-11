#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import center_by_subject_run, metrics
from run_detrended_pair_feature_selection import PAIR_CLASSES, load_checkpoints, outer_splits
from run_hierarchy_subject_calibration import (
    evaluate_scores,
    fit_source_pair_model,
    pair_scores,
)
from run_hybrid_spatial_hierarchy import selected_coarse_scores
from run_spatial_scale_feature_sweep import transform_scale
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


def preprocess_sequence(
    sequence: np.ndarray,
    records: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    if sequence.ndim != 3:
        raise ValueError(f"Expected event x time x feature sequence, got {sequence.shape}.")
    rows = []
    for lag in range(sequence.shape[1]):
        centered = center_by_subject_run(sequence[:, lag], records)
        detrended, detrend_rows = temporal_detrend_by_subject_run(
            centered, records, degree=1
        )
        sequence[:, lag] = detrended
        rows.append(
            {
                "lag": lag,
                "mean_temporal_variance_fraction": float(
                    np.mean(
                        [row["temporal_variance_fraction"] for row in detrend_rows]
                    )
                ),
            }
        )
    return sequence, rows


def second_difference_penalty(length: int) -> np.ndarray:
    if length < 3:
        return np.zeros((length, length), dtype=np.float64)
    operator = np.zeros((length - 2, length), dtype=np.float64)
    for row in range(length - 2):
        operator[row, row : row + 3] = (1.0, -2.0, 1.0)
    return operator.T @ operator


def learn_temporal_filter(
    sequence: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    classes: tuple[int, int],
    method: str = "fisher",
    covariance_shrinkage: float = 0.5,
    smoothness: float = 1.0,
) -> tuple[np.ndarray, dict]:
    pair_idx = train_idx[np.isin(y[train_idx], classes)]
    class_indices = [pair_idx[y[pair_idx] == class_id] for class_id in classes]
    class_means = np.stack(
        [sequence[indices].mean(axis=0, dtype=np.float64) for indices in class_indices]
    )
    difference = class_means[1] - class_means[0]
    signal = difference @ difference.T / difference.shape[1]

    covariance = np.zeros((sequence.shape[1], sequence.shape[1]), dtype=np.float64)
    for local_class, indices in enumerate(class_indices):
        mean = class_means[local_class]
        for index in indices:
            residual = sequence[int(index)].astype(np.float64) - mean
            covariance += residual @ residual.T / residual.shape[1]
    covariance /= max(len(pair_idx) - 2, 1)
    diagonal = np.diag(covariance)
    positive = diagonal[diagonal > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0

    if method == "svd":
        eigenvalues, eigenvectors = np.linalg.eigh(signal)
        weights = eigenvectors[:, int(np.argmax(eigenvalues))]
    elif method == "fisher":
        regularized = (1.0 - covariance_shrinkage) * covariance
        regularized += covariance_shrinkage * scale * np.eye(len(covariance))
        regularized += smoothness * scale * second_difference_penalty(len(covariance))
        regularized += max(scale * 1e-6, 1e-12) * np.eye(len(covariance))
        eigenvalues, eigenvectors = np.linalg.eigh(regularized)
        inverse_sqrt = (
            eigenvectors
            / np.sqrt(np.maximum(eigenvalues, max(scale * 1e-8, 1e-12)))
        ) @ eigenvectors.T
        whitened_signal = inverse_sqrt @ signal @ inverse_sqrt
        signal_values, signal_vectors = np.linalg.eigh(whitened_signal)
        weights = inverse_sqrt @ signal_vectors[:, int(np.argmax(signal_values))]
    else:
        raise ValueError(f"Unknown temporal filter method: {method}")

    weights /= max(float(np.linalg.norm(weights)), 1e-12)
    if float(np.sum(weights)) < 0:
        weights = -weights
    diagnostics = {
        "classes": list(classes),
        "method": method,
        "covariance_shrinkage": covariance_shrinkage,
        "smoothness": smoothness,
        "weights": weights.tolist(),
        "weight_sum": float(np.sum(weights)),
        "peak_lag": int(np.argmax(np.abs(weights))),
        "signal_trace": float(np.trace(signal)),
        "noise_trace": float(np.trace(covariance)),
    }
    return weights, diagnostics


def filtered_map(sequence: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("ntf,t->nf", sequence, weights, optimize=True).astype(np.float32)


def subject_metrics(
    coarse_scores: np.ndarray,
    pair_score_by_name: dict[str, np.ndarray],
    coarse_weight: float,
    y: np.ndarray,
    val_idx: np.ndarray,
    records: list[dict],
) -> tuple[dict[str, dict], dict[str, dict[str, dict]]]:
    overall = evaluate_scores(
        coarse_scores,
        pair_score_by_name,
        coarse_weight,
        y,
        val_idx,
        records,
    )
    subjects = np.asarray([str(records[int(index)]["subject_id"]) for index in val_idx])
    by_subject = {}
    for subject in sorted(set(subjects.tolist())):
        mask = subjects == subject
        subject_idx = val_idx[mask]
        by_subject[subject] = evaluate_scores(
            coarse_scores[mask],
            {name: scores[mask] for name, scores in pair_score_by_name.items()},
            coarse_weight,
            y,
            subject_idx,
            records,
        )
    return overall, by_subject


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["representation"], row["prediction_rule"])].append(row)
    output = []
    for (representation, rule), values in sorted(grouped.items()):
        output.append(
            {
                "representation": representation,
                "prediction_rule": rule,
                "split_count": len(values),
                "mean_balanced_accuracy": float(
                    np.mean([row["metrics"]["balanced_accuracy"] for row in values])
                ),
                "mean_macro_f1": float(
                    np.mean([row["metrics"]["macro_f1"] for row in values])
                ),
                "mean_leg_pair_accuracy": float(
                    np.mean([row["metrics"]["leg_pair_accuracy"] for row in values])
                ),
                "mean_arm_pair_accuracy": float(
                    np.mean([row["metrics"]["arm_pair_accuracy"] for row in values])
                ),
            }
        )
    return output


def paired_bootstrap(
    rows: list[dict],
    iterations: int,
    seed: int,
) -> dict:
    baseline = {
        (row["split"], row["prediction_rule"]): row
        for row in rows
        if row["representation"] == "mean"
    }
    learned = [row for row in rows if row["representation"] == "learned_arm_filter"]
    result = {}
    rng = np.random.default_rng(seed)
    for rule in sorted({row["prediction_rule"] for row in learned}):
        values = np.asarray(
            [
                row["metrics"]["balanced_accuracy"]
                - baseline[(row["split"], rule)]["metrics"]["balanced_accuracy"]
                for row in learned
                if row["prediction_rule"] == rule
            ]
        )
        samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
        result[rule] = {
            "mean_difference": float(np.mean(values)),
            "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
            "wins": int(np.sum(values > 0)),
            "ties": int(np.sum(values == 0)),
            "losses": int(np.sum(values < 0)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn a training-subject-only temporal filter for the hierarchy arm branch."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--filter-method", choices=["fisher", "svd"], default="fisher")
    parser.add_argument("--filter-covariance-shrinkage", type=float, default=0.5)
    parser.add_argument("--filter-smoothness", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split-limit", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline_json).read_text())
    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, detrend_rows = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    mean_x = sequence.mean(axis=1, dtype=np.float32)
    shape = tuple(int(value) for value in baseline["native_feature_shape"])
    if mean_x.shape[1] != int(np.prod(shape)):
        raise ValueError(f"Shape {shape} does not match {mean_x.shape[1]} features.")
    mean_pair_x, pair_shape = transform_scale(
        mean_x, shape, baseline["pair_transform"], args.batch_size
    )
    hyperparameters = {row["split"]: row for row in baseline["hyperparameters"]}
    splits = outer_splits(records, "subject", 6, baseline["subject_seeds"])
    split_by_name = {split["split"]: split for split in splits}
    split_names = sorted(hyperparameters)
    if args.split_limit is not None:
        split_names = split_names[: args.split_limit]

    rows = []
    filter_diagnostics = []
    for split_name in split_names:
        print(f"evaluating {split_name}", flush=True)
        split = split_by_name[split_name]
        hyper = hyperparameters[split_name]
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        weights, diagnostics = learn_temporal_filter(
            sequence,
            y,
            train_idx,
            PAIR_CLASSES["arm"],
            args.filter_method,
            args.filter_covariance_shrinkage,
            args.filter_smoothness,
        )
        diagnostics["split"] = split_name
        filter_diagnostics.append(diagnostics)
        arm_native = filtered_map(sequence, weights)
        arm_pair_x, arm_shape = transform_scale(
            arm_native, shape, baseline["pair_transform"], args.batch_size
        )
        if arm_shape != pair_shape:
            raise ValueError(f"Mean/filtered shapes differ: {pair_shape} vs {arm_shape}.")

        pair_models = {}
        for pair_name, pair_x in (("leg", mean_pair_x), ("arm", mean_pair_x)):
            config = hyper["selected_pair_configurations"][pair_name]
            pair_models[f"mean_{pair_name}"] = fit_source_pair_model(
                pair_x,
                y,
                train_idx,
                PAIR_CLASSES[pair_name],
                int(config["feature_count"]),
                float(config["shrinkage"]),
            )
        arm_config = hyper["selected_pair_configurations"]["arm"]
        learned_arm_model = fit_source_pair_model(
            arm_pair_x,
            y,
            train_idx,
            PAIR_CLASSES["arm"],
            int(arm_config["feature_count"]),
            float(arm_config["shrinkage"]),
        )
        coarse_scores, _ = selected_coarse_scores(
            mean_x,
            y,
            train_idx,
            val_idx,
            int(hyper["selected_coarse_feature_count"]),
            hyper["selected_coarse_method"],
            float(hyper["selected_coarse_lda_shrinkage"]),
        )
        mean_scores = {
            pair_name: pair_scores(
                mean_pair_x, val_idx, pair_models[f"mean_{pair_name}"]
            )
            for pair_name in PAIR_CLASSES
        }
        learned_scores = {
            "leg": mean_scores["leg"],
            "arm": pair_scores(arm_pair_x, val_idx, learned_arm_model),
        }
        for representation, scores in (
            ("mean", mean_scores),
            ("learned_arm_filter", learned_scores),
        ):
            overall, by_subject = subject_metrics(
                coarse_scores,
                scores,
                float(hyper["selected_coarse_weight"]),
                y,
                val_idx,
                records,
            )
            for rule, metric_row in overall.items():
                rows.append(
                    {
                        "split": split_name,
                        "representation": representation,
                        "prediction_rule": rule,
                        "metrics": metric_row,
                        "subject_metrics": {
                            subject: subject_rows[rule]
                            for subject, subject_rows in by_subject.items()
                        },
                    }
                )

    summary = summarize(rows)
    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "baseline_json": args.baseline_json,
        "sequence_key": args.sequence_key,
        "sequence_shape": list(sequence.shape),
        "pair_shape": pair_shape,
        "filter_method": args.filter_method,
        "filter_covariance_shrinkage": args.filter_covariance_shrinkage,
        "filter_smoothness": args.filter_smoothness,
        "detrend_by_lag": detrend_rows,
        "filter_diagnostics": filter_diagnostics,
        "rows": rows,
        "summary": summary,
        "paired_fold_bootstrap": paired_bootstrap(
            rows, args.bootstrap_iterations, args.bootstrap_seed
        ),
        "note": (
            "Each outer-fold arm temporal filter is learned from source subjects only. "
            "The mean baseline, spatial feature counts, covariance shrinkage, coarse model, "
            "and hierarchy weight reuse the saved nested hierarchy configuration."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "summary": summary,
                "paired_fold_bootstrap": result["paired_fold_bootstrap"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
