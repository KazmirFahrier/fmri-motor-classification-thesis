#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import apply_balanced_assignment, center_by_subject_run, metrics
from run_detrended_hierarchy_sweep import exact_scores_from_hierarchy
from run_detrended_pair_feature_selection import (
    PAIR_CLASSES,
    load_checkpoints,
    outer_splits,
    rank_pair_features,
)
from run_hybrid_spatial_hierarchy import selected_coarse_scores
from run_spatial_scale_feature_sweep import transform_scale
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


def pair_indices(indices: np.ndarray, y: np.ndarray, classes: tuple[int, int]) -> np.ndarray:
    return indices[np.isin(y[indices], classes)]


def pair_means(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    selected: np.ndarray,
    classes: tuple[int, int],
) -> np.ndarray:
    indices = pair_indices(indices, y, classes)
    return np.stack(
        [
            x[indices[y[indices] == class_id]][:, selected].mean(axis=0)
            for class_id in classes
        ]
    ).astype(np.float64)


def fit_source_pair_model(
    x: np.ndarray,
    y: np.ndarray,
    source_idx: np.ndarray,
    classes: tuple[int, int],
    feature_count: int,
    shrinkage: float,
) -> dict:
    ranking = rank_pair_features(x, y, source_idx, classes)
    selected = ranking[: min(feature_count, x.shape[1])]
    train_idx = pair_indices(source_idx, y, classes)
    train = x[train_idx][:, selected].astype(np.float64)
    means = pair_means(x, y, train_idx, selected, classes)
    target = (y[train_idx] == classes[1]).astype(np.int64)
    residuals = np.concatenate(
        [train[target == class_id] - means[class_id] for class_id in range(2)],
        axis=0,
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
    return {
        "classes": classes,
        "selected": selected,
        "source_means": means,
        "inverse_covariance": inverse,
    }


def pair_scores(
    x: np.ndarray,
    indices: np.ndarray,
    model: dict,
    means: np.ndarray | None = None,
) -> np.ndarray:
    if means is None:
        means = model["source_means"]
    inverse = model["inverse_covariance"]
    linear = means @ inverse
    intercept = -0.5 * np.sum(means * linear, axis=1)
    selected = model["selected"]
    return x[indices][:, selected].astype(np.float64) @ linear.T + intercept


def balanced_pair_prediction(scores: np.ndarray) -> np.ndarray:
    prediction = np.zeros(len(scores), dtype=np.int64)
    difference = scores[:, 1] - scores[:, 0]
    prediction[np.argsort(difference, kind="stable")[len(scores) // 2 :]] = 1
    return prediction


def pair_accuracy(
    scores: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    classes: tuple[int, int],
) -> float:
    mask = np.isin(y[indices], classes)
    target = (y[indices][mask] == classes[1]).astype(np.int64)
    return float(np.mean(balanced_pair_prediction(scores[mask]) == target))


def calibration_indices(
    subject_idx: np.ndarray,
    records: list[dict],
    run_ids: tuple[int, ...],
) -> np.ndarray:
    run_set = set(run_ids)
    return np.asarray(
        [idx for idx in subject_idx if int(records[int(idx)]["run_id"]) in run_set],
        dtype=np.int64,
    )


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    subject_idx: np.ndarray,
    calibration_runs: tuple[int, ...],
    model: dict,
    alphas: list[float],
) -> dict:
    classes = model["classes"]
    candidates = []
    source_accuracies = []
    for validation_run in calibration_runs:
        val_idx = calibration_indices(subject_idx, records, (validation_run,))
        source_scores = pair_scores(x, val_idx, model)
        source_accuracies.append(pair_accuracy(source_scores, y, val_idx, classes))
    source_accuracy = float(np.mean(source_accuracies))
    for alpha in alphas:
        accuracies = []
        for validation_run in calibration_runs:
            fit_runs = tuple(run for run in calibration_runs if run != validation_run)
            fit_idx = calibration_indices(subject_idx, records, fit_runs)
            val_idx = calibration_indices(subject_idx, records, (validation_run,))
            target_means = pair_means(
                x, y, fit_idx, model["selected"], classes
            )
            means = (1.0 - alpha) * model["source_means"] + alpha * target_means
            scores = pair_scores(x, val_idx, model, means)
            accuracies.append(pair_accuracy(scores, y, val_idx, classes))
        candidates.append((float(np.mean(accuracies)), -alpha, alpha))
    selected_accuracy, _, selected_alpha = max(candidates)
    return {
        "alpha": float(selected_alpha),
        "source_accuracy": source_accuracy,
        "selected_accuracy": float(selected_accuracy),
        "gain": float(selected_accuracy - source_accuracy),
    }


def evaluate_scores(
    coarse_scores: np.ndarray,
    pair_score_by_name: dict[str, np.ndarray],
    coarse_weight: float,
    y: np.ndarray,
    eval_idx: np.ndarray,
    records: list[dict],
) -> dict[str, dict]:
    exact_scores = exact_scores_from_hierarchy(
        coarse_scores,
        pair_score_by_name["leg"],
        pair_score_by_name["arm"],
        coarse_weight,
    )
    predictions = {
        "balanced": apply_balanced_assignment(exact_scores, eval_idx, records),
        "independent": exact_scores.argmax(axis=1).astype(np.int64),
    }
    result = {}
    for rule, prediction in predictions.items():
        row = metrics(y[eval_idx], prediction)
        row["leg_pair_accuracy"] = pair_accuracy(
            pair_score_by_name["leg"], y, eval_idx, PAIR_CLASSES["leg"]
        )
        row["arm_pair_accuracy"] = pair_accuracy(
            pair_score_by_name["arm"], y, eval_idx, PAIR_CLASSES["arm"]
        )
        result[rule] = row
    return result


def average_metric_rows(rows: list[dict]) -> dict:
    keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "leg_pair_accuracy",
        "arm_pair_accuracy",
    )
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def append_aggregated_rows(
    rows: list[dict],
    grouped: dict[tuple, list[dict]],
    base: dict,
    combination_count: int,
) -> None:
    for (branch_mode, protocol, summary_alpha, rule), values in grouped.items():
        rows.append(
            {
                **base,
                "branch_mode": branch_mode,
                "protocol": protocol,
                "summary_alpha": summary_alpha,
                "prediction_rule": rule,
                "combination_count": combination_count,
                **average_metric_rows(values),
            }
        )


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["branch_mode"],
                row["protocol"],
                row["calibration_run_count"],
                row["summary_alpha"],
                row["prediction_rule"],
            )
        ].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        result.append(
            {
                "branch_mode": key[0],
                "protocol": key[1],
                "calibration_run_count": key[2],
                "alpha": key[3],
                "prediction_rule": key[4],
                "row_count": len(values),
                **average_metric_rows(values),
            }
        )
    return result


def paired_subject_bootstrap(
    rows: list[dict],
    excluded_subjects: set[str],
    iterations: int,
    seed: int,
) -> list[dict]:
    source_rows = [
        row
        for row in rows
        if row["protocol"] == "source_only" and row["prediction_rule"] == "balanced"
    ]
    source_by_subject = {
        subject: float(np.mean([row["balanced_accuracy"] for row in source_rows if row["subject"] == subject]))
        for subject in sorted({row["subject"] for row in source_rows})
    }
    candidates = [
        row
        for row in rows
        if row["prediction_rule"] == "balanced"
        and (
            row["protocol"] == "calibration_loo_alpha"
            or (row["protocol"] == "fixed_alpha" and row["summary_alpha"] == 0.1)
        )
    ]
    grouped: dict[tuple[str, str, float, int], list[dict]] = defaultdict(list)
    for row in candidates:
        grouped[
            (
                row["branch_mode"],
                row["protocol"],
                row["summary_alpha"],
                row["calibration_run_count"],
            )
        ].append(row)
    rng = np.random.default_rng(seed)
    output = []
    for (branch_mode, protocol, alpha, count), values in sorted(grouped.items()):
        candidate_by_subject = {
            subject: float(
                np.mean([row["balanced_accuracy"] for row in values if row["subject"] == subject])
            )
            for subject in sorted({row["subject"] for row in values})
        }
        for stratum, excluded in (("all", set()), ("qc", excluded_subjects)):
            subjects = np.asarray(
                sorted(set(source_by_subject) & set(candidate_by_subject) - excluded)
            )
            differences = np.asarray(
                [candidate_by_subject[subject] - source_by_subject[subject] for subject in subjects]
            )
            samples = rng.choice(differences, size=(iterations, len(differences)), replace=True).mean(axis=1)
            output.append(
                {
                    "branch_mode": branch_mode,
                    "protocol": protocol,
                    "alpha": alpha,
                    "calibration_run_count": count,
                    "stratum": stratum,
                    "subject_count": len(subjects),
                    "source_balanced_accuracy": float(
                        np.mean([source_by_subject[subject] for subject in subjects])
                    ),
                    "calibrated_balanced_accuracy": float(
                        np.mean([candidate_by_subject[subject] for subject in subjects])
                    ),
                    "mean_difference": float(np.mean(differences)),
                    "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
                    "subjects_improved": int(np.sum(differences > 0)),
                    "subjects_tied": int(np.sum(differences == 0)),
                    "subjects_worsened": int(np.sum(differences < 0)),
                }
            )
    return output


def bootstrap_difference(
    values: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> list[float]:
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def evaluate_calibration_gates(
    gate_rows: list[dict],
    excluded_subjects: set[str],
    iterations: int,
    seed: int,
) -> list[dict]:
    output = []
    rng = np.random.default_rng(seed)
    subjects = sorted({row["subject"] for row in gate_rows})
    for count in sorted({row["calibration_run_count"] for row in gate_rows}):
        count_rows = [row for row in gate_rows if row["calibration_run_count"] == count]
        policies: dict[str, list[dict]] = {"positive_internal_gain": []}
        for row in count_rows:
            selected = dict(row)
            selected["gate_threshold"] = 0.0
            selected["calibration_applied"] = row["calibration_loo_gain"] > 0.0
            selected["gated_accuracy"] = (
                row["calibrated_accuracy"]
                if selected["calibration_applied"]
                else row["source_accuracy"]
            )
            policies["positive_internal_gain"].append(selected)

        cross_fitted = []
        for subject in subjects:
            train_rows = [row for row in count_rows if row["subject"] != subject]
            target_rows = [row for row in count_rows if row["subject"] == subject]
            values = sorted({row["calibration_loo_gain"] for row in train_rows})
            thresholds = [float("-inf"), *values, float("inf")]
            candidates = []
            for threshold in thresholds:
                accuracies = [
                    row["calibrated_accuracy"]
                    if row["calibration_loo_gain"] > threshold
                    else row["source_accuracy"]
                    for row in train_rows
                ]
                application_rate = float(
                    np.mean([row["calibration_loo_gain"] > threshold for row in train_rows])
                )
                candidates.append((float(np.mean(accuracies)), -application_rate, threshold))
            threshold = float(max(candidates)[2])
            for row in target_rows:
                selected = dict(row)
                selected["gate_threshold"] = threshold
                selected["calibration_applied"] = row["calibration_loo_gain"] > threshold
                selected["gated_accuracy"] = (
                    row["calibrated_accuracy"]
                    if selected["calibration_applied"]
                    else row["source_accuracy"]
                )
                cross_fitted.append(selected)
        policies["cross_subject_threshold"] = cross_fitted

        for policy, values in policies.items():
            for stratum, excluded in (("all", set()), ("qc", excluded_subjects)):
                stratum_subjects = [subject for subject in subjects if subject not in excluded]
                subject_rows = {
                    subject: [row for row in values if row["subject"] == subject]
                    for subject in stratum_subjects
                }
                source = np.asarray(
                    [np.mean([row["source_accuracy"] for row in subject_rows[subject]]) for subject in stratum_subjects]
                )
                universal = np.asarray(
                    [np.mean([row["calibrated_accuracy"] for row in subject_rows[subject]]) for subject in stratum_subjects]
                )
                gated = np.asarray(
                    [np.mean([row["gated_accuracy"] for row in subject_rows[subject]]) for subject in stratum_subjects]
                )
                output.append(
                    {
                        "policy": policy,
                        "calibration_run_count": count,
                        "stratum": stratum,
                        "subject_count": len(stratum_subjects),
                        "source_accuracy": float(np.mean(source)),
                        "universal_calibration_accuracy": float(np.mean(universal)),
                        "gated_accuracy": float(np.mean(gated)),
                        "gated_minus_source": float(np.mean(gated - source)),
                        "gated_minus_source_ci95": bootstrap_difference(
                            gated - source, iterations, rng
                        ),
                        "gated_minus_universal": float(np.mean(gated - universal)),
                        "gated_minus_universal_ci95": bootstrap_difference(
                            gated - universal, iterations, rng
                        ),
                        "application_rate": float(
                            np.mean([row["calibration_applied"] for row in values if row["subject"] in stratum_subjects])
                        ),
                    }
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate leakage-safe labeled subject calibration in the exact-class hierarchy."
    )
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-calibration-runs", type=int, default=5)
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--branch-modes", nargs="*", default=["arm", "leg", "both"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    parser.add_argument("--qc-excluded-subjects", nargs="*", default=["sub-42", "sub-52"])
    parser.add_argument("--split-limit", type=int)
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline_json).read_text())
    checkpoint_dir = Path(baseline["checkpoint_dir"])
    window_name = baseline["window_name"]
    features, y, records = load_checkpoints(checkpoint_dir, [window_name])
    centered = center_by_subject_run(features[window_name], records)
    coarse_x, detrend_rows = temporal_detrend_by_subject_run(centered, records, degree=1)
    shape = tuple(int(value) for value in baseline["native_feature_shape"])
    pair_x, pair_shape = transform_scale(
        coarse_x, shape, baseline["pair_transform"], args.batch_size
    )
    split_by_name = {
        split["split"]: split
        for split in outer_splits(
            records,
            "subject",
            6,
            [int(seed) for seed in baseline["subject_seeds"]],
        )
    }
    hyperparameters = {row["split"]: row for row in baseline["hyperparameters"]}
    subjects_array = np.asarray([str(record["subject_id"]) for record in records])
    rows = []
    gate_rows = []

    split_names = sorted(hyperparameters)
    if args.split_limit is not None:
        split_names = split_names[: args.split_limit]
    for split_name in split_names:
        print(f"fitting {split_name}", flush=True)
        split = split_by_name[split_name]
        hyper = hyperparameters[split_name]
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        pair_models = {}
        for pair_name, classes in PAIR_CLASSES.items():
            configuration = hyper["selected_pair_configurations"][pair_name]
            if configuration["method"] != "full_lda":
                raise ValueError(
                    f"Calibration requires full_lda pair models, got {configuration['method']}."
                )
            pair_models[pair_name] = fit_source_pair_model(
                pair_x,
                y,
                train_idx,
                classes,
                int(configuration["feature_count"]),
                float(configuration["shrinkage"]),
            )
        split_coarse_scores, _ = selected_coarse_scores(
            coarse_x,
            y,
            train_idx,
            val_idx,
            int(hyper["selected_coarse_feature_count"]),
            hyper["selected_coarse_method"],
            float(hyper["selected_coarse_lda_shrinkage"]),
        )
        val_position = {int(index): position for position, index in enumerate(val_idx)}

        for subject in sorted(set(subjects_array[val_idx].tolist())):
            subject_idx = np.flatnonzero(subjects_array == subject).astype(np.int64)
            run_ids = sorted({int(records[int(idx)]["run_id"]) for idx in subject_idx})
            for holdout_run in run_ids:
                eval_idx = calibration_indices(subject_idx, records, (holdout_run,))
                coarse_scores = split_coarse_scores[
                    np.asarray([val_position[int(index)] for index in eval_idx], dtype=np.int64)
                ]
                source_pair_scores = {
                    pair_name: pair_scores(pair_x, eval_idx, model)
                    for pair_name, model in pair_models.items()
                }
                source_metrics = evaluate_scores(
                    coarse_scores,
                    source_pair_scores,
                    float(hyper["selected_coarse_weight"]),
                    y,
                    eval_idx,
                    records,
                )
                base = {
                    "split": split_name,
                    "subject_seed": int(split["subject_seed"]),
                    "subject": subject,
                    "holdout_run": holdout_run,
                }
                for rule, metric_row in source_metrics.items():
                    rows.append(
                        {
                            **base,
                            "branch_mode": "none",
                            "protocol": "source_only",
                            "calibration_run_count": 0,
                            "summary_alpha": 0.0,
                            "prediction_rule": rule,
                            "combination_count": 1,
                            **average_metric_rows([metric_row]),
                        }
                    )

                available_runs = [run for run in run_ids if run != holdout_run]
                max_count = min(args.max_calibration_runs, len(available_runs))
                for count in range(1, max_count + 1):
                    combinations = list(itertools.combinations(available_runs, count))
                    grouped: dict[tuple, list[dict]] = defaultdict(list)
                    for calibration_runs in combinations:
                        calibration_idx = calibration_indices(
                            subject_idx, records, calibration_runs
                        )
                        target_means = {
                            pair_name: pair_means(
                                pair_x,
                                y,
                                calibration_idx,
                                model["selected"],
                                model["classes"],
                            )
                            for pair_name, model in pair_models.items()
                        }
                        selected_alphas = {}
                        if count >= 2:
                            selected_alphas = {
                                pair_name: select_alpha(
                                    pair_x,
                                    y,
                                    records,
                                    subject_idx,
                                    calibration_runs,
                                    model,
                                    args.alphas,
                                )
                                for pair_name, model in pair_models.items()
                            }
                            arm_alpha = selected_alphas["arm"]["alpha"]
                            arm_means = (
                                (1.0 - arm_alpha) * pair_models["arm"]["source_means"]
                                + arm_alpha * target_means["arm"]
                            )
                            gate_pair_scores = {
                                "leg": source_pair_scores["leg"],
                                "arm": pair_scores(
                                    pair_x, eval_idx, pair_models["arm"], arm_means
                                ),
                            }
                            gate_metrics = evaluate_scores(
                                coarse_scores,
                                gate_pair_scores,
                                float(hyper["selected_coarse_weight"]),
                                y,
                                eval_idx,
                                records,
                            )["balanced"]
                            gate_rows.append(
                                {
                                    **base,
                                    "calibration_runs": list(calibration_runs),
                                    "calibration_run_count": count,
                                    "selected_alpha": arm_alpha,
                                    "calibration_source_arm_accuracy": selected_alphas[
                                        "arm"
                                    ]["source_accuracy"],
                                    "calibration_selected_arm_accuracy": selected_alphas[
                                        "arm"
                                    ]["selected_accuracy"],
                                    "calibration_loo_gain": selected_alphas["arm"]["gain"],
                                    "source_accuracy": source_metrics["balanced"][
                                        "balanced_accuracy"
                                    ],
                                    "calibrated_accuracy": gate_metrics[
                                        "balanced_accuracy"
                                    ],
                                }
                            )
                        protocols = [
                            ("fixed_alpha", float(alpha), {name: float(alpha) for name in pair_models})
                            for alpha in args.alphas
                        ]
                        if count >= 2:
                            protocols.append(
                                (
                                    "calibration_loo_alpha",
                                    -1.0,
                                    {
                                        pair_name: diagnostics["alpha"]
                                        for pair_name, diagnostics in selected_alphas.items()
                                    },
                                )
                            )
                        for protocol, summary_alpha, alpha_by_pair in protocols:
                            for branch_mode in args.branch_modes:
                                active = set(PAIR_CLASSES) if branch_mode == "both" else {branch_mode}
                                calibrated_pair_scores = {}
                                for pair_name, model in pair_models.items():
                                    alpha = alpha_by_pair[pair_name] if pair_name in active else 0.0
                                    means = (
                                        (1.0 - alpha) * model["source_means"]
                                        + alpha * target_means[pair_name]
                                    )
                                    calibrated_pair_scores[pair_name] = pair_scores(
                                        pair_x, eval_idx, model, means
                                    )
                                evaluated = evaluate_scores(
                                    coarse_scores,
                                    calibrated_pair_scores,
                                    float(hyper["selected_coarse_weight"]),
                                    y,
                                    eval_idx,
                                    records,
                                )
                                for rule, metric_row in evaluated.items():
                                    grouped[(branch_mode, protocol, summary_alpha, rule)].append(
                                        metric_row
                                    )
                    append_aggregated_rows(
                        rows,
                        grouped,
                        {
                            **base,
                            "calibration_run_count": count,
                        },
                        len(combinations),
                    )

    summary = summarize(rows)
    source_balanced = next(
        row
        for row in summary
        if row["protocol"] == "source_only" and row["prediction_rule"] == "balanced"
    )
    baseline_balanced = next(
        row
        for row in baseline["summary"]
        if row["prediction_rule"] == "hybrid_fused_balanced"
    )
    source_rows = [
        row
        for row in rows
        if row["protocol"] == "source_only" and row["prediction_rule"] == "balanced"
    ]
    source_by_split: dict[str, list[float]] = defaultdict(list)
    for row in source_rows:
        source_by_split[row["split"]].append(row["balanced_accuracy"])
    fold_weighted_source = float(
        np.mean([np.mean(values) for values in source_by_split.values()])
    )
    baseline_reproduction_difference = (
        fold_weighted_source - baseline_balanced["mean_balanced_accuracy"]
    )
    result = {
        "baseline_json": args.baseline_json,
        "checkpoint_dir": str(checkpoint_dir),
        "window_name": window_name,
        "pair_transform": baseline["pair_transform"],
        "pair_shape": pair_shape,
        "alphas": args.alphas,
        "branch_modes": args.branch_modes,
        "mean_temporal_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in detrend_rows])
        ),
        "baseline_reproduction": {
            "reconstructed_fold_weighted_balanced_accuracy": fold_weighted_source,
            "reconstructed_subject_weighted_balanced_accuracy": source_balanced[
                "balanced_accuracy"
            ],
            "saved_baseline_balanced_accuracy": baseline_balanced["mean_balanced_accuracy"],
            "difference": baseline_reproduction_difference,
            "matches": bool(abs(baseline_reproduction_difference) < 1e-12),
        },
        "summary": summary,
        "paired_subject_bootstrap": paired_subject_bootstrap(
            rows,
            set(args.qc_excluded_subjects),
            args.bootstrap_iterations,
            args.bootstrap_seed,
        ),
        "calibration_gate_summary": evaluate_calibration_gates(
            gate_rows,
            set(args.qc_excluded_subjects),
            args.bootstrap_iterations,
            args.bootstrap_seed + 1,
        ),
        "calibration_gate_rows": gate_rows,
        "rows": rows,
        "note": (
            "Every evaluation run is excluded from target-subject calibration and alpha selection. "
            "Outer-fold feature rankings, covariance estimates, coarse models, and hierarchy weights "
            "use source subjects only. Calibration updates only selected pair-class means."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "baseline_reproduction": result["baseline_reproduction"],
                "calibration_gate_summary": result["calibration_gate_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
