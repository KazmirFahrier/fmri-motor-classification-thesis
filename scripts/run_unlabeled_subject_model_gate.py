#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    apply_balanced_assignment,
    center_by_subject_run,
    centroid_matrix,
    l2_normalize,
    metrics,
    score_with_centroids,
)
from run_clip_offset_event_sweep import coarse_metrics
from run_detrended_hierarchy_sweep import exact_scores_from_hierarchy
from run_detrended_pair_feature_selection import (
    choose_coarse_weight,
    choose_feature_counts,
    coarse_score_matrix,
    inner_splits,
    load_checkpoints,
    outer_splits,
    selected_pair_scores,
)
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


DIAGNOSTIC_NAMES = [
    "flat_mean_margin",
    "pair_mean_margin",
    "pair_minus_flat_margin",
    "flat_mean_entropy",
    "pair_mean_entropy",
    "independent_agreement",
    "balanced_agreement",
    "coarse_agreement",
    "flat_balance_penalty",
    "pair_balance_penalty",
    "flat_mean_run_imbalance",
    "pair_mean_run_imbalance",
    "flat_pseudo_template_consistency",
    "pair_pseudo_template_consistency",
    "pair_minus_flat_template_consistency",
]


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def model_scores_and_predictions(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    selected_counts: dict[str, int],
    coarse_weight: float,
) -> dict:
    flat_centroids = centroid_matrix(x[train_idx], y[train_idx])
    flat_scores = score_with_centroids(x[val_idx], flat_centroids)
    pair_scores, _ = selected_pair_scores(x, y, train_idx, val_idx, selected_counts)
    coarse_scores = coarse_score_matrix(x, y, train_idx, val_idx)
    fused_scores = exact_scores_from_hierarchy(
        coarse_scores,
        pair_scores["leg"],
        pair_scores["arm"],
        coarse_weight,
    )
    return {
        "flat_scores": flat_scores,
        "pair_scores": fused_scores,
        "flat_balanced": apply_balanced_assignment(flat_scores, val_idx, records),
        "pair_balanced": apply_balanced_assignment(fused_scores, val_idx, records),
    }


def normalized_entropy(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1) / np.log(
        scores.shape[1]
    )


def mean_margin(scores: np.ndarray) -> float:
    ordered = np.sort(scores, axis=1)
    return float(np.mean(ordered[:, -1] - ordered[:, -2]))


def run_groups(indices: np.ndarray, records: list[dict]) -> list[np.ndarray]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for local_pos, record_idx in enumerate(indices):
        grouped[int(records[int(record_idx)]["run_id"])].append(local_pos)
    return [np.asarray(grouped[run_id], dtype=np.int64) for run_id in sorted(grouped)]


def balance_penalty(scores: np.ndarray, balanced_pred: np.ndarray, groups: list[np.ndarray]) -> float:
    penalties = []
    for positions in groups:
        group_scores = scores[positions]
        independent = group_scores.argmax(axis=1)
        row_idx = np.arange(len(positions))
        penalties.append(
            float(
                np.mean(
                    group_scores[row_idx, independent]
                    - group_scores[row_idx, balanced_pred[positions]]
                )
            )
        )
    return float(np.mean(penalties))


def mean_run_imbalance(scores: np.ndarray, groups: list[np.ndarray]) -> float:
    values = []
    for positions in groups:
        pred = scores[positions].argmax(axis=1)
        expected = len(positions) // scores.shape[1]
        counts = np.bincount(pred, minlength=scores.shape[1])
        values.append(float(np.sum(np.abs(counts - expected))))
    return float(np.mean(values))


def pseudo_template_consistency(
    x_subject: np.ndarray,
    pred: np.ndarray,
    groups: list[np.ndarray],
) -> float:
    centroids_by_class: dict[int, list[np.ndarray]] = defaultdict(list)
    for positions in groups:
        for class_idx in range(4):
            class_positions = positions[pred[positions] == class_idx]
            if len(class_positions):
                centroid = x_subject[class_positions].mean(axis=0, keepdims=True)
                centroids_by_class[class_idx].append(l2_normalize(centroid)[0])
    similarities = []
    for centroids in centroids_by_class.values():
        for first_idx in range(len(centroids)):
            for second_idx in range(first_idx + 1, len(centroids)):
                similarities.append(float(centroids[first_idx] @ centroids[second_idx]))
    return float(np.mean(similarities)) if similarities else 0.0


def subject_diagnostics(
    x: np.ndarray,
    subject_idx: np.ndarray,
    records: list[dict],
    flat_scores: np.ndarray,
    pair_scores: np.ndarray,
    flat_balanced: np.ndarray,
    pair_balanced: np.ndarray,
) -> dict[str, float]:
    groups = run_groups(subject_idx, records)
    flat_independent = flat_scores.argmax(axis=1)
    pair_independent = pair_scores.argmax(axis=1)
    flat_margin = mean_margin(flat_scores)
    pair_margin = mean_margin(pair_scores)
    flat_consistency = pseudo_template_consistency(
        x[subject_idx],
        flat_balanced,
        groups,
    )
    pair_consistency = pseudo_template_consistency(
        x[subject_idx],
        pair_balanced,
        groups,
    )
    return {
        "flat_mean_margin": flat_margin,
        "pair_mean_margin": pair_margin,
        "pair_minus_flat_margin": pair_margin - flat_margin,
        "flat_mean_entropy": float(np.mean(normalized_entropy(flat_scores))),
        "pair_mean_entropy": float(np.mean(normalized_entropy(pair_scores))),
        "independent_agreement": float(np.mean(flat_independent == pair_independent)),
        "balanced_agreement": float(np.mean(flat_balanced == pair_balanced)),
        "coarse_agreement": float(np.mean((flat_independent >= 2) == (pair_independent >= 2))),
        "flat_balance_penalty": balance_penalty(flat_scores, flat_balanced, groups),
        "pair_balance_penalty": balance_penalty(pair_scores, pair_balanced, groups),
        "flat_mean_run_imbalance": mean_run_imbalance(flat_scores, groups),
        "pair_mean_run_imbalance": mean_run_imbalance(pair_scores, groups),
        "flat_pseudo_template_consistency": flat_consistency,
        "pair_pseudo_template_consistency": pair_consistency,
        "pair_minus_flat_template_consistency": pair_consistency - flat_consistency,
    }


def subject_rows(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    val_idx: np.ndarray,
    model_result: dict,
) -> list[dict]:
    val_subjects = np.asarray([str(records[int(idx)]["subject_id"]) for idx in val_idx])
    rows = []
    for subject in sorted(set(val_subjects.tolist())):
        mask = val_subjects == subject
        subject_idx = val_idx[mask]
        flat_pred = model_result["flat_balanced"][mask]
        pair_pred = model_result["pair_balanced"][mask]
        flat_accuracy = metrics(y[subject_idx], flat_pred)["balanced_accuracy"]
        pair_accuracy = metrics(y[subject_idx], pair_pred)["balanced_accuracy"]
        diagnostics = subject_diagnostics(
            x,
            subject_idx,
            records,
            model_result["flat_scores"][mask],
            model_result["pair_scores"][mask],
            flat_pred,
            pair_pred,
        )
        rows.append(
            {
                "subject": subject,
                "indices": subject_idx,
                "diagnostics": diagnostics,
                "flat_balanced_accuracy": flat_accuracy,
                "pair_balanced_accuracy": pair_accuracy,
                "pair_minus_flat_accuracy": pair_accuracy - flat_accuracy,
                "flat_pred": flat_pred,
                "pair_pred": pair_pred,
            }
        )
    return rows


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), standardized])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y)
    return {
        "mean": mean,
        "scale": scale,
        "coefficients": coefficients,
        "alpha": float(alpha),
    }


def ridge_predict(model: dict, x: np.ndarray) -> np.ndarray:
    standardized = (x - model["mean"]) / model["scale"]
    design = np.column_stack([np.ones(len(x)), standardized])
    return design @ model["coefficients"]


def select_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    alphas: list[float],
) -> tuple[float, np.ndarray, list[dict]]:
    rows = []
    predictions_by_alpha = {}
    for alpha in alphas:
        predictions = np.zeros(len(y), dtype=np.float64)
        for held_out in range(len(y)):
            train_mask = np.arange(len(y)) != held_out
            model = fit_ridge(x[train_mask], y[train_mask], alpha)
            predictions[held_out] = ridge_predict(model, x[held_out : held_out + 1])[0]
        mse = float(np.mean((predictions - y) ** 2))
        rows.append({"alpha": float(alpha), "loo_mse": mse})
        predictions_by_alpha[float(alpha)] = predictions
    selected_alpha = min(alphas, key=lambda alpha: (next(row["loo_mse"] for row in rows if row["alpha"] == alpha), -alpha))
    return float(selected_alpha), predictions_by_alpha[float(selected_alpha)], rows


def select_gate_threshold(
    predicted_delta: np.ndarray,
    flat_accuracy: np.ndarray,
    pair_accuracy: np.ndarray,
    thresholds: list[float],
) -> tuple[float, list[dict]]:
    rows = []
    for threshold in thresholds:
        choose_pair = predicted_delta > threshold
        selected_accuracy = np.where(choose_pair, pair_accuracy, flat_accuracy)
        rows.append(
            {
                "threshold": float(threshold),
                "mean_cross_fitted_selected_accuracy": float(np.mean(selected_accuracy)),
                "pair_choice_fraction": float(np.mean(choose_pair)),
            }
        )
    selected = min(
        thresholds,
        key=lambda threshold: (
            -next(
                row["mean_cross_fitted_selected_accuracy"]
                for row in rows
                if row["threshold"] == threshold
            ),
            abs(threshold),
            -threshold,
        ),
    )
    return float(selected), rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["prediction_rule"]].append(row)
    summary = []
    for rule, group in grouped.items():
        summary.append(
            {
                "prediction_rule": rule,
                "split_count": len(group),
                "mean_accuracy": float(np.mean([row["metrics"]["accuracy"] for row in group])),
                "mean_balanced_accuracy": float(
                    np.mean([row["metrics"]["balanced_accuracy"] for row in group])
                ),
                "mean_macro_f1": float(np.mean([row["metrics"]["macro_f1"] for row in group])),
                "mean_leg_vs_arm_accuracy": float(
                    np.mean([row["coarse_metrics"]["leg_vs_arm_accuracy"] for row in group])
                ),
            }
        )
    return sorted(summary, key=lambda row: -row["mean_balanced_accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn a cross-fitted unlabeled subject gate between flat and pair-selected models."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--window-name", default="offset_3_length_8")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--subject-fold-count", type=int, default=6)
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument("--subject-seeds", nargs="*", type=int, default=[11, 23, 37, 53, 71])
    parser.add_argument(
        "--feature-counts",
        nargs="*",
        type=int,
        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 13824],
    )
    parser.add_argument(
        "--coarse-weights",
        nargs="*",
        type=float,
        default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    )
    parser.add_argument("--ridge-alphas", nargs="*", type=float, default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument(
        "--gate-thresholds",
        nargs="*",
        type=float,
        default=[-0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1],
    )
    args = parser.parse_args()

    features, y, records = load_checkpoints(Path(args.checkpoint_dir), [args.window_name])
    centered = center_by_subject_run(features[args.window_name], records)
    x, group_rows = temporal_detrend_by_subject_run(centered, records, degree=1)
    splits = outer_splits(records, "subject", args.subject_fold_count, args.subject_seeds)

    result_rows = []
    gate_rows = []
    for split in splits:
        print(f'evaluating {split["split"]}', flush=True)
        inner = inner_splits(
            records,
            split["train_idx"],
            "subject",
            args.inner_subject_fold_count,
        )
        selected_counts, count_diagnostics = choose_feature_counts(
            x,
            y,
            inner,
            sorted(set(args.feature_counts)),
        )
        selected_weight, weight_diagnostics = choose_coarse_weight(
            x,
            y,
            records,
            inner,
            selected_counts,
            sorted(set(args.coarse_weights)),
        )

        meta_rows = []
        for inner_split in inner:
            inner_result = model_scores_and_predictions(
                x,
                y,
                records,
                inner_split["train_idx"],
                inner_split["val_idx"],
                selected_counts,
                selected_weight,
            )
            meta_rows.extend(
                subject_rows(x, y, records, inner_split["val_idx"], inner_result)
            )

        meta_x = np.asarray(
            [[row["diagnostics"][name] for name in DIAGNOSTIC_NAMES] for row in meta_rows],
            dtype=np.float64,
        )
        meta_delta = np.asarray(
            [row["pair_minus_flat_accuracy"] for row in meta_rows],
            dtype=np.float64,
        )
        meta_flat = np.asarray([row["flat_balanced_accuracy"] for row in meta_rows])
        meta_pair = np.asarray([row["pair_balanced_accuracy"] for row in meta_rows])
        selected_alpha, loo_predicted_delta, alpha_diagnostics = select_ridge_alpha(
            meta_x,
            meta_delta,
            sorted(set(args.ridge_alphas)),
        )
        selected_threshold, threshold_diagnostics = select_gate_threshold(
            loo_predicted_delta,
            meta_flat,
            meta_pair,
            sorted(set(args.gate_thresholds)),
        )
        gate_model = fit_ridge(meta_x, meta_delta, selected_alpha)

        outer_result = model_scores_and_predictions(
            x,
            y,
            records,
            split["train_idx"],
            split["val_idx"],
            selected_counts,
            selected_weight,
        )
        outer_subject_rows = subject_rows(x, y, records, split["val_idx"], outer_result)
        outer_x = np.asarray(
            [
                [row["diagnostics"][name] for name in DIAGNOSTIC_NAMES]
                for row in outer_subject_rows
            ],
            dtype=np.float64,
        )
        predicted_delta = ridge_predict(gate_model, outer_x)
        zero_gate_pred = outer_result["flat_balanced"].copy()
        selected_gate_pred = outer_result["flat_balanced"].copy()
        oracle_pred = outer_result["flat_balanced"].copy()
        val_subjects = np.asarray([str(records[int(idx)]["subject_id"]) for idx in split["val_idx"]])
        subject_gate_rows = []
        for row, prediction in zip(outer_subject_rows, predicted_delta):
            mask = val_subjects == row["subject"]
            zero_choose_pair = bool(prediction > 0.0)
            selected_choose_pair = bool(prediction > selected_threshold)
            oracle_choose_pair = bool(row["pair_minus_flat_accuracy"] > 0.0)
            if zero_choose_pair:
                zero_gate_pred[mask] = row["pair_pred"]
            if selected_choose_pair:
                selected_gate_pred[mask] = row["pair_pred"]
            if oracle_choose_pair:
                oracle_pred[mask] = row["pair_pred"]
            subject_gate_rows.append(
                {
                    "subject": row["subject"],
                    "diagnostics": row["diagnostics"],
                    "predicted_pair_minus_flat_accuracy": float(prediction),
                    "actual_pair_minus_flat_accuracy": row["pair_minus_flat_accuracy"],
                    "flat_balanced_accuracy": row["flat_balanced_accuracy"],
                    "pair_balanced_accuracy": row["pair_balanced_accuracy"],
                    "zero_gate_choose_pair": zero_choose_pair,
                    "selected_gate_choose_pair": selected_choose_pair,
                    "oracle_choose_pair": oracle_choose_pair,
                }
            )

        predictions = [
            ("flat_all_features_balanced", outer_result["flat_balanced"]),
            ("selected_pair_fused_balanced", outer_result["pair_balanced"]),
            ("ridge_gate_fixed_zero", zero_gate_pred),
            ("ridge_gate_selected_threshold", selected_gate_pred),
            ("oracle_subject_choice", oracle_pred),
        ]
        for rule, pred in predictions:
            result_rows.append(
                {
                    "split": split["split"],
                    "prediction_rule": rule,
                    "metrics": metrics(y[split["val_idx"]], pred),
                    "coarse_metrics": coarse_metrics(y[split["val_idx"]], pred),
                }
            )
        gate_rows.append(
            {
                "split": split["split"],
                "selected_feature_counts": selected_counts,
                "selected_coarse_weight": selected_weight,
                "selected_ridge_alpha": selected_alpha,
                "selected_gate_threshold": selected_threshold,
                "alpha_diagnostics": alpha_diagnostics,
                "threshold_diagnostics": threshold_diagnostics,
                "feature_count_diagnostics": count_diagnostics,
                "coarse_weight_diagnostics": weight_diagnostics,
                "meta_subject_count": len(meta_rows),
                "meta_pair_gain_correlation": float(
                    np.corrcoef(loo_predicted_delta, meta_delta)[0, 1]
                ),
                "subject_rows": subject_gate_rows,
                "ridge_coefficients": {
                    "intercept": float(gate_model["coefficients"][0]),
                    "standardized_features": {
                        name: float(coefficient)
                        for name, coefficient in zip(
                            DIAGNOSTIC_NAMES,
                            gate_model["coefficients"][1:],
                        )
                    },
                },
            }
        )

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "window_name": args.window_name,
        "subject_seeds": args.subject_seeds,
        "diagnostic_names": DIAGNOSTIC_NAMES,
        "mean_linear_time_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in group_rows])
        ),
        "rows": result_rows,
        "gate_rows": gate_rows,
        "summary": summarize(result_rows),
        "note": (
            "Gate diagnostics use no target labels. Ridge alpha and gate threshold are selected from "
            "cross-fitted subjects inside each outer training cohort. Oracle subject choice uses "
            "held-out labels and is an upper bound only."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(json.dumps({"out_json": args.out_json, "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
