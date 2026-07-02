#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    apply_balanced_assignment,
    balanced_assign_group,
    center_by_subject_run,
    metrics,
)
from run_clip_offset_event_sweep import coarse_metrics
from run_detrended_hierarchy_sweep import exact_scores_from_hierarchy
from run_detrended_pair_feature_selection import (
    PAIR_CLASSES,
    as_jsonable,
    choose_feature_counts,
    coarse_score_matrix,
    inner_splits,
    load_checkpoints,
    outer_splits,
    pair_accuracy,
    pair_score_matrix,
    rank_pair_features,
    selected_pair_scores,
)
from run_spatial_scale_feature_sweep import (
    feature_counts_for_dimension,
    transform_scale,
)
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


def diagonal_lda_scores(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    selected_features: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    train = x[train_idx][:, selected_features].astype(np.float64)
    val = x[val_idx][:, selected_features].astype(np.float64)
    target = (y[train_idx] >= 2).astype(np.int64)
    means = np.stack([train[target == class_id].mean(axis=0) for class_id in range(2)])
    pooled_variance = 0.5 * (
        train[target == 0].var(axis=0) + train[target == 1].var(axis=0)
    )
    positive = pooled_variance[pooled_variance > 0]
    variance_target = float(np.median(positive)) if len(positive) else 1.0
    regularized = (1.0 - shrinkage) * pooled_variance + shrinkage * variance_target
    variance_floor = max(variance_target * 1e-3, 1e-12)
    inverse_variance = 1.0 / np.maximum(regularized, variance_floor)
    linear = means * inverse_variance[None, :]
    intercept = -0.5 * np.sum(means * linear, axis=1)
    return val @ linear.T + intercept[None, :]


def full_lda_scores(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    selected_features: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    train = x[train_idx][:, selected_features].astype(np.float64)
    val = x[val_idx][:, selected_features].astype(np.float64)
    target = (y[train_idx] >= 2).astype(np.int64)
    means = np.stack([train[target == class_id].mean(axis=0) for class_id in range(2)])
    residuals = np.concatenate(
        [train[target == class_id] - means[class_id] for class_id in range(2)],
        axis=0,
    )
    covariance = residuals.T @ residuals / max(len(residuals) - 2, 1)
    diagonal = np.diag(covariance)
    positive = diagonal[diagonal > 0]
    variance_target = float(np.median(positive)) if len(positive) else 1.0
    regularized = (1.0 - shrinkage) * covariance
    regularized.flat[:: regularized.shape[0] + 1] += shrinkage * variance_target
    regularized.flat[:: regularized.shape[0] + 1] += max(
        variance_target * 1e-5, 1e-12
    )
    # The covariance is symmetric positive definite after shrinkage.  Using its
    # eigensystem avoids a pathological macOS Accelerate general-solve stall.
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    eigenvalue_floor = max(variance_target * 1e-8, 1e-15)
    inverse_projection = (eigenvectors.T @ means.T) / np.maximum(
        eigenvalues[:, None], eigenvalue_floor
    )
    linear = (eigenvectors @ inverse_projection).T
    intercept = -0.5 * np.sum(means * linear, axis=1)
    return val @ linear.T + intercept[None, :]


def coarse_scores_for_configuration(
    coarse_x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    selected_features: np.ndarray,
    method: str,
    shrinkage: float,
) -> np.ndarray:
    if method == "cosine":
        coarse_y = (y >= 2).astype(np.int64)
        return pair_score_matrix(
            coarse_x,
            coarse_y,
            train_idx,
            val_idx,
            (0, 1),
            selected_features,
        )
    if method == "diagonal_lda":
        return diagonal_lda_scores(
            coarse_x,
            y,
            train_idx,
            val_idx,
            selected_features,
            shrinkage,
        )
    if method == "full_lda":
        return full_lda_scores(
            coarse_x,
            y,
            train_idx,
            val_idx,
            selected_features,
            shrinkage,
        )
    raise ValueError(f"Unknown coarse method: {method}")


def choose_coarse_configuration(
    coarse_x: np.ndarray,
    y: np.ndarray,
    inner: list[dict],
    feature_counts: list[int],
    methods: list[str],
    lda_shrinkages: list[float],
    full_lda_max_features: int,
) -> tuple[dict, list[dict]]:
    coarse_y = (y >= 2).astype(np.int64)
    rows = []
    for split in inner:
        ranking = rank_pair_features(
            coarse_x, coarse_y, split["train_idx"], (0, 1)
        )
        for feature_count in feature_counts:
            count = min(feature_count, coarse_x.shape[1])
            for method in methods:
                if method == "full_lda" and count > full_lda_max_features:
                    continue
                shrinkages = (
                    lda_shrinkages
                    if method in {"diagonal_lda", "full_lda"}
                    else [0.0]
                )
                for shrinkage in shrinkages:
                    scores = coarse_scores_for_configuration(
                        coarse_x,
                        y,
                        split["train_idx"],
                        split["val_idx"],
                        ranking[:count],
                        method,
                        shrinkage,
                    )
                    rows.append(
                        {
                            "split": split["split"],
                            "feature_count": count,
                            "method": method,
                            "shrinkage": shrinkage,
                            "accuracy": pair_accuracy(
                                scores, coarse_y, split["val_idx"], (0, 1)
                            ),
                        }
                    )
    configurations = sorted(
        {
            (row["feature_count"], row["method"], row["shrinkage"])
            for row in rows
        }
    )
    diagnostics = []
    for count, method, shrinkage in configurations:
        accuracy = float(
            np.mean(
                [
                    row["accuracy"]
                    for row in rows
                    if row["feature_count"] == count
                    and row["method"] == method
                    and row["shrinkage"] == shrinkage
                ]
            )
        )
        diagnostics.append(
            {
                "feature_count": count,
                "method": method,
                "shrinkage": shrinkage,
                "mean_inner_accuracy": accuracy,
            }
        )
    selected = min(
        diagnostics,
        key=lambda row: (
            -row["mean_inner_accuracy"],
            row["feature_count"],
            row["method"],
            row["shrinkage"],
        ),
    )
    return {
        "feature_count": int(selected["feature_count"]),
        "method": selected["method"],
        "shrinkage": float(selected["shrinkage"]),
    }, diagnostics


def selected_coarse_scores(
    coarse_x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_count: int,
    method: str = "cosine",
    shrinkage: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    coarse_y = (y >= 2).astype(np.int64)
    ranking = rank_pair_features(coarse_x, coarse_y, train_idx, (0, 1))
    selected = ranking[: min(feature_count, coarse_x.shape[1])]
    scores = coarse_scores_for_configuration(
        coarse_x,
        y,
        train_idx,
        val_idx,
        selected,
        method,
        shrinkage,
    )
    return scores, selected


def choose_hybrid_coarse_weight(
    pair_x: np.ndarray,
    coarse_x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    inner: list[dict],
    selected_counts: dict[str, int],
    selected_coarse_configuration: dict,
    coarse_weights: list[float],
) -> tuple[float, list[dict]]:
    rows = []
    for split in inner:
        pair_scores, _ = selected_pair_scores(
            pair_x,
            y,
            split["train_idx"],
            split["val_idx"],
            selected_counts,
        )
        coarse_scores, _ = selected_coarse_scores(
            coarse_x,
            y,
            split["train_idx"],
            split["val_idx"],
            selected_coarse_configuration["feature_count"],
            selected_coarse_configuration["method"],
            selected_coarse_configuration["shrinkage"],
        )
        for weight in coarse_weights:
            scores = exact_scores_from_hierarchy(
                coarse_scores,
                pair_scores["leg"],
                pair_scores["arm"],
                weight,
            )
            prediction = apply_balanced_assignment(
                scores, split["val_idx"], records
            )
            rows.append(
                {
                    "split": split["split"],
                    "coarse_weight": weight,
                    "balanced_accuracy": metrics(
                        y[split["val_idx"]], prediction
                    )["balanced_accuracy"],
                }
            )
    means = {
        weight: float(
            np.mean(
                [row["balanced_accuracy"] for row in rows if row["coarse_weight"] == weight]
            )
        )
        for weight in coarse_weights
    }
    selected = min(means, key=lambda weight: (-means[weight], weight))
    diagnostics = [
        {"coarse_weight": weight, "mean_inner_balanced_accuracy": means[weight]}
        for weight in coarse_weights
    ]
    return float(selected), diagnostics


def hierarchical_balanced_prediction(
    coarse_scores: np.ndarray,
    pair_scores: dict[str, np.ndarray],
    val_idx: np.ndarray,
    records: list[dict],
    true_coarse: np.ndarray | None = None,
) -> np.ndarray:
    prediction = np.empty(len(val_idx), dtype=np.int64)
    groups: dict[str, list[int]] = {}
    for local_position, record_index in enumerate(val_idx):
        record = records[int(record_index)]
        key = f'{record["subject_id"]}|run-{int(record["run_id"])}'
        groups.setdefault(key, []).append(local_position)
    for positions in groups.values():
        positions = sorted(
            positions,
            key=lambda position: records[int(val_idx[position])]["event_start"],
        )
        position_array = np.asarray(positions, dtype=np.int64)
        if true_coarse is None:
            coarse_assignment = balanced_assign_group(coarse_scores[position_array])
        else:
            coarse_assignment = true_coarse[position_array]
        for coarse_class, pair_name, class_offset in (
            (0, "leg", 0),
            (1, "arm", 2),
        ):
            pair_positions = position_array[coarse_assignment == coarse_class]
            local_assignment = balanced_assign_group(pair_scores[pair_name][pair_positions])
            prediction[pair_positions] = local_assignment + class_offset
    return prediction


def evaluate_split(
    pair_x: np.ndarray,
    coarse_x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    split: dict,
    feature_counts: list[int],
    coarse_feature_counts: list[int],
    coarse_methods: list[str],
    lda_shrinkages: list[float],
    full_lda_max_features: int,
    coarse_weights: list[float],
    inner_subject_fold_count: int,
) -> tuple[list[dict], dict]:
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    inner = inner_splits(
        records, train_idx, split["family"], inner_subject_fold_count
    )
    selected_counts, count_diagnostics = choose_feature_counts(
        pair_x, y, inner, feature_counts
    )
    selected_coarse_configuration, coarse_count_diagnostics = choose_coarse_configuration(
        coarse_x,
        y,
        inner,
        coarse_feature_counts,
        coarse_methods,
        lda_shrinkages,
        full_lda_max_features,
    )
    selected_weight, weight_diagnostics = choose_hybrid_coarse_weight(
        pair_x,
        coarse_x,
        y,
        records,
        inner,
        selected_counts,
        selected_coarse_configuration,
        coarse_weights,
    )
    pair_scores, rankings = selected_pair_scores(
        pair_x, y, train_idx, val_idx, selected_counts
    )
    coarse_scores, coarse_ranking = selected_coarse_scores(
        coarse_x,
        y,
        train_idx,
        val_idx,
        selected_coarse_configuration["feature_count"],
        selected_coarse_configuration["method"],
        selected_coarse_configuration["shrinkage"],
    )
    fused_scores = exact_scores_from_hierarchy(
        coarse_scores,
        pair_scores["leg"],
        pair_scores["arm"],
        selected_weight,
    )
    prediction = apply_balanced_assignment(fused_scores, val_idx, records)
    independent = fused_scores.argmax(axis=1).astype(np.int64)
    true_coarse = (y[val_idx] >= 2).astype(np.int64)
    leg_prediction = pair_scores["leg"].argmax(axis=1).astype(np.int64)
    arm_prediction = pair_scores["arm"].argmax(axis=1).astype(np.int64) + 2
    oracle = np.where(true_coarse == 0, leg_prediction, arm_prediction)
    hard_balanced = hierarchical_balanced_prediction(
        coarse_scores, pair_scores, val_idx, records
    )
    oracle_balanced = hierarchical_balanced_prediction(
        coarse_scores, pair_scores, val_idx, records, true_coarse=true_coarse
    )
    predictions = {
        "hybrid_fused_balanced": prediction,
        "hybrid_fused_independent": independent,
        "hybrid_hard_balanced": hard_balanced,
        "hybrid_oracle_coarse": oracle,
        "hybrid_oracle_coarse_balanced": oracle_balanced,
    }
    val_subjects = np.asarray([str(records[int(index)]["subject_id"]) for index in val_idx])
    rows = []
    for rule, values in predictions.items():
        subject_metrics = {
            subject: metrics(
                y[val_idx][val_subjects == subject],
                values[val_subjects == subject],
            )
            for subject in sorted(set(val_subjects.tolist()))
        }
        rows.append(
            {
                "split": split["split"],
                "family": split["family"],
                "prediction_rule": rule,
                "metrics": metrics(y[val_idx], values),
                "coarse_metrics": coarse_metrics(y[val_idx], values),
                "subject_metrics": subject_metrics,
            }
        )
    overlap = len(set(rankings["leg"].tolist()) & set(rankings["arm"].tolist()))
    hyperparameters = {
        "split": split["split"],
        "selected_pair_feature_counts": selected_counts,
        "selected_coarse_feature_count": selected_coarse_configuration["feature_count"],
        "selected_coarse_method": selected_coarse_configuration["method"],
        "selected_coarse_lda_shrinkage": selected_coarse_configuration["shrinkage"],
        "selected_coarse_weight": selected_weight,
        "inner_feature_count_diagnostics": count_diagnostics,
        "inner_coarse_feature_count_diagnostics": coarse_count_diagnostics,
        "inner_coarse_weight_diagnostics": weight_diagnostics,
        "outer_pair_accuracy": {
            pair_name: pair_accuracy(pair_scores[pair_name], y, val_idx, classes)
            for pair_name, classes in PAIR_CLASSES.items()
        },
        "selected_feature_overlap": overlap,
        "selected_feature_jaccard": float(
            overlap
            / max(
                1,
                len(set(rankings["leg"].tolist()) | set(rankings["arm"].tolist())),
            )
        ),
        "top_coarse_feature_indices": coarse_ranking[:20].tolist(),
    }
    return rows, hyperparameters


def summarize(rows: list[dict]) -> list[dict]:
    rules = sorted(set(row["prediction_rule"] for row in rows))
    return sorted(
        [
            {
                "prediction_rule": rule,
                "split_count": sum(row["prediction_rule"] == rule for row in rows),
                "mean_accuracy": float(
                    np.mean(
                        [
                            row["metrics"]["accuracy"]
                            for row in rows
                            if row["prediction_rule"] == rule
                        ]
                    )
                ),
                "mean_balanced_accuracy": float(
                    np.mean(
                        [
                            row["metrics"]["balanced_accuracy"]
                            for row in rows
                            if row["prediction_rule"] == rule
                        ]
                    )
                ),
                "mean_macro_f1": float(
                    np.mean(
                        [
                            row["metrics"]["macro_f1"]
                            for row in rows
                            if row["prediction_rule"] == rule
                        ]
                    )
                ),
                "mean_leg_vs_arm_accuracy": float(
                    np.mean(
                        [
                            row["coarse_metrics"]["leg_vs_arm_accuracy"]
                            for row in rows
                            if row["prediction_rule"] == rule
                        ]
                    )
                ),
            }
            for rule in rules
        ],
        key=lambda row: -row["mean_accuracy"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use native coarse gating with pooled within-pair spatial specialists."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--window-name", default="offset_3_length_8")
    parser.add_argument("--feature-shape", default="24,24,24")
    parser.add_argument("--pair-transform", default="pool_2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subject-fold-count", type=int, default=6)
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument("--subject-seeds", nargs="*", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument(
        "--feature-counts",
        nargs="*",
        type=int,
        default=[32, 64, 128, 256, 512, 1024, 1728, 2048, 4096, 8192, 13824],
    )
    parser.add_argument(
        "--coarse-weights",
        nargs="*",
        type=float,
        default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    )
    parser.add_argument(
        "--coarse-feature-counts",
        nargs="*",
        type=int,
        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 13824],
    )
    parser.add_argument(
        "--coarse-methods",
        nargs="*",
        choices=["cosine", "diagonal_lda", "full_lda"],
        default=["cosine"],
    )
    parser.add_argument(
        "--lda-shrinkages",
        nargs="*",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--full-lda-max-features", type=int, default=512)
    args = parser.parse_args()

    shape = tuple(int(value) for value in args.feature_shape.split(","))
    features, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.window_name]
    )
    centered = center_by_subject_run(features[args.window_name], records)
    native_x, detrend_rows = temporal_detrend_by_subject_run(
        centered, records, degree=1
    )
    pair_x, pair_shape = transform_scale(
        native_x, shape, args.pair_transform, args.batch_size
    )
    feature_counts = feature_counts_for_dimension(
        args.feature_counts, pair_x.shape[1]
    )
    rows = []
    hyperparameters = []
    for split in outer_splits(
        records, "subject", args.subject_fold_count, args.subject_seeds
    ):
        print(f"evaluating {split['split']}", flush=True)
        split_rows, split_hyperparameters = evaluate_split(
            pair_x,
            native_x,
            y,
            records,
            split,
            feature_counts,
            feature_counts_for_dimension(
                args.coarse_feature_counts, native_x.shape[1]
            ),
            args.coarse_methods,
            sorted(set(args.lda_shrinkages)),
            args.full_lda_max_features,
            sorted(set(args.coarse_weights)),
            args.inner_subject_fold_count,
        )
        rows.extend(split_rows)
        hyperparameters.append(split_hyperparameters)

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "window_name": args.window_name,
        "native_feature_shape": shape,
        "pair_transform": args.pair_transform,
        "pair_feature_shape": pair_shape,
        "pair_feature_count": pair_x.shape[1],
        "feature_counts": feature_counts,
        "coarse_weights": sorted(set(args.coarse_weights)),
        "coarse_feature_counts": feature_counts_for_dimension(
            args.coarse_feature_counts, native_x.shape[1]
        ),
        "coarse_methods": args.coarse_methods,
        "lda_shrinkages": sorted(set(args.lda_shrinkages)),
        "full_lda_max_features": args.full_lda_max_features,
        "subject_seeds": args.subject_seeds,
        "mean_temporal_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in detrend_rows])
        ),
        "rows": rows,
        "hyperparameters": hyperparameters,
        "summary": summarize(rows),
        "note": (
            "The coarse leg-versus-arm gate uses native 24^3 features. Within-leg and within-arm "
            "specialists use a label-free spatial transform; pair feature counts and fusion weights are "
            "selected only in inner training-subject folds."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(json.dumps({"out_json": args.out_json, "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
