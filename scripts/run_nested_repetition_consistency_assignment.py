#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import l2_normalize, metrics
from run_detrended_hierarchy_sweep import exact_scores_from_hierarchy
from run_detrended_pair_feature_selection import (
    PAIR_CLASSES,
    inner_splits,
    load_checkpoints,
    outer_splits,
)
from run_learned_temporal_filter_hierarchy import (
    filtered_map,
    learn_temporal_filter,
    preprocess_sequence,
)
from run_nested_temporal_candidate_selection import arm_scores, fit_common_scores
from run_spatial_scale_feature_sweep import transform_scale


def balanced_assignments() -> np.ndarray:
    rows = []
    positions = set(range(8))
    for class_0 in itertools.combinations(range(8), 2):
        remaining_0 = positions - set(class_0)
        for class_1 in itertools.combinations(sorted(remaining_0), 2):
            remaining_1 = remaining_0 - set(class_1)
            for class_2 in itertools.combinations(sorted(remaining_1), 2):
                assignment = np.full(8, 3, dtype=np.int8)
                assignment[list(class_0)] = 0
                assignment[list(class_1)] = 1
                assignment[list(class_2)] = 2
                rows.append(assignment)
    result = np.stack(rows)
    if result.shape != (2520, 8):
        raise RuntimeError(f"Unexpected balanced assignment shape: {result.shape}")
    return result


ASSIGNMENTS = balanced_assignments()
PAIR_POSITIONS = np.asarray(
    [
        [np.flatnonzero(assignment == class_id).tolist() for class_id in range(4)]
        for assignment in ASSIGNMENTS
    ],
    dtype=np.int8,
)


def assign_group(
    exact_scores: np.ndarray,
    normalized_features: np.ndarray,
    consistency_weight: float,
) -> np.ndarray:
    centered_scores = exact_scores - exact_scores.mean(axis=1, keepdims=True)
    score_scale = max(float(np.std(centered_scores)), 1e-12)
    centered_scores /= score_scale
    model_objective = centered_scores[
        np.arange(8, dtype=np.int64)[None, :], ASSIGNMENTS
    ].sum(axis=1)
    similarities = normalized_features @ normalized_features.T
    consistency_objective = similarities[
        PAIR_POSITIONS[:, :, 0], PAIR_POSITIONS[:, :, 1]
    ].sum(axis=1)
    best = int(np.argmax(model_objective + consistency_weight * consistency_objective))
    return ASSIGNMENTS[best].astype(np.int64)


def apply_consistency_assignment(
    exact_scores: np.ndarray,
    normalized_features: np.ndarray,
    val_idx: np.ndarray,
    records: list[dict],
    consistency_weight: float,
) -> np.ndarray:
    prediction = np.empty(len(val_idx), dtype=np.int64)
    groups: dict[str, list[int]] = defaultdict(list)
    for local_position, record_index in enumerate(val_idx):
        record = records[int(record_index)]
        key = f'{record["subject_id"]}|run-{int(record["run_id"])}'
        groups[key].append(local_position)
    for positions in groups.values():
        positions = sorted(
            positions,
            key=lambda position: records[int(val_idx[position])]["event_start"],
        )
        if len(positions) != 8:
            raise ValueError(f"Expected eight events per run; found {len(positions)}")
        position_array = np.asarray(positions, dtype=np.int64)
        prediction[position_array] = assign_group(
            exact_scores[position_array],
            normalized_features[position_array],
            consistency_weight,
        )
    return prediction


def candidate_exact_scores(
    candidate: str,
    sequence: np.ndarray,
    mean_x: np.ndarray,
    mean_pair_x: np.ndarray,
    pair_shape: tuple[int, ...],
    shape: tuple[int, ...],
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hyper1024: dict,
    hyper2048: dict,
    pair_transform: str,
    batch_size: int,
) -> np.ndarray:
    hyper = hyper2048 if candidate.endswith("cap2048") else hyper1024
    arm_x = mean_pair_x
    if candidate.startswith("mean_contrast"):
        weights, _ = learn_temporal_filter(
            sequence,
            y,
            train_idx,
            PAIR_CLASSES["arm"],
            "fisher",
            0.5,
            1.0,
            2,
        )
        uniform = np.full(
            (1, sequence.shape[1]),
            1.0 / np.sqrt(sequence.shape[1]),
            dtype=np.float64,
        )
        temporal_native = filtered_map(
            sequence,
            np.concatenate([uniform, np.atleast_2d(weights)[1:2]], axis=0),
        )
        arm_x, temporal_shape = transform_scale(
            temporal_native,
            shape,
            pair_transform,
            batch_size,
        )
        if temporal_shape != pair_shape:
            raise ValueError(
                f"Mean/temporal pair shapes differ: {pair_shape} vs {temporal_shape}"
            )
    leg_scores, coarse_scores = fit_common_scores(
        mean_pair_x,
        mean_x,
        y,
        train_idx,
        val_idx,
        hyper,
    )
    arm_scores_value = arm_scores(arm_x, y, train_idx, val_idx, hyper)
    return exact_scores_from_hierarchy(
        coarse_scores,
        leg_scores,
        arm_scores_value,
        float(hyper["selected_coarse_weight"]),
    )


def subject_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    val_idx: np.ndarray,
    records: list[dict],
) -> dict:
    subjects = np.asarray([str(records[int(index)]["subject_id"]) for index in val_idx])
    return {
        subject: metrics(y_true[subjects == subject], prediction[subjects == subject])
        for subject in sorted(set(subjects.tolist()))
    }


def paired_bootstrap(
    rows: list[dict], iterations: int, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    fold_difference = np.asarray(
        [
            row["selected_metrics"]["balanced_accuracy"]
            - row["baseline_metrics"]["balanced_accuracy"]
            for row in rows
        ]
    )
    fold_samples = rng.choice(
        fold_difference, size=(iterations, len(fold_difference)), replace=True
    ).mean(axis=1)
    selected_by_subject: dict[str, list[float]] = defaultdict(list)
    baseline_by_subject: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for subject, metric_row in row["selected_subject_metrics"].items():
            selected_by_subject[subject].append(metric_row["balanced_accuracy"])
            baseline_by_subject[subject].append(
                row["baseline_subject_metrics"][subject]["balanced_accuracy"]
            )
    subjects = sorted(selected_by_subject)
    subject_difference = np.asarray(
        [
            np.mean(selected_by_subject[subject])
            - np.mean(baseline_by_subject[subject])
            for subject in subjects
        ]
    )
    subject_samples = rng.choice(
        subject_difference,
        size=(iterations, len(subject_difference)),
        replace=True,
    ).mean(axis=1)
    return {
        "fold_mean_difference": float(np.mean(fold_difference)),
        "fold_ci95": [
            float(value) for value in np.quantile(fold_samples, [0.025, 0.975])
        ],
        "subject_mean_difference": float(np.mean(subject_difference)),
        "subject_ci95": [
            float(value) for value in np.quantile(subject_samples, [0.025, 0.975])
        ],
        "subject_wins": int(np.sum(subject_difference > 0)),
        "subject_ties": int(np.sum(subject_difference == 0)),
        "subject_losses": int(np.sum(subject_difference < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested balanced decoding with within-assigned-class repetition consistency."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--baseline-cap1024-json", required=True)
    parser.add_argument("--baseline-cap2048-json", required=True)
    parser.add_argument("--nested-selection-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--weights", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split-limit", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    args = parser.parse_args()

    weights = [float(value) for value in args.weights.split(",")]
    if 0.0 not in weights:
        raise ValueError("Weights must include 0 to verify baseline reconstruction.")
    baseline1024 = json.loads(Path(args.baseline_cap1024_json).read_text())
    baseline2048 = json.loads(Path(args.baseline_cap2048_json).read_text())
    nested = json.loads(Path(args.nested_selection_json).read_text())
    if baseline1024["pair_transform"] != baseline2048["pair_transform"]:
        raise ValueError("The 1,024- and 2,048-cap pair transforms must match.")
    hyper1024 = {row["split"]: row for row in baseline1024["hyperparameters"]}
    hyper2048 = {row["split"]: row for row in baseline2048["hyperparameters"]}
    selected_candidate = {
        row["split"]: row["selected_candidate"]
        for row in nested["selected_rows"]
        if row["prediction_rule"] == "balanced"
    }

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, detrend_rows = preprocess_sequence(
        feature_dict.pop(args.sequence_key), records
    )
    mean_x = sequence.mean(axis=1, dtype=np.float32)
    normalized_mean_x = l2_normalize(mean_x)
    shape = tuple(int(value) for value in baseline1024["native_feature_shape"])
    mean_pair_x, pair_shape = transform_scale(
        mean_x, shape, baseline1024["pair_transform"], args.batch_size
    )
    split_by_name = {
        split["split"]: split
        for split in outer_splits(records, "subject", 6, baseline1024["subject_seeds"])
    }
    split_names = sorted(selected_candidate)
    if args.split_limit is not None:
        split_names = split_names[: args.split_limit]

    rows = []
    for split_name in split_names:
        candidate = selected_candidate[split_name]
        split = split_by_name[split_name]
        print(f"evaluating {split_name} ({candidate})", flush=True)
        inner_rows = []
        for inner in inner_splits(
            records,
            split["train_idx"],
            "subject",
            args.inner_subject_fold_count,
        ):
            exact_scores = candidate_exact_scores(
                candidate,
                sequence,
                mean_x,
                mean_pair_x,
                pair_shape,
                shape,
                y,
                inner["train_idx"],
                inner["val_idx"],
                hyper1024[split_name],
                hyper2048[split_name],
                baseline1024["pair_transform"],
                args.batch_size,
            )
            for weight in weights:
                prediction = apply_consistency_assignment(
                    exact_scores,
                    normalized_mean_x[inner["val_idx"]],
                    inner["val_idx"],
                    records,
                    weight,
                )
                inner_rows.append(
                    {
                        "inner_split": inner["split"],
                        "weight": weight,
                        "balanced_accuracy": metrics(
                            y[inner["val_idx"]], prediction
                        )["balanced_accuracy"],
                    }
                )
        means = {
            weight: float(
                np.mean(
                    [
                        row["balanced_accuracy"]
                        for row in inner_rows
                        if row["weight"] == weight
                    ]
                )
            )
            for weight in weights
        }
        selected_weight = min(weights, key=lambda value: (-means[value], value))
        exact_scores = candidate_exact_scores(
            candidate,
            sequence,
            mean_x,
            mean_pair_x,
            pair_shape,
            shape,
            y,
            split["train_idx"],
            split["val_idx"],
            hyper1024[split_name],
            hyper2048[split_name],
            baseline1024["pair_transform"],
            args.batch_size,
        )
        baseline_prediction = apply_consistency_assignment(
            exact_scores,
            normalized_mean_x[split["val_idx"]],
            split["val_idx"],
            records,
            0.0,
        )
        selected_prediction = apply_consistency_assignment(
            exact_scores,
            normalized_mean_x[split["val_idx"]],
            split["val_idx"],
            records,
            selected_weight,
        )
        rows.append(
            {
                "split": split_name,
                "candidate": candidate,
                "selected_weight": selected_weight,
                "inner_weight_accuracy": means,
                "baseline_metrics": metrics(y[split["val_idx"]], baseline_prediction),
                "selected_metrics": metrics(y[split["val_idx"]], selected_prediction),
                "baseline_subject_metrics": subject_metrics(
                    y[split["val_idx"]], baseline_prediction, split["val_idx"], records
                ),
                "selected_subject_metrics": subject_metrics(
                    y[split["val_idx"]], selected_prediction, split["val_idx"], records
                ),
            }
        )

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "nested_selection_json": args.nested_selection_json,
        "weights": weights,
        "assignment_count_per_run": len(ASSIGNMENTS),
        "selected_weight_counts": dict(Counter(row["selected_weight"] for row in rows)),
        "mean_baseline_balanced_accuracy": float(
            np.mean([row["baseline_metrics"]["balanced_accuracy"] for row in rows])
        ),
        "mean_selected_balanced_accuracy": float(
            np.mean([row["selected_metrics"]["balanced_accuracy"] for row in rows])
        ),
        "paired_bootstrap": paired_bootstrap(
            rows, args.bootstrap_iterations, args.bootstrap_seed
        ),
        "detrend_by_lag": detrend_rows,
        "rows": rows,
        "note": (
            "Every consistency weight is selected inside outer-training subjects. Lambda 0 "
            "is the exact score-only balanced decoder. The similarity term uses the full "
            "run-centered, linearly detrended mean event map and no held-out labels."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "selected_weight_counts",
                    "mean_baseline_balanced_accuracy",
                    "mean_selected_balanced_accuracy",
                    "paired_bootstrap",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
