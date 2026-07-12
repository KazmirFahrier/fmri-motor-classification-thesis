#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_detrended_pair_feature_selection import PAIR_CLASSES, inner_splits, load_checkpoints, outer_splits
from run_hierarchy_subject_calibration import evaluate_scores, fit_source_pair_model, pair_scores
from run_hybrid_spatial_hierarchy import selected_coarse_scores
from run_learned_temporal_filter_hierarchy import (
    filtered_map,
    learn_temporal_filter,
    preprocess_sequence,
)
from run_spatial_scale_feature_sweep import transform_scale


CANDIDATES = (
    "mean_cap1024",
    "mean_contrast_cap1024",
    "mean_contrast_cap2048",
)


def load_outer_rows(path: Path, representation: str) -> dict[tuple[str, str], dict]:
    payload = json.loads(path.read_text())
    return {
        (row["split"], row["prediction_rule"]): row
        for row in payload["rows"]
        if row["representation"] == representation
    }


def fit_common_scores(
    mean_pair_x: np.ndarray,
    mean_x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hyper: dict,
) -> tuple[np.ndarray, np.ndarray]:
    leg_config = hyper["selected_pair_configurations"]["leg"]
    leg_model = fit_source_pair_model(
        mean_pair_x,
        y,
        train_idx,
        PAIR_CLASSES["leg"],
        int(leg_config["feature_count"]),
        float(leg_config["shrinkage"]),
    )
    leg_scores = pair_scores(mean_pair_x, val_idx, leg_model)
    coarse_scores, _ = selected_coarse_scores(
        mean_x,
        y,
        train_idx,
        val_idx,
        int(hyper["selected_coarse_feature_count"]),
        hyper["selected_coarse_method"],
        float(hyper["selected_coarse_lda_shrinkage"]),
    )
    return leg_scores, coarse_scores


def arm_scores(
    arm_x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hyper: dict,
) -> np.ndarray:
    config = hyper["selected_pair_configurations"]["arm"]
    model = fit_source_pair_model(
        arm_x,
        y,
        train_idx,
        PAIR_CLASSES["arm"],
        int(config["feature_count"]),
        float(config["shrinkage"]),
    )
    return pair_scores(arm_x, val_idx, model)


def exact_accuracies(
    coarse_scores: np.ndarray,
    leg_scores: np.ndarray,
    arm_scores_value: np.ndarray,
    coarse_weight: float,
    y: np.ndarray,
    val_idx: np.ndarray,
    records: list[dict],
) -> dict[str, float]:
    result = evaluate_scores(
        coarse_scores,
        {"leg": leg_scores, "arm": arm_scores_value},
        coarse_weight,
        y,
        val_idx,
        records,
    )
    return {
        rule: float(metric_row["balanced_accuracy"])
        for rule, metric_row in result.items()
    }


def selected_summary(rows: list[dict]) -> list[dict]:
    output = []
    for rule in sorted({row["prediction_rule"] for row in rows}):
        values = [row for row in rows if row["prediction_rule"] == rule]
        output.append(
            {
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


def bootstrap_comparisons(
    selected_rows: list[dict],
    baseline_rows: dict[tuple[str, str], dict],
    iterations: int,
    seed: int,
) -> dict:
    output = {}
    rng = np.random.default_rng(seed)
    for rule in sorted({row["prediction_rule"] for row in selected_rows}):
        rule_rows = [row for row in selected_rows if row["prediction_rule"] == rule]
        fold_difference = np.asarray(
            [
                row["metrics"]["balanced_accuracy"]
                - baseline_rows[(row["split"], rule)]["metrics"]["balanced_accuracy"]
                for row in rule_rows
            ]
        )
        fold_samples = rng.choice(
            fold_difference, size=(iterations, len(fold_difference)), replace=True
        ).mean(axis=1)

        selected_by_subject: dict[str, list[float]] = defaultdict(list)
        baseline_by_subject: dict[str, list[float]] = defaultdict(list)
        for row in rule_rows:
            baseline = baseline_rows[(row["split"], rule)]
            for subject, metric_row in row["subject_metrics"].items():
                selected_by_subject[subject].append(metric_row["balanced_accuracy"])
                baseline_by_subject[subject].append(
                    baseline["subject_metrics"][subject]["balanced_accuracy"]
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
        output[rule] = {
            "fold_mean_difference": float(np.mean(fold_difference)),
            "fold_ci95": [
                float(value) for value in np.quantile(fold_samples, [0.025, 0.975])
            ],
            "subject_mean_difference": float(np.mean(subject_difference)),
            "subject_ci95": [
                float(value)
                for value in np.quantile(subject_samples, [0.025, 0.975])
            ],
            "subject_wins": int(np.sum(subject_difference > 0)),
            "subject_ties": int(np.sum(subject_difference == 0)),
            "subject_losses": int(np.sum(subject_difference < 0)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select mean/temporal-rank/covariance-cap candidates inside outer training subjects."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--baseline-cap1024-json", required=True)
    parser.add_argument("--baseline-cap2048-json", required=True)
    parser.add_argument("--candidate-cap1024-json", required=True)
    parser.add_argument("--candidate-cap2048-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument(
        "--selection-mode",
        choices=["balanced_shared", "rule_specific"],
        default="balanced_shared",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split-limit", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    args = parser.parse_args()

    baseline1024 = json.loads(Path(args.baseline_cap1024_json).read_text())
    baseline2048 = json.loads(Path(args.baseline_cap2048_json).read_text())
    hyper1024 = {row["split"]: row for row in baseline1024["hyperparameters"]}
    hyper2048 = {row["split"]: row for row in baseline2048["hyperparameters"]}
    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, detrend_rows = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    mean_x = sequence.mean(axis=1, dtype=np.float32)
    shape = tuple(int(value) for value in baseline1024["native_feature_shape"])
    mean_pair_x, pair_shape = transform_scale(
        mean_x, shape, baseline1024["pair_transform"], args.batch_size
    )

    source_rows = {
        "mean_cap1024": load_outer_rows(
            Path(args.candidate_cap1024_json), "mean"
        ),
        "mean_contrast_cap1024": load_outer_rows(
            Path(args.candidate_cap1024_json), "learned_arm_filter"
        ),
        "mean_contrast_cap2048": load_outer_rows(
            Path(args.candidate_cap2048_json), "learned_arm_filter"
        ),
    }
    split_by_name = {
        split["split"]: split
        for split in outer_splits(records, "subject", 6, baseline1024["subject_seeds"])
    }
    split_names = sorted(hyper1024)
    if args.split_limit is not None:
        split_names = split_names[: args.split_limit]

    selected_rows = []
    selection_rows = []
    for split_name in split_names:
        print(f"selecting {split_name}", flush=True)
        split = split_by_name[split_name]
        inner = inner_splits(
            records,
            split["train_idx"],
            "subject",
            args.inner_subject_fold_count,
        )
        candidate_scores: dict[str, dict[str, list[float]]] = {
            candidate: {"balanced": [], "independent": []}
            for candidate in CANDIDATES
        }
        for inner_split in inner:
            train_idx = inner_split["train_idx"]
            val_idx = inner_split["val_idx"]
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
            contrast = np.atleast_2d(weights)[1:2]
            uniform = np.full(
                (1, sequence.shape[1]),
                1.0 / np.sqrt(sequence.shape[1]),
                dtype=np.float64,
            )
            temporal_native = filtered_map(
                sequence, np.concatenate([uniform, contrast], axis=0)
            )
            temporal_pair_x, temporal_shape = transform_scale(
                temporal_native,
                shape,
                baseline1024["pair_transform"],
                args.batch_size,
            )
            if temporal_shape != pair_shape:
                raise ValueError(
                    f"Mean/temporal pair shapes differ: {pair_shape} vs {temporal_shape}."
                )

            leg1024, coarse1024 = fit_common_scores(
                mean_pair_x,
                mean_x,
                y,
                train_idx,
                val_idx,
                hyper1024[split_name],
            )
            mean_arm1024 = arm_scores(
                mean_pair_x,
                y,
                train_idx,
                val_idx,
                hyper1024[split_name],
            )
            temporal_arm1024 = arm_scores(
                temporal_pair_x,
                y,
                train_idx,
                val_idx,
                hyper1024[split_name],
            )
            leg2048, coarse2048 = fit_common_scores(
                mean_pair_x,
                mean_x,
                y,
                train_idx,
                val_idx,
                hyper2048[split_name],
            )
            temporal_arm2048 = arm_scores(
                temporal_pair_x,
                y,
                train_idx,
                val_idx,
                hyper2048[split_name],
            )
            values = {
                "mean_cap1024": exact_accuracies(
                    coarse1024,
                    leg1024,
                    mean_arm1024,
                    float(hyper1024[split_name]["selected_coarse_weight"]),
                    y,
                    val_idx,
                    records,
                ),
                "mean_contrast_cap1024": exact_accuracies(
                    coarse1024,
                    leg1024,
                    temporal_arm1024,
                    float(hyper1024[split_name]["selected_coarse_weight"]),
                    y,
                    val_idx,
                    records,
                ),
                "mean_contrast_cap2048": exact_accuracies(
                    coarse2048,
                    leg2048,
                    temporal_arm2048,
                    float(hyper2048[split_name]["selected_coarse_weight"]),
                    y,
                    val_idx,
                    records,
                ),
            }
            for candidate, accuracy_by_rule in values.items():
                for rule, accuracy in accuracy_by_rule.items():
                    candidate_scores[candidate][rule].append(accuracy)
                selection_rows.append(
                    {
                        "outer_split": split_name,
                        "inner_split": inner_split["split"],
                        "candidate": candidate,
                        "balanced_accuracy": accuracy_by_rule["balanced"],
                        "independent_accuracy": accuracy_by_rule["independent"],
                    }
                )

        means_by_rule = {
            rule: {
                candidate: float(np.mean(candidate_scores[candidate][rule]))
                for candidate in CANDIDATES
            }
            for rule in ("balanced", "independent")
        }
        for rule in ("balanced", "independent"):
            selection_rule = (
                "balanced" if args.selection_mode == "balanced_shared" else rule
            )
            means = means_by_rule[selection_rule]
            winner = min(
                CANDIDATES,
                key=lambda candidate: (
                    -means[candidate],
                    CANDIDATES.index(candidate),
                ),
            )
            selected = dict(source_rows[winner][(split_name, rule)])
            selected["selected_candidate"] = winner
            selected["selection_metric"] = selection_rule
            selected["inner_candidate_accuracy"] = means
            selected_rows.append(selected)

    summary = selected_summary(selected_rows)
    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "candidates": list(CANDIDATES),
        "selection_mode": args.selection_mode,
        "selection_metric": (
            "inner-subject-fold exact balanced accuracy shared across prediction rules"
            if args.selection_mode == "balanced_shared"
            else "matching inner-subject-fold exact accuracy for each prediction rule"
        ),
        "candidate_counts": {
            rule: dict(
                Counter(
                    row["selected_candidate"]
                    for row in selected_rows
                    if row["prediction_rule"] == rule
                )
            )
            for rule in ("balanced", "independent")
        },
        "detrend_by_lag": detrend_rows,
        "selection_rows": selection_rows,
        "selected_rows": selected_rows,
        "summary": summary,
        "bootstrap_vs_mean_cap1024": bootstrap_comparisons(
            selected_rows,
            source_rows["mean_cap1024"],
            args.bootstrap_iterations,
            args.bootstrap_seed,
        ),
        "note": (
            "Every temporal filter and candidate decision uses outer-training subjects only. "
            "The selected outer prediction is read from the matching previously evaluated "
            "candidate; held-out-subject labels never influence candidate selection."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "candidate_counts": result["candidate_counts"],
                "summary": summary,
                "bootstrap_vs_mean_cap1024": result[
                    "bootstrap_vs_mean_cap1024"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
