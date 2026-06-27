#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import apply_balanced_assignment, center_by_subject_run, metrics
from run_clip_offset_event_sweep import coarse_metrics
from run_detrended_pair_feature_selection import (
    choose_coarse_weight,
    choose_feature_counts,
    inner_splits,
    load_checkpoints,
    outer_splits,
)
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run
from run_unlabeled_subject_model_gate import model_scores_and_predictions


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def normalize_score_scale(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = float(np.sqrt(np.mean(centered**2)))
    return centered / max(scale, 1e-8)


def blended_scores(flat_scores: np.ndarray, pair_scores: np.ndarray, pair_weight: float) -> np.ndarray:
    flat = normalize_score_scale(flat_scores)
    pair = normalize_score_scale(pair_scores)
    return (1.0 - pair_weight) * flat + pair_weight * pair


def choose_pair_weight(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    inner: list[dict],
    selected_counts: dict[str, int],
    coarse_weight: float,
    pair_weights: list[float],
) -> tuple[float, list[dict]]:
    rows = []
    for split in inner:
        result = model_scores_and_predictions(
            x,
            y,
            records,
            split["train_idx"],
            split["val_idx"],
            selected_counts,
            coarse_weight,
        )
        for pair_weight in pair_weights:
            scores = blended_scores(
                result["flat_scores"],
                result["pair_scores"],
                pair_weight,
            )
            pred = apply_balanced_assignment(scores, split["val_idx"], records)
            rows.append(
                {
                    "split": split["split"],
                    "pair_weight": float(pair_weight),
                    "balanced_accuracy": metrics(y[split["val_idx"]], pred)["balanced_accuracy"],
                }
            )
    means = {
        weight: float(
            np.mean([row["balanced_accuracy"] for row in rows if row["pair_weight"] == weight])
        )
        for weight in pair_weights
    }
    selected = min(pair_weights, key=lambda weight: (-means[weight], abs(1.0 - weight)))
    diagnostics = [
        {"pair_weight": float(weight), "mean_inner_balanced_accuracy": means[weight]}
        for weight in pair_weights
    ]
    return float(selected), diagnostics


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
        description="Select a leakage-safe flat/pair score blend inside each outer subject split."
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
    parser.add_argument(
        "--pair-weights",
        nargs="*",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    args = parser.parse_args()

    features, y, records = load_checkpoints(Path(args.checkpoint_dir), [args.window_name])
    centered = center_by_subject_run(features[args.window_name], records)
    x, group_rows = temporal_detrend_by_subject_run(centered, records, degree=1)
    splits = outer_splits(records, "subject", args.subject_fold_count, args.subject_seeds)

    rows = []
    split_diagnostics = []
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
        selected_coarse_weight, coarse_diagnostics = choose_coarse_weight(
            x,
            y,
            records,
            inner,
            selected_counts,
            sorted(set(args.coarse_weights)),
        )
        selected_pair_weight, pair_weight_diagnostics = choose_pair_weight(
            x,
            y,
            records,
            inner,
            selected_counts,
            selected_coarse_weight,
            sorted(set(args.pair_weights)),
        )

        outer = model_scores_and_predictions(
            x,
            y,
            records,
            split["train_idx"],
            split["val_idx"],
            selected_counts,
            selected_coarse_weight,
        )
        selected_scores = blended_scores(
            outer["flat_scores"],
            outer["pair_scores"],
            selected_pair_weight,
        )
        selected_pred = apply_balanced_assignment(selected_scores, split["val_idx"], records)

        outer_weight_rows = []
        for pair_weight in sorted(set(args.pair_weights)):
            candidate_scores = blended_scores(
                outer["flat_scores"],
                outer["pair_scores"],
                pair_weight,
            )
            candidate_pred = apply_balanced_assignment(
                candidate_scores,
                split["val_idx"],
                records,
            )
            outer_weight_rows.append(
                {
                    "pair_weight": float(pair_weight),
                    "balanced_accuracy": metrics(
                        y[split["val_idx"]],
                        candidate_pred,
                    )["balanced_accuracy"],
                }
            )
        oracle_weight = min(
            args.pair_weights,
            key=lambda weight: (
                -next(
                    row["balanced_accuracy"]
                    for row in outer_weight_rows
                    if row["pair_weight"] == weight
                ),
                abs(1.0 - weight),
            ),
        )
        oracle_scores = blended_scores(
            outer["flat_scores"],
            outer["pair_scores"],
            oracle_weight,
        )
        oracle_pred = apply_balanced_assignment(oracle_scores, split["val_idx"], records)

        predictions = [
            ("flat_all_features_balanced", outer["flat_balanced"]),
            ("selected_pair_fused_balanced", outer["pair_balanced"]),
            ("nested_flat_pair_ensemble", selected_pred),
            ("oracle_outer_blend_weight", oracle_pred),
        ]
        for rule, pred in predictions:
            rows.append(
                {
                    "split": split["split"],
                    "prediction_rule": rule,
                    "metrics": metrics(y[split["val_idx"]], pred),
                    "coarse_metrics": coarse_metrics(y[split["val_idx"]], pred),
                }
            )
        split_diagnostics.append(
            {
                "split": split["split"],
                "selected_feature_counts": selected_counts,
                "selected_coarse_weight": selected_coarse_weight,
                "selected_pair_weight": selected_pair_weight,
                "oracle_outer_pair_weight": float(oracle_weight),
                "feature_count_diagnostics": count_diagnostics,
                "coarse_weight_diagnostics": coarse_diagnostics,
                "pair_weight_diagnostics": pair_weight_diagnostics,
                "outer_weight_rows": outer_weight_rows,
            }
        )

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "window_name": args.window_name,
        "subject_seeds": args.subject_seeds,
        "pair_weights": sorted(set(args.pair_weights)),
        "selected_pair_weight_counts": dict(
            Counter(row["selected_pair_weight"] for row in split_diagnostics)
        ),
        "mean_linear_time_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in group_rows])
        ),
        "rows": rows,
        "split_diagnostics": split_diagnostics,
        "summary": summarize(rows),
        "note": (
            "Flat and pair score scales are normalized using unlabeled events in each evaluation "
            "split. Pair blend weight is selected only from inner training-subject folds. Oracle "
            "outer blend weight uses held-out labels and is an upper bound only."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "selected_pair_weight_counts": result["selected_pair_weight_counts"],
                "summary": result["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
