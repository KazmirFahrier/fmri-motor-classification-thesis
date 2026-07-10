#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import center_by_subject_run
from run_detrended_pair_feature_selection import (
    as_jsonable,
    load_checkpoints,
    outer_splits,
)
from run_hybrid_spatial_hierarchy import evaluate_split, summarize
from run_spatial_scale_feature_sweep import (
    feature_counts_for_dimension,
    transform_scale,
)
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


def load_centered_component(
    checkpoint_dir: Path,
    key: str,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    features, labels, records = load_checkpoints(checkpoint_dir, [key])
    centered = center_by_subject_run(features[key], records)
    detrended, detrend_rows = temporal_detrend_by_subject_run(
        centered, records, degree=1
    )
    return detrended, labels, records, detrend_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test temporal-basis arm features inside the validated multi-scale "
            "coarse/fine hierarchy."
        )
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--window-prefix", default="offset_3_length_8")
    parser.add_argument("--feature-shape", default="24,24,24")
    parser.add_argument("--pair-transform", default="smooth_3")
    parser.add_argument(
        "--arm-representation",
        choices=["mean", "mean_plus_tail"],
        default="mean_plus_tail",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subject-fold-count", type=int, default=6)
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument(
        "--subject-seeds", nargs="*", type=int, default=[11, 23, 37, 51, 71]
    )
    parser.add_argument(
        "--feature-counts", nargs="*", type=int, default=[256, 512, 1024, 13824]
    )
    parser.add_argument(
        "--coarse-feature-counts",
        nargs="*",
        type=int,
        default=[64, 128, 256, 13824],
    )
    parser.add_argument(
        "--coarse-weights",
        nargs="*",
        type=float,
        default=[0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
    )
    parser.add_argument(
        "--lda-shrinkages", nargs="*", type=float, default=[0.25, 0.5, 0.75]
    )
    parser.add_argument("--full-lda-max-features", type=int, default=256)
    parser.add_argument("--pair-full-lda-max-features", type=int, default=1024)
    args = parser.parse_args()

    shape = tuple(int(value) for value in args.feature_shape.split(","))
    checkpoint_dir = Path(args.checkpoint_dir)
    mean_key = f"{args.window_prefix}_mean"
    mean_x, y, records, mean_detrend_rows = load_centered_component(
        checkpoint_dir, mean_key
    )

    if mean_x.shape[1] != int(np.prod(shape)):
        raise ValueError(f"Shape {shape} does not match {mean_x.shape[1]} features.")

    mean_pair_x, pair_shape = transform_scale(
        mean_x, shape, args.pair_transform, args.batch_size
    )
    pair_features = {
        "leg": mean_pair_x,
        "arm": mean_pair_x,
    }
    extra_detrend = {}
    if args.arm_representation == "mean_plus_tail":
        tail_key = f"{args.window_prefix}_tail_vs_body"
        tail_x, tail_y, tail_records, tail_detrend_rows = load_centered_component(
            checkpoint_dir, tail_key
        )
        if not np.array_equal(tail_y, y):
            raise ValueError("Mean and tail labels do not match.")
        if tail_records != records:
            raise ValueError("Mean and tail records do not match.")
        tail_pair_x, tail_shape = transform_scale(
            tail_x, shape, args.pair_transform, args.batch_size
        )
        if tail_shape != pair_shape:
            raise ValueError(f"Mean/tail transformed shapes differ: {pair_shape} vs {tail_shape}.")
        pair_features["arm"] = np.concatenate([mean_pair_x, tail_pair_x], axis=1)
        extra_detrend["tail_vs_body"] = tail_detrend_rows

    feature_counts = {
        pair_name: feature_counts_for_dimension(args.feature_counts, pair_x.shape[1])
        for pair_name, pair_x in pair_features.items()
    }
    rows = []
    hyperparameters = []
    for split in outer_splits(
        records, "subject", args.subject_fold_count, args.subject_seeds
    ):
        print(f"evaluating {split['split']}", flush=True)
        split_rows, split_hyperparameters = evaluate_split(
            pair_features,
            mean_x,
            y,
            records,
            split,
            feature_counts,
            ["full_lda"],
            args.pair_full_lda_max_features,
            feature_counts_for_dimension(args.coarse_feature_counts, mean_x.shape[1]),
            ["diagonal_lda", "full_lda"],
            sorted(set(args.lda_shrinkages)),
            args.full_lda_max_features,
            sorted(set(args.coarse_weights)),
            args.inner_subject_fold_count,
        )
        rows.extend(split_rows)
        hyperparameters.append(split_hyperparameters)

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "window_prefix": args.window_prefix,
        "native_feature_shape": shape,
        "pair_transform": args.pair_transform,
        "arm_representation": args.arm_representation,
        "pair_feature_shape": pair_shape,
        "pair_feature_counts": {
            pair_name: int(pair_x.shape[1]) for pair_name, pair_x in pair_features.items()
        },
        "feature_counts": feature_counts,
        "coarse_weights": sorted(set(args.coarse_weights)),
        "coarse_feature_counts": feature_counts_for_dimension(
            args.coarse_feature_counts, mean_x.shape[1]
        ),
        "lda_shrinkages": sorted(set(args.lda_shrinkages)),
        "full_lda_max_features": args.full_lda_max_features,
        "pair_full_lda_max_features": args.pair_full_lda_max_features,
        "subject_seeds": args.subject_seeds,
        "mean_temporal_variance_fraction": {
            "mean": float(
                np.mean([row["temporal_variance_fraction"] for row in mean_detrend_rows])
            ),
            **{
                name: float(
                    np.mean([row["temporal_variance_fraction"] for row in rows])
                )
                for name, rows in extra_detrend.items()
            },
        },
        "rows": rows,
        "hyperparameters": hyperparameters,
        "summary": summarize(rows),
        "note": (
            "Coarse and leg branches use the validated mean response representation. "
            "The arm branch optionally concatenates the tail-vs-body temporal basis map. "
            "All pair/coarse configurations and fusion weights are selected only inside "
            "inner training-subject folds."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(json.dumps({"out_json": args.out_json, "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
