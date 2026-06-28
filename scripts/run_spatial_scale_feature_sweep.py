#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import center_by_subject_run
from run_detrended_pair_feature_selection import (
    as_jsonable,
    evaluate_outer_split,
    load_checkpoints,
    outer_splits,
)
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


def mean_pool(
    x: np.ndarray,
    shape: tuple[int, int, int],
    factor: int,
) -> np.ndarray:
    if any(size % factor != 0 for size in shape):
        raise ValueError(f"Shape {shape} is not divisible by pooling factor {factor}.")
    volumes = x.reshape((-1, *shape))
    pooled = volumes.reshape(
        volumes.shape[0],
        shape[0] // factor,
        factor,
        shape[1] // factor,
        factor,
        shape[2] // factor,
        factor,
    ).mean(axis=(2, 4, 6))
    return pooled.reshape(len(x), -1).astype(np.float32)


def mean_smooth(
    x: np.ndarray,
    shape: tuple[int, int, int],
    kernel_size: int,
    batch_size: int,
) -> np.ndarray:
    if kernel_size % 2 != 1:
        raise ValueError("Smoothing kernel size must be odd.")
    radius = kernel_size // 2
    result = np.empty_like(x, dtype=np.float32)
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        volumes = x[start:stop].reshape((-1, *shape))
        padded = np.pad(
            volumes,
            ((0, 0), (radius, radius), (radius, radius), (radius, radius)),
            mode="reflect",
        )
        smoothed = np.zeros_like(volumes, dtype=np.float32)
        for dx in range(kernel_size):
            for dy in range(kernel_size):
                for dz in range(kernel_size):
                    smoothed += padded[
                        :,
                        dx : dx + shape[0],
                        dy : dy + shape[1],
                        dz : dz + shape[2],
                    ]
        smoothed /= float(kernel_size ** 3)
        result[start:stop] = smoothed.reshape(stop - start, -1)
    return result


def transform_scale(
    x: np.ndarray,
    shape: tuple[int, int, int],
    scale: str,
    batch_size: int,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    if scale == "native":
        return x, shape
    operation, value_text = scale.split("_", 1)
    value = int(value_text)
    if operation == "pool":
        transformed = mean_pool(x, shape, value)
        return transformed, tuple(size // value for size in shape)
    if operation == "smooth":
        return mean_smooth(x, shape, value, batch_size), shape
    raise ValueError(f"Unknown scale: {scale}")


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scale"], row["prediction_rule"])].append(row)
    result = []
    for (scale, rule), group in sorted(grouped.items()):
        result.append(
            {
                "scale": scale,
                "prediction_rule": rule,
                "split_count": len(group),
                "mean_accuracy": float(
                    np.mean([row["metrics"]["accuracy"] for row in group])
                ),
                "mean_balanced_accuracy": float(
                    np.mean([row["metrics"]["balanced_accuracy"] for row in group])
                ),
                "mean_macro_f1": float(
                    np.mean([row["metrics"]["macro_f1"] for row in group])
                ),
                "mean_leg_vs_arm_accuracy": float(
                    np.mean(
                        [row["coarse_metrics"]["leg_vs_arm_accuracy"] for row in group]
                    )
                ),
            }
        )
    return sorted(result, key=lambda row: -row["mean_balanced_accuracy"])


def feature_counts_for_dimension(base_counts: list[int], dimension: int) -> list[int]:
    return sorted({min(count, dimension) for count in base_counts} | {dimension})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare spatial smoothing and pooling inside nested pair feature selection."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--window-name", default="offset_3_length_8")
    parser.add_argument("--feature-shape", default="24,24,24")
    parser.add_argument(
        "--scales",
        nargs="*",
        default=["native", "smooth_3", "smooth_5", "pool_2", "pool_3", "pool_4"],
    )
    parser.add_argument("--subject-fold-count", type=int, default=6)
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument("--subject-seeds", nargs="*", type=int, default=[])
    parser.add_argument(
        "--feature-counts",
        nargs="*",
        type=int,
        default=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 13824],
    )
    parser.add_argument(
        "--coarse-weights",
        nargs="*",
        type=float,
        default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    shape = tuple(int(value) for value in args.feature_shape.split(","))
    features, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.window_name]
    )
    centered = center_by_subject_run(features[args.window_name], records)
    detrended, detrend_rows = temporal_detrend_by_subject_run(
        centered, records, degree=1
    )
    if detrended.shape[1] != int(np.prod(shape)):
        raise ValueError(f"Shape {shape} does not match {detrended.shape[1]} features.")
    splits = outer_splits(
        records, "subject", args.subject_fold_count, args.subject_seeds
    )

    rows = []
    hyperparameters = []
    scale_shapes = {}
    for scale in args.scales:
        print(f"transforming {scale}", flush=True)
        scaled_x, scaled_shape = transform_scale(
            detrended, shape, scale, args.batch_size
        )
        scale_shapes[scale] = scaled_shape
        counts = feature_counts_for_dimension(args.feature_counts, scaled_x.shape[1])
        for split in splits:
            print(f"evaluating {scale} {split['split']}", flush=True)
            split_rows, split_hyperparameters = evaluate_outer_split(
                scaled_x,
                y,
                records,
                split,
                counts,
                sorted(set(args.coarse_weights)),
                args.inner_subject_fold_count,
            )
            for row in split_rows:
                row["scale"] = scale
                row["scaled_feature_count"] = int(scaled_x.shape[1])
            split_hyperparameters["scale"] = scale
            split_hyperparameters["scaled_shape"] = scaled_shape
            split_hyperparameters["candidate_feature_counts"] = counts
            rows.extend(split_rows)
            hyperparameters.append(split_hyperparameters)

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "window_name": args.window_name,
        "feature_shape": shape,
        "scales": args.scales,
        "scale_shapes": scale_shapes,
        "subject_seeds": args.subject_seeds,
        "subject_fold_count": args.subject_fold_count,
        "inner_subject_fold_count": args.inner_subject_fold_count,
        "mean_temporal_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in detrend_rows])
        ),
        "rows": rows,
        "hyperparameters": hyperparameters,
        "summary": summarize(rows),
        "note": (
            "Spatial transforms are label-free. Pair-specific feature counts and hierarchy weights "
            "are selected only in inner training folds for every outer subject split."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(json.dumps({"out_json": args.out_json, "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
