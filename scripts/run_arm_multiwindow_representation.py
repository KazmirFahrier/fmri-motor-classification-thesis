#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import center_by_subject_run
from run_detrended_pair_feature_selection import (
    as_jsonable,
    inner_splits,
    load_checkpoints,
    outer_splits,
    pair_accuracy,
    rank_pair_features,
)
from run_hybrid_spatial_hierarchy import full_lda_scores
from run_spatial_scale_feature_sweep import transform_scale
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


ARM_CLASSES = (2, 3)
WINDOWS = {
    "late_8": "offset_3_length_8",
    "late_6": "offset_3_length_6",
    "canonical_6": "offset_2_length_6",
}
REPRESENTATIONS = {
    "late_8": ("late_8",),
    "late_8_plus_late_6": ("late_8", "late_6"),
    "late_8_plus_canonical_6": ("late_8", "canonical_6"),
    "all_three": ("late_8", "late_6", "canonical_6"),
}


def balanced_arm_prediction(
    scores: np.ndarray,
    y: np.ndarray,
    val_idx: np.ndarray,
    records: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arm_mask = np.isin(y[val_idx], ARM_CLASSES)
    arm_idx = val_idx[arm_mask]
    arm_scores = scores[arm_mask]
    target = (y[arm_idx] == ARM_CLASSES[1]).astype(np.int64)
    prediction = np.empty(len(arm_idx), dtype=np.int64)
    groups: dict[tuple[str, int], list[int]] = {}
    for position, index in enumerate(arm_idx):
        record = records[int(index)]
        key = (str(record["subject_id"]), int(record["run_id"]))
        groups.setdefault(key, []).append(position)
    for positions in groups.values():
        positions = np.asarray(positions, dtype=np.int64)
        difference = arm_scores[positions, 1] - arm_scores[positions, 0]
        order = np.argsort(difference, kind="stable")
        prediction[positions[order[: len(order) // 2]]] = 0
        prediction[positions[order[len(order) // 2 :]]] = 1
    return prediction, arm_scores.argmax(axis=1).astype(np.int64), target


def choose_configuration(
    representations: dict[str, np.ndarray],
    y: np.ndarray,
    inner: list[dict],
    feature_counts: list[int],
    shrinkages: list[float],
) -> tuple[dict, list[dict]]:
    rows = []
    for split in inner:
        for representation_name, x in representations.items():
            ranking = rank_pair_features(x, y, split["train_idx"], ARM_CLASSES)
            for feature_count in feature_counts:
                count = min(feature_count, x.shape[1])
                for shrinkage in shrinkages:
                    scores = full_lda_scores(
                        x,
                        y,
                        split["train_idx"],
                        split["val_idx"],
                        ranking[:count],
                        ARM_CLASSES,
                        shrinkage,
                    )
                    rows.append(
                        {
                            "split": split["split"],
                            "representation": representation_name,
                            "feature_count": count,
                            "shrinkage": shrinkage,
                            "accuracy": pair_accuracy(
                                scores, y, split["val_idx"], ARM_CLASSES
                            ),
                        }
                    )
    diagnostics = []
    configurations = sorted(
        {
            (row["representation"], row["feature_count"], row["shrinkage"])
            for row in rows
        }
    )
    for representation, feature_count, shrinkage in configurations:
        diagnostics.append(
            {
                "representation": representation,
                "feature_count": feature_count,
                "shrinkage": shrinkage,
                "mean_inner_accuracy": float(
                    np.mean(
                        [
                            row["accuracy"]
                            for row in rows
                            if row["representation"] == representation
                            and row["feature_count"] == feature_count
                            and row["shrinkage"] == shrinkage
                        ]
                    )
                ),
            }
        )
    representation_order = {name: index for index, name in enumerate(REPRESENTATIONS)}
    selected = min(
        diagnostics,
        key=lambda row: (
            -row["mean_inner_accuracy"],
            representation_order[row["representation"]],
            row["feature_count"],
            row["shrinkage"],
        ),
    )
    return selected, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a multi-window arm representation inside nested subject folds."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--feature-shape", default="24,24,24")
    parser.add_argument("--transform", default="smooth_3")
    parser.add_argument("--feature-counts", nargs="*", type=int, default=[512, 1024])
    parser.add_argument("--shrinkages", nargs="*", type=float, default=[0.5, 0.75])
    parser.add_argument("--subject-fold-count", type=int, default=6)
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument("--subject-seeds", nargs="*", type=int, default=[])
    args = parser.parse_args()

    shape = tuple(int(value) for value in args.feature_shape.split(","))
    features, y, records = load_checkpoints(
        Path(args.checkpoint_dir), list(WINDOWS.values())
    )
    transformed = {}
    detrend_rows = {}
    for short_name, window_name in WINDOWS.items():
        centered = center_by_subject_run(features[window_name], records)
        detrended, rows = temporal_detrend_by_subject_run(centered, records, degree=1)
        transformed[short_name], _ = transform_scale(
            detrended, shape, args.transform, 64
        )
        detrend_rows[short_name] = rows
    representations = {
        name: (
            transformed[parts[0]]
            if len(parts) == 1
            else np.concatenate([transformed[part] for part in parts], axis=1)
        )
        for name, parts in REPRESENTATIONS.items()
    }

    rows = []
    hyperparameters = []
    for split in outer_splits(
        records, "subject", args.subject_fold_count, args.subject_seeds
    ):
        print(f"evaluating {split['split']}", flush=True)
        inner = inner_splits(
            records,
            split["train_idx"],
            split["family"],
            args.inner_subject_fold_count,
        )
        selected, diagnostics = choose_configuration(
            representations,
            y,
            inner,
            sorted(set(args.feature_counts)),
            sorted(set(args.shrinkages)),
        )
        baseline_selected, _ = choose_configuration(
            {"late_8": representations["late_8"]},
            y,
            inner,
            sorted(set(args.feature_counts)),
            sorted(set(args.shrinkages)),
        )
        x = representations[selected["representation"]]
        ranking = rank_pair_features(x, y, split["train_idx"], ARM_CLASSES)
        selected_features = ranking[: selected["feature_count"]]
        scores = full_lda_scores(
            x,
            y,
            split["train_idx"],
            split["val_idx"],
            selected_features,
            ARM_CLASSES,
            selected["shrinkage"],
        )
        balanced, independent, target = balanced_arm_prediction(
            scores, y, split["val_idx"], records
        )
        baseline_x = representations["late_8"]
        baseline_ranking = rank_pair_features(
            baseline_x, y, split["train_idx"], ARM_CLASSES
        )
        baseline_features = baseline_ranking[: baseline_selected["feature_count"]]
        baseline_scores = full_lda_scores(
            baseline_x,
            y,
            split["train_idx"],
            split["val_idx"],
            baseline_features,
            ARM_CLASSES,
            baseline_selected["shrinkage"],
        )
        baseline_balanced, baseline_independent, _ = balanced_arm_prediction(
            baseline_scores, y, split["val_idx"], records
        )
        val_subjects = np.asarray(
            [
                str(records[int(index)]["subject_id"])
                for index in split["val_idx"]
                if y[int(index)] in ARM_CLASSES
            ]
        )
        subject_accuracy = {
            subject: {
                "balanced": float(
                    np.mean(balanced[val_subjects == subject] == target[val_subjects == subject])
                ),
                "independent": float(
                    np.mean(
                        independent[val_subjects == subject] == target[val_subjects == subject]
                    )
                ),
            }
            for subject in sorted(set(val_subjects.tolist()))
        }
        rows.append(
            {
                "split": split["split"],
                "selected_representation": selected["representation"],
                "selected_feature_count": selected["feature_count"],
                "selected_shrinkage": selected["shrinkage"],
                "balanced_accuracy": float(np.mean(balanced == target)),
                "independent_accuracy": float(np.mean(independent == target)),
                "baseline_feature_count": baseline_selected["feature_count"],
                "baseline_shrinkage": baseline_selected["shrinkage"],
                "baseline_balanced_accuracy": float(
                    np.mean(baseline_balanced == target)
                ),
                "baseline_independent_accuracy": float(
                    np.mean(baseline_independent == target)
                ),
                "subject_accuracy": subject_accuracy,
            }
        )
        hyperparameters.append(
            {
                "split": split["split"],
                "selected": selected,
                "inner_diagnostics": diagnostics,
                "top_feature_indices": selected_features[:20].tolist(),
            }
        )

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "windows": WINDOWS,
        "representations": REPRESENTATIONS,
        "transform": args.transform,
        "feature_counts": sorted(set(args.feature_counts)),
        "shrinkages": sorted(set(args.shrinkages)),
        "subject_seeds": args.subject_seeds,
        "mean_temporal_variance_fraction": {
            name: float(np.mean([row["temporal_variance_fraction"] for row in group_rows]))
            for name, group_rows in detrend_rows.items()
        },
        "rows": rows,
        "hyperparameters": hyperparameters,
        "summary": {
            "split_count": len(rows),
            "mean_balanced_accuracy": float(
                np.mean([row["balanced_accuracy"] for row in rows])
            ),
            "mean_independent_accuracy": float(
                np.mean([row["independent_accuracy"] for row in rows])
            ),
            "mean_baseline_balanced_accuracy": float(
                np.mean([row["baseline_balanced_accuracy"] for row in rows])
            ),
            "mean_baseline_independent_accuracy": float(
                np.mean([row["baseline_independent_accuracy"] for row in rows])
            ),
            "mean_selected_minus_baseline_balanced": float(
                np.mean(
                    [
                        row["balanced_accuracy"] - row["baseline_balanced_accuracy"]
                        for row in rows
                    ]
                )
            ),
            "selected_representation_counts": {
                name: sum(row["selected_representation"] == name for row in rows)
                for name in REPRESENTATIONS
            },
        },
        "note": (
            "Representation, selected feature count, and covariance shrinkage are chosen only "
            "inside each outer training-subject partition."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(json.dumps({"out_json": args.out_json, "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
