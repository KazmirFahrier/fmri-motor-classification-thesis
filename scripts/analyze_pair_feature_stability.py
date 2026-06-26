#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PAIR_NAMES = ["leg", "arm"]


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def summarize_maps(hyperparameters: list[dict], feature_shape: tuple[int, int, int]) -> dict:
    result = {}
    feature_total = int(np.prod(feature_shape))
    for pair_name in PAIR_NAMES:
        selected_sets = [
            set(row["selected_feature_indices"][pair_name])
            for row in hyperparameters
        ]
        counts = np.zeros(feature_total, dtype=np.int64)
        for selected in selected_sets:
            counts[list(selected)] += 1
        jaccards = [
            len(first & second) / max(1, len(first | second))
            for first, second in itertools.combinations(selected_sets, 2)
        ]
        order = np.argsort(-counts, kind="stable")
        top_features = []
        for feature_idx in order[:100]:
            if counts[feature_idx] == 0:
                break
            top_features.append(
                {
                    "feature_index": int(feature_idx),
                    "grid_coordinate": list(np.unravel_index(int(feature_idx), feature_shape)),
                    "selected_fold_count": int(counts[feature_idx]),
                    "selected_fold_fraction": float(counts[feature_idx] / len(selected_sets)),
                }
            )
        result[pair_name] = {
            "fold_count": len(selected_sets),
            "mean_selected_feature_count": float(np.mean([len(value) for value in selected_sets])),
            "min_selected_feature_count": int(min(len(value) for value in selected_sets)),
            "max_selected_feature_count": int(max(len(value) for value in selected_sets)),
            "mean_pairwise_jaccard": float(np.mean(jaccards)),
            "median_pairwise_jaccard": float(np.median(jaccards)),
            "features_selected_in_at_least_half_of_folds": int(np.sum(counts >= len(selected_sets) / 2)),
            "features_selected_in_at_least_80_percent_of_folds": int(
                np.sum(counts >= 0.8 * len(selected_sets))
            ),
            "top_features": top_features,
        }
    return result


def summarize_subject_deltas(rows: list[dict]) -> list[dict]:
    by_split_rule = {
        (row["split"], row["prediction_rule"]): row
        for row in rows
    }
    by_subject: dict[str, list[dict]] = defaultdict(list)
    splits = sorted(set(row["split"] for row in rows))
    for split in splits:
        baseline = by_split_rule[(split, "flat_all_features_balanced")]
        selected = by_split_rule[(split, "selected_pair_fused_balanced")]
        for subject, baseline_metrics in baseline["subject_metrics"].items():
            selected_metrics = selected["subject_metrics"][subject]
            by_subject[subject].append(
                {
                    "split": split,
                    "baseline_balanced_accuracy": baseline_metrics["balanced_accuracy"],
                    "selected_balanced_accuracy": selected_metrics["balanced_accuracy"],
                    "delta": (
                        selected_metrics["balanced_accuracy"]
                        - baseline_metrics["balanced_accuracy"]
                    ),
                }
            )

    summaries = []
    for subject, subject_rows in sorted(by_subject.items()):
        deltas = [row["delta"] for row in subject_rows]
        summaries.append(
            {
                "subject": subject,
                "repeat_count": len(subject_rows),
                "mean_baseline_balanced_accuracy": float(
                    np.mean([row["baseline_balanced_accuracy"] for row in subject_rows])
                ),
                "mean_selected_balanced_accuracy": float(
                    np.mean([row["selected_balanced_accuracy"] for row in subject_rows])
                ),
                "mean_delta": float(np.mean(deltas)),
                "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)),
                "improved_repeats": int(np.sum(np.asarray(deltas) > 0)),
                "tied_repeats": int(np.sum(np.asarray(deltas) == 0)),
                "rows": subject_rows,
            }
        )
    return sorted(summaries, key=lambda row: row["mean_delta"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize repeated-CV pair-feature map stability and per-subject deltas."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--feature-shape", nargs=3, type=int, default=[24, 24, 24])
    args = parser.parse_args()

    source = json.loads(Path(args.input_json).read_text())
    result = {
        "input_json": args.input_json,
        "feature_shape": args.feature_shape,
        "map_stability": summarize_maps(source["hyperparameters"], tuple(args.feature_shape)),
        "subject_deltas": summarize_subject_deltas(source["rows"]),
        "note": (
            "Grid coordinates refer to the resized 24x24x24 feature array and are not anatomical "
            "or MNI coordinates. Anatomical claims require affine-aware reconstruction."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "map_stability": result["map_stability"],
                "worst_subject_deltas": result["subject_deltas"][:10],
                "best_subject_deltas": result["subject_deltas"][-10:],
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
