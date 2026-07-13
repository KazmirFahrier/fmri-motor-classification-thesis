#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import center_by_subject_run, l2_normalize
from run_detrended_pair_feature_selection import load_checkpoints
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


CLASS_NAMES = [
    "Left leg movements",
    "Right leg movements",
    "Forearm movements",
    "Upper arm movements",
]


def class_centroids(
    x_norm: np.ndarray, y: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    centroids = np.stack(
        [x_norm[indices][y[indices] == class_id].mean(axis=0) for class_id in range(4)]
    )
    return l2_normalize(centroids.astype(np.float32)).astype(np.float64)


def correct_margin(similarities: np.ndarray, class_id: int) -> float:
    alternatives = np.delete(similarities, class_id)
    return float(similarities[class_id] - np.max(alternatives))


def summarize_class(
    class_id: int,
    indices: np.ndarray,
    x: np.ndarray,
    x_norm: np.ndarray,
    own_centroids: np.ndarray,
    group_centroids: np.ndarray,
) -> dict:
    class_patterns = x_norm[indices].astype(np.float64)
    raw_patterns = x[indices].astype(np.float64)
    run_centroid = l2_normalize(class_patterns.mean(axis=0, keepdims=True))[0]
    own_similarities = run_centroid @ own_centroids.T
    group_similarities = run_centroid @ group_centroids.T
    norms = np.linalg.norm(raw_patterns, axis=1)
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "pair": "leg" if class_id < 2 else "arm",
        "event_count": len(indices),
        "repeat_cosine": float(class_patterns[0] @ class_patterns[1]),
        "event_pattern_norms": norms.tolist(),
        "event_pattern_norm_ratio": float(np.min(norms) / max(float(np.max(norms)), 1e-12)),
        "own_other_run_identity_similarity": float(own_similarities[class_id]),
        "own_other_run_correct_margin": correct_margin(own_similarities, class_id),
        "own_other_run_best_class": int(np.argmax(own_similarities)),
        "group_identity_similarity": float(group_similarities[class_id]),
        "group_correct_margin": correct_margin(group_similarities, class_id),
        "group_best_class": int(np.argmax(group_similarities)),
    }


def summarize_run(class_rows: list[dict]) -> dict:
    repeats = np.asarray([row["repeat_cosine"] for row in class_rows])
    own_margins = np.asarray(
        [row["own_other_run_correct_margin"] for row in class_rows]
    )
    group_margins = np.asarray([row["group_correct_margin"] for row in class_rows])
    return {
        "mean_repeat_cosine": float(np.mean(repeats)),
        "minimum_repeat_cosine": float(np.min(repeats)),
        "negative_repeat_class_count": int(np.sum(repeats < 0)),
        "leg_mean_repeat_cosine": float(np.mean(repeats[:2])),
        "arm_mean_repeat_cosine": float(np.mean(repeats[2:])),
        "mean_own_other_run_correct_margin": float(np.mean(own_margins)),
        "negative_own_margin_class_count": int(np.sum(own_margins < 0)),
        "mean_group_correct_margin": float(np.mean(group_margins)),
        "negative_group_margin_class_count": int(np.sum(group_margins < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose weak runs into class-level repetition and template geometry."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--window-name", default="offset_3_length_8")
    parser.add_argument(
        "--focus-subjects",
        nargs="*",
        default=["sub-20", "sub-54", "sub-63", "sub-30", "sub-62"],
    )
    args = parser.parse_args()

    features, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.window_name]
    )
    x = center_by_subject_run(features[args.window_name], records)
    x, detrend_rows = temporal_detrend_by_subject_run(x, records, degree=1)
    x = x.astype(np.float32)
    x_norm = l2_normalize(x)
    subjects = np.asarray([str(row["subject_id"]) for row in records])
    run_ids = np.asarray([int(row["run_id"]) for row in records], dtype=np.int64)

    subject_results = []
    for subject in args.focus_subjects:
        if subject not in subjects:
            raise ValueError(f"Subject not found: {subject}")
        subject_indices = np.flatnonzero(subjects == subject)
        group_indices = np.flatnonzero(subjects != subject)
        group_centroids = class_centroids(x_norm, y, group_indices)
        runs = []
        for run_id in sorted(set(run_ids[subject_indices].tolist())):
            run_mask = (subjects == subject) & (run_ids == run_id)
            own_other_indices = np.flatnonzero((subjects == subject) & (run_ids != run_id))
            own_centroids = class_centroids(x_norm, y, own_other_indices)
            class_rows = []
            for class_id in range(4):
                indices = np.flatnonzero(run_mask & (y == class_id))
                if len(indices) != 2:
                    raise ValueError(
                        f"Expected two events for {subject} run {run_id} class {class_id}; "
                        f"found {len(indices)}."
                    )
                class_rows.append(
                    summarize_class(
                        class_id,
                        indices,
                        x,
                        x_norm,
                        own_centroids,
                        group_centroids,
                    )
                )
            runs.append(
                {
                    "subject": subject,
                    "run_id": run_id,
                    **summarize_run(class_rows),
                    "classes": class_rows,
                }
            )
        subject_results.append({"subject": subject, "runs": runs})

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "window_name": args.window_name,
        "focus_subjects": args.focus_subjects,
        "mean_temporal_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in detrend_rows])
        ),
        "subjects": subject_results,
        "note": (
            "This is a label-aware forensic decomposition, not a deployable QC rule. "
            "Features are subject-run centered and linearly detrended before class-level "
            "repeat and template geometry is measured."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    compact = [
        {
            "subject": subject["subject"],
            "runs": [
                {
                    key: run[key]
                    for key in (
                        "run_id",
                        "mean_repeat_cosine",
                        "minimum_repeat_cosine",
                        "negative_repeat_class_count",
                        "leg_mean_repeat_cosine",
                        "arm_mean_repeat_cosine",
                        "mean_own_other_run_correct_margin",
                        "mean_group_correct_margin",
                    )
                }
                for run in subject["runs"]
            ],
        }
        for subject in subject_results
    ]
    print(json.dumps({"out_json": args.out_json, "subjects": compact}, indent=2))


if __name__ == "__main__":
    main()
