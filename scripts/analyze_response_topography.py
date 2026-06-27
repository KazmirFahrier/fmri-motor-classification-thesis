#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    apply_balanced_assignment,
    center_by_subject_run,
    l2_normalize,
)
from run_detrended_pair_feature_selection import load_checkpoints
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


PERMUTATIONS = list(itertools.permutations(range(4)))


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def centroid_matrix(x_norm: np.ndarray, y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    centroids = np.stack(
        [x_norm[indices][y[indices] == class_id].mean(axis=0) for class_id in range(4)]
    )
    return l2_normalize(centroids.astype(np.float32)).astype(np.float64)


def best_permutation(similarity: np.ndarray) -> tuple[tuple[int, ...], float]:
    permutation = max(
        PERMUTATIONS,
        key=lambda candidate: float(
            np.mean([similarity[class_id, candidate[class_id]] for class_id in range(4)])
        ),
    )
    return permutation, float(
        np.mean([similarity[class_id, permutation[class_id]] for class_id in range(4)])
    )


def edge_mask(shape: tuple[int, int, int], width: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[:width] = True
    mask[-width:] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    mask[:, :, :width] = True
    mask[:, :, -width:] = True
    return mask.reshape(-1)


def energy_summary(raw_centroids: np.ndarray, shape: tuple[int, int, int], width: int) -> dict:
    energy = np.mean(raw_centroids.astype(np.float64) ** 2, axis=0)
    total = max(float(np.sum(energy)), 1e-12)
    probability = energy / total
    nonzero = probability > 0
    entropy = float(-np.sum(probability[nonzero] * np.log(probability[nonzero])))
    normalized_entropy = entropy / np.log(len(probability))
    sorted_energy = np.sort(energy)[::-1]
    result = {
        "normalized_energy_entropy": normalized_entropy,
        "effective_voxel_fraction": float(np.exp(entropy) / len(probability)),
        "edge_energy_fraction": float(np.sum(energy[edge_mask(shape, width)]) / total),
    }
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, int(round(fraction * len(energy))))
        result[f"top_{int(fraction * 100)}pct_energy_fraction"] = float(
            np.sum(sorted_energy[:count]) / total
        )
    return result


def repeat_geometry(x_norm: np.ndarray, y: np.ndarray, indices: np.ndarray) -> dict:
    within = []
    between = []
    for left_position, left_index in enumerate(indices):
        for right_index in indices[left_position + 1 :]:
            similarity = float(x_norm[left_index] @ x_norm[right_index])
            target = within if y[left_index] == y[right_index] else between
            target.append(similarity)
    return {
        "same_class_repeat_cosine": float(np.mean(within)),
        "different_class_cosine": float(np.mean(between)),
        "same_minus_different_cosine": float(np.mean(within) - np.mean(between)),
    }


def summarize_run(
    subject: str,
    run_id: int,
    indices: np.ndarray,
    x: np.ndarray,
    x_norm: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    group_centroids: np.ndarray,
    own_other_centroids: np.ndarray,
    shape: tuple[int, int, int],
    edge_width: int,
) -> dict:
    run_centroids = centroid_matrix(x_norm, y, indices)
    raw_centroids = np.stack(
        [x[indices][y[indices] == class_id].mean(axis=0) for class_id in range(4)]
    )
    group_similarity = run_centroids @ group_centroids.T
    group_permutation, group_best = best_permutation(group_similarity)
    own_similarity = run_centroids @ own_other_centroids.T
    own_permutation, own_best = best_permutation(own_similarity)
    group_scores = x_norm[indices].astype(np.float64) @ group_centroids.T
    group_prediction = apply_balanced_assignment(group_scores, indices, records)
    centroid_rms = np.sqrt(np.mean(raw_centroids.astype(np.float64) ** 2, axis=1))
    result = {
        "subject": subject,
        "run_id": run_id,
        "event_count": len(indices),
        "mean_class_centroid_rms": float(np.mean(centroid_rms)),
        "min_class_centroid_rms": float(np.min(centroid_rms)),
        "max_class_centroid_rms": float(np.max(centroid_rms)),
        "group_identity_similarity": float(np.mean(np.diag(group_similarity))),
        "group_best_similarity": group_best,
        "group_permutation_gain": group_best - float(np.mean(np.diag(group_similarity))),
        "group_best_permutation": list(group_permutation),
        "group_identity_class_fraction": float(
            np.mean(np.asarray(group_permutation) == np.arange(4))
        ),
        "group_balanced_accuracy": float(np.mean(y[indices] == group_prediction)),
        "own_other_run_identity_similarity": float(np.mean(np.diag(own_similarity))),
        "own_other_run_best_similarity": own_best,
        "own_other_run_permutation_gain": own_best
        - float(np.mean(np.diag(own_similarity))),
        "own_other_run_best_permutation": list(own_permutation),
        "own_other_run_identity_class_fraction": float(
            np.mean(np.asarray(own_permutation) == np.arange(4))
        ),
    }
    result.update(repeat_geometry(x_norm, y, indices))
    result.update(energy_summary(raw_centroids, shape, edge_width))
    return result


def aggregate_subject(subject: str, rows: list[dict]) -> dict:
    metric_names = [
        key
        for key, value in rows[0].items()
        if key not in {"subject", "run_id", "event_count", "group_best_permutation", "own_other_run_best_permutation"}
        and isinstance(value, (int, float))
    ]
    result = {"subject": subject, "run_count": len(rows), "runs": rows}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        result[f"mean_{metric}"] = float(np.mean(values))
        result[f"min_{metric}"] = float(np.min(values))
        result[f"max_{metric}"] = float(np.max(values))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate event-response strength, repeatability, concentration, and topographic alignment."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--window-name", default="offset_3_length_8")
    parser.add_argument("--feature-shape", default="24,24,24")
    parser.add_argument("--edge-width", type=int, default=2)
    parser.add_argument(
        "--focus-subjects",
        nargs="*",
        default=["sub-30", "sub-42", "sub-52", "sub-62"],
    )
    args = parser.parse_args()

    shape = tuple(int(value) for value in args.feature_shape.split(","))
    features, y, records = load_checkpoints(Path(args.checkpoint_dir), [args.window_name])
    x = center_by_subject_run(features[args.window_name], records)
    x, detrend_rows = temporal_detrend_by_subject_run(x, records, degree=1)
    x = x.astype(np.float32)
    if x.shape[1] != int(np.prod(shape)):
        raise ValueError(f"Feature shape {shape} does not match {x.shape[1]} voxels.")
    x_norm = l2_normalize(x)
    subjects = np.asarray([str(row["subject_id"]) for row in records])
    run_ids = np.asarray([int(row["run_id"]) for row in records], dtype=np.int64)

    subject_rows = []
    for subject in sorted(set(subjects.tolist())):
        print(f"summarizing {subject}", flush=True)
        subject_indices = np.flatnonzero(subjects == subject)
        group_indices = np.flatnonzero(subjects != subject)
        group_centroids = centroid_matrix(x_norm, y, group_indices)
        rows = []
        for run_id in sorted(set(run_ids[subject_indices].tolist())):
            indices = np.flatnonzero((subjects == subject) & (run_ids == run_id))
            own_other_indices = np.flatnonzero(
                (subjects == subject) & (run_ids != run_id)
            )
            own_other_centroids = centroid_matrix(x_norm, y, own_other_indices)
            rows.append(
                summarize_run(
                    subject,
                    run_id,
                    indices,
                    x,
                    x_norm,
                    y,
                    records,
                    group_centroids,
                    own_other_centroids,
                    shape,
                    args.edge_width,
                )
            )
        subject_rows.append(aggregate_subject(subject, rows))

    ranked_repeatability = sorted(
        subject_rows, key=lambda row: row["mean_same_class_repeat_cosine"]
    )
    ranked_own_alignment = sorted(
        subject_rows, key=lambda row: row["mean_own_other_run_identity_similarity"]
    )
    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "window_name": args.window_name,
        "feature_shape": shape,
        "edge_width": args.edge_width,
        "subject_count": len(subject_rows),
        "mean_temporal_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in detrend_rows])
        ),
        "subjects": subject_rows,
        "lowest_repeatability": ranked_repeatability[:15],
        "lowest_own_run_alignment": ranked_own_alignment[:15],
        "focus_subjects": [
            row for row in subject_rows if row["subject"] in set(args.focus_subjects)
        ],
        "note": (
            "Leave-subject-out cohort centroids are a data-driven functional consensus, not an "
            "anatomical atlas. Edge energy uses the outer feature-grid shell and is a coarse "
            "artifact indicator. All metrics use run-centered, linearly detrended event maps."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "lowest_repeatability": [
                    {
                        "subject": row["subject"],
                        "repeat_cosine": row["mean_same_class_repeat_cosine"],
                        "centroid_rms": row["mean_mean_class_centroid_rms"],
                        "own_alignment": row["mean_own_other_run_identity_similarity"],
                        "group_accuracy": row["mean_group_balanced_accuracy"],
                        "edge_energy": row["mean_edge_energy_fraction"],
                    }
                    for row in ranked_repeatability[:15]
                ],
                "focus_subjects": [
                    {
                        "subject": row["subject"],
                        "repeat_cosine": row["mean_same_class_repeat_cosine"],
                        "centroid_rms": row["mean_mean_class_centroid_rms"],
                        "own_alignment": row["mean_own_other_run_identity_similarity"],
                        "group_accuracy": row["mean_group_balanced_accuracy"],
                        "edge_energy": row["mean_edge_energy_fraction"],
                        "effective_voxel_fraction": row[
                            "mean_effective_voxel_fraction"
                        ],
                    }
                    for row in result["focus_subjects"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
