#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import center_by_subject_run, l2_normalize
from run_detrended_pair_feature_selection import load_checkpoints
from run_temporal_detrended_event_adaptation import temporal_detrend_by_subject_run


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def optimal_assignment(scores: np.ndarray) -> tuple[np.ndarray, float]:
    size = scores.shape[0]
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, tuple())}
    for row_idx in range(size):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (score_so_far, path) in states.items():
            for col_idx in range(size):
                if mask & (1 << col_idx):
                    continue
                value = float(scores[row_idx, col_idx])
                if not np.isfinite(value):
                    continue
                next_mask = mask | (1 << col_idx)
                candidate = (score_so_far + value, path + (col_idx,))
                if next_mask not in next_states or candidate[0] > next_states[next_mask][0]:
                    next_states[next_mask] = candidate
        states = next_states
    final_mask = (1 << size) - 1
    if final_mask not in states:
        raise ValueError("No complete event assignment exists.")
    total, assignment = states[final_mask]
    return np.asarray(assignment, dtype=np.int64), float(total / size)


def sorted_run_indices(
    subject_indices: np.ndarray,
    run_id: int,
    records: list[dict],
) -> np.ndarray:
    values = [idx for idx in subject_indices if int(records[idx]["run_id"]) == run_id]
    return np.asarray(
        sorted(values, key=lambda idx: int(records[idx]["event_start"])),
        dtype=np.int64,
    )


def analyze_subject(
    subject: str,
    x_norm: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    subject_indices: np.ndarray,
) -> dict:
    run_ids = sorted(set(int(records[idx]["run_id"]) for idx in subject_indices))
    run_indices = {
        run_id: sorted_run_indices(subject_indices, run_id, records)
        for run_id in run_ids
    }
    pair_rows = []
    for first_run, second_run in itertools.combinations(run_ids, 2):
        first_idx = run_indices[first_run]
        second_idx = run_indices[second_run]
        if len(first_idx) != len(second_idx):
            raise ValueError(f"Run event-count mismatch for {subject}: {first_run}, {second_run}")
        similarity = x_norm[first_idx].astype(np.float64) @ x_norm[second_idx].astype(np.float64).T
        assignment, optimal_score = optimal_assignment(similarity)
        class_match = y[first_idx] == y[second_idx][assignment]
        ordinal_match = assignment == np.arange(len(assignment))

        class_scores = similarity.copy()
        class_scores[y[first_idx, None] != y[second_idx][None, :]] = -np.inf
        class_assignment, class_score = optimal_assignment(class_scores)
        ordinal_score = float(np.mean(np.diag(similarity)))
        pair_rows.append(
            {
                "first_run": int(first_run),
                "second_run": int(second_run),
                "optimal_assignment": assignment.tolist(),
                "optimal_similarity": optimal_score,
                "optimal_class_match_fraction": float(np.mean(class_match)),
                "optimal_ordinal_match_fraction": float(np.mean(ordinal_match)),
                "class_constrained_assignment": class_assignment.tolist(),
                "class_constrained_similarity": class_score,
                "class_constraint_similarity_gap": optimal_score - class_score,
                "ordinal_identity_similarity": ordinal_score,
                "ordinal_identity_similarity_gap": optimal_score - ordinal_score,
            }
        )
    return {
        "subject": subject,
        "run_pair_count": len(pair_rows),
        "mean_optimal_class_match_fraction": float(
            np.mean([row["optimal_class_match_fraction"] for row in pair_rows])
        ),
        "mean_optimal_ordinal_match_fraction": float(
            np.mean([row["optimal_ordinal_match_fraction"] for row in pair_rows])
        ),
        "mean_class_constraint_similarity_gap": float(
            np.mean([row["class_constraint_similarity_gap"] for row in pair_rows])
        ),
        "mean_ordinal_identity_similarity_gap": float(
            np.mean([row["ordinal_identity_similarity_gap"] for row in pair_rows])
        ),
        "run_pair_rows": pair_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cross-run event matching by motor class versus event ordinal."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--window-name", default="offset_3_length_8")
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--focus-subjects",
        nargs="*",
        default=["sub-42", "sub-52", "sub-68", "sub-17", "sub-20", "sub-30", "sub-62"],
    )
    args = parser.parse_args()

    features, y, records = load_checkpoints(Path(args.checkpoint_dir), [args.window_name])
    centered = center_by_subject_run(features[args.window_name], records)
    detrended, group_rows = temporal_detrend_by_subject_run(centered, records, degree=1)
    x_norm = l2_normalize(detrended.astype(np.float32))
    subjects = np.asarray([str(record["subject_id"]) for record in records])
    rows = []
    for subject in sorted(set(subjects.tolist())):
        rows.append(
            analyze_subject(
                subject,
                x_norm,
                y,
                records,
                np.flatnonzero(subjects == subject),
            )
        )

    ranked_class = sorted(rows, key=lambda row: row["mean_optimal_class_match_fraction"])
    ranked_ordinal = sorted(
        rows,
        key=lambda row: -row["mean_optimal_ordinal_match_fraction"],
    )
    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "window_name": args.window_name,
        "subject_count": len(rows),
        "chance_class_match_fraction": 0.25,
        "chance_ordinal_match_fraction": 0.125,
        "mean_linear_time_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in group_rows])
        ),
        "rows": rows,
        "lowest_class_correspondence": ranked_class[:15],
        "highest_ordinal_correspondence": ranked_ordinal[:15],
        "focus_subjects": [row for row in rows if row["subject"] in set(args.focus_subjects)],
        "note": (
            "Optimal event matching is an oracle forensic diagnostic. Class match above 0.25 "
            "indicates cross-run motor-class correspondence; ordinal match above 0.125 indicates "
            "residual run-position correspondence. Neither is deployable classification."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "lowest_class_correspondence": [
                    {key: row[key] for key in row if key != "run_pair_rows"}
                    for row in ranked_class[:15]
                ],
                "highest_ordinal_correspondence": [
                    {key: row[key] for key in row if key != "run_pair_rows"}
                    for row in ranked_ordinal[:15]
                ],
                "focus_subjects": [
                    {key: row[key] for key in row if key != "run_pair_rows"}
                    for row in result["focus_subjects"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
