#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CLASS_NAMES = [
    "Left leg movements",
    "Right leg movements",
    "Forearm movements",
    "Upper arm movements",
]

EVENT_METRICS = [
    "normalized_dvars_percent",
    "phase_framewise_mm",
    "phase_absolute_mm",
    "center_framewise_mm",
    "center_absolute_mm",
]

RUN_METRICS = {
    "dvars_p95_percent": lambda row: row["normalized_dvars_percent"]["p95"],
    "phase_fd_p95_mm": lambda row: row["phase_translation"]["p95_framewise_mm"],
    "phase_absolute_p95_mm": lambda row: row["phase_translation"]["p95_absolute_mm"],
    "center_fd_p95_mm": lambda row: row["signal_center"]["p95_framewise_mm"],
    "center_absolute_p95_mm": lambda row: row["signal_center"]["p95_absolute_mm"],
}


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def normalize_rows(values: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    return np.nan_to_num(values / denominator)


def standardize_columns(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return np.nan_to_num((values - mean) / np.maximum(std, 1e-8))


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(first: list[float], second: list[float]) -> dict:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"count": len(x), "pearson": None, "rank": None}
    return {
        "count": len(x),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "rank": float(np.corrcoef(rank_values(x), rank_values(y))[0, 1]),
    }


def event_features(run: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = run["event_motion"]["events"]
    features = np.asarray(
        [
            [
                row[f"{prefix}{metric}"]
                for metric in EVENT_METRICS
                for prefix in ("mean_", "max_")
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    labels = np.asarray(
        [CLASS_NAMES.index(row["trial_type"]) for row in rows],
        dtype=np.int64,
    )
    return standardize_columns(features), labels


def class_centroids(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return normalize_rows(
        np.stack([features[labels == class_id].mean(axis=0) for class_id in range(4)])
    )


def confusion_matrix(labels: np.ndarray, predicted: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((4, 4), dtype=np.int64)
    for target, prediction in zip(labels, predicted):
        matrix[int(target), int(prediction)] += 1
    return matrix.tolist()


def leave_one_run_motion_accuracy(runs: list[dict]) -> dict:
    prepared = [event_features(run) for run in runs]
    labels = []
    predicted = []
    per_run = []
    for test_index, (test_x, test_y) in enumerate(prepared):
        train_x = np.concatenate(
            [values for index, (values, _) in enumerate(prepared) if index != test_index]
        )
        train_y = np.concatenate(
            [target for index, (_, target) in enumerate(prepared) if index != test_index]
        )
        centroids = class_centroids(train_x, train_y)
        test_prediction = np.argmax(normalize_rows(test_x) @ centroids.T, axis=1)
        labels.extend(test_y.tolist())
        predicted.extend(test_prediction.tolist())
        per_run.append(
            {
                "run_id": runs[test_index]["run_id"],
                "accuracy": float(np.mean(test_y == test_prediction)),
            }
        )
    y = np.asarray(labels, dtype=np.int64)
    y_hat = np.asarray(predicted, dtype=np.int64)
    return {
        "accuracy": float(np.mean(y == y_hat)),
        "chance_accuracy": 0.25,
        "confusion_matrix": confusion_matrix(y, y_hat),
        "per_run": per_run,
    }


def compare_motion_templates(first: dict, second: dict) -> dict:
    first_x, first_y = event_features(first)
    second_x, second_y = event_features(second)
    first_centroids = class_centroids(first_x, first_y)
    second_centroids = class_centroids(second_x, second_y)
    similarities = first_centroids @ second_centroids.T
    identity_similarity = float(np.mean(np.diag(similarities)))
    permutations = list(itertools.permutations(range(4)))
    scores = [
        float(np.mean([similarities[class_id, permutation[class_id]] for class_id in range(4)]))
        for permutation in permutations
    ]
    best_index = int(np.argmax(scores))
    best = permutations[best_index]
    return {
        "subject": first["subject"],
        "first_run": first["run_id"],
        "second_run": second["run_id"],
        "motion_identity_similarity": identity_similarity,
        "motion_best_similarity": scores[best_index],
        "motion_permutation_gain": scores[best_index] - identity_similarity,
        "motion_best_permutation": list(best),
        "motion_identity_class_fraction": float(
            np.mean(np.asarray(best) == np.arange(4))
        ),
        "motion_identity_is_optimal": best == (0, 1, 2, 3),
    }


def load_runs(paths: list[Path]) -> list[dict]:
    by_label = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for run in payload["runs"]:
            by_label[run["label"]] = run
    return sorted(by_label.values(), key=lambda row: (row["subject"], row["run_id"]))


def correspondence_index(path: Path) -> dict[tuple[str, int, int], dict]:
    payload = json.loads(path.read_text())
    index = {}
    for subject_row in payload["rows"]:
        subject = subject_row["subject"]
        for pair in subject_row["run_pair_rows"]:
            index[(subject, pair["first_run"], pair["second_run"])] = pair
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether image-derived motion explains event geometry or class correspondence."
    )
    parser.add_argument("--motion-json", action="append", required=True)
    parser.add_argument("--correspondence-json", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    runs = load_runs([Path(value) for value in args.motion_json])
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_subject[run["subject"]].append(run)

    run_rows = []
    for run in runs:
        row = {
            "subject": run["subject"],
            "run_id": run["run_id"],
            "source_event_geometry": run["source_event_geometry"][
                "linearly_detrended_same_minus_different_cosine"
            ],
            "source_event_accuracy": run["source_event_geometry"][
                "linearly_detrended_leave_one_event_accuracy"
            ],
        }
        row.update({name: getter(run) for name, getter in RUN_METRICS.items()})
        run_rows.append(row)

    run_correlations = {}
    for metric in RUN_METRICS:
        run_correlations[metric] = {
            target: correlation(
                [row[metric] for row in run_rows],
                [row[target] for row in run_rows],
            )
            for target in ("source_event_geometry", "source_event_accuracy")
        }
        run_correlations[metric]["within_subject"] = {
            subject: {
                target: correlation(
                    [row[metric] for row in run_rows if row["subject"] == subject],
                    [row[target] for row in run_rows if row["subject"] == subject],
                )
                for target in ("source_event_geometry", "source_event_accuracy")
            }
            for subject in sorted(by_subject)
        }

    motion_only = {
        subject: leave_one_run_motion_accuracy(sorted(group, key=lambda row: row["run_id"]))
        for subject, group in sorted(by_subject.items())
    }
    motion_pairs = []
    correspondence = correspondence_index(Path(args.correspondence_json))
    for subject, group in sorted(by_subject.items()):
        group = sorted(group, key=lambda row: row["run_id"])
        for first_index in range(len(group)):
            for second_index in range(first_index + 1, len(group)):
                row = compare_motion_templates(group[first_index], group[second_index])
                neural = correspondence.get((subject, row["first_run"], row["second_run"]))
                if neural is not None:
                    row.update(
                        {
                            "neural_identity_class_fraction": neural[
                                "identity_class_fraction"
                            ],
                            "neural_identity_template_similarity": neural[
                                "identity_template_similarity"
                            ],
                            "neural_permutation_similarity_gain": neural[
                                "permutation_similarity_gain"
                            ],
                        }
                    )
                motion_pairs.append(row)

    joined = [row for row in motion_pairs if "neural_identity_class_fraction" in row]
    pair_correlations = {
        motion_metric: {
            neural_metric: correlation(
                [row[motion_metric] for row in joined],
                [row[neural_metric] for row in joined],
            )
            for neural_metric in (
                "neural_identity_class_fraction",
                "neural_identity_template_similarity",
                "neural_permutation_similarity_gain",
            )
        }
        for motion_metric in (
            "motion_identity_similarity",
            "motion_identity_class_fraction",
            "motion_permutation_gain",
        )
    }
    within_subject_pair_correlations = {
        subject: {
            motion_metric: {
                neural_metric: correlation(
                    [
                        row[motion_metric]
                        for row in joined
                        if row["subject"] == subject
                    ],
                    [
                        row[neural_metric]
                        for row in joined
                        if row["subject"] == subject
                    ],
                )
                for neural_metric in (
                    "neural_identity_class_fraction",
                    "neural_identity_template_similarity",
                    "neural_permutation_similarity_gain",
                )
            }
            for motion_metric in (
                "motion_identity_similarity",
                "motion_identity_class_fraction",
                "motion_permutation_gain",
            )
        }
        for subject in sorted(by_subject)
    }
    subject_pair_summaries = {
        subject: {
            "pair_count": sum(row["subject"] == subject for row in motion_pairs),
            "mean_motion_identity_class_fraction": float(
                np.mean(
                    [
                        row["motion_identity_class_fraction"]
                        for row in motion_pairs
                        if row["subject"] == subject
                    ]
                )
            ),
            "mean_motion_identity_similarity": float(
                np.mean(
                    [
                        row["motion_identity_similarity"]
                        for row in motion_pairs
                        if row["subject"] == subject
                    ]
                )
            ),
        }
        for subject in sorted(by_subject)
    }
    result = {
        "motion_jsons": args.motion_json,
        "correspondence_json": args.correspondence_json,
        "subjects": sorted(by_subject),
        "run_count": len(runs),
        "run_rows": run_rows,
        "run_correlations": run_correlations,
        "motion_only_leave_one_run": motion_only,
        "motion_pairs": motion_pairs,
        "subject_pair_summaries": subject_pair_summaries,
        "motion_neural_pair_correlations": pair_correlations,
        "within_subject_motion_neural_pair_correlations": within_subject_pair_correlations,
        "note": (
            "Motion-only models use within-run standardized event summaries and leave one complete "
            "run out. Image-derived translations are proxies, not formal rigid-body parameters."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "motion_only_leave_one_run": motion_only,
                "subject_pair_summaries": subject_pair_summaries,
                "motion_neural_pair_correlations": pair_correlations,
                "within_subject_motion_neural_pair_correlations": (
                    within_subject_pair_correlations
                ),
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
