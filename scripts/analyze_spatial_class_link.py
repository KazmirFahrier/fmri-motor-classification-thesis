#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SPATIAL_METRICS = [
    "mask_dice",
    "mask_jaccard",
    "center_of_mass_distance_mm",
    "temporal_mean_map_correlation",
    "temporal_std_map_correlation",
]

CLASS_METRICS = [
    "identity_class_fraction",
    "permutation_similarity_gain",
]


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def run_id_from_label(label: str) -> int:
    return int(label.rsplit("-run-", 1)[1])


def correlation_rows(rows: list[dict]) -> list[dict]:
    result = []
    for spatial_metric in SPATIAL_METRICS:
        spatial_values = np.asarray([row[spatial_metric] for row in rows], dtype=np.float64)
        for class_metric in CLASS_METRICS:
            class_values = np.asarray([row[class_metric] for row in rows], dtype=np.float64)
            result.append(
                {
                    "spatial_metric": spatial_metric,
                    "class_metric": class_metric,
                    "pair_count": len(rows),
                    "pearson_correlation": correlation(spatial_values, class_values),
                    "rank_correlation": correlation(
                        rank_values(spatial_values),
                        rank_values(class_values),
                    ),
                }
            )
    return sorted(result, key=lambda row: -abs(row["pearson_correlation"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join native spatial run-pair stability to class-template correspondence."
    )
    parser.add_argument("--spatial-json", required=True)
    parser.add_argument("--class-json", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    spatial = json.loads(Path(args.spatial_json).read_text())
    class_data = json.loads(Path(args.class_json).read_text())
    class_lookup = {}
    for subject_row in class_data["rows"]:
        subject = subject_row["subject"]
        for pair in subject_row["run_pair_rows"]:
            key = (subject, int(pair["first_run"]), int(pair["second_run"]))
            class_lookup[key] = pair

    joined = []
    for spatial_row in spatial["comparisons"]:
        subject = spatial_row["subject"]
        first_run = run_id_from_label(spatial_row["first_label"])
        second_run = run_id_from_label(spatial_row["second_label"])
        key = (subject, min(first_run, second_run), max(first_run, second_run))
        class_row = class_lookup[key]
        joined.append(
            {
                "subject": subject,
                "first_run": key[1],
                "second_run": key[2],
                **{metric: spatial_row[metric] for metric in SPATIAL_METRICS},
                "identity_class_fraction": class_row["identity_class_fraction"],
                "identity_is_optimal": class_row["identity_is_optimal"],
                "permutation_similarity_gain": class_row["permutation_similarity_gain"],
            }
        )

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in joined:
        by_subject[row["subject"]].append(row)
    result = {
        "spatial_json": args.spatial_json,
        "class_json": args.class_json,
        "joined_pair_count": len(joined),
        "joined_rows": joined,
        "all_subject_correlations": correlation_rows(joined),
        "subject_correlations": {
            subject: correlation_rows(rows)
            for subject, rows in sorted(by_subject.items())
        },
        "note": (
            "These are forensic associations across only 15 run pairs per subject. Correlation does "
            "not establish that coverage/variance instability causes class-template failure."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "all_subject_correlations": result["all_subject_correlations"][:10],
                "subject_correlations": {
                    subject: rows[:6]
                    for subject, rows in result["subject_correlations"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
