#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def signal_summary(rows: list[dict], diagnostic_names: list[str]) -> list[dict]:
    target = np.asarray([row["actual_pair_minus_flat_accuracy"] for row in rows])
    summary = []
    for name in diagnostic_names:
        values = np.asarray([row["diagnostics"][name] for row in rows])
        summary.append(
            {
                "diagnostic": name,
                "pearson_correlation": correlation(values, target),
                "rank_correlation": correlation(rank_values(values), rank_values(target)),
                "mean_when_pair_better": float(np.mean(values[target > 0])) if np.any(target > 0) else 0.0,
                "mean_when_pair_not_better": float(np.mean(values[target <= 0]))
                if np.any(target <= 0)
                else 0.0,
            }
        )
    return sorted(summary, key=lambda row: -abs(row["pearson_correlation"]))


def subject_mean_rows(rows: list[dict], diagnostic_names: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["subject"]].append(row)
    result = []
    for subject, group in sorted(grouped.items()):
        result.append(
            {
                "subject": subject,
                "actual_pair_minus_flat_accuracy": float(
                    np.mean([row["actual_pair_minus_flat_accuracy"] for row in group])
                ),
                "predicted_pair_minus_flat_accuracy": float(
                    np.mean([row["predicted_pair_minus_flat_accuracy"] for row in group])
                ),
                "diagnostics": {
                    name: float(np.mean([row["diagnostics"][name] for row in group]))
                    for name in diagnostic_names
                },
            }
        )
    return result


def coefficient_stability(gate_rows: list[dict], diagnostic_names: list[str]) -> list[dict]:
    result = []
    for name in diagnostic_names:
        values = np.asarray(
            [
                row["ridge_coefficients"]["standardized_features"][name]
                for row in gate_rows
            ]
        )
        result.append(
            {
                "diagnostic": name,
                "mean_standardized_coefficient": float(np.mean(values)),
                "coefficient_std": float(np.std(values)),
                "positive_fraction": float(np.mean(values > 0)),
                "negative_fraction": float(np.mean(values < 0)),
            }
        )
    return sorted(result, key=lambda row: -abs(row["mean_standardized_coefficient"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether unlabeled subject diagnostics predict pair-model gains."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    source = json.loads(Path(args.input_json).read_text())
    diagnostic_names = source["diagnostic_names"]
    repeated_rows = [
        subject_row
        for gate_row in source["gate_rows"]
        for subject_row in gate_row["subject_rows"]
    ]
    mean_rows = subject_mean_rows(repeated_rows, diagnostic_names)
    repeated_actual = np.asarray(
        [row["actual_pair_minus_flat_accuracy"] for row in repeated_rows]
    )
    repeated_predicted = np.asarray(
        [row["predicted_pair_minus_flat_accuracy"] for row in repeated_rows]
    )
    subject_actual = np.asarray(
        [row["actual_pair_minus_flat_accuracy"] for row in mean_rows]
    )
    subject_predicted = np.asarray(
        [row["predicted_pair_minus_flat_accuracy"] for row in mean_rows]
    )
    result = {
        "input_json": args.input_json,
        "repeated_observation_count": len(repeated_rows),
        "subject_count": len(mean_rows),
        "gate_prediction_correlations": {
            "repeated_pearson": correlation(repeated_predicted, repeated_actual),
            "repeated_rank": correlation(
                rank_values(repeated_predicted),
                rank_values(repeated_actual),
            ),
            "subject_mean_pearson": correlation(subject_predicted, subject_actual),
            "subject_mean_rank": correlation(
                rank_values(subject_predicted),
                rank_values(subject_actual),
            ),
        },
        "repeated_signal_summary": signal_summary(repeated_rows, diagnostic_names),
        "subject_mean_signal_summary": signal_summary(mean_rows, diagnostic_names),
        "coefficient_stability": coefficient_stability(source["gate_rows"], diagnostic_names),
        "subject_mean_rows": sorted(
            mean_rows,
            key=lambda row: row["actual_pair_minus_flat_accuracy"],
        ),
        "note": (
            "These correlations use held-out outcomes for retrospective diagnosis and are not "
            "deployable gates. Repeated observations are not independent, so subject-mean summaries "
            "are the safer interpretation."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "gate_prediction_correlations": result["gate_prediction_correlations"],
                "top_subject_mean_signals": result["subject_mean_signal_summary"][:8],
                "coefficient_stability": result["coefficient_stability"][:8],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
