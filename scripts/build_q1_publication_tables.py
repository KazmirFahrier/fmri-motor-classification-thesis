#!/usr/bin/env python3
"""Build manuscript benchmark tables from frozen machine readable results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def format_value(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def build_rows(closeout: dict, confirmation: dict, paired: dict) -> list[dict]:
    frozen = closeout["frozen_results"]
    q1 = confirmation["completed_evidence"]
    nested_smoothing = q1["nested_native_smoothing_24"]
    nested_grid = q1["nested_spatial_grid"]
    pair_rows = paired["comparisons"]

    return [
        {
            "analysis": "Legacy pooled neural model",
            "protocol": "pooled historical split",
            "test_information": "historical pooled evaluation",
            "metric": "balanced accuracy",
            "all_62": frozen["legacy_full_pooled"]["balanced_accuracy"],
            "qc_60": None,
            "role": "negative control",
        },
        {
            "analysis": "Legacy neural subject holdout",
            "protocol": "independent subject holdout",
            "test_information": "single event features",
            "metric": "balanced accuracy",
            "all_62": frozen["legacy_subjectwise_holdout"]["balanced_accuracy"],
            "qc_60": None,
            "role": "negative control",
        },
        {
            "analysis": "Frozen temporal hierarchy",
            "protocol": "independent repeated nested folds",
            "test_information": "unlabeled target run adaptation",
            "metric": "subject mean balanced accuracy",
            "all_62": pair_rows["independent|all62"]["reference_mean"],
            "qc_60": pair_rows["independent|qc60"]["reference_mean"],
            "role": "frozen comparator",
        },
        {
            "analysis": "Nested native smoothing plus linear SVM",
            "protocol": "independent repeated nested folds",
            "test_information": "unlabeled target run adaptation",
            "metric": "subject mean balanced accuracy",
            "all_62": pair_rows["independent|all62"]["comparison_mean"],
            "qc_60": pair_rows["independent|qc60"]["comparison_mean"],
            "role": "primary internal conventional model",
        },
        {
            "analysis": "Nested spatial grid plus linear SVM",
            "protocol": "independent repeated nested folds",
            "test_information": "unlabeled target run adaptation",
            "metric": "fold mean balanced accuracy",
            "all_62": nested_grid["independent_balanced_accuracy"],
            "qc_60": None,
            "role": "representation confirmation",
        },
        {
            "analysis": "Conservative mean window hierarchy",
            "protocol": "complete run balanced assignment",
            "test_information": "known two per class run composition",
            "metric": "balanced accuracy",
            "all_62": frozen["conservative_mean_hierarchy"]["balanced_accuracy"],
            "qc_60": None,
            "role": "transductive baseline",
        },
        {
            "analysis": "Frozen repetition consistency hierarchy",
            "protocol": "complete run repetition consistency",
            "test_information": "known run composition and repetition structure",
            "metric": "subject mean balanced accuracy",
            "all_62": frozen["complete_balanced_run"]["subject_weighted_accuracy"],
            "qc_60": frozen["complete_balanced_run"]["qc60_subject_weighted_accuracy"],
            "role": "design constrained transductive result",
        },
        {
            "analysis": "Nested native smoothing plus balanced assignment",
            "protocol": "complete run balanced assignment",
            "test_information": "known two per class run composition",
            "metric": "subject mean balanced accuracy",
            "all_62": pair_rows["balanced|all62"]["comparison_mean"],
            "qc_60": pair_rows["balanced|qc60"]["comparison_mean"],
            "role": "secondary transductive conventional model",
        },
        {
            "analysis": "Five run arm calibration",
            "protocol": "independent personalized prediction",
            "test_information": "five labeled target subject runs",
            "metric": "accuracy",
            "all_62": frozen["five_run_labeled_personalization"]["independent_accuracy"],
            "qc_60": None,
            "role": "labeled personalization",
        },
    ]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "all_62": format_value(row["all_62"]),
                    "qc_60": format_value(row["qc_60"]),
                }
            )


def write_markdown(rows: list[dict], path: Path) -> None:
    headers = [
        "Analysis",
        "Protocol",
        "Test information",
        "Metric",
        "All 62",
        "QC60",
        "Role",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["analysis"],
                    row["protocol"],
                    row["test_information"],
                    row["metric"],
                    format_value(row["all_62"]),
                    format_value(row["qc_60"]),
                    row["role"],
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--closeout",
        type=Path,
        default=Path("experiments/confirmation/investigation_closeout.results.json"),
    )
    parser.add_argument(
        "--confirmation",
        type=Path,
        default=Path(
            "findings_2026-08-18_interpretation/experiments/q1_confirmation.results.json"
        ),
    )
    parser.add_argument(
        "--paired",
        type=Path,
        default=Path(
            "findings_2026-08-18_interpretation/experiments/"
            "frozen_vs_nested_native_smoothing.results.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("manuscript/tables")
    )
    args = parser.parse_args()

    rows = build_rows(load(args.closeout), load(args.confirmation), load(args.paired))
    write_csv(rows, args.output_dir / "main_benchmark.csv")
    write_markdown(rows, args.output_dir / "main_benchmark.md")
    print(f"wrote {len(rows)} benchmark rows to {args.output_dir}")


if __name__ == "__main__":
    main()
