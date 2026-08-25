#!/usr/bin/env python3
"""Paired subject bootstrap for the frozen hierarchy and nested preprocessing.

The frozen result stores subject metrics inside selected outer fold rows. New nested
preprocessing kernels store one mean per subject and prediction rule. This script
normalizes those two schemas and bootstraps their paired subject differences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def frozen_subject_means(rows: list[dict], rule: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["prediction_rule"] != rule:
            continue
        for subject, metrics in row["subject_metrics"].items():
            values[subject].append(float(metrics["balanced_accuracy"]))
    return {
        subject: float(np.mean(subject_values))
        for subject, subject_values in values.items()
    }


def paired_summary(
    reference: dict[str, float],
    comparison: dict[str, float],
    iterations: int,
    seed: int,
    excluded_subjects: set[str] | None = None,
) -> dict:
    excluded_subjects = excluded_subjects or set()
    subjects = sorted((set(reference) & set(comparison)) - excluded_subjects)
    if not subjects:
        raise ValueError("No shared subjects remain for paired inference.")
    reference_values = np.asarray([reference[s] for s in subjects], dtype=np.float64)
    comparison_values = np.asarray([comparison[s] for s in subjects], dtype=np.float64)
    difference = comparison_values - reference_values
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0, len(subjects), size=(iterations, len(subjects)), endpoint=False
    )
    sampled_means = difference[sampled_indices].mean(axis=1)
    low, high = np.quantile(sampled_means, [0.025, 0.975])
    return {
        "subject_count": len(subjects),
        "reference_mean": float(reference_values.mean()),
        "comparison_mean": float(comparison_values.mean()),
        "comparison_minus_reference": float(difference.mean()),
        "ci95": [float(low), float(high)],
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "comparison_wins": int(np.sum(difference > 0.0)),
        "ties": int(np.sum(difference == 0.0)),
        "comparison_losses": int(np.sum(difference < 0.0)),
        "excluded_subjects": sorted(excluded_subjects),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-json", required=True)
    parser.add_argument("--nested-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    args = parser.parse_args()

    frozen_path = Path(args.frozen_json)
    nested_path = Path(args.nested_json)
    frozen = json.loads(frozen_path.read_text())
    nested = json.loads(nested_path.read_text())
    nested_subjects = nested["per_subject_means"]

    comparisons = {}
    for rule in ("independent", "balanced"):
        reference = frozen_subject_means(frozen["selected_rows"], rule)
        comparison = {
            subject: float(value)
            for subject, value in nested_subjects[rule].items()
        }
        comparisons[f"{rule}|all62"] = paired_summary(
            reference,
            comparison,
            args.bootstrap_iterations,
            args.bootstrap_seed,
        )
        comparisons[f"{rule}|qc60"] = paired_summary(
            reference,
            comparison,
            args.bootstrap_iterations,
            args.bootstrap_seed,
            {"sub-42", "sub-52"},
        )

    payload = {
        "reference": {
            "name": frozen_path.name,
            "sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        },
        "comparison": {
            "name": nested_path.name,
            "sha256": hashlib.sha256(nested_path.read_bytes()).hexdigest(),
        },
        "unit": "subject mean balanced accuracy across repeated held out folds",
        "positive_difference_means": "nested preprocessing above frozen hierarchy",
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
        "comparisons": comparisons,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n")

    for name, result in comparisons.items():
        print(
            f"{name:20s} ref={result['reference_mean']:.4f} "
            f"new={result['comparison_mean']:.4f} "
            f"diff={result['comparison_minus_reference']:+.4f} "
            f"ci=[{result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f}] "
            f"wins/ties/losses={result['comparison_wins']}/"
            f"{result['ties']}/{result['comparison_losses']}"
        )


if __name__ == "__main__":
    main()
