#!/usr/bin/env python3
"""Paired comparison of the frozen hierarchy against standard MVPA baselines.

Marginal confidence intervals that overlap do not establish equivalence, and
non-overlapping ones are not the right test either. Both decoders are evaluated
on the same 30 outer splits and the same 62 subjects, so the comparison is
paired: the unit is the per-subject difference in balanced accuracy averaged over
the splits in which that subject was held out.

The reported interval is a subject-level bootstrap over those paired differences,
matching the resampling used elsewhere in the closeout.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def subject_means(rows: list[dict], rule: str) -> dict[str, float]:
    by_subject: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["prediction_rule"] != rule:
            continue
        for subject, metric_row in row["subject_metrics"].items():
            by_subject[subject].append(float(metric_row["balanced_accuracy"]))
    return {subject: float(np.mean(values)) for subject, values in by_subject.items()}


def paired_bootstrap(
    reference: dict[str, float],
    comparison: dict[str, float],
    iterations: int,
    seed: int,
) -> dict:
    subjects = sorted(set(reference) & set(comparison))
    if not subjects:
        raise ValueError("No shared subjects between the two result sets.")
    difference = np.asarray(
        [reference[subject] - comparison[subject] for subject in subjects]
    )
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        difference, size=(iterations, len(difference)), replace=True
    ).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "subject_count": len(subjects),
        "reference_mean": float(np.mean([reference[s] for s in subjects])),
        "comparison_mean": float(np.mean([comparison[s] for s in subjects])),
        "mean_difference": float(np.mean(difference)),
        "ci95": [float(low), float(high)],
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "subject_wins": int(np.sum(difference > 0)),
        "subject_ties": int(np.sum(difference == 0)),
        "subject_losses": int(np.sum(difference < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired subject-level comparison of the frozen decoder and standard MVPA."
    )
    parser.add_argument("--frozen-json", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    args = parser.parse_args()

    frozen = json.loads(Path(args.frozen_json).read_text())
    baseline = json.loads(Path(args.baseline_json).read_text())

    comparisons = {}
    for rule in ("independent", "balanced"):
        frozen_subjects = subject_means(frozen["selected_rows"], rule)
        for model_name in sorted({row["model"] for row in baseline["rows"]}):
            model_rows = [row for row in baseline["rows"] if row["model"] == model_name]
            comparisons[f"{rule}|frozen_vs_{model_name}"] = paired_bootstrap(
                frozen_subjects,
                subject_means(model_rows, rule),
                args.bootstrap_iterations,
                args.bootstrap_seed,
            )

    payload = {
        "frozen_json": args.frozen_json,
        "baseline_json": args.baseline_json,
        "preprocess": baseline.get("preprocess"),
        "unit": "per-subject mean balanced accuracy across the splits holding that subject out",
        "positive_difference_means": "frozen hierarchy above the standard baseline",
        "comparisons": comparisons,
        "note": (
            "Paired subject-level bootstrap on identical splits and subjects. "
            "Overlapping marginal intervals do not test this difference; the paired "
            "interval does."
        ),
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"{'comparison':44s} {'frozen':>8s} {'base':>8s} {'diff':>8s}  {'ci95':>20s}  excl0  w/t/l")
    for name, row in comparisons.items():
        ci = f"[{row['ci95'][0]:+.4f},{row['ci95'][1]:+.4f}]"
        print(
            f"{name:44s} {row['reference_mean']:8.4f} {row['comparison_mean']:8.4f} "
            f"{row['mean_difference']:+8.4f}  {ci:>20s}  {str(row['excludes_zero']):5s}  "
            f"{row['subject_wins']}/{row['subject_ties']}/{row['subject_losses']}"
        )


if __name__ == "__main__":
    main()
