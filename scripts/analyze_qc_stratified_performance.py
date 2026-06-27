#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def bootstrap_mean_interval(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> dict:
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "bootstrap_p025": float(np.percentile(samples, 2.5)),
        "bootstrap_p975": float(np.percentile(samples, 97.5)),
        "subject_count": len(values),
    }


def summarize_rule(
    rows: list[dict],
    excluded_subjects: set[str],
    repetitions: int,
    rng: np.random.Generator,
) -> dict:
    by_subject: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        for subject, subject_metrics in row.get("subject_metrics", {}).items():
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                by_subject[subject][metric].append(float(subject_metrics[metric]))

    subject_rows = []
    for subject, metric_values in sorted(by_subject.items()):
        subject_rows.append(
            {
                "subject": subject,
                "holdout_count": len(metric_values["accuracy"]),
                **{
                    f"mean_{metric}": float(np.mean(values))
                    for metric, values in metric_values.items()
                },
            }
        )
    included_rows = [
        row for row in subject_rows if row["subject"] not in excluded_subjects
    ]
    all_summary = {}
    included_summary = {}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        all_values = np.asarray(
            [row[f"mean_{metric}"] for row in subject_rows], dtype=np.float64
        )
        included_values = np.asarray(
            [row[f"mean_{metric}"] for row in included_rows], dtype=np.float64
        )
        all_summary[metric] = bootstrap_mean_interval(all_values, repetitions, rng)
        included_summary[metric] = bootstrap_mean_interval(
            included_values, repetitions, rng
        )
    return {
        "all_subjects": all_summary,
        "qc_included_subjects": included_summary,
        "accuracy_change": included_summary["accuracy"]["mean"]
        - all_summary["accuracy"]["mean"],
        "excluded_subject_metrics": [
            row for row in subject_rows if row["subject"] in excluded_subjects
        ],
        "lowest_subject_accuracy": sorted(
            subject_rows, key=lambda row: row["mean_accuracy"]
        )[:15],
        "subject_metrics": subject_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report repeated subject-fold performance with transparent QC strata."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--family", default="subject")
    parser.add_argument("--exclude-subjects", nargs="*", default=["sub-42", "sub-52"])
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260627)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text())
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in payload["rows"]:
        if row["family"] == args.family:
            grouped[row["prediction_rule"]].append(row)
    rng = np.random.default_rng(args.seed)
    summaries = {
        rule: summarize_rule(
            rows,
            set(args.exclude_subjects),
            args.bootstrap_repetitions,
            rng,
        )
        for rule, rows in sorted(grouped.items())
    }
    result = {
        "input_json": args.input_json,
        "family": args.family,
        "excluded_subjects": args.exclude_subjects,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "prediction_rules": summaries,
        "note": (
            "Each subject is first averaged across repeated outer holdouts; bootstrap intervals "
            "then resample subjects, not correlated folds. The all-subject result remains primary. "
            "The QC stratum is a sensitivity analysis using independently replicated response outliers."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "excluded_subjects": args.exclude_subjects,
                "prediction_rules": {
                    rule: {
                        "all_accuracy": summary["all_subjects"]["accuracy"],
                        "qc_accuracy": summary["qc_included_subjects"]["accuracy"],
                        "accuracy_change": summary["accuracy_change"],
                        "excluded_subject_metrics": summary[
                            "excluded_subject_metrics"
                        ],
                    }
                    for rule, summary in summaries.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
