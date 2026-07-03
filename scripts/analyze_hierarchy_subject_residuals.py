#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def correlation(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    return {
        "pearson": float(np.corrcoef(first, second)[0, 1]),
        "spearman": float(np.corrcoef(rankdata(first), rankdata(second))[0, 1]),
    }


def official_correlations(rows: list[dict], official_by_subject: dict) -> dict:
    matched = [row for row in rows if row["subject"] in official_by_subject]
    identity = np.asarray(
        [official_by_subject[row["subject"]]["group_identity_similarity"] for row in matched]
    )
    amplitude = np.asarray(
        [
            official_by_subject[row["subject"]]["mean_class_centered_map_rms"]
            for row in matched
        ]
    )
    result = {}
    for metric in ("exact_accuracy", "leg_pair_accuracy", "arm_pair_accuracy"):
        values = np.asarray([row[metric] for row in matched])
        result[f"{metric}_vs_group_identity"] = correlation(values, identity)
        result[f"{metric}_vs_class_centered_rms"] = correlation(values, amplitude)
    return result


def subject_rows(hierarchy: dict, prediction_rule: str) -> list[dict]:
    confusion_by_subject: dict[str, np.ndarray] = {}
    repeat_count: dict[str, int] = {}
    for row in hierarchy["rows"]:
        if row["prediction_rule"] != prediction_rule:
            continue
        for subject, metrics in row["subject_metrics"].items():
            confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
            confusion_by_subject.setdefault(subject, np.zeros((4, 4), dtype=np.int64))
            confusion_by_subject[subject] += confusion
            repeat_count[subject] = repeat_count.get(subject, 0) + 1

    rows = []
    for subject in sorted(confusion_by_subject):
        confusion = confusion_by_subject[subject]
        total = int(confusion.sum())
        leg_total = int(confusion[:2].sum())
        arm_total = int(confusion[2:].sum())
        exact = float(np.trace(confusion) / total)
        leg = float((confusion[0, 0] + confusion[1, 1]) / leg_total)
        arm = float((confusion[2, 2] + confusion[3, 3]) / arm_total)
        coarse = float(
            (confusion[:2, :2].sum() + confusion[2:, 2:].sum()) / total
        )
        if exact < 0.5:
            residual_type = "global_failure"
        elif leg >= 0.75 and arm <= leg - 0.15:
            residual_type = "arm_specific"
        elif arm >= 0.75 and leg <= arm - 0.15:
            residual_type = "leg_specific"
        else:
            residual_type = "mixed_or_mild"
        rows.append(
            {
                "subject": subject,
                "repeat_count": repeat_count[subject],
                "exact_accuracy": exact,
                "coarse_accuracy": coarse,
                "leg_pair_accuracy": leg,
                "arm_pair_accuracy": arm,
                "arm_minus_leg": arm - leg,
                "residual_type": residual_type,
                "confusion_matrix": confusion.tolist(),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze subject-level residuals from repeated hierarchy predictions."
    )
    parser.add_argument("--hierarchy-json", required=True)
    parser.add_argument("--official-glm-json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--prediction-rule", default="hybrid_fused_balanced")
    args = parser.parse_args()

    hierarchy = json.loads(Path(args.hierarchy_json).read_text())
    rows = subject_rows(hierarchy, args.prediction_rule)
    correlations = {}
    if args.official_glm_json:
        official = json.loads(Path(args.official_glm_json).read_text())
        official_by_subject = {row["subject"]: row for row in official["subjects"]}
        correlations = {
            "all_subjects": official_correlations(rows, official_by_subject),
            "qc_stratum": official_correlations(
                [row for row in rows if row["subject"] not in {"sub-42", "sub-52"}],
                official_by_subject,
            ),
        }
        for row in rows:
            if row["subject"] in official_by_subject:
                official_row = official_by_subject[row["subject"]]
                row["official_group_identity_similarity"] = official_row[
                    "group_identity_similarity"
                ]
                row["official_class_centered_map_rms"] = official_row[
                    "mean_class_centered_map_rms"
                ]

    counts = {
        residual_type: sum(row["residual_type"] == residual_type for row in rows)
        for residual_type in sorted({row["residual_type"] for row in rows})
    }
    result = {
        "hierarchy_json": args.hierarchy_json,
        "prediction_rule": args.prediction_rule,
        "subject_count": len(rows),
        "residual_type_counts": counts,
        "mean_exact_accuracy": float(np.mean([row["exact_accuracy"] for row in rows])),
        "mean_coarse_accuracy": float(np.mean([row["coarse_accuracy"] for row in rows])),
        "mean_leg_pair_accuracy": float(
            np.mean([row["leg_pair_accuracy"] for row in rows])
        ),
        "mean_arm_pair_accuracy": float(
            np.mean([row["arm_pair_accuracy"] for row in rows])
        ),
        "correlations": correlations,
        "worst_arm_subjects": sorted(rows, key=lambda row: row["arm_pair_accuracy"])[:10],
        "worst_exact_subjects": sorted(rows, key=lambda row: row["exact_accuracy"])[:10],
        "subjects": rows,
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "subject_count": len(rows),
                "residual_type_counts": counts,
                "mean_exact_accuracy": result["mean_exact_accuracy"],
                "mean_leg_pair_accuracy": result["mean_leg_pair_accuracy"],
                "mean_arm_pair_accuracy": result["mean_arm_pair_accuracy"],
                "correlations": correlations,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
