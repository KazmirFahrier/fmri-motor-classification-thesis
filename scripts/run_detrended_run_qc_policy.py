#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_feature_qc_subjects import leave_one_event_accuracy, pairwise_geometry
from run_balanced_event_assignment import (
    apply_balanced_assignment,
    apply_imbalance_gated_balanced_assignment,
    center_by_subject_run,
    centroid_matrix,
    metrics,
    score_with_centroids,
    split_indices,
)
from run_clip_offset_event_sweep import aggregate_events_for_offset, coarse_metrics
from run_temporal_detrended_event_adaptation import (
    temporal_detrend_by_subject_run,
)


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run_key(record: dict) -> str:
    return f'{record["subject_id"]}|run-{int(record["run_id"])}'


def summarize_runs(x: np.ndarray, y: np.ndarray, records: list[dict]) -> list[dict]:
    key_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        key_to_indices[run_key(record)].append(idx)

    rows = []
    for key, indices in sorted(key_to_indices.items()):
        subject, run_part = key.split("|")
        run_id = int(run_part.split("-")[1])
        geometry = pairwise_geometry(x, y, indices)
        loo = leave_one_event_accuracy(x, y, indices)
        rows.append(
            {
                "run_key": key,
                "subject": subject,
                "run_id": run_id,
                "event_count": int(len(indices)),
                "same_minus_different": geometry["same_minus_different_mean"],
                "within_run_leave_one_event_accuracy": loo["accuracy"],
                "within_run_leave_one_event_macro_f1": loo["macro_f1"],
            }
        )
    return rows


def build_policies(run_rows: list[dict], max_quantile: float) -> list[dict]:
    same_values = np.asarray(
        [
            row["same_minus_different"]
            for row in run_rows
            if row["same_minus_different"] is not None
        ],
        dtype=np.float64,
    )
    loo_values = np.asarray(
        [
            row["within_run_leave_one_event_accuracy"]
            for row in run_rows
            if row["within_run_leave_one_event_accuracy"] is not None
        ],
        dtype=np.float64,
    )
    quantiles = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    quantiles = [q for q in quantiles if q <= max_quantile]
    policies = [{"name": "keep_all", "metric": "none", "threshold": None, "drop_quantile": 0.0}]
    for q in quantiles[1:]:
        policies.append(
            {
                "name": f"same_minus_drop_bottom_{int(q * 100):02d}pct",
                "metric": "same_minus_different",
                "threshold": float(np.quantile(same_values, q)),
                "drop_quantile": float(q),
            }
        )
        policies.append(
            {
                "name": f"loo_drop_bottom_{int(q * 100):02d}pct",
                "metric": "within_run_leave_one_event_accuracy",
                "threshold": float(np.quantile(loo_values, q)),
                "drop_quantile": float(q),
            }
        )
    return policies


def kept_run_keys(run_rows: list[dict], policy: dict) -> set[str]:
    if policy["metric"] == "none":
        return {row["run_key"] for row in run_rows}
    metric = str(policy["metric"])
    threshold = float(policy["threshold"])
    return {
        row["run_key"]
        for row in run_rows
        if row[metric] is not None and float(row[metric]) >= threshold
    }


def indices_for_kept_runs(records: list[dict], keep: set[str]) -> np.ndarray:
    return np.asarray(
        [idx for idx, record in enumerate(records) if run_key(record) in keep],
        dtype=np.int64,
    )


def evaluate_policy(
    *,
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    policy: dict,
    keep: set[str],
    filter_validation: bool,
    split_family: str,
    subject_fold_count: int,
) -> list[dict]:
    rows = []
    keep_indices = set(indices_for_kept_runs(records, keep).tolist())
    for split in split_indices(records, split_family, subject_fold_count):
        train_idx = np.asarray(
            [idx for idx in split["train_idx"] if int(idx) in keep_indices],
            dtype=np.int64,
        )
        if filter_validation:
            val_idx = np.asarray(
                [idx for idx in split["val_idx"] if int(idx) in keep_indices],
                dtype=np.int64,
            )
        else:
            val_idx = np.asarray(split["val_idx"], dtype=np.int64)

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        if len(set(y[train_idx].tolist())) < 4:
            continue

        centroids = centroid_matrix(x[train_idx], y[train_idx])
        scores = score_with_centroids(x[val_idx], centroids)
        predictions = [
            ("independent_argmax", scores.argmax(axis=1).astype(np.int64)),
            ("balanced_subject_run_assignment", apply_balanced_assignment(scores, val_idx, records)),
            (
                "gated_balanced_imbalance_l1_4",
                apply_imbalance_gated_balanced_assignment(scores, val_idx, records, 4.0),
            ),
        ]
        for rule, pred in predictions:
            rows.append(
                {
                    "policy": policy["name"],
                    "policy_metric": policy["metric"],
                    "policy_threshold": policy["threshold"],
                    "filter_validation": bool(filter_validation),
                    "split": split["split"],
                    "family": split["family"],
                    "prediction_rule": rule,
                    "train_count": int(len(train_idx)),
                    "val_count": int(len(val_idx)),
                    "val_coverage": float(len(val_idx) / max(len(split["val_idx"]), 1)),
                    "metrics": metrics(y[val_idx], pred),
                    "coarse_metrics": coarse_metrics(y[val_idx], pred),
                }
            )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, bool, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["policy"],
                bool(row["filter_validation"]),
                row["family"],
                row["prediction_rule"],
            )
        ].append(row)

    summary = []
    for (policy, filter_validation, family, rule), group in sorted(grouped.items()):
        summary.append(
            {
                "policy": policy,
                "filter_validation": filter_validation,
                "family": family,
                "prediction_rule": rule,
                "split_count": len(group),
                "mean_accuracy": float(np.mean([row["metrics"]["accuracy"] for row in group])),
                "mean_balanced_accuracy": float(
                    np.mean([row["metrics"]["balanced_accuracy"] for row in group])
                ),
                "mean_macro_f1": float(np.mean([row["metrics"]["macro_f1"] for row in group])),
                "mean_leg_vs_arm_accuracy": float(
                    np.mean(
                        [
                            row["coarse_metrics"]["leg_vs_arm_accuracy"]
                            for row in group
                        ]
                    )
                ),
                "mean_train_count": float(np.mean([row["train_count"] for row in group])),
                "mean_val_count": float(np.mean([row["val_count"] for row in group])),
                "mean_val_coverage": float(np.mean([row["val_coverage"] for row in group])),
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            row["family"],
            row["filter_validation"],
            row["prediction_rule"],
            -row["mean_accuracy"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether post-detrend label-aware run QC helps transfer, "
            "and distinguish training-only cleanup from oracle validation exclusion."
        )
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--clip-offset", type=int, default=2)
    parser.add_argument("--max-drop-quantile", type=float, default=0.30)
    parser.add_argument("--split-family", choices=["all", "run", "subject"], default="all")
    parser.add_argument("--subject-fold-count", type=int, default=6)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    clip_x = np.load(feature_dir / "features.npy").astype(np.float32)
    clip_y = np.load(feature_dir / "labels.npy").astype(np.int64)
    clip_records = json.loads((feature_dir / "records.json").read_text())
    event_x, event_y, event_records = aggregate_events_for_offset(
        clip_x,
        clip_y,
        clip_records,
        args.clip_offset,
    )
    centered = center_by_subject_run(event_x, event_records)
    detrended, group_rows = temporal_detrend_by_subject_run(centered, event_records, degree=1)
    run_rows = summarize_runs(detrended, event_y, event_records)
    policies = build_policies(run_rows, args.max_drop_quantile)

    rows = []
    policy_rows = []
    for policy in policies:
        keep = kept_run_keys(run_rows, policy)
        policy_rows.append(
            {
                **policy,
                "kept_run_count": int(len(keep)),
                "dropped_run_count": int(len(run_rows) - len(keep)),
                "kept_event_count": int(
                    sum(
                        row["event_count"]
                        for row in run_rows
                        if row["run_key"] in keep
                    )
                ),
            }
        )
        for filter_validation in [False, True]:
            rows.extend(
                evaluate_policy(
                    x=detrended,
                    y=event_y,
                    records=event_records,
                    policy=policy,
                    keep=keep,
                    filter_validation=filter_validation,
                    split_family=args.split_family,
                    subject_fold_count=args.subject_fold_count,
                )
            )

    result = {
        "feature_dir": str(feature_dir),
        "clip_offset": int(args.clip_offset),
        "event_feature_shape": list(event_x.shape),
        "run_count": int(len(run_rows)),
        "event_count": int(len(event_y)),
        "mean_linear_time_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in group_rows])
        ),
        "policies": policy_rows,
        "runs": run_rows,
        "rows": rows,
        "summary": summarize(rows),
        "note": (
            "Run QC metrics use labels within each run, so validation filtering is an oracle/diagnostic "
            "coverage analysis. Training-only filtering tests whether low-geometry source runs poison "
            "centroids while retaining all held-out events."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "event_feature_shape": result["event_feature_shape"],
                "policy_count": len(policy_rows),
                "top_summary": result["summary"][:12],
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
