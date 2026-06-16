#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    CLASS_NAMES,
    apply_balanced_assignment,
    apply_imbalance_gated_balanced_assignment,
    center_by_subject_run,
    l2_normalize,
    metrics,
)
from run_clip_offset_event_sweep import aggregate_events_for_offset, coarse_metrics


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def centroid_matrix_from_normalized(x_norm: np.ndarray, y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    centroids = []
    for class_idx in range(len(CLASS_NAMES)):
        class_indices = indices[y[indices] == class_idx]
        if len(class_indices) == 0:
            raise ValueError(f"Missing class {class_idx} in calibration split.")
        centroids.append(x_norm[class_indices].mean(axis=0))
    return l2_normalize(np.stack(centroids, axis=0))


def blend_centroids(source: np.ndarray, subject: np.ndarray, alpha: float) -> np.ndarray:
    return l2_normalize((1.0 - float(alpha)) * source + float(alpha) * subject)


def score(x_norm: np.ndarray, indices: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return x_norm[indices].astype(np.float64) @ centroids.astype(np.float64).T


def append_prediction_rows(
    rows: list[dict],
    *,
    subject: str,
    holdout_run: int,
    protocol: str,
    calibration_runs: list[int],
    calibration_run_count: int,
    blend_alpha: float,
    scores: np.ndarray,
    val_idx: np.ndarray,
    y: np.ndarray,
    records: list[dict],
) -> None:
    prediction_variants = [
        ("independent_argmax", scores.argmax(axis=1).astype(np.int64)),
        ("balanced_subject_run_assignment", apply_balanced_assignment(scores, val_idx, records)),
        (
            "gated_balanced_imbalance_l1_4",
            apply_imbalance_gated_balanced_assignment(scores, val_idx, records, 4.0),
        ),
    ]
    for prediction_rule, pred in prediction_variants:
        rows.append(
            {
                "subject": subject,
                "holdout_run": int(holdout_run),
                "protocol": protocol,
                "prediction_rule": prediction_rule,
                "calibration_runs": calibration_runs,
                "calibration_run_count": int(calibration_run_count),
                "blend_alpha": float(blend_alpha),
                "accuracy": float(np.mean(y[val_idx] == pred)),
                "y_true": y[val_idx].tolist(),
                "y_pred": pred.tolist(),
            }
        )


def summarize_prediction_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int, float], dict[str, list]] = defaultdict(
        lambda: {"true": [], "pred": [], "split_accuracy": []}
    )
    for row in rows:
        key = (
            row["protocol"],
            row["prediction_rule"],
            int(row["calibration_run_count"]),
            float(row["blend_alpha"]),
        )
        grouped[key]["true"].extend(row["y_true"])
        grouped[key]["pred"].extend(row["y_pred"])
        grouped[key]["split_accuracy"].append(row["accuracy"])

    summary = []
    for (protocol, prediction_rule, calibration_run_count, blend_alpha), payload in sorted(grouped.items()):
        y_true = np.asarray(payload["true"], dtype=np.int64)
        y_pred = np.asarray(payload["pred"], dtype=np.int64)
        exact = metrics(y_true, y_pred)
        coarse = coarse_metrics(y_true, y_pred)
        summary.append(
            {
                "protocol": protocol,
                "prediction_rule": prediction_rule,
                "calibration_run_count": calibration_run_count,
                "blend_alpha": blend_alpha,
                "split_count": len(payload["split_accuracy"]),
                "event_count": int(len(y_true)),
                "mean_split_accuracy": float(np.mean(payload["split_accuracy"])),
                "metrics": exact,
                "coarse_metrics": coarse,
            }
        )
    return summary


def summarize_by_subject(rows: list[dict], focus_subjects: set[str]) -> list[dict]:
    grouped: dict[tuple[str, str, str, int, float], dict[str, list]] = defaultdict(
        lambda: {"true": [], "pred": [], "split_accuracy": []}
    )
    for row in rows:
        if focus_subjects and row["subject"] not in focus_subjects:
            continue
        key = (
            row["subject"],
            row["protocol"],
            row["prediction_rule"],
            int(row["calibration_run_count"]),
            float(row["blend_alpha"]),
        )
        grouped[key]["true"].extend(row["y_true"])
        grouped[key]["pred"].extend(row["y_pred"])
        grouped[key]["split_accuracy"].append(row["accuracy"])

    subject_rows = []
    for (subject, protocol, prediction_rule, calibration_run_count, blend_alpha), payload in sorted(grouped.items()):
        y_true = np.asarray(payload["true"], dtype=np.int64)
        y_pred = np.asarray(payload["pred"], dtype=np.int64)
        subject_rows.append(
            {
                "subject": subject,
                "protocol": protocol,
                "prediction_rule": prediction_rule,
                "calibration_run_count": calibration_run_count,
                "blend_alpha": blend_alpha,
                "split_count": len(payload["split_accuracy"]),
                "event_count": int(len(y_true)),
                "mean_split_accuracy": float(np.mean(payload["split_accuracy"])),
                "metrics": metrics(y_true, y_pred),
                "coarse_metrics": coarse_metrics(y_true, y_pred),
            }
        )
    return subject_rows


def best_by_calibration_count(summary: list[dict]) -> list[dict]:
    best = []
    for calibration_run_count in sorted(set(row["calibration_run_count"] for row in summary)):
        candidates = [row for row in summary if row["calibration_run_count"] == calibration_run_count]
        best.append(
            max(
                candidates,
                key=lambda row: (
                    row["metrics"]["accuracy"],
                    row["metrics"]["macro_f1"],
                    row["coarse_metrics"]["leg_vs_arm_accuracy"],
                ),
            )
        )
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether a few labeled runs from the held-out subject can calibrate "
            "subject-specific centroids on saved event features."
        )
    )
    parser.add_argument("--feature-dir", required=True, help="Directory containing features.npy, labels.npy, records.json.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--clip-offset", type=int, default=2)
    parser.add_argument("--max-calibration-runs", type=int, default=5)
    parser.add_argument(
        "--blend-alphas",
        type=float,
        nargs="*",
        default=[0.25, 0.5, 0.75, 1.0],
        help="Weight assigned to subject-specific calibration centroids when blending with source centroids.",
    )
    parser.add_argument(
        "--focus-subjects",
        nargs="*",
        default=["sub-52", "sub-42", "sub-17", "sub-20", "sub-26", "sub-27", "sub-54", "sub-63", "sub-30", "sub-62"],
    )
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    clip_x = np.load(feature_dir / "features.npy").astype(np.float32)
    clip_y = np.load(feature_dir / "labels.npy").astype(np.int64)
    clip_records = json.loads((feature_dir / "records.json").read_text())
    event_x, event_y, event_records = aggregate_events_for_offset(clip_x, clip_y, clip_records, args.clip_offset)
    x_centered = center_by_subject_run(event_x, event_records)
    x_norm = l2_normalize(x_centered.astype(np.float32))

    subjects = sorted(set(str(record["subject_id"]) for record in event_records))
    subject_to_indices = {
        subject: np.asarray(
            [idx for idx, record in enumerate(event_records) if str(record["subject_id"]) == subject],
            dtype=np.int64,
        )
        for subject in subjects
    }

    prediction_rows = []
    for subject in subjects:
        subject_indices = subject_to_indices[subject]
        run_ids = sorted(set(int(event_records[idx]["run_id"]) for idx in subject_indices))
        source_indices = np.asarray(
            [idx for idx, record in enumerate(event_records) if str(record["subject_id"]) != subject],
            dtype=np.int64,
        )
        source_centroids = centroid_matrix_from_normalized(x_norm, event_y, source_indices)

        for holdout_run in run_ids:
            val_idx = np.asarray(
                [idx for idx in subject_indices if int(event_records[idx]["run_id"]) == holdout_run],
                dtype=np.int64,
            )
            source_scores = score(x_norm, val_idx, source_centroids)
            append_prediction_rows(
                prediction_rows,
                subject=subject,
                holdout_run=holdout_run,
                protocol="source_only",
                calibration_runs=[],
                calibration_run_count=0,
                blend_alpha=0.0,
                scores=source_scores,
                val_idx=val_idx,
                y=event_y,
                records=event_records,
            )

            available_runs = [run_id for run_id in run_ids if run_id != holdout_run]
            max_k = min(args.max_calibration_runs, len(available_runs))
            for calibration_run_count in range(1, max_k + 1):
                for calibration_runs in itertools.combinations(available_runs, calibration_run_count):
                    calibration_idx = np.asarray(
                        [
                            idx
                            for idx in subject_indices
                            if int(event_records[idx]["run_id"]) in set(calibration_runs)
                        ],
                        dtype=np.int64,
                    )
                    subject_centroids = centroid_matrix_from_normalized(x_norm, event_y, calibration_idx)
                    for alpha in args.blend_alphas:
                        centroids = blend_centroids(source_centroids, subject_centroids, alpha)
                        scores = score(x_norm, val_idx, centroids)
                        protocol = "subject_only" if np.isclose(alpha, 1.0) else "source_subject_blend"
                        append_prediction_rows(
                            prediction_rows,
                            subject=subject,
                            holdout_run=holdout_run,
                            protocol=protocol,
                            calibration_runs=[int(run_id) for run_id in calibration_runs],
                            calibration_run_count=calibration_run_count,
                            blend_alpha=alpha,
                            scores=scores,
                            val_idx=val_idx,
                            y=event_y,
                            records=event_records,
                        )

    summary = summarize_prediction_rows(prediction_rows)
    subject_summary = summarize_by_subject(prediction_rows, set(args.focus_subjects))
    best_rows = best_by_calibration_count(summary)
    result = {
        "feature_dir": str(feature_dir),
        "clip_offset": int(args.clip_offset),
        "event_feature_shape": list(event_x.shape),
        "max_calibration_runs": int(args.max_calibration_runs),
        "blend_alphas": [float(alpha) for alpha in args.blend_alphas],
        "summary": summary,
        "best_by_calibration_run_count": best_rows,
        "focus_subject_summary": subject_summary,
        "rows": prediction_rows,
        "note": (
            "Features are centered within each subject-run using unlabeled run statistics. "
            "Source-only uses all non-target subjects. Calibration protocols use labeled runs "
            "from the target subject to predict a different run from the same subject."
        ),
    }

    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "clip_offset": args.clip_offset,
                "event_feature_shape": result["event_feature_shape"],
                "best_by_calibration_run_count": [
                    {
                        "calibration_run_count": row["calibration_run_count"],
                        "protocol": row["protocol"],
                        "prediction_rule": row["prediction_rule"],
                        "blend_alpha": row["blend_alpha"],
                        "accuracy": row["metrics"]["accuracy"],
                        "macro_f1": row["metrics"]["macro_f1"],
                        "leg_vs_arm_accuracy": row["coarse_metrics"]["leg_vs_arm_accuracy"],
                    }
                    for row in best_rows
                ],
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
