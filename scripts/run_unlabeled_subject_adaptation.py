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


def centroid_matrix_from_labels(x_norm: np.ndarray, y: np.ndarray, indices: np.ndarray) -> np.ndarray | None:
    centroids = []
    for class_idx in range(len(CLASS_NAMES)):
        class_indices = indices[y == class_idx]
        if len(class_indices) == 0:
            return None
        centroids.append(x_norm[class_indices].mean(axis=0))
    return l2_normalize(np.stack(centroids, axis=0))


def score(x_norm: np.ndarray, indices: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return x_norm[indices].astype(np.float64) @ centroids.astype(np.float64).T


def blend_centroids(source: np.ndarray, subject: np.ndarray, alpha: float) -> np.ndarray:
    return l2_normalize((1.0 - float(alpha)) * source + float(alpha) * subject)


def prediction_variants(scores: np.ndarray, val_idx: np.ndarray, records: list[dict]) -> list[tuple[str, np.ndarray]]:
    return [
        ("independent_argmax", scores.argmax(axis=1).astype(np.int64)),
        ("balanced_subject_run_assignment", apply_balanced_assignment(scores, val_idx, records)),
        (
            "gated_balanced_imbalance_l1_4",
            apply_imbalance_gated_balanced_assignment(scores, val_idx, records, 4.0),
        ),
    ]


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, int, float], dict[str, list]] = defaultdict(
        lambda: {"true": [], "pred": [], "split_accuracy": [], "pseudo_accuracy": []}
    )
    for row in rows:
        key = (
            row["protocol"],
            row["pseudo_label_rule"],
            row["prediction_rule"],
            int(row["calibration_run_count"]),
            float(row["blend_alpha"]),
        )
        grouped[key]["true"].extend(row["y_true"])
        grouped[key]["pred"].extend(row["y_pred"])
        grouped[key]["split_accuracy"].append(row["accuracy"])
        if row["calibration_pseudo_accuracy"] is not None:
            grouped[key]["pseudo_accuracy"].append(row["calibration_pseudo_accuracy"])

    summary_rows = []
    for (protocol, pseudo_label_rule, prediction_rule, calibration_run_count, alpha), payload in sorted(grouped.items()):
        y_true = np.asarray(payload["true"], dtype=np.int64)
        y_pred = np.asarray(payload["pred"], dtype=np.int64)
        summary_rows.append(
            {
                "protocol": protocol,
                "pseudo_label_rule": pseudo_label_rule,
                "prediction_rule": prediction_rule,
                "calibration_run_count": calibration_run_count,
                "blend_alpha": alpha,
                "split_count": len(payload["split_accuracy"]),
                "event_count": int(len(y_true)),
                "mean_split_accuracy": float(np.mean(payload["split_accuracy"])),
                "mean_calibration_pseudo_accuracy": (
                    float(np.mean(payload["pseudo_accuracy"])) if payload["pseudo_accuracy"] else None
                ),
                "metrics": metrics(y_true, y_pred),
                "coarse_metrics": coarse_metrics(y_true, y_pred),
            }
        )
    return summary_rows


def summarize_by_subject(rows: list[dict], focus_subjects: set[str]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, int, float], dict[str, list]] = defaultdict(
        lambda: {"true": [], "pred": [], "split_accuracy": [], "pseudo_accuracy": []}
    )
    for row in rows:
        if focus_subjects and row["subject"] not in focus_subjects:
            continue
        key = (
            row["subject"],
            row["protocol"],
            row["pseudo_label_rule"],
            row["prediction_rule"],
            int(row["calibration_run_count"]),
            float(row["blend_alpha"]),
        )
        grouped[key]["true"].extend(row["y_true"])
        grouped[key]["pred"].extend(row["y_pred"])
        grouped[key]["split_accuracy"].append(row["accuracy"])
        if row["calibration_pseudo_accuracy"] is not None:
            grouped[key]["pseudo_accuracy"].append(row["calibration_pseudo_accuracy"])

    subject_rows = []
    for (subject, protocol, pseudo_label_rule, prediction_rule, calibration_run_count, alpha), payload in sorted(
        grouped.items()
    ):
        y_true = np.asarray(payload["true"], dtype=np.int64)
        y_pred = np.asarray(payload["pred"], dtype=np.int64)
        subject_rows.append(
            {
                "subject": subject,
                "protocol": protocol,
                "pseudo_label_rule": pseudo_label_rule,
                "prediction_rule": prediction_rule,
                "calibration_run_count": calibration_run_count,
                "blend_alpha": alpha,
                "split_count": len(payload["split_accuracy"]),
                "event_count": int(len(y_true)),
                "mean_split_accuracy": float(np.mean(payload["split_accuracy"])),
                "mean_calibration_pseudo_accuracy": (
                    float(np.mean(payload["pseudo_accuracy"])) if payload["pseudo_accuracy"] else None
                ),
                "metrics": metrics(y_true, y_pred),
                "coarse_metrics": coarse_metrics(y_true, y_pred),
            }
        )
    return subject_rows


def best_by_calibration_count(summary_rows: list[dict]) -> list[dict]:
    best = []
    for count in sorted(set(row["calibration_run_count"] for row in summary_rows)):
        candidates = [row for row in summary_rows if row["calibration_run_count"] == count]
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


def append_evaluation_rows(
    rows: list[dict],
    *,
    subject: str,
    holdout_run: int,
    protocol: str,
    pseudo_label_rule: str,
    calibration_runs: list[int],
    calibration_run_count: int,
    blend_alpha: float,
    calibration_pseudo_accuracy: float | None,
    scores: np.ndarray,
    val_idx: np.ndarray,
    y_true: np.ndarray,
    records: list[dict],
) -> None:
    for prediction_rule, pred in prediction_variants(scores, val_idx, records):
        rows.append(
            {
                "subject": subject,
                "holdout_run": int(holdout_run),
                "protocol": protocol,
                "pseudo_label_rule": pseudo_label_rule,
                "prediction_rule": prediction_rule,
                "calibration_runs": calibration_runs,
                "calibration_run_count": int(calibration_run_count),
                "blend_alpha": float(blend_alpha),
                "calibration_pseudo_accuracy": calibration_pseudo_accuracy,
                "accuracy": float(np.mean(y_true[val_idx] == pred)),
                "y_true": y_true[val_idx].tolist(),
                "y_pred": pred.tolist(),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use pseudo-labeled target-subject calibration runs to adapt subject centroids, "
            "then evaluate a different held-out target-subject run."
        )
    )
    parser.add_argument("--feature-dir", required=True, help="Directory containing features.npy, labels.npy, records.json.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--clip-offset", type=int, default=2)
    parser.add_argument("--max-calibration-runs", type=int, default=5)
    parser.add_argument("--blend-alphas", type=float, nargs="*", default=[0.25, 0.5, 0.75, 1.0])
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

    rows = []
    skipped = []
    for subject in subjects:
        subject_indices = subject_to_indices[subject]
        run_ids = sorted(set(int(event_records[idx]["run_id"]) for idx in subject_indices))
        source_idx = np.asarray(
            [idx for idx, record in enumerate(event_records) if str(record["subject_id"]) != subject],
            dtype=np.int64,
        )
        source_centroids = centroid_matrix_from_labels(x_norm, event_y[source_idx], source_idx)
        if source_centroids is None:
            raise RuntimeError(f"Missing source class while holding out {subject}.")

        for holdout_run in run_ids:
            val_idx = np.asarray(
                [idx for idx in subject_indices if int(event_records[idx]["run_id"]) == holdout_run],
                dtype=np.int64,
            )
            source_scores = score(x_norm, val_idx, source_centroids)
            append_evaluation_rows(
                rows,
                subject=subject,
                holdout_run=holdout_run,
                protocol="source_only",
                pseudo_label_rule="none",
                calibration_runs=[],
                calibration_run_count=0,
                blend_alpha=0.0,
                calibration_pseudo_accuracy=None,
                scores=source_scores,
                val_idx=val_idx,
                y_true=event_y,
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
                    calibration_scores = score(x_norm, calibration_idx, source_centroids)
                    for pseudo_label_rule, pseudo_y in prediction_variants(
                        calibration_scores, calibration_idx, event_records
                    ):
                        pseudo_accuracy = float(np.mean(event_y[calibration_idx] == pseudo_y))
                        pseudo_centroids = centroid_matrix_from_labels(x_norm, pseudo_y, calibration_idx)
                        if pseudo_centroids is None:
                            skipped.append(
                                {
                                    "subject": subject,
                                    "holdout_run": int(holdout_run),
                                    "calibration_runs": [int(run_id) for run_id in calibration_runs],
                                    "pseudo_label_rule": pseudo_label_rule,
                                    "reason": "missing_pseudo_class",
                                }
                            )
                            continue
                        for alpha in args.blend_alphas:
                            centroids = blend_centroids(source_centroids, pseudo_centroids, alpha)
                            scores = score(x_norm, val_idx, centroids)
                            append_evaluation_rows(
                                rows,
                                subject=subject,
                                holdout_run=holdout_run,
                                protocol="unlabeled_pseudo_subject_blend",
                                pseudo_label_rule=pseudo_label_rule,
                                calibration_runs=[int(run_id) for run_id in calibration_runs],
                                calibration_run_count=calibration_run_count,
                                blend_alpha=alpha,
                                calibration_pseudo_accuracy=pseudo_accuracy,
                                scores=scores,
                                val_idx=val_idx,
                                y_true=event_y,
                                records=event_records,
                            )

    summary_rows = summarize(rows)
    result = {
        "feature_dir": str(feature_dir),
        "clip_offset": int(args.clip_offset),
        "event_feature_shape": list(event_x.shape),
        "max_calibration_runs": int(args.max_calibration_runs),
        "blend_alphas": [float(alpha) for alpha in args.blend_alphas],
        "summary": summary_rows,
        "best_by_calibration_run_count": best_by_calibration_count(summary_rows),
        "focus_subject_summary": summarize_by_subject(rows, set(args.focus_subjects)),
        "skipped": skipped,
        "rows": rows,
        "note": (
            "Features are centered within each subject-run using unlabeled run statistics. "
            "Calibration runs from the target subject are pseudo-labeled by source centroids and task-design "
            "assignment rules; their true labels are used only for final evaluation."
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
                        "pseudo_label_rule": row["pseudo_label_rule"],
                        "prediction_rule": row["prediction_rule"],
                        "blend_alpha": row["blend_alpha"],
                        "pseudo_accuracy": row["mean_calibration_pseudo_accuracy"],
                        "accuracy": row["metrics"]["accuracy"],
                        "macro_f1": row["metrics"]["macro_f1"],
                        "leg_vs_arm_accuracy": row["coarse_metrics"]["leg_vs_arm_accuracy"],
                    }
                    for row in result["best_by_calibration_run_count"]
                ],
                "skipped_count": len(skipped),
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
