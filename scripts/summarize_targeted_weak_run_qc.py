#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


MOTION_METRICS = [
    "dvars_p95_percent",
    "phase_fd_p95_mm",
    "phase_absolute_p95_mm",
    "center_fd_p95_mm",
    "center_absolute_p95_mm",
    "event_phase_fd_class_range",
    "event_dvars_class_range",
]


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def correlation_summary(rows: list[dict]) -> dict:
    geometry = np.asarray([row["event_geometry"] for row in rows])
    result = {}
    for metric in MOTION_METRICS:
        values = np.asarray([row[metric] for row in rows])
        result[metric] = {
            "pearson": correlation(values, geometry),
            "spearman": correlation(rankdata(values), rankdata(geometry)),
        }
    return result


def centered_correlation(subjects: dict) -> dict:
    centered_rows = []
    for subject, summary in subjects.items():
        rows = summary["runs"]
        geometry_mean = float(np.mean([row["event_geometry"] for row in rows]))
        metric_means = {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in MOTION_METRICS
        }
        for row in rows:
            centered_rows.append(
                {
                    "subject": subject,
                    "event_geometry": row["event_geometry"] - geometry_mean,
                    **{
                        metric: row[metric] - metric_means[metric]
                        for metric in MOTION_METRICS
                    },
                }
            )
    return correlation_summary(centered_rows)


def targeted_run_summary(label: str, run: dict) -> dict:
    default = run["event_geometry"]
    best_raw = max(
        run["event_geometry_sweep"],
        key=lambda row: row["same_minus_different_cosine"],
    )
    best_detrended = max(
        run["event_geometry_sweep"],
        key=lambda row: row["linearly_detrended_same_minus_different_cosine"],
    )
    return {
        "label": label,
        "temporal_snr_median": run["temporal_snr_median"],
        "combined_spike_fraction": run["combined_spike_fraction"],
        "default_offset": default["start_offset_volumes"],
        "default_length": default["window_length_volumes"],
        "raw_same_minus_different": default["same_minus_different_cosine"],
        "raw_leave_one_event_accuracy": default["leave_one_event_accuracy"],
        "detrended_same_minus_different": default[
            "linearly_detrended_same_minus_different_cosine"
        ],
        "detrended_leave_one_event_accuracy": default[
            "linearly_detrended_leave_one_event_accuracy"
        ],
        "best_raw": {
            "offset": best_raw["start_offset_volumes"],
            "length": best_raw["window_length_volumes"],
            "same_minus_different": best_raw["same_minus_different_cosine"],
            "leave_one_event_accuracy": best_raw["leave_one_event_accuracy"],
        },
        "best_detrended": {
            "offset": best_detrended["start_offset_volumes"],
            "length": best_detrended["window_length_volumes"],
            "same_minus_different": best_detrended[
                "linearly_detrended_same_minus_different_cosine"
            ],
            "leave_one_event_accuracy": best_detrended[
                "linearly_detrended_leave_one_event_accuracy"
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize targeted weak-run raw QC and image-derived motion."
    )
    parser.add_argument("--raw-qc-json", required=True)
    parser.add_argument("--motion-json", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    raw_qc = json.loads(Path(args.raw_qc_json).read_text())
    motion = json.loads(Path(args.motion_json).read_text())
    subjects = motion["subject_summaries"]
    result = {
        "raw_qc_json": args.raw_qc_json,
        "motion_json": args.motion_json,
        "targeted_runs": [
            targeted_run_summary(label, run)
            for label, run in raw_qc["runs"].items()
        ],
        "subject_motion_geometry_correlations": {
            subject: correlation_summary(summary["runs"])
            for subject, summary in subjects.items()
        },
        "pooled_within_subject_centered_correlations": centered_correlation(subjects),
        "note": (
            "Motion correlations use the offset-3 length-8 linearly detrended raw event "
            "geometry computed by audit_image_derived_motion.py. With six runs per subject, "
            "subject-specific correlations are descriptive and should not be treated as "
            "confirmatory tests."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
