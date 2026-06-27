#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import boto3
import nibabel as nib
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config

from audit_targeted_raw_runs import CLASS_NAMES, event_geometry, robust_z


TRIAL_TYPE_CODE_MAP = {
    0: "Rest",
    1: "Toe movements",
    2: "Ankle movements",
    3: "Left leg movements",
    4: "Right leg movements",
    5: "Forearm movements",
    6: "Upper arm movements",
    7: "Wrist movements",
    8: "Finger movements",
    9: "Eye movements",
    10: "Jaw movements",
    11: "Lip movements",
    12: "Tongue movements",
}


def normalize_trial_type(value: str) -> str:
    value = str(value).strip()
    if value in CLASS_NAMES:
        return value
    try:
        code = int(float(value))
    except ValueError:
        return value
    return TRIAL_TYPE_CODE_MAP.get(code, value)


def load_events(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    events = []
    for row in rows:
        trial_type = normalize_trial_type(row["trial_type"])
        if trial_type not in CLASS_NAMES:
            continue
        events.append(
            {
                "onset": float(row["onset"]),
                "duration": float(row["duration"]),
                "trial_type": trial_type,
            }
        )
    return events


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run_id_from_key(key: str) -> int:
    return int(key.split("_run-")[1].split("_")[0])


def list_source_files(s3, bucket: str, dataset: str, subjects: list[str]) -> dict:
    result = {}
    paginator = s3.get_paginator("list_objects_v2")
    for subject in subjects:
        prefix = f"{dataset}/{subject}/ses-1/func/"
        bold = {}
        events = {}
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = str(obj["Key"])
                if "_run-" not in key:
                    continue
                run_id = run_id_from_key(key)
                if key.endswith("_bold.nii.gz"):
                    bold[run_id] = key
                elif key.endswith("_events.tsv"):
                    events[run_id] = key
        common = sorted(set(bold) & set(events))
        result[subject] = [
            {"run_id": run_id, "bold_key": bold[run_id], "events_key": events[run_id]}
            for run_id in common
        ]
    return result


def find_cached(cache_dirs: list[Path], subject: str, run_id: int, suffix: str) -> Path | None:
    names = [
        f"{subject}_run-{run_id:02d}_{suffix}",
        f"{subject}_run-{run_id}_{suffix}",
    ]
    for cache_dir in cache_dirs:
        for name in names:
            candidate = cache_dir / name
            if candidate.exists():
                return candidate
    return None


def parabolic_peak(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return 0.0
    left = float(values[index - 1])
    center = float(values[index])
    right = float(values[index + 1])
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def phase_translation(
    volume: np.ndarray,
    reference_fft: np.ndarray,
    mask: np.ndarray,
    taper: np.ndarray,
    max_shift: int,
) -> tuple[np.ndarray, float]:
    values = volume[mask]
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[mask] = (values - values.mean()) / max(float(values.std()), 1e-8)
    spectrum = np.fft.fftn(normalized * taper)
    cross_power = spectrum * np.conj(reference_fft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-8)
    correlation = np.abs(np.fft.fftshift(np.fft.ifftn(cross_power)))

    center = np.asarray(correlation.shape) // 2
    slices = tuple(
        slice(int(axis_center - max_shift), int(axis_center + max_shift + 1))
        for axis_center in center
    )
    local = correlation[slices]
    local_peak = np.asarray(np.unravel_index(np.argmax(local), local.shape))
    peak = local_peak + center - max_shift
    subpixel = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        line_slices = [int(value) for value in peak]
        line_slices[axis] = slice(None)
        line = correlation[tuple(line_slices)]
        subpixel[axis] = parabolic_peak(line, int(peak[axis]))
    shift = peak.astype(np.float64) + subpixel - center
    quality = float(correlation[tuple(peak)] / max(float(np.mean(local)), 1e-8))
    return shift, quality


def center_trajectory(data: np.ndarray, mask: np.ndarray, voxel_sizes: np.ndarray) -> np.ndarray:
    coordinates = np.indices(mask.shape, dtype=np.float32)
    masked = data[mask].astype(np.float64, copy=False)
    baseline = np.percentile(masked, 2, axis=0, keepdims=True)
    weights = np.maximum(masked - baseline, 0.0)
    denominator = np.maximum(weights.sum(axis=0), 1e-8)
    centers = []
    for axis in range(3):
        coordinate = coordinates[axis][mask].astype(np.float64)
        centers.append((coordinate[:, None] * weights).sum(axis=0) / denominator)
    return np.stack(centers, axis=1) * voxel_sizes[None, :]


def trajectory_summary(trajectory: np.ndarray) -> dict:
    centered = trajectory - np.median(trajectory, axis=0, keepdims=True)
    absolute = np.linalg.norm(centered, axis=1)
    framewise = np.r_[0.0, np.linalg.norm(np.diff(trajectory, axis=0), axis=1)]
    return {
        "median_absolute_mm": float(np.median(absolute)),
        "p95_absolute_mm": float(np.percentile(absolute, 95)),
        "max_absolute_mm": float(np.max(absolute)),
        "median_framewise_mm": float(np.median(framewise[1:])),
        "p95_framewise_mm": float(np.percentile(framewise[1:], 95)),
        "max_framewise_mm": float(np.max(framewise[1:])),
        "outlier_frames": np.flatnonzero(robust_z(framewise[1:]) > 3.5).astype(int) + 1,
        "trajectory_mm": trajectory,
        "absolute_mm": absolute,
        "framewise_mm": framewise,
    }


def event_motion_summary(
    events: list[dict],
    repetition_time: float,
    frame_metrics: dict[str, np.ndarray],
    total_frames: int,
    start_offset: int,
    window_length: int,
) -> dict:
    event_rows = []
    occupied = np.zeros(total_frames, dtype=bool)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        start = int(round(event["onset"] / repetition_time)) + start_offset
        stop = start + window_length
        if start < 0 or stop > total_frames:
            continue
        occupied[start:stop] = True
        row = {
            "trial_type": event["trial_type"],
            "window_start": start,
            "window_stop_exclusive": stop,
        }
        for name, values in frame_metrics.items():
            window = values[start:stop]
            row[f"mean_{name}"] = float(np.mean(window))
            row[f"max_{name}"] = float(np.max(window))
        event_rows.append(row)
        by_class[event["trial_type"]].append(row)

    class_rows = {}
    for class_name in CLASS_NAMES:
        rows = by_class[class_name]
        class_rows[class_name] = {
            f"mean_{key}": float(np.mean([row[key] for row in rows]))
            for key in rows[0]
            if key.startswith("mean_") or key.startswith("max_")
        }
    rest = {
        f"mean_{name}": float(np.mean(values[~occupied]))
        for name, values in frame_metrics.items()
    }
    class_dispersion = {}
    for name in frame_metrics:
        class_means = np.asarray(
            [class_rows[class_name][f"mean_mean_{name}"] for class_name in CLASS_NAMES]
        )
        class_dispersion[name] = {
            "standard_deviation": float(np.std(class_means)),
            "range": float(np.ptp(class_means)),
        }
    return {
        "start_offset_volumes": start_offset,
        "window_length_volumes": window_length,
        "events": event_rows,
        "classes": class_rows,
        "rest": rest,
        "class_dispersion": class_dispersion,
    }


def summarize_run(
    label: str,
    bold_path: Path,
    events_path: Path,
    repetition_time: float,
    downsample: int,
    max_shift: int,
    event_offset: int,
    event_length: int,
) -> dict:
    image = nib.load(bold_path)
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD, got {data.shape}: {bold_path}")
    voxel_sizes = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    temporal_mean = data.mean(axis=-1)
    positive = temporal_mean[temporal_mean > 0]
    threshold = float(np.percentile(positive, 20))
    mask = temporal_mean > threshold

    masked = data[mask].astype(np.float32, copy=False)
    global_signal = masked.mean(axis=0, dtype=np.float64)
    dvars = np.r_[0.0, np.sqrt(np.mean(np.diff(masked, axis=1) ** 2, axis=0))]
    median_signal = max(float(np.median(np.abs(global_signal))), 1e-8)
    normalized_dvars = 100.0 * dvars / median_signal

    coarse = data[::downsample, ::downsample, ::downsample, :]
    coarse_mask = mask[::downsample, ::downsample, ::downsample]
    reference = np.median(coarse, axis=-1)
    reference_values = reference[coarse_mask]
    normalized_reference = np.zeros_like(reference, dtype=np.float32)
    normalized_reference[coarse_mask] = (
        reference_values - reference_values.mean()
    ) / max(float(reference_values.std()), 1e-8)
    windows = [np.hanning(size).astype(np.float32) for size in coarse.shape[:3]]
    taper = (
        windows[0][:, None, None]
        * windows[1][None, :, None]
        * windows[2][None, None, :]
    )
    reference_fft = np.fft.fftn(normalized_reference * taper)
    shifts = []
    phase_quality = []
    for frame in range(coarse.shape[-1]):
        shift, quality = phase_translation(
            coarse[..., frame], reference_fft, coarse_mask, taper, max_shift
        )
        shifts.append(shift)
        phase_quality.append(quality)
    phase_mm = np.stack(shifts) * (voxel_sizes * downsample)[None, :]
    center_mm = center_trajectory(data, mask, voxel_sizes)
    phase_summary = trajectory_summary(phase_mm)
    center_summary = trajectory_summary(center_mm)

    events = load_events(events_path)
    frame_metrics = {
        "normalized_dvars_percent": normalized_dvars,
        "phase_framewise_mm": phase_summary["framewise_mm"],
        "phase_absolute_mm": phase_summary["absolute_mm"],
        "center_framewise_mm": center_summary["framewise_mm"],
        "center_absolute_mm": center_summary["absolute_mm"],
    }
    motion_by_event = event_motion_summary(
        events,
        repetition_time,
        frame_metrics,
        data.shape[-1],
        event_offset,
        event_length,
    )
    geometry = event_geometry(
        data,
        mask,
        events,
        repetition_time,
        event_offset,
        event_length,
    )
    return {
        "label": label,
        "shape": list(data.shape),
        "voxel_sizes": voxel_sizes,
        "mask_voxels": int(mask.sum()),
        "global_signal_coefficient_of_variation": float(
            np.std(global_signal) / median_signal
        ),
        "normalized_dvars_percent": {
            "median": float(np.median(normalized_dvars[1:])),
            "p95": float(np.percentile(normalized_dvars[1:], 95)),
            "max": float(np.max(normalized_dvars[1:])),
            "outlier_frames": np.flatnonzero(robust_z(normalized_dvars[1:]) > 3.5).astype(int) + 1,
        },
        "phase_translation": phase_summary,
        "signal_center": center_summary,
        "phase_peak_quality": {
            "median": float(np.median(phase_quality)),
            "p10": float(np.percentile(phase_quality, 10)),
        },
        "event_motion": motion_by_event,
        "source_event_geometry": geometry,
    }


def compact_run_summary(run: dict) -> dict:
    return {
        "label": run["label"],
        "dvars_p95_percent": run["normalized_dvars_percent"]["p95"],
        "phase_fd_p95_mm": run["phase_translation"]["p95_framewise_mm"],
        "phase_absolute_p95_mm": run["phase_translation"]["p95_absolute_mm"],
        "center_fd_p95_mm": run["signal_center"]["p95_framewise_mm"],
        "center_absolute_p95_mm": run["signal_center"]["p95_absolute_mm"],
        "event_geometry": run["source_event_geometry"][
            "linearly_detrended_same_minus_different_cosine"
        ],
        "event_accuracy": run["source_event_geometry"][
            "linearly_detrended_leave_one_event_accuracy"
        ],
        "event_phase_fd_class_range": run["event_motion"]["class_dispersion"][
            "phase_framewise_mm"
        ]["range"],
        "event_dvars_class_range": run["event_motion"]["class_dispersion"][
            "normalized_dvars_percent"
        ]["range"],
    }


def summarize_subject(runs: list[dict]) -> dict:
    compact = [compact_run_summary(run) for run in runs]
    metrics = [key for key in compact[0] if key != "label"]
    summary = {"run_count": len(compact), "runs": compact}
    for metric in metrics:
        values = np.asarray([row[metric] for row in compact], dtype=np.float64)
        summary[f"mean_{metric}"] = float(np.mean(values))
        summary[f"max_{metric}"] = float(np.max(values))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive translation and event-linked motion proxies directly from raw 4D BOLD."
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--bucket", default="openneuro.org")
    parser.add_argument("--dataset", default="ds004044")
    parser.add_argument("--subjects", nargs="*", default=["sub-42", "sub-52", "sub-62"])
    parser.add_argument("--cache-dir", action="append", default=[])
    parser.add_argument("--repetition-time", type=float, default=2.0)
    parser.add_argument("--downsample", type=int, default=3)
    parser.add_argument("--max-shift", type=int, default=4)
    parser.add_argument("--event-offset", type=int, default=3)
    parser.add_argument("--event-length", type=int, default=8)
    args = parser.parse_args()

    cache_dirs = [Path(value) for value in args.cache_dir]
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    source_files = list_source_files(s3, args.bucket, args.dataset, args.subjects)
    runs = []
    sources = {}
    with tempfile.TemporaryDirectory(prefix="image-motion-") as temp_dir:
        temp_root = Path(temp_dir)
        for subject in args.subjects:
            for source in source_files[subject]:
                run_id = source["run_id"]
                label = f"{subject}-run-{run_id}"
                bold = find_cached(cache_dirs, subject, run_id, "bold.nii.gz")
                events = find_cached(cache_dirs, subject, run_id, "events.tsv")
                bold_source = "cache" if bold is not None else "streamed_temp"
                events_source = "cache" if events is not None else "streamed_temp"
                if bold is None:
                    bold = temp_root / f"{subject}_run-{run_id:02d}_bold.nii.gz"
                    print(f"downloading BOLD {label}", flush=True)
                    s3.download_file(args.bucket, source["bold_key"], str(bold))
                if events is None:
                    events = temp_root / f"{subject}_run-{run_id:02d}_events.tsv"
                    s3.download_file(args.bucket, source["events_key"], str(events))
                print(f"summarizing motion {label}", flush=True)
                run = summarize_run(
                    label,
                    bold,
                    events,
                    args.repetition_time,
                    args.downsample,
                    args.max_shift,
                    args.event_offset,
                    args.event_length,
                )
                run["subject"] = subject
                run["run_id"] = run_id
                runs.append(run)
                sources[label] = {
                    "bold": bold_source,
                    "events": events_source,
                    "bold_key": source["bold_key"],
                    "events_key": source["events_key"],
                }
                if bold_source == "streamed_temp":
                    bold.unlink(missing_ok=True)
                if events_source == "streamed_temp":
                    events.unlink(missing_ok=True)

    by_subject = {
        subject: summarize_subject([run for run in runs if run["subject"] == subject])
        for subject in args.subjects
    }
    result = {
        "bucket": args.bucket,
        "dataset": args.dataset,
        "subjects": args.subjects,
        "repetition_time": args.repetition_time,
        "downsample": args.downsample,
        "event_offset": args.event_offset,
        "event_length": args.event_length,
        "sources": sources,
        "runs": runs,
        "subject_summaries": by_subject,
        "note": (
            "These are image-derived translation proxies, not fMRIPrep rigid-body parameters or "
            "formal framewise displacement. Raw BOLD is used so acquisition motion is not hidden "
            "by the public motion-corrected derivative."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(json.dumps({"out_json": args.out_json, "subject_summaries": by_subject}, indent=2))


if __name__ == "__main__":
    main()
