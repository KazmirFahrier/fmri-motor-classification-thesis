#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


CLASS_NAMES = [
    "Left leg movements",
    "Right leg movements",
    "Forearm movements",
    "Upper arm movements",
]


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-12:
        return np.zeros_like(values)
    return 0.67448975 * (values - median) / mad


def l2_normalize(values: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    return np.nan_to_num(values / denom, nan=0.0, posinf=0.0, neginf=0.0)


def load_events(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [
        {
            "onset": float(row["onset"]),
            "duration": float(row["duration"]),
            "trial_type": row["trial_type"],
        }
        for row in rows
        if row["trial_type"] in CLASS_NAMES
    ]


def pattern_metrics(patterns: np.ndarray, labels: np.ndarray) -> dict:
    x = l2_normalize(patterns)
    similarities = x @ x.T
    same = []
    different = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            target = same if labels[left] == labels[right] else different
            target.append(float(similarities[left, right]))

    predicted = []
    for val_idx in range(len(labels)):
        train_idx = np.asarray([idx for idx in range(len(labels)) if idx != val_idx])
        centroids = []
        for class_idx in range(len(CLASS_NAMES)):
            centroid = x[train_idx][labels[train_idx] == class_idx].mean(axis=0)
            centroids.append(centroid)
        centroids = l2_normalize(np.stack(centroids))
        predicted.append(int(np.argmax(x[val_idx] @ centroids.T)))
    return {
        "same_class_cosine_mean": float(np.mean(same)),
        "different_class_cosine_mean": float(np.mean(different)),
        "same_minus_different_cosine": float(np.mean(same) - np.mean(different)),
        "leave_one_event_accuracy": float(
            np.mean(labels == np.asarray(predicted, dtype=np.int64))
        ),
    }


def event_geometry(
    data: np.ndarray,
    mask: np.ndarray,
    events: list[dict],
    repetition_time: float,
    start_offset: int,
    window_length: int,
) -> dict:
    patterns = []
    labels = []
    event_rows = []
    for event in events:
        onset_volume = int(round(event["onset"] / repetition_time))
        start = onset_volume + start_offset
        stop = min(start + window_length, data.shape[-1])
        if stop - start != window_length:
            continue
        pattern = data[..., start:stop].mean(axis=-1)[mask].astype(np.float32)
        patterns.append(pattern)
        labels.append(CLASS_NAMES.index(event["trial_type"]))
        event_rows.append(
            {
                "trial_type": event["trial_type"],
                "onset_volume": onset_volume,
                "window_start": start,
                "window_stop_exclusive": stop,
            }
        )

    patterns_array = np.stack(patterns).astype(np.float32)
    patterns_array -= patterns_array.mean(axis=0, keepdims=True)
    for row, pattern in zip(event_rows, patterns_array):
        row["centered_pattern_norm"] = float(np.linalg.norm(pattern))
    y = np.asarray(labels, dtype=np.int64)
    metrics = pattern_metrics(patterns_array, y)

    event_times = np.asarray(
        [row["window_start"] for row in event_rows],
        dtype=np.float64,
    )
    event_times -= event_times.mean()
    event_times /= max(float(np.linalg.norm(event_times)), 1e-8)
    patterns_for_detrending = patterns_array.astype(np.float64)
    time_weights = np.sum(event_times[:, None] * patterns_for_detrending, axis=0)
    fitted_time = np.outer(event_times, time_weights)
    detrended_patterns = patterns_for_detrending - fitted_time
    detrended_metrics = pattern_metrics(detrended_patterns, y)
    total_energy = float(np.sum(patterns_for_detrending ** 2))
    time_energy = float(np.sum(fitted_time ** 2))

    return {
        "start_offset_volumes": start_offset,
        "window_length_volumes": window_length,
        "event_count": len(event_rows),
        **metrics,
        "linear_time_variance_fraction": time_energy / max(total_energy, 1e-8),
        "linearly_detrended_same_minus_different_cosine": detrended_metrics[
            "same_minus_different_cosine"
        ],
        "linearly_detrended_leave_one_event_accuracy": detrended_metrics[
            "leave_one_event_accuracy"
        ],
        "events": event_rows,
    }


def summarize_run(
    bold_path: Path,
    events_path: Path,
    repetition_time: float,
    start_offset: int,
    window_length: int,
    geometry_offsets: list[int],
    geometry_window_lengths: list[int],
) -> dict:
    image = nib.load(bold_path)
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected a 4D BOLD image, got {data.shape} for {bold_path}.")

    temporal_mean = data.mean(axis=-1)
    positive = temporal_mean[temporal_mean > 0]
    mask_threshold = float(np.percentile(positive, 20))
    mask = temporal_mean > mask_threshold
    masked = data[mask].astype(np.float32, copy=False)
    global_signal = masked.mean(axis=0)
    dvars = np.sqrt(np.mean(np.diff(masked, axis=1) ** 2, axis=0))
    voxel_std = masked.std(axis=1)
    valid_tsnr = voxel_std > 1e-8
    tsnr = masked.mean(axis=1)[valid_tsnr] / voxel_std[valid_tsnr]
    global_z = robust_z(global_signal)
    dvars_z = robust_z(dvars)
    spike_frames = sorted(
        set(np.flatnonzero(np.abs(global_z) > 3.5).tolist())
        | set((np.flatnonzero(dvars_z > 3.5) + 1).tolist())
    )

    events = load_events(events_path)
    geometry = event_geometry(
        data,
        mask,
        events,
        repetition_time,
        start_offset,
        window_length,
    )
    geometry_sweep = [
        event_geometry(
            data,
            mask,
            events,
            repetition_time,
            offset,
            length,
        )
        for length in geometry_window_lengths
        for offset in geometry_offsets
        if offset + length <= 8
    ]
    return {
        "bold_path": str(bold_path),
        "events_path": str(events_path),
        "shape": list(data.shape),
        "voxel_sizes": [float(value) for value in image.header.get_zooms()[:3]],
        "repetition_time": repetition_time,
        "mask_voxels": int(mask.sum()),
        "mask_threshold": mask_threshold,
        "temporal_snr_median": float(np.median(tsnr)),
        "temporal_snr_p10": float(np.percentile(tsnr, 10)),
        "global_signal_mean": float(np.mean(global_signal)),
        "global_signal_coefficient_of_variation": float(
            np.std(global_signal) / max(abs(np.mean(global_signal)), 1e-8)
        ),
        "dvars_median": float(np.median(dvars)),
        "dvars_p95": float(np.percentile(dvars, 95)),
        "dvars_p95_to_median": float(np.percentile(dvars, 95) / max(np.median(dvars), 1e-8)),
        "global_signal_outlier_frames": np.flatnonzero(np.abs(global_z) > 3.5).tolist(),
        "dvars_outlier_frames": (np.flatnonzero(dvars_z > 3.5) + 1).tolist(),
        "combined_spike_frames": spike_frames,
        "combined_spike_fraction": float(len(spike_frames) / data.shape[-1]),
        "event_geometry": geometry,
        "event_geometry_sweep": geometry_sweep,
        "best_event_geometry": max(
            geometry_sweep,
            key=lambda row: row["same_minus_different_cosine"],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute targeted source-level BOLD QC and within-run event geometry."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("LABEL", "BOLD_NIFTI", "EVENTS_TSV"),
        required=True,
        help="Repeat for each run to compare.",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--repetition-time", type=float, default=2.0)
    parser.add_argument("--start-offset", type=int, default=2)
    parser.add_argument("--window-length", type=int, default=6)
    parser.add_argument("--geometry-offsets", default="0,1,2,3,4,5,6")
    parser.add_argument("--geometry-window-lengths", default="2,4,6")
    args = parser.parse_args()
    geometry_offsets = [int(value) for value in args.geometry_offsets.split(",")]
    geometry_window_lengths = [
        int(value) for value in args.geometry_window_lengths.split(",")
    ]

    runs = {}
    for label, bold_path, events_path in args.run:
        runs[label] = summarize_run(
            Path(bold_path),
            Path(events_path),
            args.repetition_time,
            args.start_offset,
            args.window_length,
            geometry_offsets,
            geometry_window_lengths,
        )

    result = {
        "repetition_time": args.repetition_time,
        "start_offset_volumes": args.start_offset,
        "window_length_volumes": args.window_length,
        "runs": runs,
        "rankings": {
            "lowest_temporal_snr": sorted(
                runs, key=lambda label: runs[label]["temporal_snr_median"]
            ),
            "highest_spike_fraction": sorted(
                runs,
                key=lambda label: runs[label]["combined_spike_fraction"],
                reverse=True,
            ),
            "weakest_event_geometry": sorted(
                runs,
                key=lambda label: runs[label]["event_geometry"]["same_minus_different_cosine"],
            ),
            "weakest_best_event_geometry": sorted(
                runs,
                key=lambda label: runs[label]["best_event_geometry"]["same_minus_different_cosine"],
            ),
        },
        "note": (
            "This audit uses raw source BOLD without motion correction. DVARS/global-signal spikes can flag "
            "source instability, while event geometry tests whether offset-2 event patterns are already weak "
            "before the project feature-extraction pipeline."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
