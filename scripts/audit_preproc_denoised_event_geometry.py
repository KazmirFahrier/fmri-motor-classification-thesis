#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import tempfile
from pathlib import Path

import boto3
import nibabel as nib
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config

from audit_targeted_raw_runs import pattern_metrics


CLASS_NAMES = [
    "Left leg movements",
    "Right leg movements",
    "Forearm movements",
    "Upper arm movements",
]

TRIAL_TYPE_CODE_MAP = {
    3: "Left leg movements",
    4: "Right leg movements",
    5: "Forearm movements",
    6: "Upper arm movements",
}


def load_events(path: Path, repetition_time: float) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    events = []
    for row in rows:
        value = str(row["trial_type"]).strip()
        if value not in CLASS_NAMES:
            try:
                value = TRIAL_TYPE_CODE_MAP.get(int(float(value)), value)
            except ValueError:
                pass
        if value not in CLASS_NAMES:
            continue
        onset = float(row["onset"])
        events.append(
            {
                "class_id": CLASS_NAMES.index(value),
                "event_start": int(round(onset / repetition_time)),
            }
        )
    return events


def robust_mask(data: np.ndarray) -> np.ndarray:
    temporal_mean = data.mean(axis=-1)
    positive = temporal_mean[temporal_mean > 0]
    threshold = float(np.percentile(positive, 20)) if len(positive) else 0.0
    return temporal_mean > threshold


def event_patterns(
    data: np.ndarray,
    events: list[dict],
    mask: np.ndarray,
    offset: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    patterns = []
    labels = []
    starts = []
    for event in events:
        start = int(event["event_start"]) + offset
        stop = start + length
        if stop > data.shape[-1]:
            continue
        patterns.append(data[..., start:stop].mean(axis=-1)[mask])
        labels.append(int(event["class_id"]))
        starts.append(int(event["event_start"]))
    values = np.stack(patterns).astype(np.float64)
    values -= values.mean(axis=0, keepdims=True)
    return values, np.asarray(labels, dtype=np.int64), starts


def detrend_patterns(patterns: np.ndarray, starts: list[int]) -> tuple[np.ndarray, float]:
    times = np.asarray(starts, dtype=np.float64)
    times -= times.mean()
    times /= max(float(np.linalg.norm(times)), 1e-8)
    weights = np.sum(times[:, None] * patterns, axis=0)
    fitted = np.outer(times, weights)
    return patterns - fitted, float(np.sum(fitted**2) / max(np.sum(patterns**2), 1e-8))


def rowwise_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_norm = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-8)
    second_norm = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-8)
    return np.sum(first_norm * second_norm, axis=1)


def analyze_run(
    preproc_path: Path,
    denoised_path: Path,
    events_path: Path,
    repetition_time: float,
    offset: int,
    length: int,
) -> dict:
    preproc_image = nib.load(preproc_path)
    denoised_image = nib.load(denoised_path)
    if preproc_image.shape != denoised_image.shape:
        raise ValueError(f"Stage shape mismatch: {preproc_image.shape} vs {denoised_image.shape}")
    preproc = preproc_image.get_fdata(dtype=np.float32)
    denoised = denoised_image.get_fdata(dtype=np.float32)
    preproc_mask = robust_mask(preproc)
    denoised_mask = robust_mask(denoised)
    mask = preproc_mask & denoised_mask
    events = load_events(events_path, repetition_time)
    preproc_patterns, labels, starts = event_patterns(preproc, events, mask, offset, length)
    denoised_patterns, denoised_labels, denoised_starts = event_patterns(
        denoised,
        events,
        mask,
        offset,
        length,
    )
    if not np.array_equal(labels, denoised_labels) or starts != denoised_starts:
        raise ValueError("Stage event extraction mismatch.")
    preproc_detrended, preproc_time_fraction = detrend_patterns(preproc_patterns, starts)
    denoised_detrended, denoised_time_fraction = detrend_patterns(denoised_patterns, starts)
    preproc_metrics = pattern_metrics(preproc_patterns, labels)
    denoised_metrics = pattern_metrics(denoised_patterns, labels)
    preproc_detrended_metrics = pattern_metrics(preproc_detrended, labels)
    denoised_detrended_metrics = pattern_metrics(denoised_detrended, labels)
    event_cosine = rowwise_cosine(preproc_detrended, denoised_detrended)
    result = {
        "shape": list(preproc.shape),
        "preproc_datatype": str(preproc_image.header.get_data_dtype()),
        "denoised_datatype": str(denoised_image.header.get_data_dtype()),
        "max_affine_absolute_difference": float(
            np.max(np.abs(preproc_image.affine - denoised_image.affine))
        ),
        "preproc_mask_voxels": int(preproc_mask.sum()),
        "denoised_mask_voxels": int(denoised_mask.sum()),
        "intersection_mask_voxels": int(mask.sum()),
        "mask_dice": float(
            2.0 * mask.sum() / max(float(preproc_mask.sum() + denoised_mask.sum()), 1.0)
        ),
        "event_count": len(labels),
        "preproc_metrics": preproc_metrics,
        "denoised_metrics": denoised_metrics,
        "preproc_linear_time_variance_fraction": preproc_time_fraction,
        "denoised_linear_time_variance_fraction": denoised_time_fraction,
        "preproc_detrended_metrics": preproc_detrended_metrics,
        "denoised_detrended_metrics": denoised_detrended_metrics,
        "denoised_minus_preproc_detrended_geometry": float(
            denoised_detrended_metrics["same_minus_different_cosine"]
            - preproc_detrended_metrics["same_minus_different_cosine"]
        ),
        "denoised_minus_preproc_detrended_accuracy": float(
            denoised_detrended_metrics["leave_one_event_accuracy"]
            - preproc_detrended_metrics["leave_one_event_accuracy"]
        ),
        "mean_detrended_event_pattern_cosine_between_stages": float(np.mean(event_cosine)),
        "min_detrended_event_pattern_cosine_between_stages": float(np.min(event_cosine)),
    }
    del preproc, denoised, preproc_patterns, denoised_patterns
    gc.collect()
    return result


def preproc_key(dataset: str, subject: str, run_id: int) -> str:
    return (
        f"{dataset}/derivatives/fmriprep/{subject}/"
        f"{subject}_ses-1_task-motor_run-{run_id}_space-T1w_desc-preproc_bold.nii.gz"
    )


def events_key(dataset: str, subject: str, run_id: int) -> str:
    return (
        f"{dataset}/{subject}/ses-1/func/"
        f"{subject}_ses-1_task-motor_run-{run_id:02d}_events.tsv"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare event geometry before and after the custom denoising derivative."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=4,
        metavar=("LABEL", "SUBJECT", "RUN_ID", "DENOISED_NIFTI"),
        required=True,
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--bucket", default="openneuro.org")
    parser.add_argument("--dataset", default="ds004044")
    parser.add_argument("--repetition-time", type=float, default=2.0)
    parser.add_argument("--offset", type=int, default=3)
    parser.add_argument("--length", type=int, default=8)
    args = parser.parse_args()

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    rows = {}
    with tempfile.TemporaryDirectory(prefix="stage-forensics-") as temp_dir:
        temp_root = Path(temp_dir)
        for label, subject, run_id_value, denoised_path in args.run:
            run_id = int(run_id_value)
            preproc_path = temp_root / f"{label}_preproc.nii.gz"
            events_path = temp_root / f"{label}_events.tsv"
            print(f"downloading preproc/events for {label}", flush=True)
            s3.download_file(
                args.bucket,
                preproc_key(args.dataset, subject, run_id),
                str(preproc_path),
            )
            s3.download_file(
                args.bucket,
                events_key(args.dataset, subject, run_id),
                str(events_path),
            )
            print(f"analyzing {label}", flush=True)
            row = analyze_run(
                preproc_path,
                Path(denoised_path),
                events_path,
                args.repetition_time,
                args.offset,
                args.length,
            )
            row.update({"label": label, "subject": subject, "run_id": run_id})
            rows[label] = row

    result = {
        "bucket": args.bucket,
        "dataset": args.dataset,
        "offset": args.offset,
        "length": args.length,
        "runs": rows,
        "rankings": {
            "largest_denoising_geometry_loss": sorted(
                rows,
                key=lambda label: rows[label]["denoised_minus_preproc_detrended_geometry"],
            ),
            "largest_denoising_accuracy_loss": sorted(
                rows,
                key=lambda label: rows[label]["denoised_minus_preproc_detrended_accuracy"],
            ),
            "lowest_stage_pattern_cosine": sorted(
                rows,
                key=lambda label: rows[label][
                    "mean_detrended_event_pattern_cosine_between_stages"
                ],
            ),
        },
        "note": (
            "Preprocessed files are streamed into temporary storage and deleted after each run. "
            "Event geometry uses the common preproc/denoised mask and true-time linear detrending."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
