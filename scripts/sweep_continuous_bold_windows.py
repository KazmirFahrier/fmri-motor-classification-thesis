#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


CLASS_NAMES = [
    "Left leg movements",
    "Right leg movements",
    "Forearm movements",
    "Upper arm movements",
]

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


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def zscore(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    mean = float(volume.mean())
    std = float(volume.std())
    if std < 1e-6:
        return (volume - mean).astype(np.float32, copy=False)
    return ((volume - mean) / std).astype(np.float32, copy=False)


def resize(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if tuple(volume.shape) == target_shape:
        return volume.astype(np.float32, copy=False)
    factors = [target / source for target, source in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1).astype(np.float32, copy=False)


def reproduce_thesis_transform(
    volume: np.ndarray,
    extraction_shape: tuple[int, int, int],
    feature_shape: tuple[int, int, int],
) -> np.ndarray:
    extracted = zscore(resize(volume, extraction_shape))
    return zscore(resize(extracted, feature_shape))


def load_events(path: Path, repetition_time: float) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    events = []
    for row in rows:
        trial_type = normalize_trial_type(row["trial_type"])
        if trial_type not in CLASS_NAMES:
            continue
        onset = float(row["onset"])
        events.append(
            {
                "trial_type": trial_type,
                "class_id": CLASS_NAMES.index(trial_type),
                "onset": onset,
                "event_start": int(round(onset / repetition_time)),
            }
        )
    return events


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return np.nan_to_num(x / denom, nan=0.0, posinf=0.0, neginf=0.0)


def pattern_metrics(patterns: np.ndarray, labels: np.ndarray) -> dict:
    x = l2_normalize(patterns.astype(np.float64))
    similarities = x @ x.T
    same = []
    different = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            target = same if labels[left] == labels[right] else different
            target.append(float(similarities[left, right]))

    pred = []
    for val_idx in range(len(labels)):
        train_idx = np.asarray([idx for idx in range(len(labels)) if idx != val_idx])
        centroids = []
        for class_id in range(len(CLASS_NAMES)):
            centroids.append(x[train_idx][labels[train_idx] == class_id].mean(axis=0))
        centroids = l2_normalize(np.stack(centroids))
        pred.append(int(np.argmax(x[val_idx] @ centroids.T)))
    return {
        "same_class_cosine_mean": float(np.mean(same)),
        "different_class_cosine_mean": float(np.mean(different)),
        "same_minus_different_cosine": float(np.mean(same) - np.mean(different)),
        "leave_one_event_accuracy": float(
            np.mean(labels == np.asarray(pred, dtype=np.int64))
        ),
    }


def center_and_detrend(
    patterns: np.ndarray,
    event_starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    centered = patterns.astype(np.float64)
    centered -= centered.mean(axis=0, keepdims=True)
    times = event_starts.astype(np.float64)
    times -= times.mean()
    times /= max(float(np.linalg.norm(times)), 1e-8)
    time_weights = np.sum(times[:, None] * centered, axis=0)
    fitted = np.outer(times, time_weights)
    total_energy = float(np.sum(centered ** 2))
    fitted_energy = float(np.sum(fitted ** 2))
    return centered, centered - fitted, fitted_energy / max(total_energy, 1e-8)


def load_saved_event_features(
    feature_dir: Path,
    subject: str,
    run_id: int,
    clip_offset: int,
) -> dict[tuple[int, int], np.ndarray]:
    from run_clip_offset_event_sweep import aggregate_events_for_offset

    clip_x = np.load(feature_dir / "features.npy").astype(np.float32)
    clip_y = np.load(feature_dir / "labels.npy").astype(np.int64)
    records = json.loads((feature_dir / "records.json").read_text())
    event_x, event_y, event_records = aggregate_events_for_offset(
        clip_x,
        clip_y,
        records,
        clip_offset,
    )
    return {
        (int(record["event_start"]), int(label)): event_x[idx]
        for idx, (record, label) in enumerate(zip(event_records, event_y))
        if str(record["subject_id"]) == subject and int(record["run_id"]) == run_id
    }


def reconstruction_metrics(
    direct_patterns: np.ndarray,
    labels: np.ndarray,
    events: list[dict],
    saved: dict[tuple[int, int], np.ndarray],
) -> dict:
    pairs = []
    for pattern, label, event in zip(direct_patterns, labels, events):
        saved_pattern = saved.get((int(event["event_start"]), int(label)))
        if saved_pattern is None:
            continue
        direct = pattern.astype(np.float64)
        reference = saved_pattern.astype(np.float64)
        cosine = float(
            np.sum(direct * reference)
            / max(float(np.linalg.norm(direct) * np.linalg.norm(reference)), 1e-8)
        )
        relative_rmse = float(
            np.sqrt(np.mean((direct - reference) ** 2))
            / max(float(np.sqrt(np.mean(reference ** 2))), 1e-8)
        )
        pairs.append(
            {
                "event_start": int(event["event_start"]),
                "class_id": int(label),
                "class_name": CLASS_NAMES[int(label)],
                "cosine": cosine,
                "relative_rmse": relative_rmse,
            }
        )
    return {
        "matched_event_count": len(pairs),
        "mean_cosine": float(np.mean([row["cosine"] for row in pairs])) if pairs else None,
        "min_cosine": float(np.min([row["cosine"] for row in pairs])) if pairs else None,
        "mean_relative_rmse": (
            float(np.mean([row["relative_rmse"] for row in pairs])) if pairs else None
        ),
        "events": pairs,
    }


def analyze_run(
    *,
    label: str,
    subject: str,
    run_id: int,
    bold_path: Path,
    events_path: Path,
    repetition_time: float,
    extraction_shape: tuple[int, int, int],
    feature_shape: tuple[int, int, int],
    offsets: list[int],
    lengths: list[int],
    saved_feature_dir: Path | None,
    canonical_offset: int,
    canonical_length: int,
) -> dict:
    image = nib.load(bold_path)
    data = image.get_fdata(dtype=np.float32)
    events = load_events(events_path, repetition_time)
    max_volume = image.shape[-1]
    needed_volumes = sorted(
        {
            event["event_start"] + offset + step
            for event in events
            for offset in offsets
            for length in lengths
            for step in range(length)
            if event["event_start"] + offset + length <= max_volume
        }
    )
    cache = {}
    for volume_idx in needed_volumes:
        volume = data[..., volume_idx]
        cache[volume_idx] = reproduce_thesis_transform(
            volume,
            extraction_shape,
            feature_shape,
        ).reshape(-1)

    labels = np.asarray([event["class_id"] for event in events], dtype=np.int64)
    event_starts = np.asarray([event["event_start"] for event in events], dtype=np.int64)
    windows = []
    canonical_patterns = None
    for offset in offsets:
        for length in lengths:
            if any(start + offset + length > max_volume for start in event_starts):
                continue
            patterns = np.stack(
                [
                    np.mean(
                        [cache[start + offset + step] for step in range(length)],
                        axis=0,
                    )
                    for start in event_starts
                ]
            ).astype(np.float32)
            centered, detrended, time_fraction = center_and_detrend(
                patterns,
                event_starts,
            )
            row = {
                "offset": int(offset),
                "length": int(length),
                "centered": pattern_metrics(centered, labels),
                "linear_detrended": pattern_metrics(detrended, labels),
                "linear_time_variance_fraction": time_fraction,
            }
            windows.append(row)
            if offset == canonical_offset and length == canonical_length:
                canonical_patterns = patterns

    reconstruction = None
    if saved_feature_dir is not None and canonical_patterns is not None:
        saved = load_saved_event_features(
            saved_feature_dir,
            subject,
            run_id,
            canonical_offset,
        )
        reconstruction = reconstruction_metrics(
            canonical_patterns,
            labels,
            events,
            saved,
        )

    return {
        "label": label,
        "subject": subject,
        "run_id": int(run_id),
        "bold_path": str(bold_path),
        "events_path": str(events_path),
        "source_shape": list(image.shape),
        "event_count": len(events),
        "transformed_volume_count": len(cache),
        "windows": windows,
        "best_centered": max(
            windows,
            key=lambda row: (
                row["centered"]["same_minus_different_cosine"],
                row["centered"]["leave_one_event_accuracy"],
            ),
        ),
        "best_linear_detrended": max(
            windows,
            key=lambda row: (
                row["linear_detrended"]["same_minus_different_cosine"],
                row["linear_detrended"]["leave_one_event_accuracy"],
            ),
        ),
        "canonical_reconstruction": reconstruction,
    }


def aggregate_windows(runs: dict[str, dict]) -> list[dict]:
    grouped = defaultdict(list)
    for run in runs.values():
        for row in run["windows"]:
            grouped[(row["offset"], row["length"])].append(row)
    result = []
    for (offset, length), rows in sorted(grouped.items()):
        result.append(
            {
                "offset": int(offset),
                "length": int(length),
                "run_count": len(rows),
                "mean_centered_same_minus_different": float(
                    np.mean(
                        [
                            row["centered"]["same_minus_different_cosine"]
                            for row in rows
                        ]
                    )
                ),
                "mean_centered_leave_one_event_accuracy": float(
                    np.mean(
                        [
                            row["centered"]["leave_one_event_accuracy"]
                            for row in rows
                        ]
                    )
                ),
                "mean_detrended_same_minus_different": float(
                    np.mean(
                        [
                            row["linear_detrended"]["same_minus_different_cosine"]
                            for row in rows
                        ]
                    )
                ),
                "mean_detrended_leave_one_event_accuracy": float(
                    np.mean(
                        [
                            row["linear_detrended"]["leave_one_event_accuracy"]
                            for row in rows
                        ]
                    )
                ),
                "mean_linear_time_variance_fraction": float(
                    np.mean([row["linear_time_variance_fraction"] for row in rows])
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep later/longer event windows from continuous denoised BOLD runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=5,
        metavar=("LABEL", "SUBJECT", "RUN_ID", "BOLD_NIFTI", "EVENTS_TSV"),
        required=True,
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--saved-feature-dir")
    parser.add_argument("--repetition-time", type=float, default=2.0)
    parser.add_argument("--extraction-shape", nargs=3, type=int, default=[100, 100, 100])
    parser.add_argument("--feature-shape", nargs=3, type=int, default=[24, 24, 24])
    parser.add_argument("--offsets", default="0,1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--lengths", default="2,4,6,8")
    parser.add_argument("--canonical-offset", type=int, default=2)
    parser.add_argument("--canonical-length", type=int, default=6)
    args = parser.parse_args()

    offsets = [int(value) for value in args.offsets.split(",")]
    lengths = [int(value) for value in args.lengths.split(",")]
    saved_feature_dir = Path(args.saved_feature_dir) if args.saved_feature_dir else None
    runs = {}
    for label, subject, run_id, bold_path, events_path in args.run:
        runs[label] = analyze_run(
            label=label,
            subject=subject,
            run_id=int(run_id),
            bold_path=Path(bold_path),
            events_path=Path(events_path),
            repetition_time=args.repetition_time,
            extraction_shape=tuple(args.extraction_shape),
            feature_shape=tuple(args.feature_shape),
            offsets=offsets,
            lengths=lengths,
            saved_feature_dir=saved_feature_dir,
            canonical_offset=args.canonical_offset,
            canonical_length=args.canonical_length,
        )

    aggregate = aggregate_windows(runs)
    result = {
        "repetition_time": args.repetition_time,
        "extraction_shape": args.extraction_shape,
        "feature_shape": args.feature_shape,
        "offsets": offsets,
        "lengths": lengths,
        "canonical_offset": args.canonical_offset,
        "canonical_length": args.canonical_length,
        "saved_feature_dir": str(saved_feature_dir) if saved_feature_dir else None,
        "runs": runs,
        "aggregate_windows": aggregate,
        "best_aggregate_centered": max(
            aggregate,
            key=lambda row: (
                row["mean_centered_same_minus_different"],
                row["mean_centered_leave_one_event_accuracy"],
            ),
        ),
        "best_aggregate_linear_detrended": max(
            aggregate,
            key=lambda row: (
                row["mean_detrended_same_minus_different"],
                row["mean_detrended_leave_one_event_accuracy"],
            ),
        ),
        "note": (
            "Each continuous denoised BOLD volume reproduces the thesis extraction transform "
            "(resize to 100^3, z-score, resize to feature shape, z-score). Window means are then "
            "centered within run and optionally linearly detrended using unlabeled event times."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "best_aggregate_centered": result["best_aggregate_centered"],
                "best_aggregate_linear_detrended": result[
                    "best_aggregate_linear_detrended"
                ],
                "reconstruction": {
                    label: run["canonical_reconstruction"]
                    for label, run in runs.items()
                },
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
