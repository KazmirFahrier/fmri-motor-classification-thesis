#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from run_balanced_event_assignment import (
    apply_balanced_assignment,
    apply_imbalance_gated_balanced_assignment,
    center_by_subject_run,
    centroid_matrix,
    metrics,
    score_with_centroids,
    split_indices,
)
from run_clip_offset_event_sweep import coarse_metrics
from run_temporal_detrended_event_adaptation import (
    temporal_detrend_by_subject_run,
)
from sweep_continuous_bold_windows import (
    CLASS_NAMES,
    load_events,
    reproduce_thesis_transform,
)


DEFAULT_WINDOWS = [
    (2, 6),
    (3, 6),
    (3, 8),
    (4, 2),
    (5, 4),
    (6, 2),
]


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def parse_windows(value: str) -> list[tuple[int, int]]:
    windows = []
    for item in value.split(","):
        offset, length = item.split(":")
        windows.append((int(offset), int(length)))
    return windows


def list_denoised_keys(s3, bucket: str, dataset: str) -> dict[str, list[str]]:
    paginator = s3.get_paginator("list_objects_v2")
    prefix = f"{dataset}/derivatives/fmriprep/"
    by_subject: dict[str, list[str]] = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = str(obj["Key"])
            if not key.endswith("_space-T1w_desc-preproc_bold_denoised.nii.gz"):
                continue
            subject = key.split("/")[3]
            by_subject.setdefault(subject, []).append(key)
    return {
        subject: sorted(keys, key=run_id_from_key)
        for subject, keys in sorted(by_subject.items())
    }


def run_id_from_key(key: str) -> int:
    return int(key.split("_run-")[1].split("_")[0])


def events_key(dataset: str, denoised_key: str) -> str:
    subject = denoised_key.split("/")[3]
    run_id = run_id_from_key(denoised_key)
    return (
        f"{dataset}/{subject}/ses-1/func/"
        f"{subject}_ses-1_task-motor_run-{run_id:02d}_events.tsv"
    )


def process_run(
    *,
    bold_path: Path,
    events_path: Path,
    subject: str,
    run_id: int,
    windows: list[tuple[int, int]],
    repetition_time: float,
    extraction_shape: tuple[int, int, int],
    feature_shape: tuple[int, int, int],
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict]]:
    import nibabel as nib

    image = nib.load(bold_path)
    data = image.get_fdata(dtype=np.float32)
    events = load_events(events_path, repetition_time)
    max_volume = data.shape[-1]
    required_tail = max(offset + length for offset, length in windows)
    valid_events = [
        event
        for event in events
        if event["event_start"] + required_tail <= max_volume
    ]
    if not valid_events:
        raise ValueError(
            f"No usable target events for {subject} run-{run_id:02d}; "
            f"target_events={len(events)} max_volume={max_volume} "
            f"required_tail={required_tail}"
        )
    needed_volumes = sorted(
        {
            event["event_start"] + offset + step
            for event in valid_events
            for offset, length in windows
            for step in range(length)
        }
    )
    cache = {
        volume_idx: reproduce_thesis_transform(
            data[..., volume_idx],
            extraction_shape,
            feature_shape,
        ).reshape(-1)
        for volume_idx in needed_volumes
    }

    labels = np.asarray([event["class_id"] for event in valid_events], dtype=np.int64)
    records = [
        {
            "subject_id": subject,
            "run_id": int(run_id),
            "event_start": int(event["event_start"]),
            "class_id": int(event["class_id"]),
        }
        for event in valid_events
    ]
    features = {}
    for offset, length in windows:
        name = f"offset_{offset}_length_{length}"
        features[name] = np.stack(
            [
                np.mean(
                    [
                        cache[event["event_start"] + offset + step]
                        for step in range(length)
                    ],
                    axis=0,
                )
                for event in valid_events
            ]
        ).astype(np.float32)
    return features, labels, records


def process_subject(
    *,
    s3,
    bucket: str,
    dataset: str,
    subject: str,
    keys: list[str],
    out_path: Path,
    windows: list[tuple[int, int]],
    repetition_time: float,
    extraction_shape: tuple[int, int, int],
    feature_shape: tuple[int, int, int],
) -> dict:
    subject_features: dict[str, list[np.ndarray]] = {
        f"offset_{offset}_length_{length}": []
        for offset, length in windows
    }
    labels = []
    records = []
    run_rows = []
    with tempfile.TemporaryDirectory(prefix=f"{subject}-") as temp_dir:
        temp_root = Path(temp_dir)
        for key in keys:
            run_id = run_id_from_key(key)
            bold_path = temp_root / f"{subject}_run-{run_id}_denoised.nii.gz"
            events_path = temp_root / f"{subject}_run-{run_id}_events.tsv"
            s3.download_file(bucket, key, str(bold_path))
            s3.download_file(bucket, events_key(dataset, key), str(events_path))
            try:
                run_features, run_labels, run_records = process_run(
                    bold_path=bold_path,
                    events_path=events_path,
                    subject=subject,
                    run_id=run_id,
                    windows=windows,
                    repetition_time=repetition_time,
                    extraction_shape=extraction_shape,
                    feature_shape=feature_shape,
                )
            except Exception as exc:
                run_rows.append(
                    {
                        "run_id": int(run_id),
                        "status": "skipped",
                        "reason": str(exc),
                    }
                )
            else:
                for name, values in run_features.items():
                    subject_features[name].append(values)
                labels.append(run_labels)
                records.extend(run_records)
                run_rows.append(
                    {
                        "run_id": int(run_id),
                        "status": "completed",
                        "event_count": int(len(run_records)),
                    }
                )
            finally:
                bold_path.unlink(missing_ok=True)
                events_path.unlink(missing_ok=True)

    if not labels:
        raise ValueError(f"No usable runs for {subject}; run_rows={run_rows}")

    payload = {
        name: np.concatenate(values, axis=0)
        for name, values in subject_features.items()
    }
    payload["labels"] = np.concatenate(labels, axis=0)
    payload["records_json"] = np.asarray(json.dumps(records))
    np.savez_compressed(out_path, **payload)
    return {
        "subject": subject,
        "run_count": len(keys),
        "event_count": len(records),
        "runs": run_rows,
        "checkpoint": str(out_path),
        "checkpoint_bytes": out_path.stat().st_size,
    }


def load_checkpoints(
    checkpoint_dir: Path,
    window_names: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict]]:
    features = {name: [] for name in window_names}
    labels = []
    records = []
    for path in sorted(checkpoint_dir.glob("sub-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for name in window_names:
                features[name].append(data[name].astype(np.float32))
            labels.append(data["labels"].astype(np.int64))
            records.extend(json.loads(str(data["records_json"])))
    return (
        {
            name: np.concatenate(values, axis=0)
            for name, values in features.items()
        },
        np.concatenate(labels, axis=0),
        records,
    )


def evaluate_variant(
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    subject_fold_count: int,
) -> dict:
    centered = center_by_subject_run(x, records)
    detrended, group_rows = temporal_detrend_by_subject_run(
        centered,
        records,
        degree=1,
    )
    rows = []
    for split in split_indices(records, "all", subject_fold_count):
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        centroids = centroid_matrix(detrended[train_idx], y[train_idx])
        scores = score_with_centroids(detrended[val_idx], centroids)
        predictions = [
            ("independent_argmax", scores.argmax(axis=1).astype(np.int64)),
            (
                "balanced_subject_run_assignment",
                apply_balanced_assignment(scores, val_idx, records),
            ),
            (
                "gated_balanced_imbalance_l1_4",
                apply_imbalance_gated_balanced_assignment(
                    scores,
                    val_idx,
                    records,
                    4.0,
                ),
            ),
        ]
        for rule, pred in predictions:
            rows.append(
                {
                    "split": split["split"],
                    "family": split["family"],
                    "prediction_rule": rule,
                    "train_count": int(len(train_idx)),
                    "val_count": int(len(val_idx)),
                    "metrics": metrics(y[val_idx], pred),
                    "coarse_metrics": coarse_metrics(y[val_idx], pred),
                }
            )

    summary = []
    for family in ["run", "subject"]:
        for rule in [
            "independent_argmax",
            "balanced_subject_run_assignment",
            "gated_balanced_imbalance_l1_4",
        ]:
            group = [
                row
                for row in rows
                if row["family"] == family and row["prediction_rule"] == rule
            ]
            summary.append(
                {
                    "family": family,
                    "prediction_rule": rule,
                    "count": len(group),
                    "mean_accuracy": float(
                        np.mean([row["metrics"]["accuracy"] for row in group])
                    ),
                    "mean_macro_f1": float(
                        np.mean([row["metrics"]["macro_f1"] for row in group])
                    ),
                    "mean_leg_vs_arm_accuracy": float(
                        np.mean(
                            [
                                row["coarse_metrics"]["leg_vs_arm_accuracy"]
                                for row in group
                            ]
                        )
                    ),
                    "min_accuracy": float(
                        np.min([row["metrics"]["accuracy"] for row in group])
                    ),
                    "max_accuracy": float(
                        np.max([row["metrics"]["accuracy"] for row in group])
                    ),
                }
            )
    return {
        "event_feature_shape": list(x.shape),
        "mean_linear_time_variance_fraction": float(
            np.mean([row["temporal_variance_fraction"] for row in group_rows])
        ),
        "rows": rows,
        "summary": summary,
    }


def main() -> None:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    parser = argparse.ArgumentParser(
        description="Stream continuous denoised BOLD and validate candidate HRF windows on the full cohort."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bucket", default="openneuro.org")
    parser.add_argument("--dataset", default="ds004044")
    parser.add_argument("--repetition-time", type=float, default=2.0)
    parser.add_argument("--extraction-shape", nargs=3, type=int, default=[100, 100, 100])
    parser.add_argument("--feature-shape", nargs=3, type=int, default=[24, 24, 24])
    parser.add_argument(
        "--windows",
        default=",".join(f"{offset}:{length}" for offset, length in DEFAULT_WINDOWS),
    )
    parser.add_argument("--subject-fold-count", type=int, default=6)
    parser.add_argument("--max-subjects", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "subject_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    windows = parse_windows(args.windows)
    window_names = [
        f"offset_{offset}_length_{length}"
        for offset, length in windows
    ]
    s3 = boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
    )
    by_subject = list_denoised_keys(s3, args.bucket, args.dataset)
    subjects = sorted(by_subject)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]

    progress_rows = []
    for subject_idx, subject in enumerate(subjects, start=1):
        checkpoint = checkpoint_dir / f"{subject}.npz"
        if checkpoint.exists():
            progress_rows.append(
                {
                    "subject": subject,
                    "status": "existing_checkpoint",
                    "checkpoint": str(checkpoint),
                    "checkpoint_bytes": checkpoint.stat().st_size,
                }
            )
            continue
        row = process_subject(
            s3=s3,
            bucket=args.bucket,
            dataset=args.dataset,
            subject=subject,
            keys=by_subject[subject],
            out_path=checkpoint,
            windows=windows,
            repetition_time=args.repetition_time,
            extraction_shape=tuple(args.extraction_shape),
            feature_shape=tuple(args.feature_shape),
        )
        row["status"] = "completed"
        row["subject_index"] = subject_idx
        row["subject_total"] = len(subjects)
        progress_rows.append(row)
        progress_path.write_text(
            json.dumps(
                {
                    "subjects_requested": len(subjects),
                    "subjects_completed": len(list(checkpoint_dir.glob("sub-*.npz"))),
                    "windows": windows,
                    "rows": progress_rows,
                },
                indent=2,
                default=as_jsonable,
            )
        )
        print(
            f"completed {subject} ({subject_idx}/{len(subjects)}) "
            f"checkpoint_mb={checkpoint.stat().st_size / 1e6:.1f}",
            flush=True,
        )

    features, labels, records = load_checkpoints(
        checkpoint_dir,
        window_names,
    )
    variants = {}
    for name in window_names:
        print(f"evaluating {name}", flush=True)
        variants[name] = evaluate_variant(
            features[name],
            labels,
            records,
            args.subject_fold_count,
        )

    result = {
        "bucket": args.bucket,
        "dataset": args.dataset,
        "subjects": subjects,
        "subject_count": len(subjects),
        "run_count": len(
            {
                (str(record["subject_id"]), int(record["run_id"]))
                for record in records
            }
        ),
        "event_count": len(records),
        "repetition_time": args.repetition_time,
        "extraction_shape": args.extraction_shape,
        "feature_shape": args.feature_shape,
        "windows": windows,
        "variants": variants,
        "note": (
            "Each subject checkpoint contains only transformed event features, labels, and records. "
            "Continuous denoised BOLD files are deleted immediately after each run. Evaluation uses "
            "unlabeled subject-run centering, true-time linear detrending, and fixed prediction rules."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=as_jsonable)
    )
    print(json.dumps({"out_dir": str(out_dir), "subject_count": len(subjects)}, indent=2))


if __name__ == "__main__":
    main()
