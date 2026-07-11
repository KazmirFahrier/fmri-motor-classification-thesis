#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from run_continuous_window_full_cohort import (
    as_jsonable,
    events_key,
    list_denoised_keys,
    run_id_from_key,
)
from sweep_continuous_bold_windows import load_events, reproduce_thesis_transform


DEFAULT_WINDOW = (3, 8)


def parse_window(value: str) -> tuple[int, int]:
    offset, length = value.split(":")
    return int(offset), int(length)


def temporal_basis_weights(length: int) -> dict[str, np.ndarray]:
    time = np.arange(length, dtype=np.float64)
    centered = time - time.mean()
    quadratic = centered ** 2
    quadratic -= quadratic.mean()

    weights = {
        "mean": np.full(length, 1.0 / length, dtype=np.float64),
        "linear": centered / max(float(np.sum(centered ** 2)), 1e-12),
        "quadratic": quadratic / max(float(np.sum(quadratic ** 2)), 1e-12),
    }

    half = length // 2
    early_late = np.zeros(length, dtype=np.float64)
    early_late[:half] = -1.0 / half
    early_late[half:] = 1.0 / (length - half)
    weights["early_late"] = early_late

    if length >= 4:
        tail = np.zeros(length, dtype=np.float64)
        tail[:-2] = -1.0 / (length - 2)
        tail[-2:] = 0.5
        weights["tail_vs_body"] = tail
    return weights


def make_records(subject: str, run_id: int, events: list[dict]) -> list[dict]:
    return [
        {
            "subject_id": subject,
            "run_id": int(run_id),
            "event_start": int(event["event_start"]),
            "class_id": int(event["class_id"]),
        }
        for event in events
    ]


def process_run(
    *,
    bold_path: Path,
    events_path: Path,
    subject: str,
    run_id: int,
    window: tuple[int, int],
    repetition_time: float,
    extraction_shape: tuple[int, int, int],
    feature_shape: tuple[int, int, int],
    output_mode: str = "basis",
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict]]:
    import nibabel as nib

    offset, length = window
    image = nib.load(bold_path)
    data = image.get_fdata(dtype=np.float32)
    events = load_events(events_path, repetition_time)
    max_volume = data.shape[-1]
    valid_events = [
        event for event in events if event["event_start"] + offset + length <= max_volume
    ]
    if not valid_events:
        raise ValueError(
            f"No usable target events for {subject} run-{run_id:02d}; "
            f"target_events={len(events)} max_volume={max_volume} "
            f"required_tail={offset + length}"
        )

    needed_volumes = sorted(
        {
            event["event_start"] + offset + step
            for event in valid_events
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

    if output_mode not in {"basis", "sequence", "both"}:
        raise ValueError(f"Unknown output mode: {output_mode}")
    weights = temporal_basis_weights(length) if output_mode in {"basis", "both"} else {}
    feature_lists: dict[str, list[np.ndarray]] = {name: [] for name in weights}
    sequences = []
    for event in valid_events:
        sequence = np.stack(
            [cache[event["event_start"] + offset + step] for step in range(length)],
            axis=0,
        ).astype(np.float32, copy=False)
        if output_mode in {"sequence", "both"}:
            sequences.append(sequence)
        for name, weight in weights.items():
            feature_lists[name].append((weight[:, None] * sequence).sum(axis=0))

    prefix = f"offset_{offset}_length_{length}"
    features = {
        f"{prefix}_{name}": np.stack(values).astype(np.float32)
        for name, values in feature_lists.items()
    }
    if weights:
        features[f"{prefix}_mean_linear_quadratic"] = np.concatenate(
            [
                features[f"{prefix}_mean"],
                features[f"{prefix}_linear"],
                features[f"{prefix}_quadratic"],
            ],
            axis=1,
        ).astype(np.float32)
        dynamic_parts = [
            features[f"{prefix}_linear"],
            features[f"{prefix}_quadratic"],
            features[f"{prefix}_early_late"],
        ]
        if f"{prefix}_tail_vs_body" in features:
            dynamic_parts.append(features[f"{prefix}_tail_vs_body"])
        features[f"{prefix}_dynamic"] = np.concatenate(dynamic_parts, axis=1).astype(
            np.float32
        )
    if sequences:
        features[f"{prefix}_sequence"] = np.stack(sequences).astype(np.float32)

    labels = np.asarray([event["class_id"] for event in valid_events], dtype=np.int64)
    records = make_records(subject, run_id, valid_events)
    return features, labels, records


def process_subject(
    *,
    s3,
    bucket: str,
    dataset: str,
    subject: str,
    keys: list[str],
    out_path: Path,
    window: tuple[int, int],
    repetition_time: float,
    extraction_shape: tuple[int, int, int],
    feature_shape: tuple[int, int, int],
    output_mode: str = "basis",
) -> dict:
    subject_features: dict[str, list[np.ndarray]] = {}
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
                    window=window,
                    repetition_time=repetition_time,
                    extraction_shape=extraction_shape,
                    feature_shape=feature_shape,
                    output_mode=output_mode,
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
                    subject_features.setdefault(name, []).append(values)
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
        for name, values in sorted(subject_features.items())
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
        "feature_keys": sorted(subject_features),
    }


def main() -> None:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    parser = argparse.ArgumentParser(
        description=(
            "Stream continuous denoised BOLD and save compact within-event temporal "
            "basis maps for full-cohort motor classification experiments."
        )
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bucket", default="openneuro.org")
    parser.add_argument("--dataset", default="ds004044")
    parser.add_argument("--repetition-time", type=float, default=2.0)
    parser.add_argument("--extraction-shape", nargs=3, type=int, default=[100, 100, 100])
    parser.add_argument("--feature-shape", nargs=3, type=int, default=[24, 24, 24])
    parser.add_argument(
        "--window",
        default=f"{DEFAULT_WINDOW[0]}:{DEFAULT_WINDOW[1]}",
        help="Event window as offset:length in TRs. Default is 3:8.",
    )
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument(
        "--output-mode",
        choices=["basis", "sequence", "both"],
        default="basis",
        help="Save compact temporal bases, the ordered event sequence, or both.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "subject_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    window = parse_window(args.window)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
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
            window=window,
            repetition_time=args.repetition_time,
            extraction_shape=tuple(args.extraction_shape),
            feature_shape=tuple(args.feature_shape),
            output_mode=args.output_mode,
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
                    "window": window,
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

    feature_keys = []
    if args.output_mode in {"basis", "both"}:
        feature_keys.extend(
            [
                f"offset_{window[0]}_length_{window[1]}_{name}"
                for name in (
                    list(temporal_basis_weights(window[1]))
                    + ["mean_linear_quadratic", "dynamic"]
                )
            ]
        )
    if args.output_mode in {"sequence", "both"}:
        feature_keys.append(f"offset_{window[0]}_length_{window[1]}_sequence")
    result = {
        "bucket": args.bucket,
        "dataset": args.dataset,
        "subjects": subjects,
        "subject_count": len(subjects),
        "window": window,
        "repetition_time": args.repetition_time,
        "output_mode": args.output_mode,
        "extraction_shape": args.extraction_shape,
        "feature_shape": args.feature_shape,
        "checkpoint_dir": str(checkpoint_dir),
        "feature_keys": feature_keys,
        "note": (
            "Each subject checkpoint stores the requested temporal representation derived "
            "from transformed continuous BOLD volumes. Raw NIfTI files are downloaded per "
            "run and deleted immediately after extraction."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=as_jsonable)
    )
    print(json.dumps({"out_dir": str(out_dir), "subject_count": len(subjects)}, indent=2))


if __name__ == "__main__":
    main()
