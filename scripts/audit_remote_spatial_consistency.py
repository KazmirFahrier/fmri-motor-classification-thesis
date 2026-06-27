#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config

from audit_native_spatial_consistency import compare_pair, summarize_image


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run_id_from_key(key: str) -> int:
    return int(key.split("_run-")[1].split("_")[0])


def list_denoised_keys(s3, bucket: str, dataset: str, subjects: list[str]) -> dict[str, list[str]]:
    result = {}
    paginator = s3.get_paginator("list_objects_v2")
    for subject in subjects:
        prefix = f"{dataset}/derivatives/fmriprep/{subject}/"
        keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = str(obj["Key"])
                if key.endswith("_space-T1w_desc-preproc_bold_denoised.nii.gz"):
                    keys.append(key)
        result[subject] = sorted(keys, key=run_id_from_key)
    return result


def find_cached_run(cache_dirs: list[Path], subject: str, run_id: int) -> Path | None:
    names = [
        f"{subject}_run-{run_id:02d}_denoised.nii.gz",
        f"{subject}_run-{run_id}_denoised.nii.gz",
    ]
    for cache_dir in cache_dirs:
        for name in names:
            candidate = cache_dir / name
            if candidate.exists():
                return candidate
    return None


def summarize_comparisons(rows: list[dict]) -> dict:
    comparable = [row for row in rows if row["comparable"]]
    metric_names = [
        "mask_dice",
        "mask_jaccard",
        "center_of_mass_distance_mm",
        "temporal_mean_map_correlation",
        "temporal_std_map_correlation",
    ]
    summary = {"pair_count": len(comparable)}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in comparable], dtype=np.float64)
        summary[f"mean_{metric}"] = float(np.mean(values))
        summary[f"min_{metric}"] = float(np.min(values))
        summary[f"max_{metric}"] = float(np.max(values))
    summary["lowest_mask_dice_pair"] = min(comparable, key=lambda row: row["mask_dice"])
    summary["largest_center_shift_pair"] = max(
        comparable,
        key=lambda row: row["center_of_mass_distance_mm"],
    )
    summary["lowest_std_correlation_pair"] = min(
        comparable,
        key=lambda row: row["temporal_std_map_correlation"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream missing denoised runs and compare all native T1w-space run pairs."
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--bucket", default="openneuro.org")
    parser.add_argument("--dataset", default="ds004044")
    parser.add_argument("--subjects", nargs="*", default=["sub-42", "sub-52", "sub-62"])
    parser.add_argument("--cache-dir", action="append", default=[])
    args = parser.parse_args()

    cache_dirs = [Path(value) for value in args.cache_dir]
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    keys_by_subject = list_denoised_keys(
        s3,
        args.bucket,
        args.dataset,
        args.subjects,
    )
    summaries = []
    maps = {}
    sources = {}
    with tempfile.TemporaryDirectory(prefix="spatial-forensics-") as temp_dir:
        temp_root = Path(temp_dir)
        for subject in args.subjects:
            for key in keys_by_subject[subject]:
                run_id = run_id_from_key(key)
                label = f"{subject}-run-{run_id}"
                cached = find_cached_run(cache_dirs, subject, run_id)
                if cached is not None:
                    path = cached
                    source = "cache"
                else:
                    path = temp_root / f"{subject}_run-{run_id:02d}_denoised.nii.gz"
                    print(f"downloading {label}", flush=True)
                    s3.download_file(args.bucket, key, str(path))
                    source = "streamed_temp"
                print(f"summarizing {label} source={source}", flush=True)
                summary, run_maps = summarize_image(label, subject, run_id, path)
                summaries.append(summary)
                maps[label] = run_maps
                sources[label] = {"source": source, "remote_key": key, "path": str(path)}
                if source == "streamed_temp":
                    path.unlink(missing_ok=True)

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for summary in summaries:
        by_subject[summary["subject"]].append(summary)
    comparisons = []
    subject_summaries = {}
    for subject, group in sorted(by_subject.items()):
        subject_rows = []
        group = sorted(group, key=lambda row: row["run_id"])
        for first_idx in range(len(group)):
            for second_idx in range(first_idx + 1, len(group)):
                first = group[first_idx]
                second = group[second_idx]
                row = compare_pair(first, maps[first["label"]], second, maps[second["label"]])
                row["subject"] = subject
                subject_rows.append(row)
                comparisons.append(row)
        subject_summaries[subject] = summarize_comparisons(subject_rows)

    result = {
        "bucket": args.bucket,
        "dataset": args.dataset,
        "subjects": args.subjects,
        "sources": sources,
        "runs": summaries,
        "comparisons": comparisons,
        "subject_summaries": subject_summaries,
        "note": (
            "Uncached NIfTI payloads were downloaded one at a time into a temporary directory and "
            "deleted after compact mean/std/mask maps were retained in memory."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "subject_summaries": subject_summaries,
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
