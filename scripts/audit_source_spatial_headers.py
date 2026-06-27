#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
from collections import defaultdict
from pathlib import Path

import boto3
import nibabel as nib
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run_id_from_key(key: str) -> int:
    return int(key.split("_run-")[1].split("_")[0])


def variant_from_key(key: str) -> str:
    if key.endswith("_space-T1w_desc-preproc_bold_denoised.nii.gz"):
        return "denoised_t1w"
    if key.endswith("_space-T1w_desc-preproc_bold.nii.gz"):
        return "preproc_t1w"
    if "/derivatives/" not in key and key.endswith("_bold.nii.gz"):
        return "raw_bold"
    raise ValueError(f"Unknown key variant: {key}")


def read_remote_header(s3, bucket: str, key: str) -> nib.Nifti1Header:
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        decompressor = gzip.GzipFile(fileobj=body, mode="rb")
        header_bytes = decompressor.read(544)
    finally:
        body.close()
    return nib.Nifti1Header.from_fileobj(io.BytesIO(header_bytes))


def header_row(s3, bucket: str, key: str, size: int, etag: str) -> dict:
    header = read_remote_header(s3, bucket, key)
    qform, qform_code = header.get_qform(coded=True)
    sform, sform_code = header.get_sform(coded=True)
    best_affine = sform if int(sform_code) > 0 else qform
    return {
        "key": key,
        "subject": next(part for part in key.split("/") if part.startswith("sub-")),
        "run_id": run_id_from_key(key),
        "variant": variant_from_key(key),
        "object_bytes": int(size),
        "etag": etag.strip('"'),
        "shape": [int(value) for value in header.get_data_shape()],
        "voxel_sizes": [float(value) for value in header.get_zooms()],
        "datatype": str(header.get_data_dtype()),
        "qform_code": int(qform_code),
        "sform_code": int(sform_code),
        "qform": qform.tolist(),
        "sform": sform.tolist(),
        "best_affine": best_affine.tolist(),
        "axis_codes": list(nib.aff2axcodes(best_affine)),
    }


def list_target_keys(s3, bucket: str, dataset: str, subjects: list[str]) -> list[tuple[str, int, str]]:
    targets = []
    paginator = s3.get_paginator("list_objects_v2")
    for subject in subjects:
        prefixes = [
            f"{dataset}/{subject}/ses-1/func/",
            f"{dataset}/derivatives/fmriprep/{subject}/",
        ]
        for prefix in prefixes:
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = str(obj["Key"])
                    if not key.endswith(".nii.gz") or "_run-" not in key:
                        continue
                    is_raw = "/derivatives/" not in key and key.endswith("_bold.nii.gz")
                    is_preproc = key.endswith("_space-T1w_desc-preproc_bold.nii.gz")
                    is_denoised = key.endswith("_space-T1w_desc-preproc_bold_denoised.nii.gz")
                    if is_raw or is_preproc or is_denoised:
                        targets.append((key, int(obj["Size"]), str(obj["ETag"])))
    return sorted(targets, key=lambda row: (variant_from_key(row[0]), row[0]))


def summarize_group(rows: list[dict]) -> dict:
    reference = np.asarray(rows[0]["best_affine"], dtype=np.float64)
    affines = [np.asarray(row["best_affine"], dtype=np.float64) for row in rows]
    object_sizes = np.asarray([row["object_bytes"] for row in rows], dtype=np.float64)
    return {
        "count": len(rows),
        "runs": [row["run_id"] for row in rows],
        "unique_shapes": sorted({tuple(row["shape"]) for row in rows}),
        "unique_voxel_sizes": sorted({tuple(row["voxel_sizes"]) for row in rows}),
        "unique_axis_codes": sorted({tuple(row["axis_codes"]) for row in rows}),
        "unique_qform_codes": sorted(set(row["qform_code"] for row in rows)),
        "unique_sform_codes": sorted(set(row["sform_code"] for row in rows)),
        "max_affine_absolute_difference": float(
            max(np.max(np.abs(affine - reference)) for affine in affines)
        ),
        "object_size_mean": float(np.mean(object_sizes)),
        "object_size_std": float(np.std(object_sizes)),
        "object_size_min": int(np.min(object_sizes)),
        "object_size_max": int(np.max(object_sizes)),
        "size_z_scores": {
            f'run-{row["run_id"]}': float(
                (row["object_bytes"] - np.mean(object_sizes))
                / max(float(np.std(object_sizes)), 1.0)
            )
            for row in rows
        },
        "duplicate_etags": {
            etag: [row["run_id"] for row in rows if row["etag"] == etag]
            for etag in sorted(set(row["etag"] for row in rows))
            if sum(row["etag"] == etag for row in rows) > 1
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit remote raw/preprocessed/denoised NIfTI headers without downloading images."
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--bucket", default="openneuro.org")
    parser.add_argument("--dataset", default="ds004044")
    parser.add_argument("--subjects", nargs="*", default=["sub-30", "sub-42", "sub-52", "sub-62"])
    args = parser.parse_args()

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    targets = list_target_keys(s3, args.bucket, args.dataset, args.subjects)
    rows = []
    for key, size, etag in targets:
        print(f"reading header {key}", flush=True)
        rows.append(header_row(s3, args.bucket, key, size, etag))

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["subject"], row["variant"])].append(row)
    summaries = {
        f"{subject}:{variant}": summarize_group(sorted(group, key=lambda row: row["run_id"]))
        for (subject, variant), group in sorted(grouped.items())
    }
    cross_variant_duplicates = defaultdict(list)
    for row in rows:
        cross_variant_duplicates[row["etag"]].append(
            f'{row["subject"]}:{row["variant"]}:run-{row["run_id"]}'
        )
    result = {
        "bucket": args.bucket,
        "dataset": args.dataset,
        "subjects": args.subjects,
        "row_count": len(rows),
        "rows": rows,
        "summaries": summaries,
        "cross_variant_duplicate_etags": {
            etag: labels
            for etag, labels in cross_variant_duplicates.items()
            if len(labels) > 1
        },
        "note": (
            "Only gzip-compressed NIfTI headers and S3 object metadata were read. Affine/header "
            "consistency can be ruled in or out without downloading multi-gigabyte image payloads."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "row_count": len(rows),
                "summaries": summaries,
                "cross_variant_duplicate_etags": result["cross_variant_duplicate_etags"],
            },
            indent=2,
            default=as_jsonable,
        )
    )


if __name__ == "__main__":
    main()
