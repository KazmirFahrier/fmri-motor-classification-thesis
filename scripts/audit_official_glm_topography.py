#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from pathlib import Path

import boto3
import nibabel as nib
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config


TASKS = ["LeftLeg", "RightLeg", "Forearm", "Upperarm"]
PERMUTATIONS = list(itertools.permutations(range(4)))


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def normalize_rows(values: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    return np.nan_to_num(values / denominator)


def discover_subjects(s3, bucket: str, dataset: str) -> list[str]:
    prefix = f"{dataset}/derivatives/ciftify/"
    subjects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for row in page.get("CommonPrefixes", []):
            subject = str(row["Prefix"]).rstrip("/").split("/")[-1]
            if subject.startswith("sub-") and subject != "sub-phantom":
                subjects.append(subject)
    return sorted(subjects)


def zstat_key(dataset: str, subject: str, task: str) -> str:
    directory = (
        f"{dataset}/derivatives/ciftify/{subject}/results/"
        "ses-1_task-motor_hp200_s4_level2.feat"
    )
    filename = (
        f"{subject}_ses-1_task-motor_level2_zstat_{task}_hp200_s4.dscalar.nii"
    )
    return f"{directory}/{filename}"


def load_subject_maps(
    s3,
    bucket: str,
    dataset: str,
    subject: str,
    temp_root: Path,
) -> np.ndarray:
    maps = []
    for task in TASKS:
        path = temp_root / f"{subject}_{task}.dscalar.nii"
        s3.download_file(bucket, zstat_key(dataset, subject, task), str(path))
        values = np.asarray(nib.load(path).get_fdata(dtype=np.float32)).reshape(-1)
        maps.append(values)
        path.unlink(missing_ok=True)
    return np.stack(maps).astype(np.float32)


def best_permutation(similarity: np.ndarray) -> tuple[tuple[int, ...], float]:
    permutation = max(
        PERMUTATIONS,
        key=lambda candidate: float(
            np.mean([similarity[class_id, candidate[class_id]] for class_id in range(4)])
        ),
    )
    score = float(
        np.mean([similarity[class_id, permutation[class_id]] for class_id in range(4)])
    )
    return permutation, score


def mean_off_diagonal(similarity: np.ndarray) -> float:
    mask = ~np.eye(similarity.shape[0], dtype=bool)
    return float(np.mean(similarity[mask]))


def summarize_subject(
    subject: str,
    raw_maps: np.ndarray,
    centered_maps: np.ndarray,
    normalized_maps: np.ndarray,
    group_templates: np.ndarray,
) -> dict:
    similarity = normalized_maps @ group_templates.T
    permutation, best_score = best_permutation(similarity)
    raw_normalized = normalize_rows(raw_maps.astype(np.float64))
    raw_similarity = raw_normalized @ raw_normalized.T
    centered_similarity = normalized_maps @ normalized_maps.T
    return {
        "subject": subject,
        "grayordinate_count": raw_maps.shape[1],
        "mean_raw_zstat": float(np.mean(raw_maps)),
        "mean_positive_zstat": float(np.mean(np.maximum(raw_maps, 0.0))),
        "mean_raw_map_rms": float(np.mean(np.sqrt(np.mean(raw_maps ** 2, axis=1)))),
        "mean_class_centered_map_rms": float(
            np.mean(np.sqrt(np.mean(centered_maps ** 2, axis=1)))
        ),
        "raw_within_subject_mean_class_cosine": mean_off_diagonal(raw_similarity),
        "centered_within_subject_mean_class_cosine": mean_off_diagonal(
            centered_similarity
        ),
        "group_identity_similarity": float(np.mean(np.diag(similarity))),
        "group_best_similarity": best_score,
        "group_permutation_gain": best_score - float(np.mean(np.diag(similarity))),
        "group_best_permutation": list(permutation),
        "group_identity_is_optimal": permutation == (0, 1, 2, 3),
        "group_identity_class_fraction": float(
            np.mean(np.asarray(permutation) == np.arange(4))
        ),
        "group_similarity_matrix": similarity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit authors' official level-2 GLM maps against cohort class topography."
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--bucket", default="openneuro.org")
    parser.add_argument("--dataset", default="ds004044")
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument(
        "--focus-subjects", nargs="*", default=["sub-30", "sub-42", "sub-52", "sub-62"]
    )
    args = parser.parse_args()

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    subjects = args.subjects or discover_subjects(s3, args.bucket, args.dataset)
    raw = []
    completed_subjects = []
    failures = []
    with tempfile.TemporaryDirectory(prefix="official-glm-") as temp_dir:
        temp_root = Path(temp_dir)
        for subject in subjects:
            print(f"downloading official GLM maps {subject}", flush=True)
            try:
                raw.append(
                    load_subject_maps(
                        s3, args.bucket, args.dataset, subject, temp_root
                    )
                )
                completed_subjects.append(subject)
            except Exception as error:
                failures.append({"subject": subject, "error": str(error)})

    raw_maps = np.stack(raw).astype(np.float32)
    valid = np.all(np.isfinite(raw_maps), axis=(0, 1)) & (
        np.max(np.abs(raw_maps), axis=(0, 1)) > 0
    )
    raw_maps = raw_maps[:, :, valid]
    centered = raw_maps - raw_maps.mean(axis=1, keepdims=True)
    normalized = normalize_rows(centered.reshape(-1, centered.shape[-1])).reshape(
        centered.shape
    )

    rows = []
    for subject_index, subject in enumerate(completed_subjects):
        train_indices = np.asarray(
            [index for index in range(len(completed_subjects)) if index != subject_index]
        )
        group_templates = normalize_rows(normalized[train_indices].mean(axis=0))
        rows.append(
            summarize_subject(
                subject,
                raw_maps[subject_index],
                centered[subject_index],
                normalized[subject_index],
                group_templates,
            )
        )

    ranked = sorted(rows, key=lambda row: row["group_identity_similarity"])
    result = {
        "bucket": args.bucket,
        "dataset": args.dataset,
        "tasks": TASKS,
        "subject_count": len(completed_subjects),
        "valid_grayordinate_count": int(valid.sum()),
        "failures": failures,
        "subjects": rows,
        "lowest_group_identity_similarity": ranked[:15],
        "focus_subjects": [
            row for row in rows if row["subject"] in set(args.focus_subjects)
        ],
        "note": (
            "These are the dataset authors' six-run level-2 HCP/Ciftify z-stat maps. The four "
            "maps are centered across class within each subject before leave-subject-out cohort "
            "matching, isolating relative somatotopic pattern from common movement activation."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "subject_count": len(completed_subjects),
                "failures": failures,
                "lowest_group_identity_similarity": [
                    {
                        "subject": row["subject"],
                        "identity_similarity": row["group_identity_similarity"],
                        "identity_class_fraction": row[
                            "group_identity_class_fraction"
                        ],
                        "permutation_gain": row["group_permutation_gain"],
                        "centered_rms": row["mean_class_centered_map_rms"],
                    }
                    for row in ranked[:15]
                ],
                "focus_subjects": [
                    {
                        "subject": row["subject"],
                        "identity_similarity": row["group_identity_similarity"],
                        "identity_class_fraction": row[
                            "group_identity_class_fraction"
                        ],
                        "permutation_gain": row["group_permutation_gain"],
                        "centered_rms": row["mean_class_centered_map_rms"],
                    }
                    for row in result["focus_subjects"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
