#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def center_of_mass(mask: np.ndarray) -> np.ndarray:
    coordinates = np.argwhere(mask)
    return coordinates.mean(axis=0) if len(coordinates) else np.zeros(3, dtype=np.float64)


def bounding_box(mask: np.ndarray) -> list[list[int]]:
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return [[0, 0], [0, 0], [0, 0]]
    return [
        [int(coordinates[:, axis].min()), int(coordinates[:, axis].max())]
        for axis in range(3)
    ]


def summarize_image(label: str, subject: str, run_id: int, path: Path) -> tuple[dict, dict]:
    image = nib.load(path)
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image for {path}, got {data.shape}.")
    finite_fraction = float(np.mean(np.isfinite(data)))
    data = np.nan_to_num(data, copy=False)
    mean_map = data.mean(axis=-1, dtype=np.float64).astype(np.float32)
    std_map = data.std(axis=-1, dtype=np.float64).astype(np.float32)
    positive = mean_map[mean_map > 0]
    threshold = float(np.percentile(positive, 20)) if len(positive) else 0.0
    mask = mean_map > threshold
    com_voxel = center_of_mass(mask)
    com_world = nib.affines.apply_affine(image.affine, com_voxel)
    summary = {
        "label": label,
        "subject": subject,
        "run_id": int(run_id),
        "path": str(path),
        "shape": list(data.shape),
        "voxel_sizes": [float(value) for value in image.header.get_zooms()[:3]],
        "axis_codes": list(nib.aff2axcodes(image.affine)),
        "affine": image.affine.tolist(),
        "finite_fraction": finite_fraction,
        "nonzero_fraction": float(np.mean(data != 0)),
        "mean_intensity": float(np.mean(mean_map[mask])),
        "mean_map_std": float(np.std(mean_map[mask])),
        "median_temporal_std": float(np.median(std_map[mask])),
        "mask_threshold": threshold,
        "mask_voxels": int(mask.sum()),
        "mask_fraction": float(np.mean(mask)),
        "mask_bounding_box": bounding_box(mask),
        "mask_center_of_mass_voxel": com_voxel.tolist(),
        "mask_center_of_mass_world": com_world.tolist(),
    }
    maps = {
        "mean": mean_map,
        "std": std_map,
        "mask": mask,
        "affine": image.affine.copy(),
    }
    del data
    gc.collect()
    return summary, maps


def correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    left = first[mask].astype(np.float64)
    right = second[mask].astype(np.float64)
    if len(left) == 0 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def compare_pair(first_summary: dict, first_maps: dict, second_summary: dict, second_maps: dict) -> dict:
    if first_maps["mean"].shape != second_maps["mean"].shape:
        return {
            "first_label": first_summary["label"],
            "second_label": second_summary["label"],
            "comparable": False,
            "reason": "shape_mismatch",
        }
    intersection = first_maps["mask"] & second_maps["mask"]
    union = first_maps["mask"] | second_maps["mask"]
    first_world = np.asarray(first_summary["mask_center_of_mass_world"])
    second_world = np.asarray(second_summary["mask_center_of_mass_world"])
    return {
        "first_label": first_summary["label"],
        "second_label": second_summary["label"],
        "comparable": True,
        "max_affine_absolute_difference": float(
            np.max(np.abs(first_maps["affine"] - second_maps["affine"]))
        ),
        "mask_intersection_voxels": int(intersection.sum()),
        "mask_union_voxels": int(union.sum()),
        "mask_dice": float(
            2.0
            * intersection.sum()
            / max(float(first_maps["mask"].sum() + second_maps["mask"].sum()), 1.0)
        ),
        "mask_jaccard": float(intersection.sum() / max(float(union.sum()), 1.0)),
        "center_of_mass_distance_mm": float(np.linalg.norm(first_world - second_world)),
        "temporal_mean_map_correlation": correlation(
            first_maps["mean"],
            second_maps["mean"],
            intersection,
        ),
        "temporal_std_map_correlation": correlation(
            first_maps["std"],
            second_maps["std"],
            intersection,
        ),
        "mean_absolute_temporal_mean_difference": float(
            np.mean(np.abs(first_maps["mean"][intersection] - second_maps["mean"][intersection]))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare native T1w-space coverage and summary maps across cached denoised runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=4,
        metavar=("LABEL", "SUBJECT", "RUN_ID", "NIFTI"),
        required=True,
    )
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    summaries = []
    maps = {}
    for label, subject, run_id, path in args.run:
        print(f"summarizing {label}", flush=True)
        summary, run_maps = summarize_image(label, subject, int(run_id), Path(path))
        summaries.append(summary)
        maps[label] = run_maps

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for summary in summaries:
        by_subject[summary["subject"]].append(summary)
    comparisons = []
    for subject, group in sorted(by_subject.items()):
        for first_idx in range(len(group)):
            for second_idx in range(first_idx + 1, len(group)):
                first = group[first_idx]
                second = group[second_idx]
                row = compare_pair(first, maps[first["label"]], second, maps[second["label"]])
                row["subject"] = subject
                comparisons.append(row)

    result = {
        "runs": summaries,
        "comparisons": comparisons,
        "note": (
            "Masks use the same temporal-mean 20th-percentile rule for every run. Mean/std-map "
            "correlations and mask overlap test gross T1w-space registration/coverage consistency, "
            "not task-class correspondence."
        ),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2, default=as_jsonable))
    print(json.dumps({"out_json": args.out_json, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
