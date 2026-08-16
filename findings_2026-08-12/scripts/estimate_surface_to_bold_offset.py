#!/usr/bin/env python3
"""Estimate the frame offset between ciftify surfaces and fmriprep BOLD.

Sampling the fmriprep `space-T1w` BOLD directly at ciftify surface coordinates puts
only about 89% of cortical vertices inside the field of view, and the
surface-versus-volume centroids disagree by roughly 19 mm in `y`. A pure
translation with zero `x` and tens of millimetres in `y`/`z` is the signature of the
FreeSurfer surface-RAS versus scanner-RAS `c_ras` offset: the surfaces are in
FreeSurfer surface coordinates while the BOLD is in scanner coordinates.

The exact value lives in the FreeSurfer `orig.mgz` header, and ds004044 ships no
FreeSurfer volume, no T1w reference, and no transform in either derivative tree.
It therefore has to be recovered from the data.

## Objective

For a candidate offset, the score is the **fraction of midthickness vertices that
fall inside the BOLD brain mask**. Cortex should lie inside the brain, so a correct
offset drives this close to 1. The measure is bounded, interpretable, insensitive to
intensity scaling, and cannot be inflated by drifting into a bright region the way a
mean-intensity objective can.

A coarse-to-fine search over translations is used rather than a general rigid
search, because the hypothesis under test is specifically a translation. The script
reports whether a translation alone suffices; if the best achievable in-mask
fraction stays low, the two spaces differ by more than `c_ras` and a full rigid
registration is required instead.

## Validation

`c_ras` is a property of each subject's anatomical acquisition, so estimates need
not be identical across subjects, but for one study on one scanner they should be
tightly clustered. Running several subjects and inspecting the spread is the check
on whether the estimator is recovering a real quantity or fitting noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from project_bold_to_surface import read_surface


def vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area-weighted outward vertex normals."""
    v0, v1, v2 = (vertices[triangles[:, i]] for i in range(3))
    face = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(vertices)
    for column in range(3):
        np.add.at(normals, triangles[:, column], face)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    normals /= lengths
    # A closed cortical surface should point outward; flip if the mesh winding
    # produced inward normals, judged against the vector from the centroid.
    outward = vertices - vertices.mean(axis=0)
    if float(np.sum(normals * outward)) < 0:
        normals = -normals
    return normals


def sample_nearest(
    volume: np.ndarray,
    affine: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour sample; returns values and an in-bounds flag."""
    inverse = np.linalg.inv(affine)
    voxel = np.rint(points @ inverse[:3, :3].T + inverse[:3, 3]).astype(np.int64)
    shape = np.asarray(volume.shape[:3])
    inside = np.all((voxel >= 0) & (voxel < shape), axis=1)
    values = np.zeros(len(points), dtype=np.float64)
    if np.any(inside):
        index = voxel[inside]
        values[inside] = volume[index[:, 0], index[:, 1], index[:, 2]]
    return values, inside


def boundary_contrast(
    volume: np.ndarray,
    affine: np.ndarray,
    vertices: np.ndarray,
    normals: np.ndarray,
    offset: np.ndarray,
    step_mm: float,
) -> float:
    """Boundary-based objective: normalised contrast across the white surface.

    The white surface separates white matter from grey matter, and in a T1w image
    white matter is the brighter of the two. Sampling a short distance inside and
    outside along the surface normal and maximising the normalised difference is
    the standard boundary-based registration cost, and unlike a mask overlap it
    needs no skull stripping and is insensitive to intensity scaling.
    """
    shifted = vertices + offset
    inner, ok_in = sample_nearest(volume, affine, shifted - step_mm * normals)
    outer, ok_out = sample_nearest(volume, affine, shifted + step_mm * normals)
    valid = ok_in & ok_out & ((inner + outer) > 0)
    if np.sum(valid) < 0.5 * len(vertices):
        return -1.0
    contrast = (inner[valid] - outer[valid]) / (inner[valid] + outer[valid])
    return float(np.mean(contrast))


def brain_mask(mean_volume: np.ndarray, percentile: float) -> np.ndarray:
    """Threshold the mean EPI at a percentile of its non-zero voxels."""
    finite = mean_volume[np.isfinite(mean_volume) & (mean_volume > 0)]
    if finite.size == 0:
        raise ValueError("Mean volume has no positive voxels.")
    return mean_volume > np.percentile(finite, percentile)


def in_mask_fraction(
    points: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    offset: np.ndarray,
) -> float:
    inverse = np.linalg.inv(affine)
    shifted = points + offset
    voxel = np.rint(shifted @ inverse[:3, :3].T + inverse[:3, 3]).astype(np.int64)
    shape = np.asarray(mask.shape)
    inside = np.all((voxel >= 0) & (voxel < shape), axis=1)
    if not np.any(inside):
        return 0.0
    index = voxel[inside]
    hits = mask[index[:, 0], index[:, 1], index[:, 2]]
    # Vertices outside the field of view count as misses, not as absent data.
    return float(np.sum(hits) / len(points))


def search_offset(
    points: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    coarse_range: int,
    coarse_step: int,
    fine_step: float,
) -> tuple[np.ndarray, float, list[dict]]:
    trace = []
    best_offset = np.zeros(3)
    best_score = in_mask_fraction(points, mask, affine, best_offset)
    trace.append({"stage": "identity", "offset": [0.0, 0.0, 0.0], "score": best_score})

    grid = np.arange(-coarse_range, coarse_range + 1, coarse_step, dtype=float)
    for dx in grid:
        for dy in grid:
            for dz in grid:
                candidate = np.array([dx, dy, dz])
                score = in_mask_fraction(points, mask, affine, candidate)
                if score > best_score:
                    best_score, best_offset = score, candidate
    trace.append(
        {"stage": "coarse", "offset": best_offset.tolist(), "score": best_score}
    )

    step = float(coarse_step) / 2.0
    while step >= fine_step:
        improved = True
        while improved:
            improved = False
            for axis in range(3):
                for direction in (-1.0, 1.0):
                    candidate = best_offset.copy()
                    candidate[axis] += direction * step
                    score = in_mask_fraction(points, mask, affine, candidate)
                    if score > best_score:
                        best_score, best_offset, improved = score, candidate, True
        step /= 2.0
    trace.append(
        {"stage": "fine", "offset": best_offset.tolist(), "score": best_score}
    )
    return best_offset, best_score, trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover the surface-to-BOLD translation for one subject."
    )
    parser.add_argument("--bold", required=True)
    parser.add_argument("--white", required=True)
    parser.add_argument("--pial", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--mask-percentile", type=float, default=60.0)
    parser.add_argument("--coarse-range", type=int, default=36)
    parser.add_argument("--coarse-step", type=int, default=6)
    parser.add_argument("--fine-step", type=float, default=0.5)
    args = parser.parse_args()

    image = nib.load(args.bold)
    data = np.asanyarray(image.dataobj, dtype=np.float32)
    mean_volume = data.mean(axis=3) if data.ndim == 4 else data
    del data
    mask = brain_mask(mean_volume, args.mask_percentile)

    white, _ = read_surface(Path(args.white))
    pial, _ = read_surface(Path(args.pial))
    midthickness = (white + pial) / 2.0

    offset, score, trace = search_offset(
        midthickness,
        mask,
        image.affine,
        args.coarse_range,
        args.coarse_step,
        args.fine_step,
    )

    payload = {
        "bold": args.bold,
        "white": args.white,
        "volume_shape": list(mean_volume.shape),
        "mask_percentile": args.mask_percentile,
        "mask_voxel_fraction": float(mask.mean()),
        "vertex_count": int(len(midthickness)),
        "identity_in_mask_fraction": trace[0]["score"],
        "estimated_offset_mm": [float(v) for v in offset],
        "in_mask_fraction": score,
        "improvement": score - trace[0]["score"],
        "translation_sufficient": bool(score >= 0.90),
        "trace": trace,
        "note": (
            "Offset is applied to surface coordinates to bring them into the BOLD "
            "frame. A high in-mask fraction under a pure translation supports the "
            "c_ras hypothesis; a low one means a full rigid registration is needed."
        ),
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "trace"}, indent=2))


if __name__ == "__main__":
    main()
