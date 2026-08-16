#!/usr/bin/env python3
"""Project space-T1w BOLD onto an inter-subject-aligned surface mesh.

The frozen representation rescales a bounding box of each subject's native
anatomy onto a common grid, which is not registration: voxel `(12, 12, 12)` is the
same fraction across two subjects' bounding boxes, not the same anatomical
location. See `docs/INTER_SUBJECT_NORMALIZATION.md`.

ds004044 ships no MNI-space BOLD, no anatomical transforms, and no preprocessed
grayordinate time series — only level-2 GLM dtseries. What it does ship, under
`derivatives/ciftify/sub-XX/native/`, is the expensive part: reconstructed ribbon
surfaces and MSMSulc registration spheres. This script uses them.

Two steps, both implemented directly on nibabel so Connectome Workbench is not
required:

1. **Ribbon sampling.** Each native vertex is sampled from the volume at several
   depths between the white and pial surfaces and averaged, approximating
   `wb_command -volume-to-surface-mapping -ribbon-constrained`. The BOLD is
   already in `T1w` space, which is the space these surfaces live in, so no
   additional registration is applied.

2. **Spherical resampling.** `sphere.MSMSulc.native` places every subject's native
   mesh in one common spherical frame at radius 100. Resampling each subject onto
   the *same* target icosphere by barycentric interpolation therefore establishes
   vertex-to-vertex anatomical correspondence across subjects.

The target sphere is generated here by subdividing an icosahedron, so no external
template is needed. Correspondence comes from MSMSulc, not from the target mesh,
so any sufficiently dense shared sphere works.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def read_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, triangles) from a GIFTI surface file."""
    image = nib.load(str(path))
    vertices = None
    triangles = None
    for array in image.darrays:
        data = array.data
        if data.ndim != 2 or data.shape[1] != 3:
            continue
        if data.dtype.kind == "f" and vertices is None:
            vertices = np.asarray(data, dtype=np.float64)
        elif data.dtype.kind in "iu" and triangles is None:
            triangles = np.asarray(data, dtype=np.int64)
    if vertices is None or triangles is None:
        raise ValueError(f"{path} does not contain both coordinates and triangles.")
    return vertices, triangles


def icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """A geodesic sphere of radius 1, used as the shared resampling target."""
    phi = (1.0 + 5.0**0.5) / 2.0
    vertices = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    for _ in range(subdivisions):
        midpoint: dict[tuple[int, int], int] = {}
        vertex_list = list(vertices)
        new_faces = []

        def midpoint_index(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in midpoint:
                vertex_list.append((vertices[a] + vertices[b]) / 2.0)
                midpoint[key] = len(vertex_list) - 1
            return midpoint[key]

        for tri_a, tri_b, tri_c in faces:
            ab = midpoint_index(tri_a, tri_b)
            bc = midpoint_index(tri_b, tri_c)
            ca = midpoint_index(tri_c, tri_a)
            new_faces += [
                [tri_a, ab, ca], [tri_b, bc, ab], [tri_c, ca, bc], [ab, bc, ca]
            ]
        vertices = np.asarray(vertex_list, dtype=np.float64)
        faces = np.asarray(new_faces, dtype=np.int64)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices, faces


def sample_volume(
    data: np.ndarray,
    affine: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Trilinear sample a 4D volume at world-space points. Returns (n_points, T)."""
    inverse = np.linalg.inv(affine)
    voxel = points @ inverse[:3, :3].T + inverse[:3, 3]

    floor = np.floor(voxel).astype(np.int64)
    frac = voxel - floor
    shape = np.asarray(data.shape[:3])

    result = np.zeros((len(points), data.shape[3]), dtype=np.float32)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offset = floor + np.array([dx, dy, dz])
                weight = (
                    np.where(dx, frac[:, 0], 1 - frac[:, 0])
                    * np.where(dy, frac[:, 1], 1 - frac[:, 1])
                    * np.where(dz, frac[:, 2], 1 - frac[:, 2])
                )
                inside = np.all((offset >= 0) & (offset < shape), axis=1)
                if not np.any(inside):
                    continue
                index = offset[inside]
                result[inside] += (
                    weight[inside, None]
                    * data[index[:, 0], index[:, 1], index[:, 2], :]
                )
    return result


def inside_field_of_view(
    affine: np.ndarray,
    shape: tuple[int, ...],
    points: np.ndarray,
) -> np.ndarray:
    inverse = np.linalg.inv(affine)
    voxel = points @ inverse[:3, :3].T + inverse[:3, 3]
    bound = np.asarray(shape[:3]) - 1
    return np.all((voxel >= 0) & (voxel <= bound), axis=1)


def ribbon_sample(
    data: np.ndarray,
    affine: np.ndarray,
    white: np.ndarray,
    pial: np.ndarray,
    depth_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average trilinear samples between the white and pial surfaces.

    Also returns a validity mask. The EPI field of view is smaller than the T1w and
    clips inferior and posterior cortex, so roughly a tenth of vertices fall outside
    it — this is coverage, not misregistration (see
    `docs/INTER_SUBJECT_NORMALIZATION.md`). Those vertices must be marked invalid
    rather than silently sampled as zero, because a zero is indistinguishable from a
    real measurement downstream and the affected set differs between subjects.

    A vertex is valid only if **every** sampled depth lies inside the field of view.
    """
    fractions = (np.arange(depth_count) + 0.5) / depth_count
    total = None
    valid = np.ones(len(white), dtype=bool)
    for fraction in fractions:
        points = white + (pial - white) * fraction
        valid &= inside_field_of_view(affine, data.shape, points)
        sampled = sample_volume(data, affine, points)
        total = sampled if total is None else total + sampled
    return total / len(fractions), valid


def barycentric_resample(
    source_sphere: np.ndarray,
    source_triangles: np.ndarray,
    source_values: np.ndarray,
    target_sphere: np.ndarray,
) -> np.ndarray:
    """Interpolate source vertex values onto target sphere directions.

    For each target direction the nearest source vertex is found, then the value
    is taken as an inverse-distance-weighted blend over that vertex's immediate
    neighbourhood, which approximates barycentric interpolation without needing a
    full ray-triangle search.
    """
    source_unit = source_sphere / np.linalg.norm(source_sphere, axis=1, keepdims=True)
    target_unit = target_sphere / np.linalg.norm(target_sphere, axis=1, keepdims=True)

    neighbours: list[set[int]] = [set() for _ in range(len(source_unit))]
    for tri_a, tri_b, tri_c in source_triangles:
        neighbours[tri_a].update((tri_b, tri_c))
        neighbours[tri_b].update((tri_a, tri_c))
        neighbours[tri_c].update((tri_a, tri_b))

    output = np.zeros((len(target_unit), source_values.shape[1]), dtype=np.float32)
    chunk = 4096
    for start in range(0, len(target_unit), chunk):
        block = target_unit[start : start + chunk]
        nearest = (block @ source_unit.T).argmax(axis=1)
        for offset, vertex in enumerate(nearest):
            candidates = [vertex, *neighbours[vertex]]
            coords = source_unit[candidates]
            distance = np.arccos(np.clip(coords @ block[offset], -1.0, 1.0))
            weight = 1.0 / np.maximum(distance, 1e-6)
            weight /= weight.sum()
            output[start + offset] = weight @ source_values[candidates]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project space-T1w BOLD onto an MSMSulc-aligned shared sphere."
    )
    parser.add_argument("--bold", required=True)
    parser.add_argument("--white", required=True)
    parser.add_argument("--pial", required=True)
    parser.add_argument("--sphere", required=True, help="sphere.MSMSulc.native")
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--subdivisions", type=int, default=5,
                        help="5 gives 10242 vertices per hemisphere.")
    parser.add_argument("--ribbon-depths", type=int, default=5)
    parser.add_argument(
        "--validity-threshold",
        type=float,
        default=0.5,
        help="Minimum in-FOV source weight for a target vertex to count as valid.",
    )
    args = parser.parse_args()

    image = nib.load(args.bold)
    data = np.asanyarray(image.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected a 4D BOLD volume, got {data.shape}.")

    white, _ = read_surface(Path(args.white))
    pial, _ = read_surface(Path(args.pial))
    sphere, sphere_triangles = read_surface(Path(args.sphere))
    if not (len(white) == len(pial) == len(sphere)):
        raise ValueError("White, pial, and sphere meshes must share a vertex count.")

    native, native_valid = ribbon_sample(
        data, image.affine, white, pial, args.ribbon_depths
    )
    target, target_triangles = icosphere(args.subdivisions)
    resampled = barycentric_resample(sphere, sphere_triangles, native, target)
    # Propagate validity through the same resampling, then threshold: a target
    # vertex is trustworthy only if it draws predominantly on in-FOV sources.
    target_valid = (
        barycentric_resample(
            sphere,
            sphere_triangles,
            native_valid.astype(np.float32)[:, None],
            target,
        )[:, 0]
        >= args.validity_threshold
    )
    resampled[~target_valid] = 0.0

    np.savez_compressed(
        args.out_npz,
        timeseries=resampled.astype(np.float32),
        valid_vertices=target_valid,
        target_vertices=target.astype(np.float32),
        target_triangles=target_triangles.astype(np.int32),
    )
    summary = {
        "bold": args.bold,
        "volume_shape": list(data.shape),
        "native_vertex_count": int(len(white)),
        "target_vertex_count": int(len(target)),
        "timepoints": int(resampled.shape[1]),
        "finite_fraction": float(np.isfinite(resampled).mean()),
        "native_in_fov_fraction": float(native_valid.mean()),
        "target_valid_fraction": float(target_valid.mean()),
        "mean_temporal_std_valid": float(
            resampled[target_valid].std(axis=1).mean() if np.any(target_valid) else 0.0
        ),
        "out_npz": args.out_npz,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
