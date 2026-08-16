#!/usr/bin/env python3
"""Build a cortical-ribbon mask in the frozen 24^3 feature grid.

This is the decisive control for the coverage question. The hyperaligned surface sits
`0.026` behind the volumetric baseline, and the leading explanation is that the
volumetric bounding box retains subcortex, cerebellum, and white matter while the
surface represents cortex alone. Two cheap probes were consistent with that and
neither confirmed it: feature concatenation added nothing, and a radial partition
turned out to separate brain from non-brain rather than cortex from subcortex.

Restricting the volumetric decoder to cortex removes the coverage advantage while
holding representation, folds, preprocessing, and decoder fixed. If volumetric
accuracy then falls to roughly the hyperaligned surface level, the residual gap is
coverage and the alignment story is complete. If it does not, something else in the
surface pipeline is costing accuracy.

## Why this is affordable

Labelling the grid needs each subject's BOLD geometry, but not their BOLD data. The
NIfTI header is 348 bytes, so an HTTP range request for the first 64 KB of each
gzipped volume yields the shape and affine for a few megabytes across the cohort.
Only the white and pial surfaces are downloaded in full, about 870 MB rather than the
12 GB a full re-download would cost.

## Mapping

`reproduce_thesis_transform` resizes the BOLD volume to `100^3` and then to `24^3`,
both with linear interpolation, so a source voxel `i` lands at `i * 24 / shape_i`. The
ribbon is sampled at several depths between the white and pial surfaces, mapped from
world coordinates to BOLD voxel indices via the affine, then to grid indices by that
scaling.

Subjects differ in anatomy, so the per-subject masks differ. The output is a
**frequency map** — the fraction of subjects for whom each grid voxel is ribbon —
which is thresholded to give a single mask applied to every subject, keeping the
feature space identical across the cohort.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import urllib.request
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (REPO_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from project_bold_to_surface import read_surface  # noqa: E402


BUCKET = "https://s3.amazonaws.com/openneuro.org/ds004044"


def fetch(url: str, destination: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as response, destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            return True
        except Exception as error:  # noqa: BLE001
            print(f"    fetch attempt {attempt + 1}: {error}", flush=True)
    return False


def bold_geometry(subject: str, run_id: int = 1) -> tuple[tuple[int, ...], np.ndarray] | None:
    """Shape and affine from the gzip prefix, without downloading the volume."""
    url = (
        f"{BUCKET}/derivatives/fmriprep/{subject}/"
        f"{subject}_ses-1_task-motor_run-{run_id}_space-T1w_desc-preproc_bold_denoised.nii.gz"
    )
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"Range": "bytes=0-65535"})
            with urllib.request.urlopen(request, timeout=120) as response:
                chunk = response.read()
            raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(chunk)
            header = nib.Nifti1Header.from_fileobj(io.BytesIO(raw[:348]), check=False)
            return tuple(header.get_data_shape()[:3]), header.get_best_affine()
        except Exception as error:  # noqa: BLE001
            print(f"    header attempt {attempt + 1}: {error}", flush=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Cortical ribbon frequency map in the 24^3 grid.")
    parser.add_argument("--subjects-from", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--depths", type=int, default=7)
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    subjects = sorted(p.stem for p in Path(args.subjects_from).glob("sub-*.npz"))
    print(f"{len(subjects)} subjects", flush=True)

    grid = args.grid
    counts = np.zeros(grid**3, dtype=np.int32)
    used = 0
    per_subject = []

    for subject in subjects:
        geometry = bold_geometry(subject)
        if geometry is None:
            print(f"{subject}: no header, skipped", flush=True)
            continue
        shape, affine = geometry
        inverse = np.linalg.inv(affine)

        voxels = set()
        ok = True
        for hemisphere in ("L", "R"):
            paths = {}
            for kind in ("white", "pial"):
                name = f"{subject}.{hemisphere}.{kind}.native.surf.gii"
                local = work / name
                if not local.exists() and not fetch(
                    f"{BUCKET}/derivatives/ciftify/{subject}/native/{name}", local
                ):
                    ok = False
                    break
                paths[kind] = local
            if not ok:
                break
            white, _ = read_surface(paths["white"])
            pial, _ = read_surface(paths["pial"])
            for fraction in (np.arange(args.depths) + 0.5) / args.depths:
                points = white + (pial - white) * fraction
                voxel = np.rint(points @ inverse[:3, :3].T + inverse[:3, 3]).astype(np.int64)
                inside = np.all(
                    (voxel >= 0) & (voxel < np.asarray(shape)), axis=1
                )
                voxel = voxel[inside]
                # Two linear resizes compose to a single scaling per axis.
                scaled = np.rint(
                    voxel * (grid / np.asarray(shape))
                ).astype(np.int64)
                scaled = np.clip(scaled, 0, grid - 1)
                flat = (scaled[:, 0] * grid + scaled[:, 1]) * grid + scaled[:, 2]
                voxels.update(flat.tolist())
        if not ok:
            print(f"{subject}: missing surfaces, skipped", flush=True)
            continue

        index = np.fromiter(voxels, dtype=np.int64)
        counts[index] += 1
        used += 1
        per_subject.append({"subject": subject, "ribbon_voxels": int(len(index))})
        for leftover in work.glob(f"{subject}.*"):
            leftover.unlink(missing_ok=True)
        print(f"{subject}: {len(index)} ribbon voxels ({len(index)/grid**3:.3f})", flush=True)

    frequency = counts / max(used, 1)
    np.savez_compressed(
        args.out_npz,
        frequency=frequency.astype(np.float32),
        subjects_used=used,
        grid=grid,
    )
    summary = {
        "subjects_used": used,
        "grid": grid,
        "depths": args.depths,
        "voxels_total": grid**3,
        "coverage_at_thresholds": {
            str(t): int((frequency >= t).sum()) for t in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "per_subject": per_subject,
    }
    Path(args.out_npz).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["coverage_at_thresholds"], indent=2))


if __name__ == "__main__":
    main()
