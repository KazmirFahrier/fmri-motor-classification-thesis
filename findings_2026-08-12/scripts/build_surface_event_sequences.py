#!/usr/bin/env python3
"""Build surface-projected event sequences for the whole cohort.

Produces a drop-in replacement for the frozen `(48, 8, 13824)` volumetric
checkpoints, with the volumetric bounding-box rescale replaced by projection onto
an inter-subject-aligned cortical surface. Output shape is `(48, 8, V)` where `V`
is the combined left and right shared-sphere vertex count, and the archive carries
the same `labels` and `records_json` keys, so every downstream decoder runs against
it unchanged.

## Why this is the interesting comparison

The frozen representation applies `zscore(resize(volume))` to `space-T1w` BOLD,
which rescales a bounding box of native anatomy and gives no anatomical
correspondence between subjects. This path instead samples the cortical ribbon and
resamples every subject onto one shared sphere through their MSMSulc registration,
so vertex *k* means the same anatomical location in every subject. Running the same
decoder on both isolates the contribution of genuine inter-subject alignment.

## Mechanics

- Volumes are chosen exactly as the frozen extraction chooses them:
  `event_start + offset + step` for `step` in `range(length)`, so the temporal
  window is identical and only the spatial representation differs.
- The resampling operator depends only on the two spheres, not on the data, so it
  is built **once per subject and hemisphere** and applied to every volume. Rebuilding
  it per volume would dominate the runtime.
- Each run's NIfTI is streamed, used, and deleted immediately. Peak disk stays at
  roughly one run.
- Vertices outside the EPI field of view are flagged rather than silently zeroed;
  the per-subject valid mask is stored alongside the sequences.
- Work is checkpointed per subject, so an interrupted run resumes by skipping
  subjects whose output already exists.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (REPO_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from project_bold_to_surface import (  # noqa: E402
    icosphere,
    inside_field_of_view,
    read_surface,
    sample_volume,
)
from sweep_continuous_bold_windows import load_events  # noqa: E402


BUCKET = "https://s3.amazonaws.com/openneuro.org/ds004044"
HEMISPHERES = ("L", "R")


def fetch(url: str, destination: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as response, destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            return True
        except Exception as error:  # noqa: BLE001 - network failures are expected
            print(f"    fetch attempt {attempt + 1} failed: {error}", flush=True)
            time.sleep(3 * (attempt + 1))
    return False


def build_resampler(
    source_sphere: np.ndarray,
    source_triangles: np.ndarray,
    target_sphere: np.ndarray,
    neighbour_cap: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute barycentric-style weights from source vertices to target ones.

    Returns index and weight arrays of shape ``(n_target, neighbour_cap)``. Depends
    only on geometry, so it is reused across every volume of every run.
    """
    source_unit = source_sphere / np.linalg.norm(source_sphere, axis=1, keepdims=True)
    target_unit = target_sphere / np.linalg.norm(target_sphere, axis=1, keepdims=True)

    neighbours: list[set[int]] = [set() for _ in range(len(source_unit))]
    for tri_a, tri_b, tri_c in source_triangles:
        neighbours[tri_a].update((tri_b, tri_c))
        neighbours[tri_b].update((tri_a, tri_c))
        neighbours[tri_c].update((tri_a, tri_b))

    indices = np.zeros((len(target_unit), neighbour_cap), dtype=np.int64)
    weights = np.zeros((len(target_unit), neighbour_cap), dtype=np.float32)
    chunk = 2048
    for start in range(0, len(target_unit), chunk):
        block = target_unit[start : start + chunk]
        nearest = (block @ source_unit.T).argmax(axis=1)
        for offset, vertex in enumerate(nearest):
            candidates = [vertex, *sorted(neighbours[vertex])][:neighbour_cap]
            coords = source_unit[candidates]
            distance = np.arccos(np.clip(coords @ block[offset], -1.0, 1.0))
            weight = 1.0 / np.maximum(distance, 1e-6)
            weight /= weight.sum()
            indices[start + offset, : len(candidates)] = candidates
            weights[start + offset, : len(candidates)] = weight
    return indices, weights


def apply_resampler(
    values: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """values: (n_source, k) -> (n_target, k)."""
    return np.einsum("tn,tnk->tk", weights, values[indices])


def process_subject(
    subject: str,
    out_dir: Path,
    work_dir: Path,
    target: np.ndarray,
    args: argparse.Namespace,
) -> dict | None:
    destination = out_dir / f"{subject}.npz"
    if destination.exists():
        print(f"{subject}: already done, skipping", flush=True)
        return None

    surfaces: dict[str, dict[str, np.ndarray]] = {}
    resamplers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for hemisphere in HEMISPHERES:
        paths = {}
        for kind in ("white", "pial", "sphere.MSMSulc"):
            name = f"{subject}.{hemisphere}.{kind}.native.surf.gii"
            local = work_dir / name
            if not local.exists() and not fetch(
                f"{BUCKET}/derivatives/ciftify/{subject}/native/{name}", local
            ):
                print(f"{subject}: missing {name}", flush=True)
                return None
            paths[kind] = local
        white, _ = read_surface(paths["white"])
        pial, _ = read_surface(paths["pial"])
        sphere, sphere_triangles = read_surface(paths["sphere.MSMSulc"])
        surfaces[hemisphere] = {"white": white, "pial": pial}
        resamplers[hemisphere] = build_resampler(sphere, sphere_triangles, target)
        print(f"{subject}.{hemisphere}: {len(white)} native vertices", flush=True)

    sequences = []
    labels = []
    records = []
    valid_mask = np.ones(len(target) * len(HEMISPHERES), dtype=bool)

    for run_id in range(1, args.run_count + 1):
        # Raw BIDS pads the run index (`run-01`); the fmriprep derivatives do not
        # (`run-1`). Both forms are needed.
        stem = f"{subject}_ses-1_task-motor_run-{run_id}"
        raw_stem = f"{subject}_ses-1_task-motor_run-{run_id:02d}"
        events_local = work_dir / f"{raw_stem}_events.tsv"
        if not events_local.exists() and not fetch(
            f"{BUCKET}/{subject}/ses-1/func/{raw_stem}_events.tsv", events_local
        ):
            print(f"{subject} run-{run_id}: no events", flush=True)
            return None
        events = load_events(events_local, args.repetition_time)

        bold_local = work_dir / f"{stem}_bold.nii.gz"
        if not fetch(
            f"{BUCKET}/derivatives/fmriprep/{subject}/"
            f"{stem}_space-T1w_desc-preproc_bold_denoised.nii.gz",
            bold_local,
        ):
            print(f"{subject} run-{run_id}: no BOLD", flush=True)
            return None

        image = nib.load(str(bold_local))
        data = np.asanyarray(image.dataobj, dtype=np.float32)
        max_volume = data.shape[3] - 1
        usable = [
            event
            for event in events
            if event["event_start"] + args.offset + args.length - 1 <= max_volume
        ]
        needed = sorted(
            {
                event["event_start"] + args.offset + step
                for event in usable
                for step in range(args.length)
            }
        )
        subset = data[..., needed]
        del data

        per_hemisphere = []
        for hemisphere in HEMISPHERES:
            white = surfaces[hemisphere]["white"]
            pial = surfaces[hemisphere]["pial"]
            fractions = (np.arange(args.ribbon_depths) + 0.5) / args.ribbon_depths
            total = None
            in_fov = np.ones(len(white), dtype=bool)
            for fraction in fractions:
                points = white + (pial - white) * fraction
                in_fov &= inside_field_of_view(image.affine, subset.shape, points)
                sampled = sample_volume(subset, image.affine, points)
                total = sampled if total is None else total + sampled
            native = total / len(fractions)
            indices, weights = resamplers[hemisphere]
            per_hemisphere.append(apply_resampler(native, indices, weights))
            hemi_valid = (
                apply_resampler(in_fov.astype(np.float32)[:, None], indices, weights)[:, 0]
                >= args.validity_threshold
            )
            slot = HEMISPHERES.index(hemisphere)
            valid_mask[slot * len(target) : (slot + 1) * len(target)] &= hemi_valid
        del subset

        projected = np.concatenate(per_hemisphere, axis=0)  # (2V, n_needed)
        position = {volume: column for column, volume in enumerate(needed)}
        for event in usable:
            columns = [
                position[event["event_start"] + args.offset + step]
                for step in range(args.length)
            ]
            sequences.append(projected[:, columns].T.astype(np.float32))
            labels.append(event["class_id"])
            records.append(
                {
                    "subject_id": subject,
                    "run_id": run_id,
                    "event_start": int(event["event_start"]),
                    "class_id": int(event["class_id"]),
                }
            )
        bold_local.unlink(missing_ok=True)
        print(
            f"{subject} run-{run_id}: {len(usable)} events, {len(needed)} volumes",
            flush=True,
        )

    sequence_array = np.stack(sequences).astype(np.float32)
    np.savez_compressed(
        destination,
        **{f"offset_{args.offset}_length_{args.length}_sequence": sequence_array},
        labels=np.asarray(labels, dtype=np.int64),
        valid_vertices=valid_mask,
        records_json=json.dumps(records),
    )
    for leftover in work_dir.glob(f"{subject}.*"):
        leftover.unlink(missing_ok=True)
    summary = {
        "subject": subject,
        "shape": list(sequence_array.shape),
        "valid_vertex_fraction": float(valid_mask.mean()),
        "events": len(labels),
    }
    print(f"{subject}: saved {summary}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cohort-wide surface-projected event sequence extraction."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument(
        "--subjects-from",
        help=(
            "Directory of existing sub-*.npz checkpoints to take the subject list "
            "from. Strongly preferred over --subject-count: the cohort is NOT numbered "
            "contiguously (15, 19, 28, 40, 41 and 64 are absent, and it runs to 68), so "
            "a generated range both fails on subjects that do not exist and silently "
            "omits ones that do. Taking the list from the frozen checkpoints also "
            "guarantees the surface cohort matches the volumetric cohort exactly, which "
            "is required for a paired comparison."
        ),
    )
    parser.add_argument("--subject-count", type=int, default=62)
    parser.add_argument("--run-count", type=int, default=6)
    parser.add_argument("--offset", type=int, default=3)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--repetition-time", type=float, default=2.0)
    parser.add_argument("--ribbon-depths", type=int, default=5)
    parser.add_argument("--subdivisions", type=int, default=5)
    parser.add_argument("--validity-threshold", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.subjects:
        subjects = args.subjects
    elif args.subjects_from:
        subjects = sorted(
            path.stem for path in Path(args.subjects_from).glob("sub-*.npz")
        )
        if not subjects:
            raise SystemExit(f"No sub-*.npz in {args.subjects_from}")
        print(f"cohort from {args.subjects_from}: {len(subjects)} subjects", flush=True)
    else:
        # Fallback only. The cohort is not contiguously numbered, so this will both
        # fail on absent subjects and omit real ones; prefer --subjects-from.
        subjects = [f"sub-{index:02d}" for index in range(1, args.subject_count + 1)]
    target, _ = icosphere(args.subdivisions)
    print(f"target sphere: {len(target)} vertices per hemisphere", flush=True)

    summaries = []
    started = time.time()
    for subject in subjects:
        try:
            summary = process_subject(subject, out_dir, work_dir, target, args)
            if summary:
                summaries.append(summary)
        except Exception as error:  # noqa: BLE001 - one bad subject must not stop the cohort
            print(f"{subject}: FAILED {type(error).__name__}: {error}", flush=True)
        print(f"  elapsed {time.time() - started:.0f}s", flush=True)

    (out_dir / "extraction_summary.json").write_text(
        json.dumps(
            {
                "offset": args.offset,
                "length": args.length,
                "subdivisions": args.subdivisions,
                "vertices_per_hemisphere": int(len(target)),
                "ribbon_depths": args.ribbon_depths,
                "subjects_completed": len(summaries),
                "summaries": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
