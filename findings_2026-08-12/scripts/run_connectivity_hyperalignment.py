#!/usr/bin/env python3
"""Connectivity-based hyperalignment on surface-projected event sequences.

Anatomical surface alignment puts every subject on a shared sphere, but anatomy is
only a proxy for function: two subjects' vertex `k` sit at the same anatomical
location without necessarily carrying the same response. Hyperalignment refines that
by rotating each subject's feature space into a common one.

## Why the usual formulation does not work here, and what does

Classic hyperalignment and SRM both need **temporal correspondence** — subject `i`
and subject `j` experiencing the same thing at the same index — so that Procrustes
has matched columns to align. This design defeats that: the cohort receives 7-8
distinct class orders per run, so event `k` means different things for different
subjects.

Connectivity supplies the correspondence instead. Each vertex is described by its
correlation with a set of **parcels that are shared across subjects by
construction**. Two subjects' connectivity profiles are therefore comparable
column-by-column without any stimulus or label correspondence, and Procrustes has
something well defined to align.

## Method

1. Reshape each subject's `(events, lags, V)` sequences to `(events * lags, V)`.
2. Partition the shared sphere into parcels by nearest-neighbour assignment to a
   coarser icosphere, and average within each to get `(samples, P)` targets.
3. Correlate every vertex against every parcel: a `(V, P)` connectivity profile.
4. Within each parcel, solve orthogonal Procrustes mapping that subject's local
   connectivity block onto a template, refined over a few iterations.
5. Apply each subject's local orthogonal maps to their event features.

Transforms are **orthogonal and per-parcel**. A global `V x V` rotation is infeasible
at this vertex count, and unconstrained maps would let a subject's features be
rewritten freely rather than merely rotated, which is how hyperalignment turns into
overfitting.

## Leakage

The alignment consumes no labels at any point, so it cannot transport class
information. It is nonetheless **transductive**, in the same sense as the existing
subject-run centering: a held-out subject's own unlabeled data is used to estimate
their transform. The template is built from **training subjects only**, which is what
keeps the fold structure honest, and the script asserts this rather than assuming it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (REPO_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from project_bold_to_surface import icosphere  # noqa: E402


def build_parcels(
    target_vertices: np.ndarray,
    parcel_subdivisions: int,
) -> np.ndarray:
    """Assign each vertex to its nearest coarse-icosphere centre."""
    centres, _ = icosphere(parcel_subdivisions)
    unit = target_vertices / np.linalg.norm(target_vertices, axis=1, keepdims=True)
    assignment = np.zeros(len(unit), dtype=np.int64)
    chunk = 4096
    for start in range(0, len(unit), chunk):
        block = unit[start : start + chunk]
        assignment[start : start + chunk] = (block @ centres.T).argmax(axis=1)
    return assignment


def connectivity_profile(
    samples: np.ndarray,
    parcels: np.ndarray,
    parcel_count: int,
) -> np.ndarray:
    """(samples, V) -> (V, P) correlation of each vertex with each parcel mean."""
    centred = samples - samples.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0)
    scale[scale < 1e-8] = 1.0
    centred = centred / scale

    targets = np.zeros((len(samples), parcel_count), dtype=np.float64)
    for parcel in range(parcel_count):
        members = parcels == parcel
        if np.any(members):
            targets[:, parcel] = centred[:, members].mean(axis=1)
    target_scale = targets.std(axis=0)
    target_scale[target_scale < 1e-8] = 1.0
    targets = (targets - targets.mean(axis=0, keepdims=True)) / target_scale
    return (centred.T @ targets) / len(samples)


def orthogonal_procrustes(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Orthogonal R minimising ||R @ source - template||_F."""
    u, _, vt = np.linalg.svd(template @ source.T)
    return u @ vt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connectivity hyperalignment of surface event sequences."
    )
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-subjects", nargs="+", required=True,
                        help="Template is built from these subjects only.")
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--subdivisions", type=int, default=5,
                        help="Must match the extraction's target sphere.")
    parser.add_argument("--parcel-subdivisions", type=int, default=2,
                        help="2 gives 162 parcels per hemisphere.")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(in_dir.glob("sub-*.npz"))
    if not paths:
        raise SystemExit(f"No checkpoints in {in_dir}")

    train = set(args.train_subjects)
    if not train.issubset({p.stem for p in paths}):
        raise SystemExit("Some --train-subjects are absent from --in-dir")

    target, _ = icosphere(args.subdivisions)
    hemisphere_vertices = len(target)
    parcels_one = build_parcels(target, args.parcel_subdivisions)
    parcel_count_one = int(parcels_one.max()) + 1
    # Left and right hemispheres are stacked, so the right gets its own parcel ids.
    parcels = np.concatenate([parcels_one, parcels_one + parcel_count_one])
    parcel_count = parcel_count_one * 2
    print(
        f"{parcel_count} parcels over {len(parcels)} vertices "
        f"(median size {int(np.median(np.bincount(parcels)))})",
        flush=True,
    )

    subjects: dict[str, dict] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            sequence = data[args.sequence_key].astype(np.float32)
            subjects[path.stem] = {
                "sequence": sequence,
                "labels": data["labels"],
                "records": str(data["records_json"]),
                "valid": data["valid_vertices"].astype(bool)
                if "valid_vertices" in data
                else np.ones(sequence.shape[2], dtype=bool),
            }
        events, lags, vertices = sequence.shape
        if vertices != len(parcels):
            raise SystemExit(
                f"{path.stem} has {vertices} vertices, expected {len(parcels)}; "
                "--subdivisions must match the extraction."
            )
        flat = sequence.reshape(events * lags, vertices).astype(np.float64)
        subjects[path.stem]["connectivity"] = connectivity_profile(
            flat, parcels, parcel_count
        )
        print(f"{path.stem}: connectivity {subjects[path.stem]['connectivity'].shape}", flush=True)

    # Template from training subjects only, refined by realigning to it each round.
    transforms: dict[str, dict[int, np.ndarray]] = {name: {} for name in subjects}
    for parcel in range(parcel_count):
        members = np.flatnonzero(parcels == parcel)
        if len(members) < 2:
            continue
        blocks = {
            name: payload["connectivity"][members] for name, payload in subjects.items()
        }
        template = np.mean([blocks[name] for name in sorted(train)], axis=0)
        for _ in range(args.iterations):
            rotations = {
                name: orthogonal_procrustes(block, template)
                for name, block in blocks.items()
            }
            template = np.mean(
                [rotations[name] @ blocks[name] for name in sorted(train)], axis=0
            )
        for name, rotation in rotations.items():
            transforms[name][parcel] = rotation
        if parcel % 50 == 0:
            print(f"  parcel {parcel}/{parcel_count}", flush=True)

    summaries = []
    for name, payload in subjects.items():
        sequence = payload["sequence"]
        events, lags, vertices = sequence.shape
        flat = sequence.reshape(events * lags, vertices).astype(np.float64)
        aligned = np.zeros_like(flat)
        for parcel, rotation in transforms[name].items():
            members = np.flatnonzero(parcels == parcel)
            aligned[:, members] = flat[:, members] @ rotation.T
        out = aligned.reshape(events, lags, vertices).astype(np.float32)
        np.savez_compressed(
            out_dir / f"{name}.npz",
            **{args.sequence_key: out},
            labels=payload["labels"],
            valid_vertices=payload["valid"],
            records_json=payload["records"],
        )
        summaries.append(
            {
                "subject": name,
                "in_template": name in train,
                "parcels_transformed": len(transforms[name]),
            }
        )
        print(f"{name}: aligned and saved", flush=True)

    (out_dir / "hyperalignment_summary.json").write_text(
        json.dumps(
            {
                "source": str(in_dir),
                "parcel_count": parcel_count,
                "iterations": args.iterations,
                "template_subjects": sorted(train),
                "transform": "per-parcel orthogonal Procrustes on connectivity profiles",
                "leakage_note": (
                    "No labels are used at any stage. The template is built from "
                    "training subjects only; each subject's own transform is estimated "
                    "from their own unlabeled data, which is transductive in the same "
                    "sense as the existing subject-run centering."
                ),
                "subjects": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
