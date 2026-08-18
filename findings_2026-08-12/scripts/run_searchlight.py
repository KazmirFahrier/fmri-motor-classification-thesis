#!/usr/bin/env python3
"""Volumetric searchlight over the frozen feature grid.

Searchlight analysis is the standard MVPA localisation method and is absent from all
129 commits. Every accuracy this project reports is whole-grid, so nothing yet says
*where* the discriminative signal lives. The existing feature-stability maps are
explicitly described as model-selection frequencies rather than localisation, and the
project has been careful not to over-read them.

A searchlight answers a different and cleaner question: how well can each small
neighbourhood decode on its own? That is interpretable without any claim about the
whole-brain model's weights.

## Why this is affordable here

A searchlight is usually expensive because each sphere needs its own cross-validated
fit. Here each sphere holds a few dozen voxels against roughly 2480 training events,
so the fit is tiny and direct — no dual basis is needed, since the feature count is
already far below the sample count. The cost is dominated by the number of spheres,
which is why only in-brain voxels are visited.

## Interpretation limits, which are severe

The `24^3` grid is a **bounding-box rescale of native anatomy**, not a registered
space. A voxel index does not correspond to the same anatomical location across
subjects, and the surface work in this round established that this rescale is
nonetheless competitive with anatomy-based surface alignment for decoding.

So a searchlight map here shows **where in the rescaled bounding box** information
concentrates. It supports statements about spatial extent and concentration, and
about gross location such as central versus peripheral or superior versus inferior.
It does **not** license anatomical labels like "M1" or "SMA". Any such claim requires
the normalisation this project does not have.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (REPO_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from run_balanced_event_assignment import metrics  # noqa: E402
from run_detrended_pair_feature_selection import (  # noqa: E402
    load_checkpoints,
    outer_splits,
)
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402
from sklearn.svm import SVC  # noqa: E402


def sphere_offsets(radius: int) -> np.ndarray:
    span = range(-radius, radius + 1)
    offsets = [
        (dx, dy, dz)
        for dx in span
        for dy in span
        for dz in span
        if dx * dx + dy * dy + dz * dz <= radius * radius
    ]
    return np.asarray(offsets, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Searchlight over the 24^3 grid.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fixed-c", type=float, default=0.0001)
    parser.add_argument("--mask-npz", help="Ribbon frequency map; limits which centres are visited.")
    parser.add_argument("--mask-threshold", type=float, default=0.1)
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--report-every", type=int, default=500)
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    x = sequence.mean(axis=1, dtype=np.float32)
    del sequence
    grid = args.grid
    if x.shape[1] != grid**3:
        raise SystemExit(f"{x.shape[1]} features is not {grid}^3")

    centres = np.arange(grid**3)
    if args.mask_npz:
        with np.load(args.mask_npz) as data:
            frequency = data["frequency"]
        centres = np.flatnonzero(frequency >= args.mask_threshold)
        print(f"visiting {len(centres)} of {grid**3} voxels", flush=True)

    offsets = sphere_offsets(args.radius)
    print(f"sphere radius {args.radius}: {len(offsets)} voxels", flush=True)

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    accuracy = np.zeros(grid**3, dtype=np.float32)
    counts = np.zeros(grid**3, dtype=np.int32)

    coords = np.stack(np.unravel_index(centres, (grid, grid, grid)), axis=1)
    started = time.time()
    for position, (flat_index, centre) in enumerate(zip(centres, coords)):
        neighbours = centre + offsets
        inside = np.all((neighbours >= 0) & (neighbours < grid), axis=1)
        neighbours = neighbours[inside]
        columns = (neighbours[:, 0] * grid + neighbours[:, 1]) * grid + neighbours[:, 2]
        block = x[:, columns]

        fold_scores = []
        for split in splits:
            train_idx, val_idx = split["train_idx"], split["val_idx"]
            mean = block[train_idx].mean(axis=0)
            scale = block[train_idx].std(axis=0)
            scale[scale < 1e-8] = 1.0
            model = SVC(C=args.fixed_c, kernel="linear", decision_function_shape="ovr")
            model.fit((block[train_idx] - mean) / scale, y[train_idx])
            prediction = model.predict((block[val_idx] - mean) / scale)
            fold_scores.append(metrics(y[val_idx], prediction)["balanced_accuracy"])
        accuracy[flat_index] = float(np.mean(fold_scores))
        counts[flat_index] = len(columns)

        if (position + 1) % args.report_every == 0:
            elapsed = time.time() - started
            rate = (position + 1) / elapsed
            print(
                f"  {position + 1}/{len(centres)} centres "
                f"[{elapsed:.0f}s, {rate:.1f}/s, eta {(len(centres) - position - 1) / rate:.0f}s]",
                flush=True,
            )

    visited = accuracy[centres]
    np.savez_compressed(
        args.out_npz,
        accuracy=accuracy,
        counts=counts,
        centres=centres,
        grid=grid,
        radius=args.radius,
    )
    summary = {
        "checkpoint_dir": args.checkpoint_dir,
        "grid": grid,
        "radius": args.radius,
        "sphere_voxels": int(len(offsets)),
        "centres_visited": int(len(centres)),
        "outer_split_count": len(splits),
        "chance": 0.25,
        "max_searchlight_accuracy": float(visited.max()),
        "mean_searchlight_accuracy": float(visited.mean()),
        "fraction_above_0.40": float((visited > 0.40).mean()),
        "fraction_above_0.50": float((visited > 0.50).mean()),
        "whole_grid_reference": 0.8051,
        "interpretation_limit": (
            "The grid is a bounding-box rescale, not a registered space. This map "
            "supports statements about spatial extent and gross location, not "
            "anatomical labels."
        ),
    }
    Path(args.out_npz).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
