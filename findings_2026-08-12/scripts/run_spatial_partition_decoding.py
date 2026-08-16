#!/usr/bin/env python3
"""Decode from radial partitions of the volumetric grid.

The hyperaligned surface remains `0.026` behind the volumetric baseline, and adding
surface features to the volumetric ones improves nothing. The leading explanation is
coverage: the volumetric bounding box retains subcortex, cerebellum, and white
matter, while the surface represents cortex alone.

The decisive test is a ribbon-masked volumetric decoder, but labelling the `24^3`
grid anatomically needs every subject's surfaces and BOLD affines re-downloaded.
This is a cheap proxy that uses only data already on disk.

## The proxy, and its limits

`reproduce_thesis_transform` rescales each subject's bounding box onto a common
`24^3` grid. That is not registration, but it does impose a coarse and consistent
geometry: cortex lies near the outer surface of the box and subcortical structures
lie near its centre. Partitioning voxels by radial distance from the grid centre
therefore separates roughly-cortical from roughly-central signal.

This is **crude**. The bounding box includes non-brain voxels, the brain is not
spherical, cerebellum is inferior rather than central, and no partition boundary
corresponds to an anatomical one. It cannot substitute for the ribbon-masked control.
What it can do is answer a coarser question that still bears on the interpretation:
**does the central part of the volume carry class information independently of the
periphery?** If it does not, the coverage explanation weakens considerably, because
the structures the surface discards would be contributing little.
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

from run_balanced_event_assignment import apply_balanced_assignment, metrics  # noqa: E402
from run_detrended_pair_feature_selection import (  # noqa: E402
    load_checkpoints,
    outer_splits,
)
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402
from run_standard_mvpa_baseline import (  # noqa: E402
    dual_basis,
    fit_projected,
    standardize,
)


def radial_partitions(grid: int, edges: list[float]) -> dict[str, np.ndarray]:
    """Normalised radius of each voxel from the grid centre, bucketed by `edges`."""
    axis = (np.arange(grid) - (grid - 1) / 2.0) / ((grid - 1) / 2.0)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2 + zz**2).ravel()
    radius = radius / radius.max()
    partitions = {}
    bounds = [0.0, *edges, 1.0001]
    for lower, upper in zip(bounds[:-1], bounds[1:]):
        mask = (radius >= lower) & (radius < upper)
        if mask.sum() >= 50:
            partitions[f"r_{lower:.2f}_{min(upper, 1.0):.2f}"] = mask
    partitions["all"] = np.ones(grid**3, dtype=bool)
    return partitions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode from radial shells of the volumetric feature grid."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--edges", nargs="+", type=float, default=[0.4, 0.6, 0.8])
    parser.add_argument("--fixed-c", type=float, default=0.0001)
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--split-limit", type=int)
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    mean_x = sequence.mean(axis=1, dtype=np.float32)
    del sequence
    if mean_x.shape[1] != args.grid**3:
        raise SystemExit(f"{mean_x.shape[1]} features is not {args.grid}^3")

    partitions = radial_partitions(args.grid, args.edges)
    for name, mask in partitions.items():
        print(f"  {name}: {int(mask.sum())} voxels", flush=True)

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    rows = []
    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        for name, mask in partitions.items():
            block = mean_x[:, mask]
            mean, scale = standardize(block, train_idx)
            z_train, z_val = dual_basis(
                (block[train_idx] - mean) / scale, (block[val_idx] - mean) / scale
            )
            scores = fit_projected(
                "linear_svm", z_train, y[train_idx], z_val,
                z_train @ z_train.T, z_val @ z_train.T, args.fixed_c, 0,
            )
            for rule, prediction in (
                ("independent", scores.argmax(axis=1).astype(np.int64)),
                ("balanced", apply_balanced_assignment(scores, val_idx, records)),
            ):
                rows.append(
                    {
                        "split": split["split"],
                        "partition": name,
                        "voxels": int(mask.sum()),
                        "prediction_rule": rule,
                        "balanced_accuracy": float(
                            metrics(y[val_idx], prediction)["balanced_accuracy"]
                        ),
                    }
                )
        print(f"{split['split']} done", flush=True)

    summary = {}
    for name in partitions:
        for rule in ("independent", "balanced"):
            values = [
                r["balanced_accuracy"] for r in rows
                if r["partition"] == name and r["prediction_rule"] == rule
            ]
            summary[f"{name}|{rule}"] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values)),
                "voxels": int(partitions[name].sum()),
            }

    Path(args.out_json).write_text(
        json.dumps(
            {
                "checkpoint_dir": args.checkpoint_dir,
                "grid": args.grid,
                "edges": args.edges,
                "fixed_c": args.fixed_c,
                "outer_split_count": len(splits),
                "caveat": (
                    "Radial distance in a bounding-box rescale is a crude proxy for "
                    "cortical versus central anatomy. No partition boundary is "
                    "anatomical. This does not replace a ribbon-masked control."
                ),
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\n{'partition|rule':28s} {'voxels':>7s} {'mean':>8s} {'sd':>8s}")
    for name, row in summary.items():
        print(f"{name:28s} {row['voxels']:7d} {row['mean']:8.4f} {row['sd']:8.4f}")


if __name__ == "__main__":
    main()
