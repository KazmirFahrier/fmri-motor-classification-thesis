#!/usr/bin/env python3
"""Sweep the spatial smoothing kernel as a preprocessing choice.

`smooth_3` is applied as a fixed transform inside the hierarchy's pair specialists and
was never swept as a preprocessing choice for the pipeline as a whole. The coverage map
records it as open on exactly those grounds.

It needs **no re-extraction**. `mean_smooth` in
`scripts/run_spatial_scale_feature_sweep.py` operates on the extracted 13824-feature
vectors, reshaping them to the `24^3` grid and box-filtering there, so the whole sweep
runs on the frozen checkpoints.

## Why the answer is not obvious

The distributed-signal finding cuts both ways. If class information is spread across
many weakly informative voxels, smoothing should help by averaging correlated
neighbours and raising per-feature signal-to-noise. If it is carried by fine spatial
detail that survives the `24^3` downsampling, smoothing should destroy it.

The `24^3` grid is already a heavy downsampling of a `~100^3` volume, so each feature
already pools a substantial neighbourhood. That argues the marginal value of further
smoothing is small — which is a prediction this sweep can check rather than assume.

Both the fixed-threshold and the nested estimate are reported. The project has twice
measured that choosing a preprocessing parameter with the whole cohort visible flatters
the result — ANOVA selection lost more than half its gain to nesting, and the temporal
window lost all of it — so a fixed-kernel number here would not be quotable on its own.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "findings_2026-08-12" / "scripts",
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from run_balanced_event_assignment import apply_balanced_assignment, metrics  # noqa: E402
from run_detrended_pair_feature_selection import (  # noqa: E402
    inner_splits,
    load_checkpoints,
    outer_splits,
)
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402
from run_spatial_scale_feature_sweep import mean_smooth  # noqa: E402
from run_standard_mvpa_baseline import dual_basis, fit_projected, standardize  # noqa: E402


def scores_for(
    block: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    c_value: float,
    model: str,
) -> np.ndarray:
    mean, scale = standardize(block, train_idx)
    z_train, z_eval = dual_basis(
        (block[train_idx] - mean) / scale, (block[eval_idx] - mean) / scale
    )
    return fit_projected(
        model, z_train, y[train_idx], z_eval,
        z_train @ z_train.T, z_eval @ z_train.T, c_value, 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial smoothing kernel sweep.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--model", default="linear_svm")
    parser.add_argument("--fixed-c", type=float, default=0.0001)
    parser.add_argument("--kernels", nargs="+", type=int, default=[1, 3, 5, 7])
    parser.add_argument("--feature-shape", nargs=3, type=int, default=[24, 24, 24])
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split-limit", type=int)
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    base = sequence.mean(axis=1, dtype=np.float32)
    shape = tuple(args.feature_shape)
    print(f"{base.shape[0]} events, grid {shape}", flush=True)

    # Smoothing is label-free, so every kernel is applied once up front and reused
    # across folds and across the inner selection.
    smoothed = {}
    for kernel in args.kernels:
        smoothed[kernel] = (
            base if kernel == 1
            else mean_smooth(base, shape, kernel, args.batch_size)
        )
        print(f"kernel {kernel} prepared", flush=True)

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    fixed = {kernel: {"independent": [], "balanced": []} for kernel in args.kernels}
    nested = {"independent": [], "balanced": []}
    selections = []

    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]

        for kernel in args.kernels:
            s = scores_for(
                smoothed[kernel], y, train_idx, val_idx, args.fixed_c, args.model
            )
            for rule, prediction in (
                ("independent", s.argmax(axis=1).astype(np.int64)),
                ("balanced", apply_balanced_assignment(s, val_idx, records)),
            ):
                fixed[kernel][rule].append(
                    float(metrics(y[val_idx], prediction)["balanced_accuracy"])
                )

        # Nested: choose the kernel on inner subject folds of the training set only.
        inner = inner_splits(
            records, train_idx, "subject", args.inner_subject_fold_count
        )
        inner_means = {}
        for kernel in args.kernels:
            values = []
            for inner_split in inner:
                s = scores_for(
                    smoothed[kernel], y, inner_split["train_idx"],
                    inner_split["val_idx"], args.fixed_c, args.model,
                )
                values.append(
                    metrics(y[inner_split["val_idx"]],
                            s.argmax(axis=1))["balanced_accuracy"]
                )
            inner_means[kernel] = float(np.mean(values))
        best = max(args.kernels, key=lambda k: inner_means[k])

        s = scores_for(smoothed[best], y, train_idx, val_idx, args.fixed_c, args.model)
        for rule, prediction in (
            ("independent", s.argmax(axis=1).astype(np.int64)),
            ("balanced", apply_balanced_assignment(s, val_idx, records)),
        ):
            nested[rule].append(
                float(metrics(y[val_idx], prediction)["balanced_accuracy"])
            )
        selections.append({"split": split["split"], "kernel": best,
                           "inner_means": inner_means})
        print(f"{split['split']} selected kernel {best} "
              f"nested={nested['independent'][-1]:.4f}", flush=True)

    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "model": args.model,
        "outer_split_count": len(splits),
        "kernels": args.kernels,
        "fixed": {
            str(kernel): {
                rule: {"mean": float(np.mean(values)), "sd": float(np.std(values))}
                for rule, values in fixed[kernel].items()
            }
            for kernel in args.kernels
        },
        "nested": {
            rule: {"mean": float(np.mean(values)), "sd": float(np.std(values))}
            for rule, values in nested.items()
        },
        "selected_kernel_counts": dict(Counter(s["kernel"] for s in selections)),
        "selections": selections,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))

    print("\n  kernel   independent   balanced")
    for kernel in args.kernels:
        print(f"  {kernel:<8} {np.mean(fixed[kernel]['independent']):.4f}        "
              f"{np.mean(fixed[kernel]['balanced']):.4f}")
    print(f"  nested   {np.mean(nested['independent']):.4f}        "
          f"{np.mean(nested['balanced']):.4f}")
    print(f"\nselected kernels: {dict(Counter(s['kernel'] for s in selections))}")


if __name__ == "__main__":
    main()
