#!/usr/bin/env python3
"""Paired subject-level test of the smoothing kernel's effect.

The sweep established that `smooth_3` is worth `+0.0177` as a point estimate and is
selected in all 30 folds. Every other head-to-head claim in this project carries a
paired difference with a bootstrap CI over subjects, and this one needs the same
treatment before it can be quoted — particularly because it bears on the headline
comparison between the frozen hierarchy and conventional MVPA.

Both conditions are scored on **identical folds within a single process**, so the
difference is genuinely paired rather than a comparison of two separately-run point
estimates.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
    load_checkpoints,
    outer_splits,
)
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402
from run_spatial_scale_feature_sweep import mean_smooth  # noqa: E402
from run_standard_mvpa_baseline import dual_basis, fit_projected, standardize  # noqa: E402


def paired_bootstrap(
    reference: dict[str, float],
    comparison: dict[str, float],
    iterations: int,
    seed: int,
) -> dict:
    subjects = sorted(set(reference) & set(comparison))
    difference = np.asarray([reference[s] - comparison[s] for s in subjects])
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        difference, size=(iterations, len(difference)), replace=True
    ).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "subject_count": len(subjects),
        "reference_mean": float(np.mean([reference[s] for s in subjects])),
        "comparison_mean": float(np.mean([comparison[s] for s in subjects])),
        "mean_difference": float(difference.mean()),
        "ci95": [float(low), float(high)],
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "subject_wins": int((difference > 0).sum()),
        "subject_ties": int((difference == 0).sum()),
        "subject_losses": int((difference < 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired test of the smoothing kernel.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--model", default="linear_svm")
    parser.add_argument("--fixed-c", type=float, default=0.0001)
    parser.add_argument("--kernels", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--feature-shape", nargs=3, type=int, default=[24, 24, 24])
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    base = sequence.mean(axis=1, dtype=np.float32)
    shape = tuple(args.feature_shape)

    blocks = {
        k: base if k == 1 else mean_smooth(base, shape, k, args.batch_size)
        for k in args.kernels
    }
    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)

    # per_subject[kernel][rule][subject] -> list of fold accuracies
    per_subject = {
        k: {rule: defaultdict(list) for rule in ("independent", "balanced")}
        for k in args.kernels
    }

    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        grouped: dict[str, list[int]] = defaultdict(list)
        for local_pos, record_idx in enumerate(val_idx):
            grouped[str(records[int(record_idx)]["subject_id"])].append(local_pos)

        for kernel in args.kernels:
            block = blocks[kernel]
            mean, scale = standardize(block, train_idx)
            z_train, z_val = dual_basis(
                (block[train_idx] - mean) / scale, (block[val_idx] - mean) / scale
            )
            scores = fit_projected(
                args.model, z_train, y[train_idx], z_val,
                z_train @ z_train.T, z_val @ z_train.T, args.fixed_c, 0,
            )
            for rule, prediction in (
                ("independent", scores.argmax(axis=1).astype(np.int64)),
                ("balanced", apply_balanced_assignment(scores, val_idx, records)),
            ):
                for subject, positions in grouped.items():
                    per_subject[kernel][rule][subject].append(
                        float(metrics(y[val_idx][positions],
                                      prediction[positions])["balanced_accuracy"])
                    )
        print(f"{split['split']} done", flush=True)

    means = {
        k: {rule: {s: float(np.mean(v)) for s, v in per_subject[k][rule].items()}
            for rule in per_subject[k]}
        for k in args.kernels
    }

    comparisons = {}
    reference = args.kernels[-1]
    for kernel in args.kernels[:-1]:
        for rule in ("independent", "balanced"):
            comparisons[f"{rule}|kernel{reference}_vs_kernel{kernel}"] = paired_bootstrap(
                means[reference][rule], means[kernel][rule],
                args.bootstrap_iterations, args.bootstrap_seed,
            )

    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "model": args.model,
        "outer_split_count": len(splits),
        "kernels": args.kernels,
        "paired": comparisons,
        "per_subject_means": means,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))

    for name, entry in comparisons.items():
        print(f"\n{name}")
        print(f"  {entry['reference_mean']:.4f} vs {entry['comparison_mean']:.4f}")
        print(f"  difference {entry['mean_difference']:+.4f} "
              f"CI95 [{entry['ci95'][0]:+.4f}, {entry['ci95'][1]:+.4f}]  "
              f"excludes zero: {entry['excludes_zero']}")
        print(f"  wins/ties/losses {entry['subject_wins']}/"
              f"{entry['subject_ties']}/{entry['subject_losses']}")


if __name__ == "__main__":
    main()
