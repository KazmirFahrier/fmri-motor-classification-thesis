#!/usr/bin/env python3
"""Within-run permutation null for twelve-class decoding.

The twelve-class result is quoted against an analytic chance of `1/12`. This project's
standing practice is to measure the null rather than assume it — the four-class work
established that the transductive preprocessing is worth `+0.52`, which is exactly the
kind of mechanism that could inflate an apparent effect without any class information,
and the only way to rule that out is to hold the preprocessing fixed and vary the
labels.

Labels are shuffled **within each subject-run**, preserving the two-per-class
composition that the design guarantees for all twelve conditions. The preprocessing is
label-free and so is identical under permutation.

## Cost, and how it is contained

Twelve classes means one-versus-one fits 66 binary problems per model, so a naive null
would be prohibitive. The kernel `X X^T` does not depend on the labels, so it is built
**once per split** and every permutation for that split reuses it; only the SVM fit
repeats. Splits are the outer loop and permutations the inner one, which also bounds
memory to a single split's kernels — caching all thirty would need roughly 16 GB.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "findings_2026-08-12" / "scripts",
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from run_detrended_pair_feature_selection import (  # noqa: E402
    load_checkpoints,
    outer_splits,
)
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402
from run_permutation_test import build_run_positions, shuffle_within_run  # noqa: E402
from run_standard_mvpa_baseline import standardize  # noqa: E402


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> float:
    recalls = []
    for class_idx in range(class_count):
        mask = y_true == class_idx
        if int(mask.sum()):
            recalls.append(float((y_pred[mask] == class_idx).sum()) / int(mask.sum()))
    return float(np.mean(recalls)) if recalls else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Twelve-class permutation null.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--fixed-c", type=float, default=0.0001)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--permutation-seed", type=int, default=29)
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--split-limit", type=int)
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    block = sequence.mean(axis=1, dtype=np.float32)
    del sequence
    class_count = int(y.max()) + 1

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    run_positions = build_run_positions(records)
    rng = np.random.default_rng(args.permutation_seed)
    # Draw every permutation up front so each split scores the same label sets.
    permuted_labels = [
        shuffle_within_run(y, run_positions, rng) for _ in range(args.permutations)
    ]

    observed_per_split = []
    null_per_split = np.zeros((len(splits), args.permutations), dtype=np.float64)

    for split_index, split in enumerate(splits):
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        mean, scale = standardize(block, train_idx)
        x_train = ((block[train_idx] - mean) / scale).astype(np.float64)
        x_val = ((block[val_idx] - mean) / scale).astype(np.float64)
        kernel_train = x_train @ x_train.T
        kernel_val = x_val @ x_train.T
        del x_train, x_val

        def fit_score(labels: np.ndarray) -> float:
            model = SVC(C=args.fixed_c, kernel="precomputed",
                        decision_function_shape="ovr", random_state=0)
            model.fit(kernel_train, labels[train_idx])
            return balanced_accuracy(
                labels[val_idx], model.predict(kernel_val).astype(np.int64), class_count
            )

        observed_per_split.append(fit_score(y))
        for perm_index, labels in enumerate(permuted_labels):
            null_per_split[split_index, perm_index] = fit_score(labels)
        print(f"{split['split']} observed {observed_per_split[-1]:.4f} "
              f"null {null_per_split[split_index].mean():.4f}", flush=True)
        del kernel_train, kernel_val

    observed = float(np.mean(observed_per_split))
    draws = null_per_split.mean(axis=0)
    p_value = float((np.sum(draws >= observed) + 1) / (len(draws) + 1))
    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "class_count": class_count,
        "analytic_chance": 1.0 / class_count,
        "outer_split_count": len(splits),
        "permutations": args.permutations,
        "observed": observed,
        "null_mean": float(draws.mean()),
        "null_sd": float(draws.std(ddof=1)),
        "null_min": float(draws.min()),
        "null_max": float(draws.max()),
        "p_value": p_value,
        "z": float((observed - draws.mean()) / draws.std(ddof=1)),
        "null_draws": draws.tolist(),
        "observed_per_split": observed_per_split,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"\nobserved        {observed:.4f}")
    print(f"analytic chance {1/class_count:.4f}")
    print(f"empirical null  {draws.mean():.4f} (sd {draws.std(ddof=1):.4f}, "
          f"max {draws.max():.4f})")
    print(f"z {payload['z']:.1f}   p {p_value:.4f}")


if __name__ == "__main__":
    main()
