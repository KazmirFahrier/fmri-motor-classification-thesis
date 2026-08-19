#!/usr/bin/env python3
"""Temporal generalization matrix: is the motor code stationary across the response?

Every analysis in this project so far collapses the eight lags by averaging, which
presumes the code is stationary. That presumption has never been tested. The standard
design (King & Dehaene, 2014) trains at lag *i*, tests at lag *j*, and reads the
structure of the resulting matrix: broad off-diagonal generalization means one stable
pattern, a narrow diagonal band means the pattern evolves.

This is **not** a repeat of the temporal-basis investigation, which asked whether
*weighting* the lags helps and found it did not. That is a question about the optimal
readout. This asks whether the underlying code is the same thing at every lag, which
is what determines whether averaging is discarding structure or exploiting redundancy.

## Why it is cheap

For a linear SVM the precomputed kernel is exactly the inner-product matrix, so
``kernel_train = X_i X_i^T`` and ``kernel_val = X_j X_i^T`` — no dual basis or SVD is
needed at all. One fit per training lag serves all eight test lags, so the 64-cell
matrix costs eight fits per fold rather than sixty-four.

Standardization comes from the **training lag**, applied unchanged to the test lag, so
the classifier is genuinely transported across time rather than refitted.
"""
from __future__ import annotations

import argparse
import json
import sys
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

from run_balanced_event_assignment import apply_balanced_assignment, metrics  # noqa: E402
from run_detrended_pair_feature_selection import (  # noqa: E402
    inner_splits,
    load_checkpoints,
    outer_splits,
)
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402
from run_standard_mvpa_baseline import standardize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal generalization matrix.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--fixed-c", type=float, default=0.0001)
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--split-limit", type=int)
    parser.add_argument(
        "--per-lag-standardization", action="store_true",
        help="Standardize the test lag with its own training-subject statistics rather "
             "than carrying the training lag's. Across lags with very different signal "
             "amplitude, reusing the training lag's mean leaves a systematic offset in "
             "the test features, which shifts every decision value and can drive "
             "transfer below chance without the underlying code having changed. "
             "Estimated on training subjects only, so no leakage is introduced.",
    )
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    lag_count = sequence.shape[1]
    print(f"{sequence.shape[0]} events, {lag_count} lags", flush=True)

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    # matrix[rule][train_lag][test_lag] accumulates per-fold accuracies
    matrix = {
        rule: [[[] for _ in range(lag_count)] for _ in range(lag_count)]
        for rule in ("independent", "balanced")
    }

    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        for train_lag in range(lag_count):
            block_train_source = sequence[:, train_lag, :]
            mean, scale = standardize(block_train_source, train_idx)
            x_train = ((block_train_source[train_idx] - mean) / scale).astype(np.float64)

            kernel_train = x_train @ x_train.T
            model = SVC(
                C=args.fixed_c,
                kernel="precomputed",
                decision_function_shape="ovr",
                random_state=0,
            )
            model.fit(kernel_train, y[train_idx])

            for test_lag in range(lag_count):
                # By default the training lag's standardization is carried over, so the
                # decoder is transported unchanged. With --per-lag-standardization the
                # test lag is instead standardized by its own training-subject
                # statistics, which removes the cross-phase offset described above while
                # still transporting the fitted weights.
                if args.per_lag_standardization:
                    test_mean, test_scale = standardize(
                        sequence[:, test_lag, :], train_idx
                    )
                else:
                    test_mean, test_scale = mean, scale
                x_val = (
                    (sequence[:, test_lag, :][val_idx] - test_mean) / test_scale
                ).astype(np.float64)
                scores = model.decision_function(x_val @ x_train.T).astype(np.float64)
                for rule, prediction in (
                    ("independent", scores.argmax(axis=1).astype(np.int64)),
                    ("balanced", apply_balanced_assignment(scores, val_idx, records)),
                ):
                    matrix[rule][train_lag][test_lag].append(
                        float(metrics(y[val_idx], prediction)["balanced_accuracy"])
                    )
        print(f"{split['split']} done", flush=True)

    summary = {
        rule: {
            "mean": [[float(np.mean(cell)) for cell in row] for row in matrix[rule]],
            "sd": [[float(np.std(cell)) for cell in row] for row in matrix[rule]],
        }
        for rule in matrix
    }

    mean_indep = np.asarray(summary["independent"]["mean"])
    diagonal = float(np.mean(np.diag(mean_indep)))
    off = float(np.mean(mean_indep[~np.eye(lag_count, dtype=bool)]))
    # A stable code generalizes off-diagonal as well as on; a dynamic one does not.
    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "outer_split_count": len(splits),
        "lag_count": lag_count,
        "fixed_c": args.fixed_c,
        "per_lag_standardization": bool(args.per_lag_standardization),
        "matrix": summary,
        "diagonal_mean": diagonal,
        "off_diagonal_mean": off,
        "stationarity_ratio": off / diagonal if diagonal else None,
        "best_train_lag": int(np.argmax(mean_indep.mean(axis=1))),
        "best_test_lag": int(np.argmax(mean_indep.mean(axis=0))),
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))

    print("\nindependent-rule matrix (rows = train lag, cols = test lag)")
    print("      " + "".join(f"  t{j}   " for j in range(lag_count)))
    for i, row in enumerate(summary["independent"]["mean"]):
        print(f"  L{i}  " + "".join(f"{v:.3f} " for v in row))
    print(f"\ndiagonal      {diagonal:.4f}")
    print(f"off-diagonal  {off:.4f}")
    print(f"ratio         {off / diagonal:.4f}")


if __name__ == "__main__":
    main()
