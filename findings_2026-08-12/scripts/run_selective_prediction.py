#!/usr/bin/env python3
"""Accuracy as a function of coverage, when the decoder may abstain.

Every accuracy this project reports forces a prediction on every event. That is the
right default for a benchmark, but the frozen protocol also carries a **deployment
rule**, and a deployed decoder can decline to answer. If accuracy on the confident
70% of events is far above the all-events figure, that is a materially different
proposition from `0.83` on everything, and it is the number a brain-computer-interface
reader would actually want.

## Confidence measure

The margin between the top two class scores, per event. It needs no probability
calibration, it is invariant to the score scale, and it is the standard choice for
selective prediction with a discriminative model.

## What is and is not fitted

Nothing here is fitted. The margin is computed from the same held-out scores the
accuracy is computed from, and coverage is swept post hoc. This is a **descriptive**
curve of how accuracy and abstention trade off, not a tuned rejection policy — a
deployed threshold would have to be selected on training data, and its held-out
behaviour would be slightly worse than this curve suggests. That caveat belongs with
any figure drawn from this.

Reported for both prediction rules. The balanced rule assigns whole runs jointly, so
abstention there means declining individual events from an otherwise complete
assignment, which is a coherent but different operation from declining to predict at
all.
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

from run_balanced_event_assignment import apply_balanced_assignment  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy-coverage curve.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--model", default="linear_svm")
    parser.add_argument("--fixed-c", type=float, default=0.0001)
    parser.add_argument("--coverages", nargs="+", type=float,
                        default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    x = sequence.mean(axis=1, dtype=np.float32)
    del sequence

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)

    pooled = {"independent": [], "balanced": []}
    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        mean, scale = standardize(x, train_idx)
        z_train, z_val = dual_basis(
            (x[train_idx] - mean) / scale, (x[val_idx] - mean) / scale
        )
        scores = fit_projected(
            args.model, z_train, y[train_idx], z_val,
            z_train @ z_train.T, z_val @ z_train.T, args.fixed_c, 0,
        )
        ordered = np.sort(scores, axis=1)
        margin = ordered[:, -1] - ordered[:, -2]
        truth = y[val_idx]
        for rule, prediction in (
            ("independent", scores.argmax(axis=1).astype(np.int64)),
            ("balanced", apply_balanced_assignment(scores, val_idx, records)),
        ):
            pooled[rule].append(
                np.stack([margin, (prediction == truth).astype(float)], axis=1)
            )
        print(f"{split['split']} done", flush=True)

    curves = {}
    for rule, blocks in pooled.items():
        table = np.concatenate(blocks, axis=0)
        order = np.argsort(-table[:, 0])  # most confident first
        correct = table[order, 1]
        points = []
        for coverage in args.coverages:
            keep = max(int(round(coverage * len(correct))), 1)
            points.append(
                {
                    "coverage": coverage,
                    "events_kept": keep,
                    "accuracy": float(correct[:keep].mean()),
                }
            )
        curves[rule] = points

    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "model": args.model,
        "confidence": "margin between the top two class scores",
        "outer_split_count": len(splits),
        "total_events": int(sum(len(b) for b in pooled["independent"])),
        "curves": curves,
        "caveat": (
            "Descriptive, not a tuned policy. The threshold is swept post hoc on the "
            "same held-out scores the accuracy is computed from; a deployed threshold "
            "chosen on training data would perform slightly worse."
        ),
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"\n{'coverage':>9s} {'independent':>12s} {'balanced':>10s}")
    for a, b in zip(curves["independent"], curves["balanced"]):
        print(f"{a['coverage']:9.2f} {a['accuracy']:12.4f} {b['accuracy']:10.4f}")


if __name__ == "__main__":
    main()
