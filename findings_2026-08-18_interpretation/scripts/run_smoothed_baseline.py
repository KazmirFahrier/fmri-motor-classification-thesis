#!/usr/bin/env python3
"""Conventional MVPA baseline given the same spatial preprocessing as the hierarchy.

The frozen hierarchy's pair specialists use `smooth_3`; the repository records it as
"the validated spatial choice" and instructs keeping it for both. The conventional-MVPA
baselines this project compares against were never given it, so the reported advantage
of `+0.026` conflates the decoder with a preprocessing step only one side received.

This emits results in the same `rows` schema as `run_standard_mvpa_baseline.py`, so
`compare_frozen_vs_standard_mvpa.py` can run the identical paired subject-level
bootstrap against the frozen decoder's own `selected_rows`. That makes the comparison
paired on identical folds and subjects rather than a difference of two separately-run
point estimates.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoothed conventional MVPA baseline.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--models", nargs="+", default=["linear_svm", "logistic_l2"])
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--fixed-c", type=float, default=0.0001)
    parser.add_argument("--feature-shape", nargs=3, type=int, default=[24, 24, 24])
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split-limit", type=int)
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    block = sequence.mean(axis=1, dtype=np.float32)
    if args.kernel > 1:
        block = mean_smooth(block, tuple(args.feature_shape), args.kernel, args.batch_size)
        print(f"applied smooth_{args.kernel}", flush=True)

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    rows = []
    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        mean, scale = standardize(block, train_idx)
        z_train, z_val = dual_basis(
            (block[train_idx] - mean) / scale, (block[val_idx] - mean) / scale
        )
        kernel_train, kernel_val = z_train @ z_train.T, z_val @ z_train.T

        grouped: dict[str, list[int]] = defaultdict(list)
        for local_pos, record_idx in enumerate(val_idx):
            grouped[str(records[int(record_idx)]["subject_id"])].append(local_pos)

        for model in args.models:
            scores = fit_projected(
                model, z_train, y[train_idx], z_val,
                kernel_train, kernel_val, args.fixed_c, 0,
            )
            for rule, prediction in (
                ("independent", scores.argmax(axis=1).astype(np.int64)),
                ("balanced", apply_balanced_assignment(scores, val_idx, records)),
            ):
                rows.append({
                    "model": model,
                    "split": split["split"],
                    "prediction_rule": rule,
                    "metrics": metrics(y[val_idx], prediction),
                    "subject_metrics": {
                        subject: metrics(y[val_idx][positions], prediction[positions])
                        for subject, positions in sorted(grouped.items())
                    },
                })
        print(f"{split['split']} done", flush=True)

    summary = {}
    for model in args.models:
        for rule in ("independent", "balanced"):
            values = [
                r["metrics"]["balanced_accuracy"] for r in rows
                if r["model"] == model and r["prediction_rule"] == rule
            ]
            summary[f"{model}|{rule}"] = float(np.mean(values))

    Path(args.out_json).write_text(json.dumps({
        "checkpoint_dir": args.checkpoint_dir,
        "preprocess": f"frozen + smooth_{args.kernel}",
        "smoothing_kernel": args.kernel,
        "fixed_c": args.fixed_c,
        "outer_split_count": len(splits),
        "summary": summary,
        "rows": rows,
    }, indent=2))
    for name, value in summary.items():
        print(f"  {name:<32} {value:.4f}")


if __name__ == "__main__":
    main()
