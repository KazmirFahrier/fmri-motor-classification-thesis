#!/usr/bin/env python3
"""Does "preprocessing dominates the decoder" hold at twelve classes?

The project's most robust finding is that preprocessing matters far more than decoder
choice. Four-class numbers: unlabeled subject-run centering is worth `+0.52`, per-lag
detrending `+0.034`, and `smooth_3` `+0.0177` — while the frozen hierarchy's advantage
over a preprocessing-matched linear SVM is `+0.0040` and does not exclude zero.

That claim now carries the paper, so it should not rest on one class count. Four classes
is a small, unusually confusable subset — three of the frozen four are among the four
worst-decoded conditions in the set — and a decomposition measured only there could be
a property of that subset rather than of the data.

This runs the identical decomposition at twelve classes. Each stage adds one label-free
step, so the increments are directly comparable with the four-class table.
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

from run_balanced_event_assignment import center_by_subject_run  # noqa: E402
from run_detrended_pair_feature_selection import (  # noqa: E402
    load_checkpoints,
    outer_splits,
)
from run_spatial_scale_feature_sweep import mean_smooth  # noqa: E402
from run_standard_mvpa_baseline import standardize, subject_zscore  # noqa: E402
from run_temporal_detrended_event_adaptation import (  # noqa: E402
    temporal_detrend_by_subject_run,
)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> float:
    recalls = []
    for class_idx in range(class_count):
        mask = y_true == class_idx
        if int(mask.sum()):
            recalls.append(float((y_pred[mask] == class_idx).sum()) / int(mask.sum()))
    return float(np.mean(recalls)) if recalls else 0.0


def build_stage(
    raw: np.ndarray,
    records: list[dict],
    stage: str,
    shape: tuple,
    batch_size: int,
) -> np.ndarray:
    """Return the event x feature block under one preprocessing stage.

    Every step here is label-free, which is what makes the whole decomposition a
    statement about preprocessing rather than about leakage.

    At twelve classes ``raw`` is 8928 x 8 x 13824, just under 4 GB. Copying it per stage
    doubles that and drives the machine into swap, so the mean over lags is accumulated
    one lag at a time and ``raw`` is never duplicated.
    """
    if stage == "none":
        return raw.mean(axis=1, dtype=np.float32)
    if stage == "subject_centering":
        return subject_zscore(raw.mean(axis=1, dtype=np.float32), records)

    detrend = stage in ("run_centering_detrend", "run_centering_detrend_smooth")
    block = np.zeros(raw.shape[::2], dtype=np.float32)
    for lag in range(raw.shape[1]):
        lag_block = center_by_subject_run(np.ascontiguousarray(raw[:, lag]), records)
        if detrend:
            lag_block, _ = temporal_detrend_by_subject_run(lag_block, records, degree=1)
        block += lag_block
        del lag_block
    block /= float(raw.shape[1])

    if stage == "run_centering_detrend_smooth":
        block = mean_smooth(block, shape, 3, batch_size)
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description="Twelve-class preprocessing decomposition.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
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
    raw = feature_dict.pop(args.sequence_key)
    class_count = int(y.max()) + 1
    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    stages = ["none", "subject_centering", "run_centering",
              "run_centering_detrend", "run_centering_detrend_smooth"]
    results = {}
    for stage in stages:
        block = build_stage(raw, records, stage, tuple(args.feature_shape), args.batch_size)
        values = []
        for split in splits:
            train_idx, val_idx = split["train_idx"], split["val_idx"]
            mean, scale = standardize(block, train_idx)
            x_train = ((block[train_idx] - mean) / scale).astype(np.float64)
            x_val = ((block[val_idx] - mean) / scale).astype(np.float64)
            model = SVC(C=args.fixed_c, kernel="precomputed",
                        decision_function_shape="ovr", random_state=0)
            model.fit(x_train @ x_train.T, y[train_idx])
            values.append(balanced_accuracy(
                y[val_idx], model.predict(x_val @ x_train.T).astype(np.int64), class_count
            ))
        results[stage] = {"mean": float(np.mean(values)), "sd": float(np.std(values))}
        print(f"{stage:<34} {np.mean(values):.4f}", flush=True)
        del block

    increments = {}
    for previous, stage in zip(stages[1:], stages[2:]):
        increments[f"{previous} -> {stage}"] = (
            results[stage]["mean"] - results[previous]["mean"]
        )
    increments["none -> subject_centering"] = (
        results["subject_centering"]["mean"] - results["none"]["mean"]
    )

    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "class_count": class_count,
        "chance": 1.0 / class_count,
        "outer_split_count": len(splits),
        "stages": results,
        "increments": increments,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print("\nincrements")
    for name, value in increments.items():
        print(f"  {name:<48} {value:+.4f}")


if __name__ == "__main__":
    main()
