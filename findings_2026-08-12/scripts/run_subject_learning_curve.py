#!/usr/bin/env python3
"""Accuracy as a function of the number of training subjects.

The project reports a single estimate at the full cohort size and has never asked
whether that estimate is near a ceiling. The answer changes what the remaining work
should be:

- If the curve is **still rising** at the largest available training size, sample
  size is a live constraint. More subjects would help, the current number
  understates what the method can reach, and reopening the external-cohort search
  is worthwhile.
- If the curve is **flat**, more subjects will not help and the bottleneck is the
  representation or the cross-subject alignment. Effort should go to normalization
  and functional alignment instead.

Two axes are supported, and they answer different questions.

`--vary subjects` subsamples the training subjects. `--vary runs` keeps every
training subject and subsamples the runs each contributes. The second is the
practically actionable axis: collecting more runs from subjects already enrolled is
far cheaper than recruiting new ones, so if the run axis is still climbing while the
subject axis has saturated, the cheap intervention is also the effective one.

## Design

For each outer fold the held-out subjects are held **fixed**, and only the training
side varies, so every point on a curve is evaluated against the same target. Several
independent draws are taken at each size to separate the trend from draw-to-draw
noise. On the run axis, runs are drawn independently within each subject so the curve
is not confounded with any particular run ordering.

Everything leakage-sensitive is recomputed inside each draw: standardisation, the
dual basis, and the classifier are all fitted on that draw's subjects alone.
Regularisation is fixed at the value the full nested analysis selected most often,
since re-running inner selection inside every draw would multiply cost by the grid
size and would itself vary with training size in a way that confounds the curve.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (REPO_ROOT / "scripts", REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from run_balanced_event_assignment import apply_balanced_assignment, metrics  # noqa: E402
from run_detrended_pair_feature_selection import (  # noqa: E402
    load_checkpoints,
    outer_splits,
)
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402
from run_standard_mvpa_baseline import (  # noqa: E402
    correlation_centroid_scores,
    dual_basis,
    fit_projected,
    standardize,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learning curve over the number of training subjects."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--models", nargs="+",
                        default=["linear_svm", "logistic_l2", "correlation_centroid"])
    parser.add_argument("--fixed-c", nargs="+", type=float, default=[0.0001, 0.01, 0.0])
    parser.add_argument(
        "--vary",
        choices=["subjects", "runs"],
        default="subjects",
        help=(
            "subjects: subsample training subjects, all six runs each. "
            "runs: keep every training subject and subsample their runs. The second "
            "axis is the practically actionable one, since collecting more runs from "
            "enrolled subjects is far cheaper than recruiting more subjects."
        ),
    )
    parser.add_argument("--subject-counts", nargs="+", type=int,
                        default=[6, 10, 16, 24, 32, 40, 51])
    parser.add_argument("--run-counts", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--split-limit", type=int)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    fixed_c = dict(zip(args.models, args.fixed_c))

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    mean_x = sequence.mean(axis=1, dtype=np.float32)
    del sequence
    class_count = int(y.max()) + 1
    subjects = np.asarray([str(record["subject_id"]) for record in records])

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    started = time.time()

    runs = np.asarray([int(record["run_id"]) for record in records])
    for split in splits:
        train_subjects = sorted(set(subjects[split["train_idx"]].tolist()))
        val_idx = split["val_idx"]
        if args.vary == "subjects":
            available = len(train_subjects)
            counts = args.subject_counts
        else:
            available = len(set(runs[split["train_idx"]].tolist()))
            counts = args.run_counts
        for count in counts:
            if count > available:
                continue
            draw_count = 1 if count == available else args.draws
            for draw in range(draw_count):
                if args.vary == "subjects":
                    chosen = rng.choice(train_subjects, size=count, replace=False)
                    train_idx = split["train_idx"][
                        np.isin(subjects[split["train_idx"]], chosen)
                    ]
                else:
                    # Every training subject is kept; each contributes a random
                    # subset of their own runs, drawn independently per subject so
                    # the curve is not confounded with a particular run ordering.
                    keep = np.zeros(len(split["train_idx"]), dtype=bool)
                    block_subjects = subjects[split["train_idx"]]
                    block_runs = runs[split["train_idx"]]
                    for subject in train_subjects:
                        member = block_subjects == subject
                        subject_runs = sorted(set(block_runs[member].tolist()))
                        picked = rng.choice(
                            subject_runs, size=min(count, len(subject_runs)), replace=False
                        )
                        keep |= member & np.isin(block_runs, picked)
                    train_idx = split["train_idx"][keep]

                mean, scale = standardize(mean_x, train_idx)
                z_train, z_val = dual_basis(
                    (mean_x[train_idx] - mean) / scale,
                    (mean_x[val_idx] - mean) / scale,
                )
                kernel_train = z_train @ z_train.T
                kernel_val = z_val @ z_train.T

                for model in args.models:
                    if model == "correlation_centroid":
                        scores = correlation_centroid_scores(
                            mean_x, y, train_idx, val_idx, class_count
                        )
                    else:
                        scores = fit_projected(
                            model, z_train, y[train_idx], z_val,
                            kernel_train, kernel_val, fixed_c[model], 0,
                        )
                    for rule, prediction in (
                        ("independent", scores.argmax(axis=1).astype(np.int64)),
                        ("balanced", apply_balanced_assignment(scores, val_idx, records)),
                    ):
                        rows.append(
                            {
                                "split": split["split"],
                                "varied_count": int(count),
                                "vary_axis": args.vary,
                                "draw": int(draw),
                                "model": model,
                                "prediction_rule": rule,
                                "train_event_count": int(len(train_idx)),
                                "balanced_accuracy": float(
                                    metrics(y[val_idx], prediction)["balanced_accuracy"]
                                ),
                            }
                        )
            print(
                f"{split['split']} {args.vary}={count} done [{time.time() - started:.0f}s]",
                flush=True,
            )

    curve: dict[str, list[dict]] = defaultdict(list)
    for model in args.models:
        for rule in ("independent", "balanced"):
            for count in sorted({r["varied_count"] for r in rows}):
                values = [
                    r["balanced_accuracy"] for r in rows
                    if r["model"] == model and r["prediction_rule"] == rule
                    and r["varied_count"] == count
                ]
                if values:
                    curve[f"{model}|{rule}"].append(
                        {
                            "varied_count": count,
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "n_estimates": len(values),
                        }
                    )

    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "vary_axis": args.vary,
        "subject_counts": args.subject_counts,
        "run_counts": args.run_counts,
        "draws_per_count": args.draws,
        "outer_split_count": len(splits),
        "subject_seeds": args.subject_seeds,
        "fixed_c": fixed_c,
        "design": (
            "Held-out subjects are fixed per fold; only the training subject set is "
            "subsampled. Standardisation, the dual basis, and the classifier are "
            "refitted inside every draw."
        ),
        "limitation": (
            "C is fixed rather than reselected per draw, because inner selection "
            "would itself vary with training size and confound the curve."
        ),
        "curve": dict(curve),
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))

    axis_counts = args.subject_counts if args.vary == "subjects" else args.run_counts
    print(f"\n{'model|rule':34s} " + " ".join(f"{c:>7d}" for c in axis_counts))
    for name, points in curve.items():
        by_count = {p["varied_count"]: p["mean"] for p in points}
        cells = " ".join(
            f"{by_count[c]:7.4f}" if c in by_count else "      -"
            for c in axis_counts
        )
        print(f"{name:34s} {cells}")


if __name__ == "__main__":
    main()
