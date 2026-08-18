#!/usr/bin/env python3
"""What predicts a subject's decodability?

Per-subject accuracy ranges from near chance to near ceiling. The project has
case-by-case forensics on the weak subjects but no systematic model, which leaves the
QC-60 exclusion resting on procedural grounds alone: it was prespecified, therefore it
is not a fishing expedition. That defence is sound but thin.

If a subject-level measure computed **without labels from the held-out fold** predicts
accuracy, the exclusion becomes principled rather than merely prespecified, and the
manuscript gains a defensible criterion. If nothing predicts it, that is equally worth
stating plainly — it makes exclusion harder to justify and should not be hidden.

## Candidate predictors

All are computed per subject from the same preprocessed sequence the decoders see.

- **Split-half pattern reliability.** Runs are split into two halves, class-mean
  patterns computed in each, and the four patterns correlated across halves. This is
  the measure the plan singles out, and it is the closest thing to a data-quality
  index the checkpoints support.
- **Discriminability ratio.** Between-class centroid distance over within-class spread,
  a direct multivariate signal-to-noise proxy.
- **Temporal stability.** Mean correlation between consecutive lags, which falls when a
  subject's response is noisy or badly timed.
- **Residual scale.** Mean absolute value after preprocessing, catching subjects whose
  signal is unusually large or small.

Accuracy is the per-subject mean over the five outer folds in which that subject is
held out, so a subject's own data never contributes to the model predicting it.
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

from run_balanced_event_assignment import metrics  # noqa: E402
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


def correlate(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else 0.0


def subject_predictors(
    block: np.ndarray,
    sequence: np.ndarray,
    labels: np.ndarray,
    runs: np.ndarray,
    class_count: int,
) -> dict[str, float]:
    unique_runs = np.unique(runs)
    first, second = unique_runs[::2], unique_runs[1::2]
    half_a, half_b = np.isin(runs, first), np.isin(runs, second)

    reliabilities = []
    for class_idx in range(class_count):
        rows_a = block[half_a & (labels == class_idx)]
        rows_b = block[half_b & (labels == class_idx)]
        if len(rows_a) and len(rows_b):
            reliabilities.append(correlate(rows_a.mean(axis=0), rows_b.mean(axis=0)))
    reliability = float(np.mean(reliabilities)) if reliabilities else 0.0

    centroids = np.stack(
        [block[labels == c].mean(axis=0) for c in range(class_count)]
    )
    between = float(
        np.mean([
            np.linalg.norm(centroids[i] - centroids[j])
            for i in range(class_count) for j in range(i + 1, class_count)
        ])
    )
    within = float(
        np.mean([
            np.linalg.norm(block[labels == c] - centroids[c], axis=1).mean()
            for c in range(class_count)
        ])
    )

    lag_pairs = [
        correlate(sequence[:, lag].ravel(), sequence[:, lag + 1].ravel())
        for lag in range(sequence.shape[1] - 1)
    ]

    return {
        "split_half_reliability": reliability,
        "discriminability_ratio": between / within if within > 1e-12 else 0.0,
        "temporal_stability": float(np.mean(lag_pairs)),
        "residual_scale": float(np.mean(np.abs(block))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictors of subject decodability.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--model", default="linear_svm")
    parser.add_argument("--fixed-c", type=float, default=0.0001)
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--split-limit", type=int)
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    block = sequence.mean(axis=1, dtype=np.float32)
    class_count = int(y.max()) + 1

    by_subject: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_subject[str(record["subject_id"])].append(index)

    predictors = {}
    for subject, rows in sorted(by_subject.items()):
        idx = np.asarray(rows)
        predictors[subject] = subject_predictors(
            block[idx], sequence[idx], y[idx],
            np.asarray([records[int(i)]["run_id"] for i in idx]),
            class_count,
        )
    print(f"predictors computed for {len(predictors)} subjects", flush=True)

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    accuracies: dict[str, list[float]] = defaultdict(list)
    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        mean, scale = standardize(block, train_idx)
        x_all = ((block - mean) / scale).astype(np.float32)
        z_train, z_val = dual_basis(x_all[train_idx], x_all[val_idx])
        scores = fit_projected(
            args.model, z_train, y[train_idx], z_val,
            z_train @ z_train.T, z_val @ z_train.T, args.fixed_c, 0,
        )
        prediction = scores.argmax(axis=1).astype(np.int64)
        grouped: dict[str, list[int]] = defaultdict(list)
        for local_pos, record_idx in enumerate(val_idx):
            grouped[str(records[int(record_idx)]["subject_id"])].append(local_pos)
        for subject, positions in grouped.items():
            accuracies[subject].append(
                float(metrics(y[val_idx][positions],
                              prediction[positions])["balanced_accuracy"])
            )
        print(f"{split['split']} done", flush=True)

    subjects = sorted(set(accuracies) & set(predictors))
    target = np.asarray([float(np.mean(accuracies[s])) for s in subjects])
    names = ["split_half_reliability", "discriminability_ratio",
             "temporal_stability", "residual_scale"]
    design = np.stack([[predictors[s][n] for s in subjects] for n in names], axis=1)

    univariate = {
        name: correlate(design[:, i], target) for i, name in enumerate(names)
    }

    # Multiple regression on standardized predictors, so coefficients are comparable.
    centered = (design - design.mean(axis=0)) / np.where(
        design.std(axis=0) < 1e-12, 1.0, design.std(axis=0)
    )
    matrix = np.column_stack([np.ones(len(subjects)), centered])
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    fitted = matrix @ coefficients
    residual = target - fitted
    total = target - target.mean()
    r_squared = float(1.0 - (residual @ residual) / (total @ total))

    order = np.argsort(target)
    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "model": args.model,
        "subject_count": len(subjects),
        "outer_split_count": len(splits),
        "accuracy": {"mean": float(target.mean()), "sd": float(target.std()),
                     "min": float(target.min()), "max": float(target.max())},
        "univariate_correlation": univariate,
        "multiple_regression": {
            "r_squared": r_squared,
            "intercept": float(coefficients[0]),
            "standardized_coefficients": {
                name: float(coefficients[i + 1]) for i, name in enumerate(names)
            },
        },
        "per_subject": {
            s: {"accuracy": float(np.mean(accuracies[s])), **predictors[s]}
            for s in subjects
        },
        "weakest_subjects": [subjects[i] for i in order[:5]],
        "strongest_subjects": [subjects[i] for i in order[-5:]],
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))

    print(f"\naccuracy {target.mean():.4f} (sd {target.std():.4f}, "
          f"range {target.min():.4f}-{target.max():.4f})")
    print("\nunivariate correlation with accuracy")
    for name, value in sorted(univariate.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:<26} r = {value:+.4f}")
    print(f"\nmultiple regression R2 = {r_squared:.4f}")
    for name, value in payload["multiple_regression"]["standardized_coefficients"].items():
        print(f"  {name:<26} beta = {value:+.4f}")
    print(f"\nweakest:   {', '.join(payload['weakest_subjects'])}")
    print(f"strongest: {', '.join(payload['strongest_subjects'])}")


if __name__ == "__main__":
    main()
