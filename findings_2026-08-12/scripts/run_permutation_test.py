#!/usr/bin/env python3
"""Label-shuffled permutation null for the decoders under the frozen protocol.

Every accuracy reported by this project so far is stated without a null
distribution. That is a gap a Q1 venue will flag, and it matters more here than
usual: a large share of the headline accuracy traces to an unlabeled subject-run
centering step, so a reader is entitled to ask whether the remaining signal is a
genuine label association or an artifact of the design and the preprocessing.

## Why this is cheap

Every feature-side operation in the pipeline is **label-free**:

- `center_by_subject_run` uses only subject and run identifiers and the feature
  values.
- Per-lag linear detrending uses event time.
- Per-feature standardisation uses training rows.
- The dual basis and the linear kernels are functions of the features alone.

None of them changes when labels are permuted, so all of it is computed once per
fold and reused across every permutation. Only the classifier is refitted. This is
also why permuting labels is a *valid* null here rather than a leaky one: nothing
upstream of the classifier can smuggle label information into the features.

## Exchangeability

Labels are shuffled **within each subject-run**. This preserves the exact
two-events-per-class composition of every run, which the balanced assignment rule
depends on, and it preserves subject and run structure. The null therefore asks
the sharpest available question: beyond run-level structure that the design
guarantees, is there a real association between spatial pattern and class?

Shuffling globally would break run composition, make the balanced rule inapplicable,
and produce an easier and less honest null.

## Regularisation under the null

Refitting the full nested inner selection inside every permutation would multiply
cost by the size of the `C` grid times the inner fold count. Instead `C` is fixed at
the value the unpermuted analysis selected most often, and that value is recorded in
the output. This is the conventional compromise and it is stated as a limitation
rather than hidden.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
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


def shuffle_within_run(
    y: np.ndarray,
    run_positions: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute labels inside each subject-run, preserving class composition."""
    permuted = y.copy()
    for positions in run_positions:
        permuted[positions] = rng.permutation(y[positions])
    return permuted


def build_run_positions(records: list[dict]) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[f'{record["subject_id"]}|run-{int(record["run_id"])}'].append(index)
    return [np.asarray(v, dtype=np.int64) for _, v in sorted(grouped.items())]


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    val_idx: np.ndarray,
    records: list[dict],
) -> dict[str, float]:
    return {
        "independent": metrics(
            labels[val_idx], scores.argmax(axis=1).astype(np.int64)
        )["balanced_accuracy"],
        "balanced": metrics(
            labels[val_idx], apply_balanced_assignment(scores, val_idx, records)
        )["balanced_accuracy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Within-run label-shuffled permutation null under the frozen protocol."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--models", nargs="+",
                        default=["linear_svm", "logistic_l2", "correlation_centroid"])
    parser.add_argument("--fixed-c", nargs="+", type=float, default=[0.0001, 0.01, 0.0],
                        help="Per model, in --models order. Ignored for correlation_centroid.")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
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

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    # Every one of these is label-free, so it is computed once and reused for all
    # permutations. This is the entire reason the test is affordable.
    print(f"precomputing label-free structure for {len(splits)} folds", flush=True)
    started = time.time()
    prepared = []
    for split in splits:
        mean, scale = standardize(mean_x, split["train_idx"])
        z_train, z_val = dual_basis(
            (mean_x[split["train_idx"]] - mean) / scale,
            (mean_x[split["val_idx"]] - mean) / scale,
        )
        prepared.append(
            {
                "split": split,
                "z_train": z_train,
                "z_val": z_val,
                "kernel_train": z_train @ z_train.T,
                "kernel_val": z_val @ z_train.T,
            }
        )
    print(f"  done in {time.time() - started:.0f}s", flush=True)

    run_positions = build_run_positions(records)
    rng = np.random.default_rng(args.seed)

    def run_pass(labels: np.ndarray) -> dict[str, dict[str, float]]:
        totals: dict[str, dict[str, list[float]]] = {
            model: {"independent": [], "balanced": []} for model in args.models
        }
        for item in prepared:
            split = item["split"]
            train_idx, val_idx = split["train_idx"], split["val_idx"]
            for model in args.models:
                if model == "correlation_centroid":
                    scores = correlation_centroid_scores(
                        mean_x, labels, train_idx, val_idx, class_count
                    )
                else:
                    scores = fit_projected(
                        model,
                        item["z_train"],
                        labels[train_idx],
                        item["z_val"],
                        item["kernel_train"],
                        item["kernel_val"],
                        fixed_c[model],
                        0,
                    )
                for rule, value in evaluate(scores, labels, val_idx, records).items():
                    totals[model][rule].append(value)
        return {
            model: {rule: float(np.mean(values)) for rule, values in rules.items()}
            for model, rules in totals.items()
        }

    print("observed pass", flush=True)
    observed = run_pass(y)
    print(json.dumps(observed, indent=2), flush=True)

    null_draws: dict[str, dict[str, list[float]]] = {
        model: {"independent": [], "balanced": []} for model in args.models
    }
    started = time.time()
    for index in range(args.permutations):
        permuted = shuffle_within_run(y, run_positions, rng)
        result = run_pass(permuted)
        for model, rules in result.items():
            for rule, value in rules.items():
                null_draws[model][rule].append(value)
        if (index + 1) % 25 == 0:
            elapsed = time.time() - started
            print(
                f"  permutation {index + 1}/{args.permutations} "
                f"[{elapsed:.0f}s, {elapsed / (index + 1):.2f}s each]",
                flush=True,
            )

    summary = {}
    for model in args.models:
        for rule in ("independent", "balanced"):
            draws = np.asarray(null_draws[model][rule])
            observed_value = observed[model][rule]
            # Add-one estimator: p is never reported as exactly zero.
            exceedances = int(np.sum(draws >= observed_value))
            p_value = (exceedances + 1) / (len(draws) + 1)
            summary[f"{model}|{rule}"] = {
                "observed": observed_value,
                "null_mean": float(draws.mean()),
                "null_std": float(draws.std()),
                "null_q95": float(np.quantile(draws, 0.95)),
                "null_max": float(draws.max()),
                "exceedances": exceedances,
                "p_value": p_value,
                "p_value_is_upper_bound": exceedances == 0,
                "z_against_null": float(
                    (observed_value - draws.mean()) / max(draws.std(), 1e-12)
                ),
            }

    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "permutations": args.permutations,
        "outer_split_count": len(splits),
        "exchangeability": "labels shuffled within each subject-run, preserving two events per class",
        "fixed_c": fixed_c,
        "label_free_precomputation": (
            "Centering, detrending, standardisation, the dual basis, and the kernels "
            "depend only on features, subject/run identifiers, and event time. They are "
            "invariant under label permutation and are computed once per fold."
        ),
        "limitation": (
            "C is fixed at the value most often selected by the unpermuted nested "
            "analysis rather than reselected inside each permutation, which would "
            "multiply cost by the grid size times the inner fold count."
        ),
        "observed": observed,
        "summary": summary,
        "null_draws": {
            model: {rule: values for rule, values in rules.items()}
            for model, rules in null_draws.items()
        },
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"\n{'model|rule':38s} {'observed':>9s} {'null mean':>10s} {'null q95':>9s} {'p':>9s} {'z':>7s}")
    for name, row in summary.items():
        print(
            f"{name:38s} {row['observed']:9.4f} {row['null_mean']:10.4f} "
            f"{row['null_q95']:9.4f} {row['p_value']:9.5f} {row['z_against_null']:7.1f}"
        )


if __name__ == "__main__":
    main()
