#!/usr/bin/env python3
"""Score-level ensemble across decoders and representations.

The paired comparison showed the frozen hierarchy beats the linear SVM on 40 of 62
subjects and **loses on 21**, with the corrected CNN close behind both. Decoders that
disagree on a third of subjects may have partly decorrelated errors, and combining
them is the cheapest remaining route to a better headline number.

Feature-level concatenation of volumetric and surface features was already tried and
gave exactly nothing (`+0.000006`). Score-level combination is a different operation:
concatenation asks one regularised model to weigh 34308 correlated features at once,
whereas ensembling lets each decoder fit its own representation and only then combines
their opinions. The first result does not predict the second.

## Members

Within each fold, on each representation supplied, three decoders are fitted and
their validation score matrices retained. Scores are z-scored per fold and per
decoder before combination so that a decoder with a wider score range cannot dominate
by scale alone.

## Weighting

Two schemes, both honest about selection:

- **uniform** averages the z-scored score matrices, with no fitting at all.
- **inner-selected** picks weights on a coarse simplex using the inner subject folds
  of the current outer fold, so weight selection never sees held-out subjects.

The uniform result is reported alongside the selected one because a selected
ensemble that fails to beat uniform averaging has not earned its extra complexity.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (REPO_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from run_balanced_event_assignment import apply_balanced_assignment, metrics  # noqa: E402
from run_detrended_pair_feature_selection import (  # noqa: E402
    inner_splits,
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


def zscore_scores(scores: np.ndarray) -> np.ndarray:
    centred = scores - scores.mean(axis=1, keepdims=True)
    scale = centred.std(axis=1, keepdims=True)
    return centred / np.maximum(scale, 1e-9)


def anova_statistic(block: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    grand = block.mean(axis=0)
    between = np.zeros(block.shape[1], dtype=np.float64)
    within = np.zeros(block.shape[1], dtype=np.float64)
    for class_id in range(class_count):
        class_rows = block[labels == class_id]
        if len(class_rows) < 2:
            continue
        centre = class_rows.mean(axis=0)
        between += len(class_rows) * (centre - grand) ** 2
        within += ((class_rows - centre) ** 2).sum(axis=0)
    return between / np.maximum(within, 1e-12)


def decoder_scores(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    class_count: int,
    fixed_c: dict,
    anova_quantile: float | None = None,
) -> dict[str, np.ndarray]:
    if anova_quantile:
        # Selection uses training rows and training labels only, exactly as in the
        # standalone nested run. Applied per representation, since the two have very
        # different feature counts.
        statistic = anova_statistic(x[train_idx], y[train_idx], class_count)
        keep = statistic > np.quantile(statistic, anova_quantile)
        if keep.sum() < 20:
            keep = statistic >= np.sort(statistic)[-20]
        x = x[:, keep]
    mean, scale = standardize(x, train_idx)
    z_train, z_eval = dual_basis(
        (x[train_idx] - mean) / scale, (x[eval_idx] - mean) / scale
    )
    kernel_train = z_train @ z_train.T
    kernel_eval = z_eval @ z_train.T
    out = {}
    for model, c_value in fixed_c.items():
        if model == "correlation_centroid":
            out[model] = correlation_centroid_scores(
                x, y, train_idx, eval_idx, class_count
            )
        else:
            out[model] = fit_projected(
                model, z_train, y[train_idx], z_eval,
                kernel_train, kernel_eval, c_value, 0,
            )
    return {k: zscore_scores(v) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score-level decoder ensemble.")
    parser.add_argument("--representations", nargs="+", required=True,
                        help="name=/path/to/checkpoints pairs")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--inner-subject-fold-count", type=int, default=4)
    parser.add_argument("--split-limit", type=int)
    parser.add_argument(
        "--anova-quantile",
        type=float,
        help=(
            "Apply ANOVA feature selection within each representation before fitting, "
            "dropping this fraction of features. Tests whether selection and ensembling "
            "stack, since each was worth about +0.006 alone."
        ),
    )
    args = parser.parse_args()

    fixed_c = {"linear_svm": 0.0001, "logistic_l2": 0.01, "correlation_centroid": 0.0}

    sources = {}
    records = None
    y = None
    for entry in args.representations:
        name, path = entry.split("=", 1)
        feature_dict, y_here, records_here = load_checkpoints(
            Path(path), [args.sequence_key]
        )
        sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records_here)
        sources[name] = sequence.mean(axis=1, dtype=np.float32)
        del sequence
        if records is None:
            records, y = records_here, y_here
        else:
            if not np.array_equal(y, y_here):
                raise SystemExit(f"{name}: label vector differs")
            for a, b in zip(records, records_here):
                if (str(a["subject_id"]), int(a["run_id"]), int(a["event_start"])) != (
                    str(b["subject_id"]), int(b["run_id"]), int(b["event_start"])
                ):
                    raise SystemExit(f"{name}: event order differs")
        print(f"{name}: {sources[name].shape}", flush=True)

    class_count = int(y.max()) + 1
    members = [f"{rep}|{model}" for rep in sources for model in fixed_c]
    print(f"{len(members)} ensemble members: {members}", flush=True)

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    # Coarse simplex over member weights, kept small so inner selection stays honest.
    grid = [w for w in itertools.product([0, 1, 2], repeat=len(members)) if sum(w) > 0]
    print(f"{len(grid)} candidate weightings", flush=True)

    rows = []
    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        inner = inner_splits(records, train_idx, "subject", args.inner_subject_fold_count)

        # Inner-fold member scores, used only for weight selection.
        inner_scores, inner_truth = [], []
        for inner_split in inner:
            block = {}
            for rep, x in sources.items():
                got = decoder_scores(
                    x, y, inner_split["train_idx"], inner_split["val_idx"],
                    class_count, fixed_c, args.anova_quantile,
                )
                for model, s in got.items():
                    block[f"{rep}|{model}"] = s
            inner_scores.append(block)
            inner_truth.append(y[inner_split["val_idx"]])

        best_weights, best_score = None, -1.0
        for weights in grid:
            total = 0.0
            for block, truth in zip(inner_scores, inner_truth):
                combined = sum(
                    w * block[m] for w, m in zip(weights, members) if w
                )
                total += metrics(truth, combined.argmax(axis=1))["balanced_accuracy"]
            if total > best_score:
                best_score, best_weights = total, weights

        outer_block = {}
        for rep, x in sources.items():
            got = decoder_scores(
                x, y, train_idx, val_idx, class_count, fixed_c, args.anova_quantile
            )
            for model, s in got.items():
                outer_block[f"{rep}|{model}"] = s

        variants = {
            "uniform": sum(outer_block[m] for m in members),
            "inner_selected": sum(
                w * outer_block[m] for w, m in zip(best_weights, members) if w
            ),
        }
        for m in members:
            variants[f"single:{m}"] = outer_block[m]

        for name, combined in variants.items():
            for rule, prediction in (
                ("independent", combined.argmax(axis=1).astype(np.int64)),
                ("balanced", apply_balanced_assignment(combined, val_idx, records)),
            ):
                rows.append(
                    {
                        "split": split["split"],
                        "variant": name,
                        "weights": list(best_weights) if name == "inner_selected" else None,
                        "prediction_rule": rule,
                        "balanced_accuracy": float(
                            metrics(y[val_idx], prediction)["balanced_accuracy"]
                        ),
                    }
                )
        shown = [r for r in rows if r["split"] == split["split"]
                 and r["variant"] == "inner_selected" and r["prediction_rule"] == "independent"]
        print(f"{split['split']} selected={best_weights} indep={shown[0]['balanced_accuracy']:.4f}", flush=True)

    summary = {}
    for variant in sorted({r["variant"] for r in rows}):
        for rule in ("independent", "balanced"):
            values = [
                r["balanced_accuracy"] for r in rows
                if r["variant"] == variant and r["prediction_rule"] == rule
            ]
            summary[f"{variant}|{rule}"] = {
                "mean": float(np.mean(values)), "sd": float(np.std(values))
            }

    Path(args.out_json).write_text(
        json.dumps(
            {
                "representations": args.representations,
                "members": members,
                "outer_split_count": len(splits),
                "weight_grid_size": len(grid),
                "anova_quantile": args.anova_quantile,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\n{'variant|rule':44s} {'mean':>8s} {'sd':>8s}")
    for name, row in sorted(summary.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{name:44s} {row['mean']:8.4f} {row['sd']:8.4f}")


if __name__ == "__main__":
    main()
