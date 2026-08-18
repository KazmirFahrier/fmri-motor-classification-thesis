#!/usr/bin/env python3
"""What is the frozen hierarchy doing that a flat linear decoder is not?

The frozen decoder beats a flat linear SVM by `+0.0262`. The project has never asked
*which component* of the hierarchy earns that. It combines at least three ingredients
that a flat SVM lacks:

1. **Two-stage structure** — a coarse leg-versus-arm decision, then a within-pair
   decision, rather than one four-way boundary.
2. **Pair-specific feature selection** — each pair specialist ranks features for its
   own contrast instead of sharing one feature set.
3. **Covariance-aware scoring** with capped feature counts.

Feature selection has already been partly isolated: giving the flat baseline ANOVA
selection narrows the gap from `+0.026` to `+0.020`, so roughly a quarter of the
advantage is selection rather than structure. This script isolates the other two by
building flat-SVM analogues of the hierarchy's shape:

- `flat` — one four-way SVM, the existing baseline.
- `hierarchical_hard` — a leg-vs-arm SVM, then a within-pair SVM chosen by that
  decision. Errors at the coarse stage are unrecoverable, which is the known weakness
  of naive hierarchies and is why the frozen decoder fuses scores instead.
- `hierarchical_fused` — coarse and pair scores combined additively with a weight
  selected on inner folds, so no hard routing decision is made. This is the frozen
  hierarchy's shape, without its covariance machinery.
- `pairwise_selected` — per-pair ANOVA feature selection feeding the fused form,
  adding ingredient 2.

If `hierarchical_fused` recovers most of the `+0.026`, the hierarchy's advantage is
structural and reproducible with ordinary classifiers. If it does not, the covariance
scoring is doing the work and the structure is incidental.

An earlier diagnostic already found that a *naive* two-stage centroid classifier did
not beat flat centroids, and that an oracle given the true coarse group reached only
`0.7238`. Those results used centroids, not discriminative models, so this is not a
repeat — but they set the expectation that hard routing will disappoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    dual_basis,
    fit_projected,
    standardize,
)

LEG, ARM = (0, 1), (2, 3)


def anova_keep(block, labels, class_count, quantile):
    grand = block.mean(axis=0)
    between = np.zeros(block.shape[1])
    within = np.zeros(block.shape[1])
    for c in range(class_count):
        rows = block[labels == c]
        if len(rows) < 2:
            continue
        centre = rows.mean(axis=0)
        between += len(rows) * (centre - grand) ** 2
        within += ((rows - centre) ** 2).sum(axis=0)
    stat = between / np.maximum(within, 1e-12)
    keep = stat > np.quantile(stat, quantile)
    return keep if keep.sum() >= 20 else stat >= np.sort(stat)[-20]


def binary_scores(x, y, train_idx, eval_idx, positive, c_value, mask=None):
    """Signed margin for a binary contrast, positive meaning class `positive[1]`."""
    block = x[:, mask] if mask is not None else x
    sel = train_idx[np.isin(y[train_idx], positive)]
    mean, scale = standardize(block, sel)
    z_train, z_eval = dual_basis(
        (block[sel] - mean) / scale, (block[eval_idx] - mean) / scale
    )
    target = (y[sel] == positive[1]).astype(np.int64)
    s = fit_projected(
        "linear_svm", z_train, target, z_eval,
        z_train @ z_train.T, z_eval @ z_train.T, c_value, 0,
    )
    # OvR on two classes yields a single column in some sklearn paths.
    return s[:, 1] - s[:, 0] if s.ndim == 2 and s.shape[1] == 2 else np.ravel(s)


def coarse_scores(x, y, train_idx, eval_idx, c_value, mask=None):
    block = x[:, mask] if mask is not None else x
    mean, scale = standardize(block, train_idx)
    z_train, z_eval = dual_basis(
        (block[train_idx] - mean) / scale, (block[eval_idx] - mean) / scale
    )
    target = np.isin(y[train_idx], ARM).astype(np.int64)
    s = fit_projected(
        "linear_svm", z_train, target, z_eval,
        z_train @ z_train.T, z_eval @ z_train.T, c_value, 0,
    )
    return s[:, 1] - s[:, 0] if s.ndim == 2 and s.shape[1] == 2 else np.ravel(s)


def assemble(coarse, leg, arm, weight):
    """Four-way scores from a coarse margin and two within-pair margins."""
    out = np.zeros((len(coarse), 4))
    out[:, 0] = -weight * coarse - leg
    out[:, 1] = -weight * coarse + leg
    out[:, 2] = weight * coarse - arm
    out[:, 3] = weight * coarse + arm
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Decompose the hierarchy's advantage.")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    ap.add_argument("--fixed-c", type=float, default=0.0001)
    ap.add_argument("--weights", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--anova-quantile", type=float, default=0.9)
    ap.add_argument("--outer-fold-count", type=int, default=6)
    ap.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    ap.add_argument("--inner-subject-fold-count", type=int, default=4)
    ap.add_argument("--split-limit", type=int)
    args = ap.parse_args()

    fd, y, records = load_checkpoints(Path(args.checkpoint_dir), [args.sequence_key])
    seq, _ = preprocess_sequence(fd.pop(args.sequence_key), records)
    x = seq.mean(axis=1, dtype=np.float32)
    del seq
    class_count = int(y.max()) + 1

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit:
        splits = splits[: args.split_limit]

    rows, chosen_weights = [], []
    for split in splits:
        tr, va = split["train_idx"], split["val_idx"]

        # flat baseline
        mean, scale = standardize(x, tr)
        z_tr, z_va = dual_basis((x[tr] - mean) / scale, (x[va] - mean) / scale)
        flat = fit_projected(
            "linear_svm", z_tr, y[tr], z_va, z_tr @ z_tr.T, z_va @ z_tr.T, args.fixed_c, 0
        )

        c_out = coarse_scores(x, y, tr, va, args.fixed_c)
        leg_out = binary_scores(x, y, tr, va, LEG, args.fixed_c)
        arm_out = binary_scores(x, y, tr, va, ARM, args.fixed_c)

        # hard routing
        hard = np.zeros((len(va), 4))
        is_arm = c_out > 0
        hard[~is_arm, 0] = -leg_out[~is_arm]
        hard[~is_arm, 1] = leg_out[~is_arm]
        hard[~is_arm, 2:] = -1e6
        hard[is_arm, 2] = -arm_out[is_arm]
        hard[is_arm, 3] = arm_out[is_arm]
        hard[is_arm, :2] = -1e6

        # fused, weight chosen on inner folds
        inner = inner_splits(records, tr, "subject", args.inner_subject_fold_count)
        inner_cache = []
        for isp in inner:
            inner_cache.append(
                (
                    coarse_scores(x, y, isp["train_idx"], isp["val_idx"], args.fixed_c),
                    binary_scores(x, y, isp["train_idx"], isp["val_idx"], LEG, args.fixed_c),
                    binary_scores(x, y, isp["train_idx"], isp["val_idx"], ARM, args.fixed_c),
                    y[isp["val_idx"]],
                )
            )
        best_w, best = None, -1.0
        for w in args.weights:
            score = np.mean([
                metrics(t, assemble(c, l, a, w).argmax(1))["balanced_accuracy"]
                for c, l, a, t in inner_cache
            ])
            if score > best:
                best, best_w = score, w
        chosen_weights.append(best_w)
        fused = assemble(c_out, leg_out, arm_out, best_w)

        # per-pair ANOVA selection feeding the fused form
        mask_c = anova_keep(x[tr], np.isin(y[tr], ARM).astype(int), 2, args.anova_quantile)
        sel_leg = tr[np.isin(y[tr], LEG)]
        sel_arm = tr[np.isin(y[tr], ARM)]
        mask_l = anova_keep(x[sel_leg], y[sel_leg], class_count, args.anova_quantile)
        mask_a = anova_keep(x[sel_arm], y[sel_arm], class_count, args.anova_quantile)
        fused_sel = assemble(
            coarse_scores(x, y, tr, va, args.fixed_c, mask_c),
            binary_scores(x, y, tr, va, LEG, args.fixed_c, mask_l),
            binary_scores(x, y, tr, va, ARM, args.fixed_c, mask_a),
            best_w,
        )

        for name, s in (
            ("flat", flat),
            ("hierarchical_hard", hard),
            ("hierarchical_fused", fused),
            ("pairwise_selected", fused_sel),
        ):
            for rule, pred in (
                ("independent", s.argmax(1).astype(np.int64)),
                ("balanced", apply_balanced_assignment(s, va, records)),
            ):
                rows.append({
                    "split": split["split"], "variant": name, "prediction_rule": rule,
                    "balanced_accuracy": float(metrics(y[va], pred)["balanced_accuracy"]),
                })
        got = [r for r in rows if r["split"] == split["split"] and r["prediction_rule"] == "independent"]
        print(f"{split['split']} w={best_w} " + " ".join(f"{r['variant']}={r['balanced_accuracy']:.4f}" for r in got), flush=True)

    summary = {}
    for v in ("flat", "hierarchical_hard", "hierarchical_fused", "pairwise_selected"):
        for rule in ("independent", "balanced"):
            vals = [r["balanced_accuracy"] for r in rows if r["variant"] == v and r["prediction_rule"] == rule]
            summary[f"{v}|{rule}"] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}

    Path(args.out_json).write_text(json.dumps({
        "checkpoint_dir": args.checkpoint_dir,
        "outer_split_count": len(splits),
        "selected_fusion_weights": dict(Counter(chosen_weights)),
        "frozen_reference": {"independent": 0.8314, "balanced": 0.8806},
        "summary": summary, "rows": rows,
    }, indent=2))
    print(f"\n{'variant|rule':40s} {'mean':>8s} {'sd':>8s}")
    for k, v in sorted(summary.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{k:40s} {v['mean']:8.4f} {v['sd']:8.4f}")


if __name__ == "__main__":
    main()
