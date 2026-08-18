#!/usr/bin/env python3
"""Representational geometry: which classes are confused, and is the structure shared?

Prior diagnostics in this project observed, from a single method, that coarse
leg-versus-arm routing is strong and the within-pair stage is the bottleneck. That is
an observation about one decoder. If the same structure appears in the raw
representational geometry and in the confusion matrices of every decoder built here,
it stops being a property of the hierarchy and becomes a property of the data — which
is a much stronger thing for the manuscript to claim.

Two independent views are computed.

**Confusion structure.** Pooled confusion matrices over the 30 folds, per decoder and
per prediction rule. The design-constrained rule is included because it redistributes
errors rather than only removing them, and the shape of that redistribution has never
been looked at.

**Representational dissimilarity.** Per subject, the correlation distance between
class-mean patterns, giving a 4x4 RDM (Kriegeskorte, 2008). Two questions follow: does
the group-mean RDM show the coarse/fine split, and do individual subjects agree with
each other? Inter-subject RDM agreement is the test of whether the geometry is shared
or idiosyncratic, and it is computed on the off-diagonal entries only.

RDMs are computed on the same preprocessed sequence the decoders see, so the geometry
described here is the geometry they actually operate on rather than a separate view of
the raw data.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "findings_2026-08-12" / "scripts",
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from run_balanced_event_assignment import apply_balanced_assignment  # noqa: E402
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

# Frozen four, in class-id order. Ids 0/1 are the leg pair (a laterality contrast) and
# 2/3 the arm pair (a proximal-distal one).
CLASS_NAMES = ["Left leg", "Right leg", "Forearm", "Upper arm"]
LEG, ARM = (0, 1), (2, 3)


def rdm_for(block: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    """Correlation-distance RDM between class-mean patterns."""
    centroids = np.stack(
        [block[labels == c].mean(axis=0) for c in range(class_count)]
    ).astype(np.float64)
    centroids -= centroids.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    similarity = (centroids / norms) @ (centroids / norms).T
    return 1.0 - similarity


def main() -> None:
    parser = argparse.ArgumentParser(description="RSA and confusion structure.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--models", nargs="+",
                        default=["linear_svm", "logistic_l2", "correlation_centroid"])
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
    print(f"{block.shape[0]} events, {class_count} classes", flush=True)

    # ---- Representational dissimilarity, per subject -------------------------------
    by_subject: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_subject[str(record["subject_id"])].append(index)

    subject_rdms, subject_ids = [], []
    for subject, rows in sorted(by_subject.items()):
        idx = np.asarray(rows)
        subject_rdms.append(rdm_for(block[idx], y[idx], class_count))
        subject_ids.append(subject)
    subject_rdms = np.stack(subject_rdms)
    group_rdm = subject_rdms.mean(axis=0)

    triu = np.triu_indices(class_count, k=1)
    vectors = subject_rdms[:, triu[0], triu[1]]
    agreements = [
        float(np.corrcoef(vectors[a], vectors[b])[0, 1])
        for a, b in combinations(range(len(subject_ids)), 2)
    ]

    coarse = float(np.mean([group_rdm[a, b] for a in LEG for b in ARM]))
    within_leg = float(group_rdm[LEG[0], LEG[1]])
    within_arm = float(group_rdm[ARM[0], ARM[1]])

    # ---- Confusion structure, pooled over folds ------------------------------------
    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    confusion = {
        model: {rule: np.zeros((class_count, class_count), dtype=np.int64)
                for rule in ("independent", "balanced")}
        for model in args.models
    }

    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        mean, scale = standardize(block, train_idx)
        x_all = ((block - mean) / scale).astype(np.float32)
        z_train, z_val = dual_basis(x_all[train_idx], x_all[val_idx])
        kernel_train, kernel_val = z_train @ z_train.T, z_val @ z_train.T

        for model in args.models:
            if model == "correlation_centroid":
                scores = correlation_centroid_scores(
                    x_all, y, train_idx, val_idx, class_count
                )
            else:
                scores = fit_projected(
                    model, z_train, y[train_idx], z_val,
                    kernel_train, kernel_val, args.fixed_c, 0,
                )
            for rule, prediction in (
                ("independent", scores.argmax(axis=1).astype(np.int64)),
                ("balanced", apply_balanced_assignment(scores, val_idx, records)),
            ):
                for true_label, predicted in zip(y[val_idx], prediction):
                    confusion[model][rule][int(true_label), int(predicted)] += 1
        print(f"{split['split']} done", flush=True)

    def summarize_confusion(counts: np.ndarray) -> dict:
        rows = counts.sum(axis=1, keepdims=True).astype(np.float64)
        rows[rows == 0] = 1.0
        normalized = counts / rows
        errors = counts.sum() - np.trace(counts)
        within_pair = sum(
            counts[a, b] for pair in (LEG, ARM) for a, b in ((pair[0], pair[1]), (pair[1], pair[0]))
        )
        return {
            "counts": counts.tolist(),
            "row_normalized": normalized.tolist(),
            "accuracy": float(np.trace(counts) / counts.sum()),
            "within_pair_error_fraction": float(within_pair / errors) if errors else None,
            "per_class_accuracy": np.diag(normalized).tolist(),
        }

    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "class_names": CLASS_NAMES,
        "outer_split_count": len(splits),
        "rdm": {
            "group_mean": group_rdm.tolist(),
            "subject_ids": subject_ids,
            "coarse_leg_vs_arm": coarse,
            "within_leg_pair": within_leg,
            "within_arm_pair": within_arm,
            "coarse_to_fine_ratio": coarse / max((within_leg + within_arm) / 2, 1e-12),
            "inter_subject_agreement_mean": float(np.mean(agreements)),
            "inter_subject_agreement_sd": float(np.std(agreements)),
            "inter_subject_agreement_pairs": len(agreements),
        },
        "confusion": {
            model: {rule: summarize_confusion(confusion[model][rule])
                    for rule in confusion[model]}
            for model in args.models
        },
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))

    print("\ngroup RDM (correlation distance between class means)")
    print("             " + "".join(f"{n[:9]:>11}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:>10} " + "".join(f"{v:11.4f}" for v in group_rdm[i]))
    print(f"\ncoarse leg-vs-arm distance   {coarse:.4f}")
    print(f"within leg pair              {within_leg:.4f}")
    print(f"within arm pair              {within_arm:.4f}")
    print(f"coarse:fine ratio            {payload['rdm']['coarse_to_fine_ratio']:.3f}")
    print(f"inter-subject RDM agreement  {np.mean(agreements):.4f} "
          f"(sd {np.std(agreements):.4f}, {len(agreements)} pairs)")
    for model in args.models:
        entry = payload["confusion"][model]["independent"]
        print(f"\n{model}: acc {entry['accuracy']:.4f}, "
              f"{entry['within_pair_error_fraction']:.3f} of errors are within-pair")


if __name__ == "__main__":
    main()
