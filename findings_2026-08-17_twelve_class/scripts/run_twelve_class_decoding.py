#!/usr/bin/env python3
"""Decode all twelve movement conditions, and test the somatotopic prediction.

The dataset defines twelve conditions and this project has only ever used four, so the
problem the dataset was built for has never been attempted here. This runs it under the
frozen 30-fold protocol, and then asks the question the four-class cross-contrast
transfer result raised.

## The question this exists to answer

Four-class analysis found that a decoder trained on left-versus-right leg reads
forearm-versus-upper-arm at `0.6525` on held-out subjects, 30/30 folds, against a
permutation null at chance. Something organises those four classes along an axis
crossing the limb boundary. Three accounts fit and four classes cannot separate them:
a somatotopic gradient, movement amplitude, or a laterality confound in the arm
conditions.

Twelve conditions separate them, because they supply a **graded somatotopic ordering**.
The classical homunculus runs toe, ankle, leg, trunk, upper arm, forearm, wrist, finger,
then face. If the shared axis is somatotopic, representational distance should increase
with separation along that ordering, and the face conditions — anatomically remote from
every limb — should sit far from all of them. If the axis is amplitude or laterality,
distance should not track the ordering.

That makes this a falsifiable prediction rather than a description, which is what the
four-class result could not supply.

## Comparability

Class ids `0-3` are the frozen four in their original order, and the four-class subset
of this data reproduces the existing checkpoints to `1.9e-06` — float32 rounding in the
resize. Every number here is therefore directly comparable with the rest of the project.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
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

from run_detrended_pair_feature_selection import (  # noqa: E402
    load_checkpoints,
    outer_splits,
)
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402
from run_standard_mvpa_baseline import standardize  # noqa: E402

# Class id -> (name, somatotopic rank). Ranks follow the classical homunculus ordering
# from distal lower limb through the trunk to the face. Ties are given equal rank where
# the ordering is genuinely ambiguous (left/right leg are the same somatotopic level
# in opposite hemispheres; jaw and lip are adjacent face territory).
CLASSES = {
    0:  ("Left leg",   3.0),
    1:  ("Right leg",  3.0),
    2:  ("Forearm",    6.0),
    3:  ("Upper arm",  5.0),
    4:  ("Toe",        1.0),
    5:  ("Ankle",      2.0),
    6:  ("Wrist",      7.0),
    7:  ("Finger",     8.0),
    8:  ("Eye",       11.0),
    9:  ("Jaw",        9.0),
    10: ("Lip",        9.5),
    11: ("Tongue",    10.0),
}
BODY_PART = {
    **{i: "lower_limb" for i in (0, 1, 4, 5)},
    **{i: "upper_limb" for i in (2, 3, 6, 7)},
    **{i: "face" for i in (8, 9, 10, 11)},
}


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> float:
    """Mean per-class recall.

    The repository's shared ``metrics`` builds its confusion matrix from a module-level
    four-element ``CLASS_NAMES`` and raises on a twelve-class problem. This is the same
    quantity generalised over class count, kept local so the shared helper is untouched.
    """
    recalls = []
    for class_idx in range(class_count):
        mask = y_true == class_idx
        support = int(mask.sum())
        if support:
            recalls.append(float((y_pred[mask] == class_idx).sum()) / support)
    return float(np.mean(recalls)) if recalls else 0.0


def rdm_for(block: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    centroids = np.stack(
        [block[labels == c].mean(axis=0) for c in range(class_count)]
    ).astype(np.float64)
    centroids -= centroids.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    unit = centroids / norms
    return 1.0 - unit @ unit.T


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denominator = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / denominator) if denominator > 1e-12 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Twelve-class decoding and RSA.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
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
    del sequence
    class_count = int(y.max()) + 1
    print(f"{block.shape[0]} events, {class_count} classes, "
          f"chance {1/class_count:.4f}", flush=True)

    # ---- Representational geometry, per subject ------------------------------------
    by_subject: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_subject[str(record["subject_id"])].append(index)
    subject_rdms = np.stack([
        rdm_for(block[np.asarray(rows)], y[np.asarray(rows)], class_count)
        for _, rows in sorted(by_subject.items())
    ])
    group_rdm = subject_rdms.mean(axis=0)

    triu = np.triu_indices(class_count, k=1)
    ranks = np.asarray([CLASSES[i][1] for i in range(class_count)])
    somatotopic_distance = np.abs(ranks[triu[0]] - ranks[triu[1]])
    observed_distance = group_rdm[triu[0], triu[1]]
    somatotopic_rho = spearman(somatotopic_distance, observed_distance)

    # Same-part versus different-part, the coarse version of the same prediction.
    same_part = np.asarray([
        BODY_PART[int(a)] == BODY_PART[int(b)] for a, b in zip(*triu)
    ])
    payload_geometry = {
        "group_rdm": group_rdm.tolist(),
        "somatotopic_rank": {CLASSES[i][0]: CLASSES[i][1] for i in range(class_count)},
        "somatotopic_spearman": somatotopic_rho,
        "within_body_part_mean": float(observed_distance[same_part].mean()),
        "between_body_part_mean": float(observed_distance[~same_part].mean()),
        "inter_subject_agreement_mean": float(np.mean([
            np.corrcoef(subject_rdms[a][triu], subject_rdms[b][triu])[0, 1]
            for a, b in combinations(range(len(subject_rdms)), 2)
        ])),
    }

    # ---- Twelve-class decoding -----------------------------------------------------
    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    accuracies, confusion = [], np.zeros((class_count, class_count), dtype=np.int64)
    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        mean, scale = standardize(block, train_idx)
        x_train = ((block[train_idx] - mean) / scale).astype(np.float64)
        x_val = ((block[val_idx] - mean) / scale).astype(np.float64)
        model = SVC(C=args.fixed_c, kernel="precomputed",
                    decision_function_shape="ovr", random_state=0)
        model.fit(x_train @ x_train.T, y[train_idx])
        prediction = model.predict(x_val @ x_train.T).astype(np.int64)
        accuracies.append(
            balanced_accuracy(y[val_idx], prediction, class_count)
        )
        for truth, predicted in zip(y[val_idx], prediction):
            confusion[int(truth), int(predicted)] += 1
        print(f"{split['split']} {accuracies[-1]:.4f}", flush=True)

    rows = confusion.sum(axis=1, keepdims=True).astype(np.float64)
    rows[rows == 0] = 1.0
    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "class_names": {str(i): CLASSES[i][0] for i in range(class_count)},
        "class_count": class_count,
        "chance": 1.0 / class_count,
        "outer_split_count": len(splits),
        "accuracy": {
            "mean": float(np.mean(accuracies)),
            "sd": float(np.std(accuracies)),
            "folds": accuracies,
        },
        "confusion_counts": confusion.tolist(),
        "confusion_row_normalized": (confusion / rows).tolist(),
        "per_class_accuracy": {
            CLASSES[i][0]: float((confusion / rows)[i, i]) for i in range(class_count)
        },
        "geometry": payload_geometry,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))

    print(f"\n12-class accuracy {np.mean(accuracies):.4f} "
          f"(sd {np.std(accuracies):.4f}, chance {1/class_count:.4f})")
    print(f"\nsomatotopic ordering: Spearman rho = {somatotopic_rho:+.4f}")
    print(f"  within body part  {payload_geometry['within_body_part_mean']:.4f}")
    print(f"  between body part {payload_geometry['between_body_part_mean']:.4f}")
    print("\nper-class accuracy")
    for i in range(class_count):
        print(f"  {CLASSES[i][0]:<12} {(confusion/rows)[i, i]:.4f}")


if __name__ == "__main__":
    main()
