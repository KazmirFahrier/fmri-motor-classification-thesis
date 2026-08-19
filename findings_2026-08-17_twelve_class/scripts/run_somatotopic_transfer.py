#!/usr/bin/env python3
"""Does cross-contrast transfer follow the somatotopic axis?

Four-class analysis found that a decoder trained on left-versus-right leg reads
forearm-versus-upper-arm at `0.6525`, 30/30 folds, against a null at chance. Twelve-class
decoding then showed the representational geometry recovers the homunculus. This joins
the two: if the shared decision axis is somatotopic, a decoder trained on one
within-limb contrast should transfer to another, and transfer should be **directional**
along the body axis.

## The mapping is fixed a priori, which is what makes this a test

For a binary problem, choosing whichever of two label mappings scores higher on
held-out data guarantees an above-chance result regardless of the data. Every contrast
here is therefore oriented by **somatotopic rank before any model is fitted**: the
lower-ranked (more distal, or more inferior on the homunculus) member is class 0 and the
higher-ranked member is class 1. "Aligned" transfer then means the two decision axes
point the same way along the body, which is a directional prediction the geometry makes
and an amplitude or laterality account does not.

Left-versus-right leg is deliberately **excluded** from the directed set: its two members
share a somatotopic rank, so its orientation along the axis is undefined and any mapping
would be arbitrary. It is reported separately, because it is the contrast the original
finding used and its ambiguity is precisely what twelve classes exist to resolve.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
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

RANK = {0: 3.0, 1: 3.0, 2: 6.0, 3: 5.0, 4: 1.0, 5: 2.0,
        6: 7.0, 7: 8.0, 8: 11.0, 9: 9.0, 10: 9.5, 11: 10.0}
NAME = {0: "Left leg", 1: "Right leg", 2: "Forearm", 3: "Upper arm", 4: "Toe",
        5: "Ankle", 6: "Wrist", 7: "Finger", 8: "Eye", 9: "Jaw", 10: "Lip",
        11: "Tongue"}

# (low-rank class, high-rank class). Orientation is set by RANK, never by the result.
DIRECTED = {
    "toe_ankle":   (4, 5),
    "upperarm_forearm": (3, 2),
    "wrist_finger": (6, 7),
    "jaw_lip":     (9, 10),
    "lip_tongue":  (10, 11),
}
UNDIRECTED = {"leg_left_right": (0, 1)}


def fit_transfer(
    gram: np.ndarray,
    train_rows: np.ndarray,
    eval_rows: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
) -> np.ndarray:
    model = SVC(C=c_value, kernel="precomputed", random_state=0)
    model.fit(gram[np.ix_(train_rows, train_rows)], y_train)
    return model.decision_function(gram[np.ix_(eval_rows, train_rows)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Somatotopic transfer.")
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

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    contrasts = {**DIRECTED, **UNDIRECTED}
    results = {f"{a}->{b}": [] for a, b in permutations(contrasts, 2)}
    within = {name: [] for name in contrasts}

    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        mean, scale = standardize(block, train_idx)
        x_all = ((block - mean) / scale).astype(np.float64)
        gram = x_all @ x_all.T
        del x_all

        rows = {
            name: {
                "train": train_idx[np.isin(y[train_idx], pair)],
                "val": val_idx[np.isin(y[val_idx], pair)],
            }
            for name, pair in contrasts.items()
        }
        # class 1 is always the higher somatotopic rank
        targets = {
            name: (y[rows[name]["train"]] == pair[1]).astype(np.int64)
            for name, pair in contrasts.items()
        }

        for name, pair in contrasts.items():
            decision = fit_transfer(gram, rows[name]["train"], rows[name]["val"],
                                    targets[name], args.fixed_c)
            truth = (y[rows[name]["val"]] == pair[1]).astype(np.int64)
            within[name].append(float(np.mean((decision > 0).astype(np.int64) == truth)))

        for source, target in permutations(contrasts, 2):
            decision = fit_transfer(gram, rows[source]["train"], rows[target]["val"],
                                    targets[source], args.fixed_c)
            truth = (y[rows[target]["val"]] == contrasts[target][1]).astype(np.int64)
            results[f"{source}->{target}"].append(
                float(np.mean((decision > 0).astype(np.int64) == truth))
            )
        del gram
        print(f"{split['split']} done", flush=True)

    def describe(values):
        a = np.asarray(values)
        sem = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        return {"mean": float(a.mean()),
                "ci95": [float(a.mean() - 1.96 * sem), float(a.mean() + 1.96 * sem)],
                "folds_above_chance": int((a > 0.5).sum()), "fold_count": len(a)}

    transfer = {k: describe(v) for k, v in results.items()}
    within_summary = {k: describe(v) for k, v in within.items()}

    # Directed-only analysis: does transfer decay with somatotopic separation?
    midpoint = {n: (RANK[p[0]] + RANK[p[1]]) / 2 for n, p in contrasts.items()}
    distances, accuracies, labels = [], [], []
    for source, target in permutations(DIRECTED, 2):
        distances.append(abs(midpoint[source] - midpoint[target]))
        accuracies.append(transfer[f"{source}->{target}"]["mean"])
        labels.append(f"{source}->{target}")
    distances, accuracies = np.asarray(distances), np.asarray(accuracies)
    centered_d, centered_a = distances - distances.mean(), accuracies - accuracies.mean()
    denominator = np.linalg.norm(centered_d) * np.linalg.norm(centered_a)
    distance_r = float(centered_d @ centered_a / denominator) if denominator > 1e-12 else 0.0

    directed_means = [transfer[f"{s}->{t}"]["mean"] for s, t in permutations(DIRECTED, 2)]
    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "outer_split_count": len(splits),
        "chance": 0.5,
        "contrast_definition": {
            n: {"class0": NAME[p[0]], "class1": NAME[p[1]],
                "rank0": RANK[p[0]], "rank1": RANK[p[1]], "directed": n in DIRECTED}
            for n, p in contrasts.items()
        },
        "within_contrast": within_summary,
        "transfer": transfer,
        "directed_summary": {
            "mean_transfer": float(np.mean(directed_means)),
            "pairs_above_chance": int(np.sum(np.asarray(directed_means) > 0.5)),
            "pair_count": len(directed_means),
            "correlation_with_somatotopic_distance": distance_r,
            "pairs": [
                {"pair": lab, "distance": float(d), "transfer": float(a)}
                for lab, d, a in zip(labels, distances, accuracies)
            ],
        },
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))

    print("\nwithin-contrast (ceiling for each)")
    for name, entry in within_summary.items():
        print(f"  {name:<22} {entry['mean']:.4f}")
    print("\ndirected transfer, a-priori somatotopic orientation")
    for source, target in permutations(DIRECTED, 2):
        e = transfer[f"{source}->{target}"]
        print(f"  {source:>20} -> {target:<20} {e['mean']:.4f}  "
              f"{e['folds_above_chance']}/{e['fold_count']} folds")
    print(f"\nmean directed transfer      {np.mean(directed_means):.4f}")
    print(f"pairs above chance          {int(np.sum(np.asarray(directed_means) > 0.5))}"
          f"/{len(directed_means)}")
    print(f"correlation with distance   r = {distance_r:+.4f}")


if __name__ == "__main__":
    main()
