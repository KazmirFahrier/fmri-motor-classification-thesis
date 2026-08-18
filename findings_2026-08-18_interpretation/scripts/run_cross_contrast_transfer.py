#!/usr/bin/env python3
"""Cross-contrast transfer: do the two pairs share a decision axis?

The frozen four classes form two pairs that are different *kinds* of contrast. Left
leg versus right leg is a **laterality** distinction; forearm versus upper arm is a
**proximal-distal** one. Whether a decoder trained on one can read the other tests
whether the fine stage is solving one problem twice or two unrelated problems.

The plan recorded an expectation of failure, on the grounds that the two contrasts are
conceptually unrelated. The representational geometry computed alongside this
contradicts that expectation: in the group RDM, left leg sits closer to forearm
(`1.264`) than to upper arm (`1.764`), and right leg reverses it. That is a structure
crossing the limb boundary, and it predicts positive transfer under the mapping
`left leg -> forearm`, `right leg -> upper arm`.

The mapping direction is therefore **predicted in advance from the RDM**, not chosen
by whichever mapping scores higher — picking the better of two mappings after seeing
held-out accuracy would be selection on the test set, and for a binary problem it
would guarantee a number above chance no matter what the data contained.

The swapped row is reported for completeness but carries **no independent evidence**:
for a binary decision it is exactly `1 - aligned`, an arithmetic identity rather than
a second observation. The evidence for a shared axis is that the *a priori* mapping
clears chance reliably across the 30 folds and does so in **both** transfer
directions.

Transfer is evaluated on **held-out subjects** under the same 30-fold protocol as
everything else, so nothing subject-specific can carry the effect. A within-contrast
reference — train and test on the same pair, held-out subjects — gives the ceiling
each transfer number should be read against.

## The control this result needs

This project's preprocessing is transductive: unlabeled subject-run centering is worth
`+0.52`, far more than any decoder choice. Centering makes the four class means within
a run sum to approximately zero, which imposes a dependency between them, and a
reviewer's first question will be whether that dependency alone manufactures apparent
transfer between two contrasts that share no biology.

`--permutations` answers it directly. Labels are shuffled **within each subject-run**,
preserving the two-per-class composition, and the entire transfer analysis is rerun.
The preprocessing is label-free and therefore identical under permutation, so the null
holds the suspected mechanism fixed and varies only the label mapping. If centering
were generating the effect, the null would reproduce it.
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

from run_detrended_pair_feature_selection import (  # noqa: E402
    load_checkpoints,
    outer_splits,
)
from run_permutation_test import build_run_positions, shuffle_within_run  # noqa: E402
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402
from run_standard_mvpa_baseline import standardize  # noqa: E402

CLASS_NAMES = ["Left leg", "Right leg", "Forearm", "Upper arm"]
PAIRS = {"leg": (0, 1), "arm": (2, 3)}


def binary_fit_predict(
    gram: np.ndarray,
    train_rows: np.ndarray,
    eval_rows: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
) -> np.ndarray:
    """Binary linear SVM on a slice of a precomputed Gram matrix.

    The Gram matrix ``X X^T`` does not depend on the labels, so it is built once per
    split and every permutation indexes into it. That is what makes a 200-draw null
    affordable: permutations change only which rows are training rows, never the inner
    products between them.
    """
    model = SVC(C=c_value, kernel="precomputed", random_state=0)
    model.fit(gram[np.ix_(train_rows, train_rows)], y_train)
    return model.decision_function(gram[np.ix_(eval_rows, train_rows)]).astype(np.float64)


def analyse(
    gram_for: dict,
    labels: np.ndarray,
    splits: list[dict],
    c_value: float,
    verbose: bool,
) -> dict[str, list[float]]:
    """Run the full transfer analysis for one labelling.

    ``gram_for`` maps a split name to its precomputed Gram matrix over all events.
    Standardization and the Gram are both label-free, so they are computed once and
    reused across permutations rather than recomputed each time.
    """
    conditions = {
        "leg_to_arm_aligned": [], "leg_to_arm_swapped": [],
        "arm_to_leg_aligned": [], "arm_to_leg_swapped": [],
        "leg_within": [], "arm_within": [],
    }
    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        gram = gram_for[split["split"]]

        pair_rows = {
            name: {
                "train": train_idx[np.isin(labels[train_idx], pair)],
                "val": val_idx[np.isin(labels[val_idx], pair)],
            }
            for name, pair in PAIRS.items()
        }
        fitted = {
            name: (
                pair_rows[name]["train"],
                (labels[pair_rows[name]["train"]] == pair[1]).astype(np.int64),
            )
            for name, pair in PAIRS.items()
        }

        for source, target in (("leg", "arm"), ("arm", "leg")):
            train_rows, y_train = fitted[source]
            eval_rows = pair_rows[target]["val"]
            decision = binary_fit_predict(gram, train_rows, eval_rows, y_train, c_value)
            truth = (labels[eval_rows] == PAIRS[target][1]).astype(np.int64)
            aligned = float(np.mean((decision > 0).astype(np.int64) == truth))
            conditions[f"{source}_to_{target}_aligned"].append(aligned)
            conditions[f"{source}_to_{target}_swapped"].append(1.0 - aligned)

        for name, pair in PAIRS.items():
            train_rows, y_train = fitted[name]
            eval_rows = pair_rows[name]["val"]
            decision = binary_fit_predict(gram, train_rows, eval_rows, y_train, c_value)
            truth = (labels[eval_rows] == pair[1]).astype(np.int64)
            conditions[f"{name}_within"].append(
                float(np.mean((decision > 0).astype(np.int64) == truth))
            )
        if verbose:
            print(
                f"{split['split']} leg->arm {conditions['leg_to_arm_aligned'][-1]:.3f} "
                f"arm->leg {conditions['arm_to_leg_aligned'][-1]:.3f}",
                flush=True,
            )
    return conditions


def describe(values: list[float]) -> dict:
    array = np.asarray(values)
    sem = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci95": [float(array.mean() - 1.96 * sem), float(array.mean() + 1.96 * sem)],
        "folds_above_chance": int((array > 0.5).sum()),
        "fold_count": len(array),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-contrast transfer.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--fixed-c", type=float, default=0.0001)
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--split-limit", type=int)
    parser.add_argument("--permutations", type=int, default=0,
                        help="Within-run label permutations for the null. 0 skips it.")
    parser.add_argument("--permutation-seed", type=int, default=17)
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    block = sequence.mean(axis=1, dtype=np.float32)

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    # Standardization and the Gram depend only on the split, never on labels, so both
    # are built once here and every permutation reuses them.
    gram_for = {}
    for split in splits:
        mean, scale = standardize(block, split["train_idx"])
        x_all = ((block - mean) / scale).astype(np.float64)
        gram_for[split["split"]] = x_all @ x_all.T
        print(f"gram built for {split['split']}", flush=True)

    conditions = analyse(gram_for, y, splits, args.fixed_c, verbose=True)
    summary = {name: describe(values) for name, values in conditions.items()}

    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "class_names": CLASS_NAMES,
        "pairs": {k: list(v) for k, v in PAIRS.items()},
        "outer_split_count": len(splits),
        "chance": 0.5,
        "conditions": summary,
        "fold_values": conditions,
    }

    if args.permutations:
        run_positions = build_run_positions(records)
        rng = np.random.default_rng(args.permutation_seed)
        null = {name: [] for name in conditions}
        for index in range(args.permutations):
            permuted = shuffle_within_run(y, run_positions, rng)
            result = analyse(gram_for, permuted, splits, args.fixed_c, verbose=False)
            for name, values in result.items():
                null[name].append(float(np.mean(values)))
            if (index + 1) % 10 == 0:
                print(
                    f"permutation {index + 1}/{args.permutations} "
                    f"leg->arm null {np.mean(null['leg_to_arm_aligned']):.4f}",
                    flush=True,
                )
        payload["null"] = {}
        for name in ("leg_to_arm_aligned", "arm_to_leg_aligned",
                     "leg_within", "arm_within"):
            draws = np.asarray(null[name])
            observed = summary[name]["mean"]
            # One-sided p with the +1 correction, the standard permutation estimate.
            p_value = float((np.sum(draws >= observed) + 1) / (len(draws) + 1))
            payload["null"][name] = {
                "observed": observed,
                "null_mean": float(draws.mean()),
                "null_sd": float(draws.std(ddof=1)) if len(draws) > 1 else 0.0,
                "null_max": float(draws.max()),
                "p_value": p_value,
                "z": float((observed - draws.mean()) / draws.std(ddof=1))
                if len(draws) > 1 and draws.std(ddof=1) > 1e-12 else None,
                "permutations": len(draws),
            }
        payload["null_draws"] = {k: v for k, v in null.items()}

    Path(args.out_json).write_text(json.dumps(payload, indent=2))

    print("\n                          mean     CI95              folds>chance")
    for name in ("leg_within", "arm_within", "leg_to_arm_aligned",
                 "arm_to_leg_aligned", "leg_to_arm_swapped", "arm_to_leg_swapped"):
        e = summary[name]
        print(f"  {name:<22} {e['mean']:.4f}  "
              f"[{e['ci95'][0]:.4f}, {e['ci95'][1]:.4f}]   "
              f"{e['folds_above_chance']}/{e['fold_count']}")

    if args.permutations:
        print(f"\nwithin-run permutation null ({args.permutations} draws)")
        print("                          observed   null mean    null sd    z        p")
        for name, entry in payload["null"].items():
            z = f"{entry['z']:.1f}" if entry["z"] is not None else "n/a"
            print(f"  {name:<22} {entry['observed']:.4f}     "
                  f"{entry['null_mean']:.4f}     {entry['null_sd']:.4f}   "
                  f"{z:>6}   {entry['p_value']:.4f}")


if __name__ == "__main__":
    main()
