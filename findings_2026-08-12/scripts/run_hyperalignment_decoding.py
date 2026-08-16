#!/usr/bin/env python3
"""Connectivity hyperalignment with a per-fold template, evaluated end to end.

The surface representation carries more class information than the volumetric one
within subject (`0.7332` versus `0.6912`) yet transfers worse across subjects
(`0.7460` versus `0.8051`). That dissociation is a transfer failure on a
representation that is *not* information-poor, which is exactly the deficit a
functional alignment is meant to repair.

## Why the template is rebuilt inside every fold

A standalone alignment pass would build one template, align everyone, and hand the
result to the decoder. That is cheaper and it is wrong: subjects held out in a given
fold would have helped shape the feature space that fold is evaluated in.

The alignment consumes no labels, so a shared template could not transport class
information, and one could argue it is no worse than the unlabeled subject-run
centering the frozen protocol already applies to held-out runs. That argument is
probably right, but it is an argument, and it is avoidable: rebuilding the template
from each fold's training subjects costs a few seconds per fold and removes the
question entirely. The conservative version is the one a reviewer cannot attack.

Connectivity profiles themselves are fold-independent — they use no labels and no
template — so they are computed once and reused.

## Leakage summary

- Connectivity profiles: each subject's own unlabeled data. Transductive, label-free.
- Template: training subjects of the current fold only. Asserted, not assumed.
- Each subject's rotation: their own profile against that template.
- Standardisation, dual basis, classifier: training rows of the current fold only.
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
for _extra in (REPO_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from project_bold_to_surface import icosphere  # noqa: E402
from run_balanced_event_assignment import apply_balanced_assignment, metrics  # noqa: E402
from run_connectivity_hyperalignment import (  # noqa: E402
    build_parcels,
    connectivity_profile,
    orthogonal_procrustes,
)
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
        description="Per-fold connectivity hyperalignment plus decoding."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--models", nargs="+",
                        default=["linear_svm", "logistic_l2", "correlation_centroid"])
    parser.add_argument("--fixed-c", nargs="+", type=float, default=[0.0001, 0.01, 0.0])
    parser.add_argument("--subdivisions", type=int, default=5)
    parser.add_argument("--parcel-subdivisions", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--split-limit", type=int)
    args = parser.parse_args()

    fixed_c = dict(zip(args.models, args.fixed_c))
    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    events, lags, vertices = sequence.shape
    mean_x = sequence.mean(axis=1, dtype=np.float32)
    flat = sequence.reshape(events * lags, vertices).astype(np.float32)
    del sequence
    class_count = int(y.max()) + 1
    subjects = np.asarray([str(record["subject_id"]) for record in records])
    subject_list = sorted(set(subjects.tolist()))

    target, _ = icosphere(args.subdivisions)
    parcels_one = build_parcels(target, args.parcel_subdivisions)
    parcel_count_one = int(parcels_one.max()) + 1
    parcels = np.concatenate([parcels_one, parcels_one + parcel_count_one])
    parcel_count = parcel_count_one * 2
    if len(parcels) != vertices:
        raise SystemExit(f"{vertices} vertices but parcels cover {len(parcels)}.")
    members = [np.flatnonzero(parcels == p) for p in range(parcel_count)]
    print(f"{parcel_count} parcels, median size {int(np.median([len(m) for m in members]))}", flush=True)

    # Fold-independent and label-free, so computed once.
    started = time.time()
    profiles = {}
    for subject in subject_list:
        rows_for_subject = np.flatnonzero(subjects == subject)
        block = np.concatenate(
            [np.arange(i * lags, (i + 1) * lags) for i in rows_for_subject]
        )
        profiles[subject] = connectivity_profile(
            flat[block].astype(np.float64), parcels, parcel_count
        )
    print(f"connectivity profiles in {time.time() - started:.0f}s", flush=True)

    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    rows = []
    started = time.time()
    for split in splits:
        train_subjects = sorted(set(subjects[split["train_idx"]].tolist()))
        val_subjects = set(subjects[split["val_idx"]].tolist())
        if set(train_subjects) & val_subjects:
            raise RuntimeError("Split isolation violated before alignment.")

        aligned = np.zeros_like(mean_x)
        for parcel in range(parcel_count):
            index = members[parcel]
            if len(index) < 2:
                aligned[:, index] = mean_x[:, index]
                continue
            blocks = {s: profiles[s][index] for s in subject_list}
            template = np.mean([blocks[s] for s in train_subjects], axis=0)
            for _ in range(args.iterations):
                rotations = {
                    s: orthogonal_procrustes(block, template)
                    for s, block in blocks.items()
                }
                template = np.mean(
                    [rotations[s] @ blocks[s] for s in train_subjects], axis=0
                )
            for subject in subject_list:
                rows_for_subject = subjects == subject
                aligned[np.ix_(rows_for_subject, index)] = (
                    mean_x[np.ix_(rows_for_subject, index)] @ rotations[subject].T
                )

        train_idx, val_idx = split["train_idx"], split["val_idx"]
        mean, scale = standardize(aligned, train_idx)
        z_train, z_val = dual_basis(
            (aligned[train_idx] - mean) / scale, (aligned[val_idx] - mean) / scale
        )
        kernel_train = z_train @ z_train.T
        kernel_val = z_val @ z_train.T
        for model in args.models:
            if model == "correlation_centroid":
                scores = correlation_centroid_scores(
                    aligned, y, train_idx, val_idx, class_count
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
                        "model": model,
                        "prediction_rule": rule,
                        "balanced_accuracy": float(
                            metrics(y[val_idx], prediction)["balanced_accuracy"]
                        ),
                    }
                )
        done = [r for r in rows if r["split"] == split["split"]
                and r["model"] == "linear_svm" and r["prediction_rule"] == "independent"]
        print(
            f"{split['split']} linear_svm indep={done[0]['balanced_accuracy']:.4f} "
            f"[{time.time() - started:.0f}s]",
            flush=True,
        )

    summary = {}
    for model in args.models:
        for rule in ("independent", "balanced"):
            values = [
                r["balanced_accuracy"] for r in rows
                if r["model"] == model and r["prediction_rule"] == rule
            ]
            summary[f"{model}|{rule}"] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values)),
                "folds": len(values),
            }

    Path(args.out_json).write_text(
        json.dumps(
            {
                "checkpoint_dir": args.checkpoint_dir,
                "parcel_count": parcel_count,
                "iterations": args.iterations,
                "template": "training subjects of each fold only, rebuilt per fold",
                "outer_split_count": len(splits),
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\n{'model|rule':34s} {'mean':>8s} {'sd':>8s}")
    for name, row in summary.items():
        print(f"{name:34s} {row['mean']:8.4f} {row['sd']:8.4f}")


if __name__ == "__main__":
    main()
