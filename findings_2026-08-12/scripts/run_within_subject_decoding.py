#!/usr/bin/env python3
"""Within-subject leave-one-run-out decoding, to separate representation quality
from cross-subject alignment.

The surface-projected representation decodes worse across subjects than the
volumetric one. Two very different explanations produce that same result:

1. **Alignment did not help enough** to pay for the cortex-only coverage the surface
   path accepts — it discards subcortex, cerebellum, and roughly a tenth of cortex
   that falls outside the EPI field of view.
2. **The projection degraded the data.** Ribbon sampling and barycentric resampling
   are smoothing operations; if they blurred away spatial detail, the surface
   representation is simply less informative, and cross-subject accuracy would drop
   for reasons that have nothing to do with alignment.

Cross-subject accuracy cannot distinguish these, but **within-subject** accuracy can.
Training and testing inside one subject removes inter-subject alignment from the
problem entirely, so what remains is how much class information the representation
carries at all.

- If within-subject accuracy is comparable between representations, the projection
  preserved the signal and the cross-subject gap is about alignment and coverage.
- If the surface is markedly worse within subject, the projection lost information
  and the cross-subject comparison is confounded by that loss rather than measuring
  alignment.

Leave-one-run-out is the natural split: it holds out an entire run, so no event from
a test run contributes to training, and the six runs give six folds per subject.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (REPO_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from run_balanced_event_assignment import apply_balanced_assignment, metrics  # noqa: E402
from run_detrended_pair_feature_selection import load_checkpoints  # noqa: E402
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402
from run_standard_mvpa_baseline import (  # noqa: E402
    correlation_centroid_scores,
    dual_basis,
    fit_projected,
    standardize,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-one-run-out decoding inside each subject."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--label", required=True, help="Name for this representation.")
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--models", nargs="+",
                        default=["linear_svm", "logistic_l2", "correlation_centroid"])
    parser.add_argument("--fixed-c", nargs="+", type=float, default=[0.0001, 0.01, 0.0])
    parser.add_argument("--subject-limit", type=int)
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
    runs = np.asarray([int(record["run_id"]) for record in records])
    subject_list = sorted(set(subjects.tolist()))
    if args.subject_limit:
        subject_list = subject_list[: args.subject_limit]

    rows = []
    for subject in subject_list:
        member = np.flatnonzero(subjects == subject)
        for held_out in sorted(set(runs[member].tolist())):
            val_idx = member[runs[member] == held_out]
            train_idx = member[runs[member] != held_out]
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
                            "subject": subject,
                            "held_out_run": int(held_out),
                            "model": model,
                            "prediction_rule": rule,
                            "balanced_accuracy": float(
                                metrics(y[val_idx], prediction)["balanced_accuracy"]
                            ),
                        }
                    )
        print(f"{subject} done", flush=True)

    summary = {}
    for model in args.models:
        for rule in ("independent", "balanced"):
            values = [
                r["balanced_accuracy"] for r in rows
                if r["model"] == model and r["prediction_rule"] == rule
            ]
            by_subject = defaultdict(list)
            for r in rows:
                if r["model"] == model and r["prediction_rule"] == rule:
                    by_subject[r["subject"]].append(r["balanced_accuracy"])
            per_subject = np.asarray([np.mean(v) for v in by_subject.values()])
            summary[f"{model}|{rule}"] = {
                "mean": float(np.mean(values)),
                "subject_mean": float(np.mean(per_subject)),
                "subject_sd": float(np.std(per_subject)),
                "fold_count": len(values),
            }

    payload = {
        "label": args.label,
        "checkpoint_dir": args.checkpoint_dir,
        "feature_count": int(mean_x.shape[1]),
        "protocol": "leave-one-run-out within each subject; no cross-subject transfer",
        "subject_count": len(subject_list),
        "summary": summary,
        "rows": rows,
        "note": (
            "Within-subject accuracy measures how much class information the "
            "representation carries, with inter-subject alignment removed from the "
            "problem. Comparing two representations here isolates representation "
            "quality from alignment."
        ),
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"\n{'model|rule':34s} {'mean':>8s} {'subj mean':>10s} {'subj sd':>8s}")
    for name, row in summary.items():
        print(f"{name:34s} {row['mean']:8.4f} {row['subject_mean']:10.4f} {row['subject_sd']:8.4f}")


if __name__ == "__main__":
    main()
