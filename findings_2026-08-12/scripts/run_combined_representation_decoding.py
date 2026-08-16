#!/usr/bin/env python3
"""Decode from volumetric and surface features concatenated.

The hyperaligned surface sits `0.026` behind the volumetric baseline, and the
leading explanation is coverage: the volumetric bounding box retains subcortex and
cerebellum, the surface does not. The direct test is a ribbon-masked volumetric
decoder, but labelling the `24^3` bounding-box grid anatomically requires
re-downloading every subject's surfaces *and* BOLD affines, several gigabytes of
transfer for one control.

Concatenating the two representations bounds the same question much more cheaply,
and asks something independently useful.

- If the combined decoder is **no better than volumetric alone**, the surface carries
  no information the volumetric representation lacks. That is what one expects if the
  volumetric grid already contains everything the surface has, plus the structures
  the surface discards.
- If the combined decoder is **better than either**, the two are complementary: the
  surface contributes something — plausibly the improved within-subject signal and
  the functional correspondence — that the volumetric grid does not capture. That
  would also raise the project's headline number, which no other open item currently
  promises.

Both representations are standardised on training rows before concatenation so that
neither dominates the joint scale purely because of its units, and the joint feature
space is then fitted in the same exact dual basis used everywhere else.
"""
from __future__ import annotations

import argparse
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
    load_checkpoints,
    outer_splits,
)
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402
from run_standard_mvpa_baseline import (  # noqa: E402
    dual_basis,
    fit_projected,
    standardize,
)


def load_mean(checkpoint_dir: str, sequence_key: str):
    feature_dict, y, records = load_checkpoints(Path(checkpoint_dir), [sequence_key])
    sequence, _ = preprocess_sequence(feature_dict.pop(sequence_key), records)
    mean_x = sequence.mean(axis=1, dtype=np.float32)
    del sequence
    return mean_x, y, records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode from concatenated volumetric and surface features."
    )
    parser.add_argument("--volumetric-dir", required=True)
    parser.add_argument("--surface-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--models", nargs="+", default=["linear_svm", "logistic_l2"])
    parser.add_argument("--fixed-c", nargs="+", type=float, default=[0.0001, 0.01])
    parser.add_argument("--outer-fold-count", type=int, default=6)
    parser.add_argument("--subject-seeds", nargs="+", type=int, default=[11, 23, 37, 51, 71])
    parser.add_argument("--split-limit", type=int)
    args = parser.parse_args()

    fixed_c = dict(zip(args.models, args.fixed_c))
    vol_x, y_vol, records = load_mean(args.volumetric_dir, args.sequence_key)
    sur_x, y_sur, records_sur = load_mean(args.surface_dir, args.sequence_key)

    # The two checkpoint sets are built independently, so event correspondence is
    # verified rather than assumed. A silent misalignment here would invent a result.
    if len(records) != len(records_sur):
        raise SystemExit(f"{len(records)} volumetric events vs {len(records_sur)} surface")
    if not np.array_equal(y_vol, y_sur):
        raise SystemExit("Label vectors differ between representations.")
    for left, right in zip(records, records_sur):
        if (
            str(left["subject_id"]) != str(right["subject_id"])
            or int(left["run_id"]) != int(right["run_id"])
            or int(left["event_start"]) != int(right["event_start"])
        ):
            raise SystemExit(f"Event mismatch: {left} vs {right}")
    print(
        f"verified {len(records)} matched events; "
        f"volumetric {vol_x.shape[1]} + surface {sur_x.shape[1]} features",
        flush=True,
    )

    y = y_vol
    splits = outer_splits(records, "subject", args.outer_fold_count, args.subject_seeds)
    if args.split_limit is not None:
        splits = splits[: args.split_limit]

    rows = []
    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        blocks_train, blocks_val = [], []
        for source in (vol_x, sur_x):
            mean, scale = standardize(source, train_idx)
            blocks_train.append((source[train_idx] - mean) / scale)
            blocks_val.append((source[val_idx] - mean) / scale)
        combined_train = np.concatenate(blocks_train, axis=1)
        combined_val = np.concatenate(blocks_val, axis=1)
        z_train, z_val = dual_basis(combined_train, combined_val)
        kernel_train = z_train @ z_train.T
        kernel_val = z_val @ z_train.T
        for model in args.models:
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
        shown = [
            r for r in rows if r["split"] == split["split"]
            and r["model"] == "linear_svm" and r["prediction_rule"] == "independent"
        ]
        print(f"{split['split']} linear_svm indep={shown[0]['balanced_accuracy']:.4f}", flush=True)

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
                "volumetric_dir": args.volumetric_dir,
                "surface_dir": args.surface_dir,
                "feature_count": int(vol_x.shape[1] + sur_x.shape[1]),
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
