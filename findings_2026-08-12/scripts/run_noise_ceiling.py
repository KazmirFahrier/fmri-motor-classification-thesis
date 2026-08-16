#!/usr/bin/env python3
"""Estimate how much accuracy the data can support at all.

Every accuracy in this project is reported against a `0.25` chance floor and no
ceiling, so "the frozen decoder beats standard MVPA by `+0.026`" has no scale. If
measurement noise caps achievable accuracy near `0.85`, then `0.8314` is close to
optimal and that margin is most of what remains. If the cap is `0.95`, the margin is
genuinely modest. The two readings support different papers.

Two complementary estimates, both from existing checkpoints.

## 1. Split-half pattern reliability

Each subject's per-class mean response is computed from odd runs and from even runs,
and the two are correlated. Spearman-Brown corrects the correlation from half-length
to full-length, giving the reliability of a class pattern estimated from all six runs.
This is the standard neuroimaging noise ceiling and it bounds how well *any* method
can characterise a subject's response.

## 2. Centroid-count extrapolation

Reliability is not accuracy, so a second estimate works in accuracy units directly.
Within each subject, events are classified against class centroids built from `k` of
that subject's other runs, for `k = 1 … 5`. Accuracy rises with `k` because the
centroids denoise. Extrapolating that curve to infinite runs estimates the accuracy
attainable against a *perfectly estimated* template for that subject — an upper bound
on what any amount of training data could buy, holding the representation fixed.

This is deliberately a **within-subject** bound. It removes the cross-subject transfer
problem entirely, so it is an optimistic ceiling: no cross-subject decoder should be
expected to exceed it, and the gap between it and the observed cross-subject accuracy
is the part attributable to transfer rather than to noise.
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

from run_detrended_pair_feature_selection import load_checkpoints  # noqa: E402
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def spearman_brown(correlation: float, factor: float = 2.0) -> float:
    if correlation <= -1.0:
        return -1.0
    return factor * correlation / (1.0 + (factor - 1.0) * correlation)


def main() -> None:
    parser = argparse.ArgumentParser(description="Noise ceiling for the frozen representation.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--label", default="volumetric")
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    x = sequence.mean(axis=1, dtype=np.float32)
    del sequence
    subjects = np.asarray([str(r["subject_id"]) for r in records])
    runs = np.asarray([int(r["run_id"]) for r in records])
    class_count = int(y.max()) + 1
    subject_list = sorted(set(subjects.tolist()))

    # --- 1. split-half reliability -------------------------------------------------
    reliabilities = []
    for subject in subject_list:
        member = subjects == subject
        odd = member & (runs % 2 == 1)
        even = member & (runs % 2 == 0)
        per_class = []
        for class_id in range(class_count):
            a = x[odd & (y == class_id)].mean(axis=0)
            b = x[even & (y == class_id)].mean(axis=0)
            a = a - a.mean()
            b = b - b.mean()
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            per_class.append(float(a @ b / denom) if denom > 0 else 0.0)
        reliabilities.append(
            {
                "subject": subject,
                "half_correlation": float(np.mean(per_class)),
                "full_reliability": spearman_brown(float(np.mean(per_class))),
                "per_class": [round(v, 4) for v in per_class],
            }
        )
        print(f"{subject}: half r={np.mean(per_class):+.4f}", flush=True)

    half = np.array([r["half_correlation"] for r in reliabilities])
    full = np.array([r["full_reliability"] for r in reliabilities])

    # --- 2. centroid-count extrapolation -------------------------------------------
    rng = np.random.default_rng(0)
    curve = {}
    for k in range(1, 6):
        accuracies = []
        for subject in subject_list:
            member = np.flatnonzero(subjects == subject)
            subject_runs = sorted(set(runs[member].tolist()))
            for held_out in subject_runs:
                others = [r for r in subject_runs if r != held_out]
                for _ in range(3):
                    picked = rng.choice(others, size=k, replace=False)
                    train = member[np.isin(runs[member], picked)]
                    val = member[runs[member] == held_out]
                    centroids = np.stack(
                        [
                            x[train][y[train] == c].mean(axis=0)
                            if np.any(y[train] == c)
                            else np.zeros(x.shape[1], dtype=np.float32)
                            for c in range(class_count)
                        ]
                    )
                    scores = l2_normalize(x[val]) @ l2_normalize(centroids).T
                    accuracies.append(float((scores.argmax(1) == y[val]).mean()))
        curve[k] = float(np.mean(accuracies))
        print(f"centroids from k={k} runs: {curve[k]:.4f}", flush=True)

    # Extrapolate accuracy = A - B/k to infinite runs.
    ks = np.array(sorted(curve), dtype=float)
    vals = np.array([curve[int(k)] for k in ks])
    design = np.stack([np.ones_like(ks), 1.0 / ks], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, vals, rcond=None)
    ceiling = float(coefficients[0])

    payload = {
        "label": args.label,
        "checkpoint_dir": args.checkpoint_dir,
        "split_half_reliability": {
            "mean_half_correlation": float(half.mean()),
            "mean_full_reliability_spearman_brown": float(full.mean()),
            "median_full_reliability": float(np.median(full)),
            "min": float(full.min()),
            "max": float(full.max()),
            "per_subject": reliabilities,
        },
        "centroid_extrapolation": {
            "curve": {str(k): round(v, 6) for k, v in curve.items()},
            "model": "accuracy = A - B/k fitted over k = 1..5 runs of centroid data",
            "estimated_ceiling": round(ceiling, 6),
            "caveat": (
                "Within-subject and therefore optimistic: it removes cross-subject "
                "transfer entirely. No cross-subject decoder should be expected to "
                "exceed it; the gap between it and observed cross-subject accuracy is "
                "the share attributable to transfer rather than to measurement noise."
            ),
        },
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"\nmean split-half r = {half.mean():.4f}")
    print(f"Spearman-Brown full reliability = {full.mean():.4f}")
    print(f"centroid curve = {[round(curve[k], 4) for k in sorted(curve)]}")
    print(f"extrapolated within-subject ceiling = {ceiling:.4f}")


if __name__ == "__main__":
    main()
