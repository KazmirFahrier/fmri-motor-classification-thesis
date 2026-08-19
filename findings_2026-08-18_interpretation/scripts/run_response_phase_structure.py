#!/usr/bin/env python3
"""Is the late-lag block the haemodynamic undershoot?

Temporal generalization over the full 16-lag response shows two separate blocks. Lags
2-10 generalize broadly among themselves — the plateau the eight-lag window measured,
where the off/on-diagonal ratio is `0.93`. Lags 11-15 form a second block with its own
diagonal near `0.45`, well above the `0.25` chance level, and they **anti-generalize**
with the first: training at lag 12 and testing at lag 4 gives `0.123`, less than half of
chance.

Systematically-wrong prediction is not the signature of absent signal, which would sit
*at* chance. It is the signature of signal with an inverted sign — which is what the
post-stimulus BOLD undershoot would produce, since the response dips below baseline
after the peak and the spatial pattern reverses with it.

This tests that mechanism directly rather than inferring it from the accuracy matrix.
For each subject, class-mean patterns are computed at every lag and correlated across
lags. The undershoot account predicts a specific structure: strong positive correlation
within the plateau, strong **negative** correlation between plateau and late lags, and
positive correlation again within the late block.

The correlations are on class-mean patterns after the project's per-lag preprocessing.
That preprocessing centers and detrends each lag **independently**, so it cannot couple
one lag to another and cannot manufacture a cross-lag anti-correlation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "findings_2026-08-12" / "scripts",
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from run_detrended_pair_feature_selection import load_checkpoints  # noqa: E402
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402

# Three phases, not two. Lags 8-10 are the response crossing back through baseline and
# correlate positively with both neighbours, so including them in a "plateau" block
# averages the peak-versus-undershoot inversion away — which is exactly the mistake an
# earlier version of this analysis made. PEAK and UNDERSHOOT are kept disjoint and the
# transition is reported separately.
PEAK = range(3, 8)
TRANSITION = range(8, 11)
UNDERSHOOT = range(11, 16)
PLATEAU = range(2, 11)   # retained for continuity with the eight-lag window
LATE = range(11, 16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Response-phase pattern structure.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_0_length_16_sequence")
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence, _ = preprocess_sequence(feature_dict.pop(args.sequence_key), records)
    lag_count = sequence.shape[1]
    class_count = int(y.max()) + 1

    by_subject: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_subject[str(record["subject_id"])].append(index)

    # For each subject and lag, the **discriminative** component of the class means:
    # the across-class mean pattern is removed, leaving only what separates the classes.
    #
    # Correlating raw class means instead would be dominated by the class-common
    # activation, which is by far the largest component and is shared by every class.
    # Two lags can have highly similar overall patterns while the directions that
    # separate the classes differ completely, and it is only those directions a linear
    # decoder transports.
    patterns = []
    for _, rows in sorted(by_subject.items()):
        idx = np.asarray(rows)
        per_lag = []
        for lag in range(lag_count):
            block = sequence[idx, lag, :]
            centroids = np.stack([
                block[y[idx] == c].mean(axis=0) for c in range(class_count)
            ]).astype(np.float64)
            centroids -= centroids.mean(axis=0, keepdims=True)
            per_lag.append(centroids.ravel())
        patterns.append(np.stack(per_lag))
    patterns = np.stack(patterns)  # subject x lag x (class*feature)

    norms = np.linalg.norm(patterns, axis=2, keepdims=True)
    norms[norms < 1e-12] = 1.0
    unit = patterns / norms
    # subject-wise lag x lag correlation, then averaged
    correlation = np.einsum("slf,smf->slm", unit, unit).mean(axis=0)

    def block_mean(rows, cols, exclude_diagonal: bool) -> float:
        values = [
            correlation[i, j] for i in rows for j in cols
            if not (exclude_diagonal and i == j)
        ]
        return float(np.mean(values))

    # Class-to-class similarity across phases. A positive overall correlation with
    # below-chance transfer implies the class geometry **rotates**: if class A's late
    # pattern resembles class B's plateau pattern, a decoder carried across the boundary
    # misassigns systematically rather than guessing. The test is whether the matched
    # (same-class) similarity exceeds the mismatched (different-class) similarity.
    #
    # Note the across-class mean is already zero here: subject-run centering with two
    # events per class per run forces the centroids to sum to zero, so no further
    # removal of a class-common component is needed or possible.
    def phase_similarity(rows, cols):
        """Same-class versus different-class centroid similarity between two phases."""
        per_subject_m, per_subject_x = [], []
        for subject_index in range(patterns.shape[0]):
            m_vals, x_vals = [], []
            for i in rows:
                for j in cols:
                    a = patterns[subject_index, i].reshape(class_count, -1)
                    b = patterns[subject_index, j].reshape(class_count, -1)
                    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
                    similarity = a @ b.T
                    m_vals.append(float(np.diag(similarity).mean()))
                    x_vals.append(float(
                        similarity[~np.eye(class_count, dtype=bool)].mean()))
            per_subject_m.append(float(np.mean(m_vals)))
            per_subject_x.append(float(np.mean(x_vals)))
        m = np.asarray(per_subject_m); x = np.asarray(per_subject_x)
        return {
            "same_class": float(m.mean()),
            "different_class": float(x.mean()),
            "difference": float((m - x).mean()),
            "subjects_same_above_different": int(((m - x) > 0).sum()),
            "subjects_same_class_negative": int((m < 0).sum()),
            "subject_count": len(m),
        }

    phase_pairs = {
        "peak_vs_undershoot": phase_similarity(PEAK, UNDERSHOOT),
        "peak_vs_transition": phase_similarity(PEAK, TRANSITION),
        "transition_vs_undershoot": phase_similarity(TRANSITION, UNDERSHOOT),
        "plateau_vs_late_blockwide": phase_similarity(PLATEAU, LATE),
    }

    matched, mismatched = [], []
    per_subject_matched, per_subject_mismatched = [], []
    for subject_index in range(patterns.shape[0]):
        m_vals, x_vals = [], []
        for i in PLATEAU:
            for j in LATE:
                a = patterns[subject_index, i].reshape(class_count, -1)
                b = patterns[subject_index, j].reshape(class_count, -1)
                a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
                similarity = a @ b.T
                diagonal = np.diag(similarity)
                off = similarity[~np.eye(class_count, dtype=bool)]
                m_vals.append(float(diagonal.mean()))
                x_vals.append(float(off.mean()))
        per_subject_matched.append(float(np.mean(m_vals)))
        per_subject_mismatched.append(float(np.mean(x_vals)))
        matched.extend(m_vals); mismatched.extend(x_vals)

    difference = np.asarray(per_subject_matched) - np.asarray(per_subject_mismatched)
    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "phase_pairs": phase_pairs,
        "cross_phase_class_similarity": {
            "matched_same_class": float(np.mean(matched)),
            "mismatched_different_class": float(np.mean(mismatched)),
            "difference": float(difference.mean()),
            "subjects_with_positive_difference": int((difference > 0).sum()),
            "subject_count": len(difference),
        },
        "lag_count": lag_count,
        "correlation_matrix": correlation.tolist(),
        "plateau_lags": list(PLATEAU),
        "late_lags": list(LATE),
        "within_plateau": block_mean(PLATEAU, PLATEAU, True),
        "within_late": block_mean(LATE, LATE, True),
        "plateau_vs_late": block_mean(PLATEAU, LATE, False),
        "subject_count": patterns.shape[0],
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))

    print("class-mean pattern correlation across lags (mean over subjects)")
    print("      " + "".join(f"{j:>7}" for j in range(lag_count)))
    for i in range(lag_count):
        print(f"  L{i:<3}" + "".join(f"{correlation[i, j]:+7.2f}" for j in range(lag_count)))
    print(f"\nwithin plateau (lags 2-10)   {payload['within_plateau']:+.4f}")
    print(f"within late    (lags 11-15)  {payload['within_late']:+.4f}")
    print(f"plateau vs late              {payload['plateau_vs_late']:+.4f}")
    c = payload["cross_phase_class_similarity"]
    print("\ncross-phase class-to-class similarity (plateau vs late)")
    print(f"  same class        {c['matched_same_class']:+.4f}")
    print(f"  different class   {c['mismatched_different_class']:+.4f}")
    print(f"  difference        {c['difference']:+.4f}  "
          f"({c['subjects_with_positive_difference']}/{c['subject_count']} subjects positive)")
    print("\nby phase pair: same-class vs different-class centroid similarity")
    print("                              same     diff     n subj same<0")
    for name, entry in phase_pairs.items():
        print(f"  {name:<28} {entry['same_class']:+.4f}  {entry['different_class']:+.4f}"
              f"   {entry['subjects_same_class_negative']}/{entry['subject_count']}")


if __name__ == "__main__":
    main()
