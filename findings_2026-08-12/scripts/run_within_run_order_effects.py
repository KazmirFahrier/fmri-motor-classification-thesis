#!/usr/bin/env python3
"""First versus second presentation of each class within a run.

Checking the design revealed that every run is **palindromic**: four classes followed
by their mirror, e.g. `sub-01` run 1 is `[1, 3, 0, 2, 2, 0, 3, 1]`. That is what
guarantees exactly two events per class per run, and it is the structure the frozen
repetition-consistency decoder exploits when it assumes two events of each class.

It also means the two presentations of a class sit at **mirrored positions**: the
class appearing first appears last, the second appears second-to-last. So position in
the run is confounded with repetition in a specific, known way, and nobody has checked
whether the two presentations differ.

Three questions, all cheap and all from existing checkpoints:

1. **Amplitude.** Does the second presentation evoke a weaker response? Repetition
   suppression is well documented and would show up as reduced pattern magnitude.
2. **Pattern.** Do the two presentations carry the *same* spatial pattern, or does the
   representation drift? Measured as the correlation between first and second
   presentation patterns, against a same-class-different-run baseline.
3. **Decodability.** Is one presentation easier to classify than the other?

Any of these mattering has consequences. The repetition-consistency decoder treats
the two events of a class as exchangeable evidence; if they systematically differ, it
could weight them, and the design confound between position and repetition would need
stating in the manuscript.
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

from run_detrended_pair_feature_selection import load_checkpoints  # noqa: E402
from run_learned_temporal_filter_hierarchy import preprocess_sequence  # noqa: E402


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Within-run repetition and order effects.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip centering/detrending; amplitude effects survive it poorly.")
    args = parser.parse_args()

    feature_dict, y, records = load_checkpoints(
        Path(args.checkpoint_dir), [args.sequence_key]
    )
    sequence = feature_dict.pop(args.sequence_key)
    if not args.no_preprocess:
        sequence, _ = preprocess_sequence(sequence, records)
    x = sequence.mean(axis=1, dtype=np.float32)
    del sequence

    # Order events within each subject-run by onset, then label each class occurrence
    # as first or second.
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(str(record["subject_id"]), int(record["run_id"]))].append(index)

    palindromic = 0
    total_runs = 0
    pairs = []
    for key, indices in sorted(grouped.items()):
        ordered = sorted(indices, key=lambda i: int(records[i]["event_start"]))
        labels = [int(y[i]) for i in ordered]
        total_runs += 1
        if labels == labels[::-1]:
            palindromic += 1
        seen: dict[int, int] = {}
        for position, index in enumerate(ordered):
            label = int(y[index])
            if label in seen:
                pairs.append(
                    {
                        "subject": key[0],
                        "run": key[1],
                        "class_id": label,
                        "first_index": seen[label],
                        "second_index": index,
                        "first_position": ordered.index(seen[label]),
                        "second_position": position,
                    }
                )
            else:
                seen[label] = index

    print(f"palindromic runs: {palindromic}/{total_runs}", flush=True)

    first = np.array([p["first_index"] for p in pairs])
    second = np.array([p["second_index"] for p in pairs])

    # 1. amplitude
    amp_first = np.linalg.norm(x[first], axis=1)
    amp_second = np.linalg.norm(x[second], axis=1)
    ratio = amp_second / np.maximum(amp_first, 1e-9)

    # 2. pattern similarity, against a same-class-different-run baseline
    nf, ns = l2_normalize(x[first]), l2_normalize(x[second])
    within_pair = np.sum(nf * ns, axis=1)

    by_subject_class = defaultdict(list)
    for p in pairs:
        by_subject_class[(p["subject"], p["class_id"])].append(p)
    across_run = []
    rng = np.random.default_rng(0)
    for entries in by_subject_class.values():
        if len(entries) < 2:
            continue
        for _ in range(4):
            a, b = rng.choice(len(entries), size=2, replace=False)
            u = l2_normalize(x[entries[a]["first_index"]])
            v = l2_normalize(x[entries[b]["second_index"]])
            across_run.append(float(u @ v))
    across_run = np.asarray(across_run)

    rng2 = np.random.default_rng(1)
    boot = rng2.choice(ratio, size=(20000, len(ratio)), replace=True).mean(axis=1)
    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "preprocessed": not args.no_preprocess,
        "palindromic_runs": f"{palindromic}/{total_runs}",
        "pair_count": len(pairs),
        "amplitude": {
            "mean_first": float(amp_first.mean()),
            "mean_second": float(amp_second.mean()),
            "mean_ratio_second_over_first": float(ratio.mean()),
            "ratio_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "suppression_present": bool(np.quantile(boot, 0.975) < 1.0),
        },
        "pattern_similarity": {
            "within_pair_same_run": float(within_pair.mean()),
            "same_class_across_runs": float(across_run.mean()),
            "difference": float(within_pair.mean() - across_run.mean()),
            "note": (
                "If the within-run pair is more similar than the same class across "
                "runs, the two presentations share run-specific structure beyond class "
                "identity, which is exactly what the repetition-consistency decoder "
                "exploits."
            ),
        },
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "checkpoint_dir"}, indent=2))


if __name__ == "__main__":
    main()
