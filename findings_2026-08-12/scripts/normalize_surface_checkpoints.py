#!/usr/bin/env python3
"""Apply the frozen pipeline's per-volume normalisation to surface checkpoints.

The volumetric pipeline z-scores **each volume across the whole grid**, twice, inside
`reproduce_thesis_transform`. The surface extraction deliberately stores raw sampled
BOLD instead, because the correct normalisation depends on which vertices are valid
and that is not known until the field-of-view mask exists.

Without this step the comparison between representations is confounded: the
volumetric features are z-scored and the surface features are raw intensities on the
order of several hundred, so subject-level and run-level scale differences that the
volumetric path removes would survive in the surface path. Any accuracy difference
would then be partly about normalisation rather than about spatial correspondence.

Two details matter:

- Normalisation is computed **over valid vertices only**. Invalid vertices are stored
  as zero, and including them would drag the mean toward zero by an amount that
  varies with each subject's field-of-view coverage — reintroducing exactly the kind
  of between-subject nuisance the step is meant to remove.
- Invalid vertices are left at zero afterwards. After centring, zero is the mean, so
  they contribute nothing to a covariance or an inner product rather than acting as
  an extreme value.

The output directory is a drop-in for `load_checkpoints`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalize_sequence(
    sequence: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Z-score each (event, lag) volume across valid vertices."""
    out = np.zeros_like(sequence)
    values = sequence[:, :, valid]
    mean = values.mean(axis=2, keepdims=True)
    std = values.std(axis=2, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    out[:, :, valid] = (values - mean) / std
    return out.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Z-score surface checkpoints per volume over valid vertices."
    )
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument(
        "--intersect-valid",
        action="store_true",
        help=(
            "Restrict every subject to the intersection of all subjects' valid "
            "vertices, so the feature space is identical across subjects."
        ),
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(in_dir.glob("sub-*.npz"))
    if not paths:
        raise SystemExit(f"No checkpoints in {in_dir}")

    shared = None
    if args.intersect_valid:
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                valid = data["valid_vertices"].astype(bool)
            shared = valid if shared is None else (shared & valid)
        print(
            f"shared valid vertices: {int(shared.sum())} of {len(shared)} "
            f"({shared.mean():.4f})",
            flush=True,
        )

    summaries = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            sequence = data[args.sequence_key].astype(np.float32)
            labels = data["labels"]
            records = str(data["records_json"])
            valid = data["valid_vertices"].astype(bool)
        use = shared if shared is not None else valid
        normalized = normalize_sequence(sequence, use)
        np.savez_compressed(
            out_dir / path.name,
            **{args.sequence_key: normalized},
            labels=labels,
            valid_vertices=use,
            records_json=records,
        )
        summaries.append(
            {
                "subject": path.stem,
                "valid_fraction": float(use.mean()),
                "std_over_valid": float(normalized[:, :, use].std()),
            }
        )
        print(f"{path.stem}: valid {use.mean():.4f}", flush=True)

    (out_dir / "normalization_summary.json").write_text(
        json.dumps(
            {
                "source": str(in_dir),
                "intersect_valid": args.intersect_valid,
                "shared_valid_count": int(shared.sum()) if shared is not None else None,
                "subjects": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
