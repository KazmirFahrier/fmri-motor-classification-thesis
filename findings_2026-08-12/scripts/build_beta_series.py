#!/usr/bin/env python3
"""Least-Squares-Separate (LSS) trial-wise beta estimation.

The canonical MVPA feature is a per-trial GLM beta, not a raw window mean. This
project has only ever used window means, and a beta series is absent from all 129
commits. A reviewer will expect the comparison regardless of which wins.

## Why the existing checkpoints cannot answer this

The frozen checkpoints hold eight volumes per event, offsets 3 to 10. Events are 16 s
apart and the haemodynamic response extends well past that, so adjacent trials
overlap. A window mean silently attributes the tail of the previous trial to the
current one. LSS exists precisely to separate them, and doing so requires the
continuous run time series, which the checkpoints do not retain. Hence the
re-extraction.

## LSS

For each trial a design matrix is built with:

- one regressor for **that trial alone**,
- one regressor for **all other trials combined**,
- polynomial drift terms.

The trial's own beta is taken from that fit, and the process repeats per trial. This
is Mumford's LSS, and it is preferred over fitting all trials in a single model (LSA)
when trials are closely spaced, because LSA's collinearity between adjacent trial
regressors inflates the variance of every estimate.

Betas are passed through the **same** `reproduce_thesis_transform` as the frozen
pipeline, so the only thing that differs from the existing checkpoints is how each
trial's amplitude is estimated. Output shape is `(48, 1, 13824)` — one map per trial
rather than eight lags — which flows through the existing loaders unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (REPO_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from sweep_continuous_bold_windows import (  # noqa: E402
    load_events,
    reproduce_thesis_transform,
)


BUCKET = "https://s3.amazonaws.com/openneuro.org/ds004044"


def fetch(url: str, destination: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=600) as response, destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            return True
        except Exception as error:  # noqa: BLE001
            print(f"    fetch attempt {attempt + 1}: {error}", flush=True)
            time.sleep(3 * (attempt + 1))
    return False


def double_gamma_hrf(times: np.ndarray) -> np.ndarray:
    """Canonical SPM-style double-gamma haemodynamic response."""
    from math import gamma as gamma_fn

    def gamma_pdf(t, shape, scale):
        out = np.zeros_like(t)
        positive = t > 0
        out[positive] = (
            (t[positive] ** (shape - 1) * np.exp(-t[positive] / scale))
            / (gamma_fn(shape) * scale**shape)
        )
        return out

    peak = gamma_pdf(times, 6.0, 1.0)
    undershoot = gamma_pdf(times, 16.0, 1.0)
    response = peak - undershoot / 6.0
    return response / np.max(np.abs(response))


def build_regressor(
    onsets: list[float],
    duration: float,
    volume_count: int,
    repetition_time: float,
    upsample: int = 16,
) -> np.ndarray:
    """Box-car at `onsets` convolved with the canonical HRF, sampled at TR."""
    fine_step = repetition_time / upsample
    fine_length = volume_count * upsample
    stick = np.zeros(fine_length)
    for onset in onsets:
        start = int(round(onset / fine_step))
        stop = int(round((onset + duration) / fine_step))
        stick[max(start, 0) : min(stop, fine_length)] = 1.0
    kernel = double_gamma_hrf(np.arange(0, 32, fine_step))
    convolved = np.convolve(stick, kernel)[:fine_length]
    return convolved[:: upsample][:volume_count]


def main() -> None:
    parser = argparse.ArgumentParser(description="LSS trial-wise beta series.")
    parser.add_argument("--subjects-from", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--run-count", type=int, default=6)
    parser.add_argument("--repetition-time", type=float, default=2.0)
    parser.add_argument("--event-duration", type=float, default=16.0)
    parser.add_argument("--drift-order", type=int, default=3)
    parser.add_argument("--extraction-shape", nargs=3, type=int, default=[100, 100, 100])
    parser.add_argument("--feature-shape", nargs=3, type=int, default=[24, 24, 24])
    args = parser.parse_args()

    out_dir, work = Path(args.out_dir), Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    subjects = sorted(p.stem for p in Path(args.subjects_from).glob("sub-*.npz"))
    print(f"{len(subjects)} subjects", flush=True)

    started = time.time()
    expected_events = args.run_count * 8
    for subject in subjects:
        destination = out_dir / f"{subject}.npz"
        if destination.exists():
            # Existence alone is not proof of completeness. A run interrupted after
            # fewer runs leaves a structurally valid file with too few events, and
            # skipping it silently corrupts the cohort -- the balanced assignment rule
            # requires eight events per run. Validate before skipping.
            try:
                with np.load(destination, allow_pickle=False) as existing:
                    found = existing["offset_3_length_8_sequence"].shape[0]
            except Exception:  # noqa: BLE001
                found = -1
            if found == expected_events:
                print(f"{subject}: done already ({found} events)", flush=True)
                continue
            print(
                f"{subject}: incomplete ({found} events, expected {expected_events}) "
                f"- re-extracting",
                flush=True,
            )
            destination.unlink(missing_ok=True)

        betas, labels, records = [], [], []
        ok = True
        for run_id in range(1, args.run_count + 1):
            stem = f"{subject}_ses-1_task-motor_run-{run_id}"
            raw_stem = f"{subject}_ses-1_task-motor_run-{run_id:02d}"
            events_local = work / f"{raw_stem}_events.tsv"
            if not events_local.exists() and not fetch(
                f"{BUCKET}/{subject}/ses-1/func/{raw_stem}_events.tsv", events_local
            ):
                ok = False
                break
            events = load_events(events_local, args.repetition_time)

            bold_local = work / f"{stem}_bold.nii.gz"
            if not fetch(
                f"{BUCKET}/derivatives/fmriprep/{subject}/"
                f"{stem}_space-T1w_desc-preproc_bold_denoised.nii.gz",
                bold_local,
            ):
                ok = False
                break

            image = nib.load(str(bold_local))
            data = np.asanyarray(image.dataobj, dtype=np.float32)
            volume_count = data.shape[3]
            flat = data.reshape(-1, volume_count).T  # (T, V)
            del data

            drift = np.stack(
                [
                    np.linspace(-1, 1, volume_count) ** power
                    for power in range(args.drift_order + 1)
                ],
                axis=1,
            )
            onsets = [event["onset"] for event in events]

            for index, event in enumerate(events):
                others = [o for j, o in enumerate(onsets) if j != index]
                own = build_regressor(
                    [event["onset"]], args.event_duration, volume_count, args.repetition_time
                )
                rest = (
                    build_regressor(others, args.event_duration, volume_count, args.repetition_time)
                    if others
                    else np.zeros(volume_count)
                )
                design = np.column_stack([own, rest, drift])
                solution, *_ = np.linalg.lstsq(design, flat, rcond=None)
                beta_volume = solution[0].reshape(image.shape[:3])
                betas.append(
                    reproduce_thesis_transform(
                        beta_volume,
                        tuple(args.extraction_shape),
                        tuple(args.feature_shape),
                    ).reshape(-1)
                )
                labels.append(event["class_id"])
                records.append(
                    {
                        "subject_id": subject,
                        "run_id": run_id,
                        "event_start": int(event["event_start"]),
                        "class_id": int(event["class_id"]),
                    }
                )
            bold_local.unlink(missing_ok=True)
            print(f"{subject} run-{run_id}: {len(events)} betas", flush=True)

        if not ok:
            print(f"{subject}: FAILED, skipped", flush=True)
            continue

        array = np.stack(betas).astype(np.float32)[:, None, :]  # (events, 1, features)
        np.savez_compressed(
            destination,
            **{"offset_3_length_8_sequence": array},
            labels=np.asarray(labels, dtype=np.int64),
            records_json=json.dumps(records),
        )
        print(
            f"{subject}: saved {array.shape} [{time.time() - started:.0f}s]", flush=True
        )


if __name__ == "__main__":
    main()
