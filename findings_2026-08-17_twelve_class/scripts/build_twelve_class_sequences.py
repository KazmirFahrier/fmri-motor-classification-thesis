#!/usr/bin/env python3
"""Extract all twelve movement conditions, not the four the project has used.

The dataset defines **twelve** movement conditions — toe, ankle, left leg, right leg,
finger, wrist, forearm, upper arm, jaw, lip, tongue, eyes — and every run contains
each of them exactly twice, for 24 movement blocks per run. The frozen pipeline
extracts eight of those, the two repeats each of left leg, right leg, forearm and
upper arm, so roughly two thirds of the available movement events have never been
used and the 12-class problem the dataset was built for has never been attempted.

See `../findings_2026-08-12/LITERATURE_REVIEW.md` for how this was found and verified
against the raw events files.

## Why this is the highest-value extension

The subject-count learning curve showed the cohort is **saturated**: one more subject
is worth `+0.0008`. The runs-per-subject curve was still climbing. Events per subject
is therefore the axis with headroom, and this triples it — 144 events per subject
instead of 48.

It also enables analyses the four-class design cannot support:

- **Somatotopic gradient.** The classical homunculus predicts an ordering from toe
  through leg, trunk, arm and hand to face. Whether representational geometry recovers
  that ordering is a neuroscience result, and the literature predicts it will do so
  only partially.
- **Face conditions as a reference.** Jaw, lip, tongue and eye are anatomically remote
  from limb representations and should separate easily, giving the upper bound the
  project currently lacks.
- **Within-limb gradients.** Toe and ankle extend the leg pair; wrist and finger extend
  the arm pair. This tests whether the fine within-limb stage — repeatedly identified
  as the bottleneck — improves or degrades with more graded classes.

## Comparability

The temporal window, the spatial transform, and the file layout are **identical** to
the frozen pipeline, so the four-class subset of this output should reproduce the
existing checkpoints. The only difference is which events are retained. That is
deliberate: it keeps every existing result directly comparable, and it means the
four-class rows can be verified against known numbers before any 12-class claim is
made.

Output is `(144, 8, 13824)` per subject with the same key names, so existing loaders
and decoders run against it unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (REPO_ROOT / "scripts",):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from sweep_continuous_bold_windows import reproduce_thesis_transform  # noqa: E402

BUCKET = "https://s3.amazonaws.com/openneuro.org/ds004044"

# Code 0 is rest and is excluded; 1-12 are the movement conditions. This mirrors the
# project's own TRIAL_TYPE_CODE_MAP so the four shared classes keep their identity.
CONDITIONS = {
    1: "Toe movements", 2: "Ankle movements", 3: "Left leg movements",
    4: "Right leg movements", 5: "Forearm movements", 6: "Upper arm movements",
    7: "Wrist movements", 8: "Finger movements", 9: "Eye movements",
    10: "Jaw movements", 11: "Lip movements", 12: "Tongue movements",
}
# The frozen four, kept as ids 0-3 so a four-class subset of this output lines up
# exactly with the existing checkpoints.
FROZEN_FOUR = [3, 4, 5, 6]
CLASS_ID = {code: index for index, code in enumerate(FROZEN_FOUR)}
for code in sorted(CONDITIONS):
    if code not in CLASS_ID:
        CLASS_ID[code] = len(CLASS_ID)


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


def load_all_events(path: Path, repetition_time: float) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    events = []
    for row in rows:
        try:
            code = int(float(str(row["trial_type"]).strip()))
        except ValueError:
            continue
        if code not in CONDITIONS:
            continue  # rest, or anything unrecognised
        onset = float(row["onset"])
        events.append({
            "code": code,
            "trial_type": CONDITIONS[code],
            "class_id": CLASS_ID[code],
            "onset": onset,
            "event_start": int(round(onset / repetition_time)),
        })
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description="Twelve-condition event sequence extraction.")
    ap.add_argument("--subjects-from", required=True,
                    help="Directory of existing sub-*.npz, used for the authoritative "
                         "subject list. The cohort is NOT contiguously numbered.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--run-count", type=int, default=6)
    ap.add_argument("--offset", type=int, default=3)
    ap.add_argument("--length", type=int, default=8)
    ap.add_argument("--repetition-time", type=float, default=2.0)
    ap.add_argument("--extraction-shape", nargs=3, type=int, default=[100, 100, 100])
    ap.add_argument("--feature-shape", nargs=3, type=int, default=[24, 24, 24])
    args = ap.parse_args()

    out_dir, work = Path(args.out_dir), Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    subjects = sorted(p.stem for p in Path(args.subjects_from).glob("sub-*.npz"))
    key = f"offset_{args.offset}_length_{args.length}_sequence"
    expected = args.run_count * len(CONDITIONS) * 2
    print(f"{len(subjects)} subjects; expecting {expected} events each", flush=True)

    started = time.time()
    for subject in subjects:
        destination = out_dir / f"{subject}.npz"
        if destination.exists():
            try:
                with np.load(destination, allow_pickle=False) as existing:
                    found = existing[key].shape[0]
            except Exception:  # noqa: BLE001
                found = -1
            if found == expected:
                print(f"{subject}: done already ({found} events)", flush=True)
                continue
            print(f"{subject}: incomplete ({found}/{expected}) - re-extracting", flush=True)
            destination.unlink(missing_ok=True)

        sequences, labels, records, ok = [], [], [], True
        for run_id in range(1, args.run_count + 1):
            stem = f"{subject}_ses-1_task-motor_run-{run_id}"
            raw_stem = f"{subject}_ses-1_task-motor_run-{run_id:02d}"
            events_local = work / f"{raw_stem}_events.tsv"
            if not events_local.exists() and not fetch(
                f"{BUCKET}/{subject}/ses-1/func/{raw_stem}_events.tsv", events_local
            ):
                ok = False
                break
            events = load_all_events(events_local, args.repetition_time)

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
            max_volume = data.shape[3] - 1
            usable = [
                e for e in events
                if e["event_start"] + args.offset + args.length - 1 <= max_volume
            ]
            needed = sorted({
                e["event_start"] + args.offset + step
                for e in usable for step in range(args.length)
            })
            # Transform each needed volume once; events share volumes only rarely here
            # but the cache costs nothing and mirrors the frozen extraction.
            cache = {
                v: reproduce_thesis_transform(
                    data[..., v], tuple(args.extraction_shape), tuple(args.feature_shape)
                ).reshape(-1)
                for v in needed
            }
            del data
            for e in usable:
                sequences.append(np.stack(
                    [cache[e["event_start"] + args.offset + s] for s in range(args.length)],
                    axis=0,
                ).astype(np.float32))
                labels.append(e["class_id"])
                records.append({
                    "subject_id": subject, "run_id": run_id,
                    "event_start": int(e["event_start"]), "class_id": int(e["class_id"]),
                    "code": int(e["code"]), "trial_type": e["trial_type"],
                })
            bold_local.unlink(missing_ok=True)
            print(f"{subject} run-{run_id}: {len(usable)} events", flush=True)

        if not ok:
            print(f"{subject}: FAILED, skipped", flush=True)
            continue

        array = np.stack(sequences).astype(np.float32)
        np.savez_compressed(
            destination, **{key: array},
            labels=np.asarray(labels, dtype=np.int64),
            records_json=json.dumps(records),
        )
        counts = np.bincount(np.asarray(labels), minlength=len(CONDITIONS))
        print(f"{subject}: saved {array.shape}, per-class {counts.tolist()} "
              f"[{time.time() - started:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
