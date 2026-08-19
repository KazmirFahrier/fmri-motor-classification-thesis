"""Extract all twelve conditions across the full haemodynamic response.

The temporal generalization result showed the motor code is stationary — off-diagonal
generalization is 93% of on-diagonal — but it could only say that about the plateau and
the decay. The frozen checkpoints start at `offset_3`, so the **rising phase has never
been measured**, and the claim "the code is stationary" is currently a claim about two
thirds of the response.

`offset 0, length 16` is the longest window this design permits without losing data.
The latest event starts at volume 216 and each run has 232 volumes, so `216 + 15 = 231`
lands exactly on the final volume: **every one of the 2976 events survives**. Length 18
would drop 3% of them.

The window also strictly contains the frozen one — lags 3 through 10 here are the
existing `offset_3_length_8` checkpoints — so the four-class subset can be validated
against known numbers before any claim about the rising phase is made.

Sharded by subject. Kaggle allocates 4 CPUs and the transform costs about 21 s per run
against roughly 11 s of download, so the job is CPU-bound here rather than
bandwidth-bound as it is locally, and several shards in parallel is what makes it
worthwhile.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

SHARD, SHARDS = 0, 3
OFFSET, LENGTH = 0, 16
REPETITION_TIME = 2.0
EXTRACTION_SHAPE, FEATURE_SHAPE = (100, 100, 100), (24, 24, 24)
BUCKET = "https://s3.amazonaws.com/openneuro.org/ds004044"

# Non-contiguous: 15, 19, 28, 40, 41 and 64 are absent and the cohort runs to 68.
# Generating this with range() would request five subjects that do not exist and
# silently omit five that do.
SUBJECTS = [f"sub-{n:02d}" for n in (
    list(range(1, 15)) + [16, 17, 18] + list(range(20, 28)) + list(range(29, 40))
    + list(range(42, 64)) + [65, 66, 67, 68]
)]
# Frozen four keep ids 0-3 so that subset lines up with the existing checkpoints; the
# remaining eight follow in condition order, matching build_twelve_class_sequences.py.
CLASS_ID = {3: 0, 4: 1, 5: 2, 6: 3}
for _code in (1, 2, 7, 8, 9, 10, 11, 12):
    CLASS_ID[_code] = len(CLASS_ID)


def zscore(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    mean, std = float(volume.mean()), float(volume.std())
    if std < 1e-6:
        return (volume - mean).astype(np.float32, copy=False)
    return ((volume - mean) / std).astype(np.float32, copy=False)


def resize(volume: np.ndarray, target: tuple) -> np.ndarray:
    if tuple(volume.shape) == target:
        return volume.astype(np.float32, copy=False)
    return zoom(volume, [t / s for t, s in zip(target, volume.shape)],
                order=1).astype(np.float32, copy=False)


def transform(volume: np.ndarray) -> np.ndarray:
    return zscore(resize(zscore(resize(volume, EXTRACTION_SHAPE)), FEATURE_SHAPE)).reshape(-1)


def fetch(url: str, destination: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=600) as response, \
                    destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            return True
        except Exception as error:  # noqa: BLE001
            print(f"    fetch attempt {attempt + 1}: {error}", flush=True)
            time.sleep(3 * (attempt + 1))
    return False


def main() -> None:
    assert len(CLASS_ID) == 12, f"expected 12 conditions, got {len(CLASS_ID)}"
    out_dir = Path("/kaggle/working/seq"); out_dir.mkdir(parents=True, exist_ok=True)
    work = Path("/kaggle/working/tmp"); work.mkdir(parents=True, exist_ok=True)
    subjects = SUBJECTS[SHARD::SHARDS]
    key = f"offset_{OFFSET}_length_{LENGTH}_sequence"
    print(f"shard {SHARD}/{SHARDS}: {len(subjects)} subjects", flush=True)

    started = time.time()
    for subject in subjects:
        destination = out_dir / f"{subject}.npz"
        if destination.exists():
            print(f"{subject}: already done", flush=True)
            continue

        sequences, labels, records, ok = [], [], [], True
        for run_id in range(1, 7):
            stem = f"{subject}_ses-1_task-motor_run-{run_id}"
            raw_stem = f"{subject}_ses-1_task-motor_run-{run_id:02d}"

            events_local = work / f"{raw_stem}_events.tsv"
            if not events_local.exists() and not fetch(
                f"{BUCKET}/{subject}/ses-1/func/{raw_stem}_events.tsv", events_local
            ):
                ok = False; break
            with events_local.open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            events = []
            for row in rows:
                try:
                    code = int(float(str(row["trial_type"]).strip()))
                except ValueError:
                    continue
                if code not in CLASS_ID:
                    continue
                events.append({
                    "class_id": CLASS_ID[code],
                    "event_start": int(round(float(row["onset"]) / REPETITION_TIME)),
                })

            bold_local = work / f"{stem}_bold.nii.gz"
            if not fetch(
                f"{BUCKET}/derivatives/fmriprep/{subject}/"
                f"{stem}_space-T1w_desc-preproc_bold_denoised.nii.gz", bold_local
            ):
                ok = False; break

            data = np.asanyarray(nib.load(str(bold_local)).dataobj, dtype=np.float32)
            max_volume = data.shape[3] - 1
            usable = [e for e in events
                      if e["event_start"] + OFFSET + LENGTH - 1 <= max_volume]
            cache = {
                v: transform(data[..., v])
                for v in sorted({e["event_start"] + OFFSET + s
                                 for e in usable for s in range(LENGTH)})
            }
            del data
            for e in usable:
                sequences.append(np.stack(
                    [cache[e["event_start"] + OFFSET + s] for s in range(LENGTH)],
                    axis=0).astype(np.float32))
                labels.append(e["class_id"])
                records.append({"subject_id": subject, "run_id": run_id,
                                "event_start": int(e["event_start"]),
                                "class_id": int(e["class_id"])})
            bold_local.unlink(missing_ok=True)
            print(f"{subject} run-{run_id}: {len(usable)}/{len(events)} events",
                  flush=True)

        if not ok:
            print(f"{subject}: FAILED", flush=True); continue
        np.savez_compressed(
            destination, **{key: np.stack(sequences).astype(np.float32)},
            labels=np.asarray(labels, dtype=np.int64),
            records_json=json.dumps(records))
        elapsed = time.time() - started
        print(f"{subject}: saved {len(sequences)} events, {elapsed/60:.1f} min elapsed",
              flush=True)

    shutil.rmtree(work, ignore_errors=True)
    print(f"shard {SHARD} complete in {(time.time()-started)/60:.1f} min", flush=True)


main()
