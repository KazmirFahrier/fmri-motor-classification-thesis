"""Extract at several spatial grid resolutions to quantify a disclosed design choice.

The manuscript discloses that the `24^3` feature grid was chosen with all 62 subjects
visible, alongside the temporal window and the covariance caps. Two of those have since
been quantified — nesting removed the temporal window's entire apparent gain and more
than half of the ANOVA threshold's — but the grid never has, because unlike those it
cannot be tested without re-extracting the cohort.

That makes it the last cheap disclosed choice still unmeasured, and re-extraction is
precisely the work worth moving off the local machine.

The grid sweep found `32^3` beats the disclosed `24^3` by `+0.0135` on the four-class
problem. That was measured on the frozen four, which are three of the four worst-decoded
conditions in the dataset and an unusually confusable subset — the same reason the
preprocessing decomposition had to be repeated at twelve classes before it could carry
the paper.

This extracts all twelve conditions at `32^3` so the resolution finding can be checked
against the full problem rather than a hard corner of it.

The temporal window, the extraction shape and the class mapping are all identical to the
frozen pipeline. The grid is the only thing that varies.
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

SHARD, SHARDS = 1, 3
OFFSET, LENGTH = 3, 8
REPETITION_TIME = 2.0
EXTRACTION_SHAPE = (100, 100, 100)
GRIDS = [(32, 32, 32)]
BUCKET = "https://s3.amazonaws.com/openneuro.org/ds004044"

SUBJECTS = [f"sub-{n:02d}" for n in (
    list(range(1, 15)) + [16, 17, 18] + list(range(20, 28)) + list(range(29, 40))
    + list(range(42, 64)) + [65, 66, 67, 68]
)]
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
    out_dir = Path("/kaggle/working/seq"); out_dir.mkdir(parents=True, exist_ok=True)
    work = Path("/kaggle/working/tmp"); work.mkdir(parents=True, exist_ok=True)
    subjects = SUBJECTS[SHARD::SHARDS]
    print(f"shard {SHARD}/{SHARDS}: {len(subjects)} subjects, grids {GRIDS}", flush=True)

    started = time.time()
    for subject in subjects:
        destination = out_dir / f"{subject}.npz"
        if destination.exists():
            print(f"{subject}: already done", flush=True); continue

        per_grid = {g: [] for g in GRIDS}
        labels, records, ok = [], [], True
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
                if code in CLASS_ID:
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
            needed = sorted({e["event_start"] + OFFSET + s
                             for e in usable for s in range(LENGTH)})
            # The 100^3 intermediate is shared by every grid, so it is computed once.
            cache = {}
            for v in needed:
                intermediate = zscore(resize(data[..., v], EXTRACTION_SHAPE))
                cache[v] = {g: zscore(resize(intermediate, g)).reshape(-1) for g in GRIDS}
            del data
            for e in usable:
                for g in GRIDS:
                    per_grid[g].append(np.stack(
                        [cache[e["event_start"] + OFFSET + s][g] for s in range(LENGTH)],
                        axis=0).astype(np.float32))
                labels.append(e["class_id"])
                records.append({"subject_id": subject, "run_id": run_id,
                                "event_start": int(e["event_start"]),
                                "class_id": int(e["class_id"])})
            bold_local.unlink(missing_ok=True)
            print(f"{subject} run-{run_id}: {len(usable)} events", flush=True)

        if not ok:
            print(f"{subject}: FAILED", flush=True); continue
        payload = {
            f"grid{g[0]}_offset_{OFFSET}_length_{LENGTH}_sequence":
                np.stack(per_grid[g]).astype(np.float32)
            for g in GRIDS
        }
        np.savez_compressed(destination, **payload,
                            labels=np.asarray(labels, dtype=np.int64),
                            records_json=json.dumps(records))
        print(f"{subject}: saved, {(time.time()-started)/60:.1f} min elapsed", flush=True)

    shutil.rmtree(work, ignore_errors=True)
    print(f"shard {SHARD} complete in {(time.time()-started)/60:.1f} min", flush=True)


main()
