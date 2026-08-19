"""Smooth at native resolution, the way neuroimaging normally does it.

Spatial smoothing is the third-largest preprocessing effect measured in this project —
`+0.0177` at four classes, `+0.0202` at twelve, both larger than any decoder difference.
But it is currently applied *after* the volume has been resampled to a `24^3` grid, as a
box filter over that grid. That is not what the field means by smoothing: standard
practice is a Gaussian kernel applied at the acquired resolution, before any
downsampling.

The distinction matters because the `24^3` resampling already averages a large
neighbourhood, so a post-hoc box filter is smoothing something twice while a native
Gaussian smooths the data once, correctly, at the scale the noise actually lives.

If native smoothing beats the post-hoc box filter, the project's largest remaining
preprocessing lever has been under-exploited, and the effect that already exceeds every
decoder difference gets larger. If it does not, the cheap post-hoc version is vindicated
and the pipeline needs no change — which is worth knowing either way, because a reviewer
will ask why smoothing was done on the downsampled grid.

Four kernels are extracted in one pass per volume, so the download is paid once. `sigma
0` is the unsmoothed control and must reproduce the frozen checkpoints exactly; if it
does not, nothing else here can be trusted.
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
from scipy.ndimage import gaussian_filter, zoom

SHARD, SHARDS = 1, 3
OFFSET, LENGTH = 3, 8
REPETITION_TIME = 2.0
EXTRACTION_SHAPE = (100, 100, 100)
FEATURE_SHAPE = (24, 24, 24)
# Sigma in voxels of the acquired volume. The data are ~2.4 mm isotropic, so these are
# roughly 0, 4, 6 and 8 mm FWHM — the range routinely used for motor-cortex analyses.
SIGMAS = [0.0, 0.7, 1.1, 1.4]
BUCKET = "https://s3.amazonaws.com/openneuro.org/ds004044"

SUBJECTS = [f"sub-{n:02d}" for n in (
    list(range(1, 15)) + [16, 17, 18] + list(range(20, 28)) + list(range(29, 40))
    + list(range(42, 64)) + [65, 66, 67, 68]
)]
CLASS_ID = {3: 0, 4: 1, 5: 2, 6: 3}


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
    print(f"shard {SHARD}/{SHARDS}: {len(subjects)} subjects, sigmas {SIGMAS}", flush=True)

    started = time.time()
    for subject in subjects:
        destination = out_dir / f"{subject}.npz"
        if destination.exists():
            print(f"{subject}: already done", flush=True); continue

        per_sigma = {sg: [] for sg in SIGMAS}
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
            # Smoothing happens on the acquired volume, before any resampling. The
            # sigma-0 entry skips the filter entirely rather than calling it with 0.
            cache = {}
            for v in needed:
                volume = data[..., v]
                entry = {}
                for sg in SIGMAS:
                    smoothed = volume if sg == 0.0 else gaussian_filter(volume, sigma=sg)
                    intermediate = zscore(resize(smoothed, EXTRACTION_SHAPE))
                    entry[sg] = zscore(resize(intermediate, FEATURE_SHAPE)).reshape(-1)
                cache[v] = entry
            del data
            for e in usable:
                for sg in SIGMAS:
                    per_sigma[sg].append(np.stack(
                        [cache[e["event_start"] + OFFSET + s][sg] for s in range(LENGTH)],
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
            f"sigma{str(sg).replace('.', 'p')}_offset_{OFFSET}_length_{LENGTH}_sequence":
                np.stack(per_sigma[sg]).astype(np.float32)
            for sg in SIGMAS
        }
        np.savez_compressed(destination, **payload,
                            labels=np.asarray(labels, dtype=np.int64),
                            records_json=json.dumps(records))
        print(f"{subject}: saved, {(time.time()-started)/60:.1f} min elapsed", flush=True)

    shutil.rmtree(work, ignore_errors=True)
    print(f"shard {SHARD} complete in {(time.time()-started)/60:.1f} min", flush=True)


main()
