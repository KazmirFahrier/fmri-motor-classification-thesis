"""Benchmark Kaggle as an extraction host for this project.

Locally, extraction is bound by download bandwidth: the process sits at under 1% CPU
and the link saturates at 14.1 MB/s, so four parallel workers bought 2x rather than 4x.
The question this answers is whether Kaggle's datacenter link to the OpenNeuro S3
bucket lifts that ceiling enough to be worth moving extraction off the local machine,
and — critically — whether the CPU cost of the spatial transform becomes the new
bottleneck once bandwidth is no longer the limit.

That second question is the one that decides the design. With a fast link and only a
few cores, a Kaggle run could easily be *slower* than the local one, and the honest way
to find out is to measure both halves separately rather than assume.

Nothing here writes results the project will cite. It is a timing probe.
"""
from __future__ import annotations

import json
import shutil
import time
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

BUCKET = "https://s3.amazonaws.com/openneuro.org/ds004044"
SUBJECT, RUNS = "sub-01", (1, 2)


def zscore(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    mean, std = float(volume.mean()), float(volume.std())
    if std < 1e-6:
        return (volume - mean).astype(np.float32, copy=False)
    return ((volume - mean) / std).astype(np.float32, copy=False)


def resize(volume: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    if tuple(volume.shape) == target:
        return volume.astype(np.float32, copy=False)
    factors = [t / s for t, s in zip(target, volume.shape)]
    return zoom(volume, factors, order=1).astype(np.float32, copy=False)


def transform(volume: np.ndarray) -> np.ndarray:
    return zscore(resize(zscore(resize(volume, (100, 100, 100))), (24, 24, 24)))


report = {"cpu_count": None, "runs": []}
try:
    import os
    report["cpu_count"] = os.cpu_count()
except Exception:
    pass

work = Path("/kaggle/working/bench")
work.mkdir(parents=True, exist_ok=True)

for run_id in RUNS:
    stem = f"{SUBJECT}_ses-1_task-motor_run-{run_id}"
    url = (
        f"{BUCKET}/derivatives/fmriprep/{SUBJECT}/"
        f"{stem}_space-T1w_desc-preproc_bold_denoised.nii.gz"
    )
    local = work / f"{stem}.nii.gz"

    started = time.time()
    with urllib.request.urlopen(url, timeout=600) as response, local.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    download_seconds = time.time() - started
    size_mb = local.stat().st_size / 1048576

    started = time.time()
    image = nib.load(str(local))
    data = np.asanyarray(image.dataobj, dtype=np.float32)
    load_seconds = time.time() - started

    # Transform 40 volumes and extrapolate, so the probe stays short.
    sample = min(40, data.shape[3])
    started = time.time()
    for index in range(sample):
        transform(data[..., index])
    per_volume = (time.time() - started) / sample

    entry = {
        "run": run_id,
        "size_mb": round(size_mb, 1),
        "download_seconds": round(download_seconds, 1),
        "download_mb_per_s": round(size_mb / download_seconds, 1),
        "decompress_load_seconds": round(load_seconds, 1),
        "volumes": int(data.shape[3]),
        "seconds_per_volume": round(per_volume, 4),
        "projected_transform_seconds": round(per_volume * data.shape[3], 1),
    }
    entry["projected_total_seconds_per_run"] = round(
        entry["download_seconds"] + entry["decompress_load_seconds"]
        + entry["projected_transform_seconds"], 1
    )
    report["runs"].append(entry)
    print(json.dumps(entry, indent=2), flush=True)

    del data
    local.unlink(missing_ok=True)

mean_total = float(np.mean([r["projected_total_seconds_per_run"] for r in report["runs"]]))
report["mean_seconds_per_run"] = round(mean_total, 1)
# 62 subjects x 6 runs is the full cohort pass.
report["projected_full_cohort_hours"] = round(mean_total * 372 / 3600, 2)
report["local_reference"] = {
    "mb_per_s": 14.1,
    "runs_per_minute_four_workers": 2.66,
    "note": "local process sat at under 1% CPU; bandwidth-bound",
}

Path("/kaggle/working/benchmark.json").write_text(json.dumps(report, indent=2))
print("\n=== SUMMARY ===")
print(f"cpus                     {report['cpu_count']}")
print(f"mean seconds per run     {report['mean_seconds_per_run']}")
print(f"projected cohort hours   {report['projected_full_cohort_hours']}")
print(f"local link was 14.1 MB/s; Kaggle measured "
      f"{np.mean([r['download_mb_per_s'] for r in report['runs']]):.1f} MB/s")
