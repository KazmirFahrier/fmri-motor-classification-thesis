from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from fmri_pipeline.data.datasets import ClipDataset, TransformConfig


def test_clip_dataset_contiguous_with_hrf_shift(tmp_path: Path):
    class_name = "Forearm movements"
    class_dir = tmp_path / "batch_01" / class_name
    class_dir.mkdir(parents=True)

    rows = []
    sample_id = 0
    subject_id = "sub-01"
    run_id = 1
    for vol_id in range(1, 7):
        filepath = class_dir / f"{subject_id}_run-{run_id}_vol-{vol_id:03d}.nii.gz"
        data = np.random.randn(8, 8, 8).astype(np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(filepath))
        rows.append(
            {
                "sample_id": sample_id,
                "filepath": str(filepath),
                "class_name": class_name,
                "class_id": 0,
                "subject_id": subject_id,
                "run_id": run_id,
                "vol_id": vol_id,
                "batch_id": "batch_01",
                "exists": True,
            }
        )
        sample_id += 1

    df = pd.DataFrame(rows)
    ds = ClipDataset(
        manifest_df=df,
        sample_ids=df["sample_id"].tolist(),
        class_names=[class_name],
        transform_cfg=TransformConfig(target_shape=(8, 8, 8), normalization="zscore"),
        clip_length=3,
        clip_stride=1,
        clip_window_stride=1,
        hrf_shift=1,
        train=False,
        seed=42,
    )

    assert len(ds) >= 1
    x, y = ds[0]
    assert x.shape == (3, 1, 8, 8, 8)
    assert int(y.item()) == 0


def test_clip_dataset_event_window_default_uses_unshifted_volumes(tmp_path: Path):
    class_name = "Left leg movements"
    class_dir = tmp_path / "batch_01" / class_name
    class_dir.mkdir(parents=True)

    rows = []
    subject_id = "sub-01"
    run_id = 1
    for sample_id, vol_id in enumerate(range(72, 80)):
        filepath = class_dir / f"{subject_id}_run-{run_id}_vol-{vol_id:03d}.nii.gz"
        data = np.random.randn(8, 8, 8).astype(np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(filepath))
        rows.append(
            {
                "sample_id": sample_id,
                "filepath": str(filepath),
                "class_name": class_name,
                "class_id": 0,
                "subject_id": subject_id,
                "run_id": run_id,
                "vol_id": vol_id,
                "batch_id": "batch_01",
                "exists": True,
            }
        )

    df = pd.DataFrame(rows)
    ds = ClipDataset(
        manifest_df=df,
        sample_ids=df["sample_id"].tolist(),
        class_names=[class_name],
        transform_cfg=TransformConfig(target_shape=(8, 8, 8), normalization="zscore"),
        clip_length=6,
        clip_stride=1,
        clip_window_stride=1,
        hrf_shift=0,
        train=False,
        seed=42,
    )

    assert ds.clips[0]["vol_ids"] == [72, 73, 74, 75, 76, 77]
