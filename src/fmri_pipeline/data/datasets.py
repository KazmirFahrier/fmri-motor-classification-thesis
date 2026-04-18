from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def _ensure_3d(data: np.ndarray) -> np.ndarray:
    if data.ndim == 3:
        return data
    if data.ndim == 4:
        # Safety fallback if a 4D file is encountered; use first volume.
        return data[..., 0]
    raise ValueError(f"Expected 3D/4D NIfTI data, got shape={data.shape}")


def load_volume(filepath: str) -> np.ndarray:
    img = nib.load(filepath)
    data = img.get_fdata(dtype=np.float32)
    return _ensure_3d(data)


def resize_volume(volume: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    if list(volume.shape) == list(target_shape):
        return volume.astype(np.float32, copy=False)
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resized = zoom(volume, factors, order=1)
    return resized.astype(np.float32, copy=False)


def zscore_normalize(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = float(volume.mean())
    std = float(volume.std())
    if std < eps:
        return (volume - mean).astype(np.float32, copy=False)
    return ((volume - mean) / std).astype(np.float32, copy=False)


def random_crop_3d(
    volume: np.ndarray,
    crop_shape: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    d, h, w = volume.shape
    cd, ch, cw = crop_shape
    if cd > d or ch > h or cw > w:
        return volume

    d0 = int(rng.integers(0, d - cd + 1))
    h0 = int(rng.integers(0, h - ch + 1))
    w0 = int(rng.integers(0, w - cw + 1))
    return volume[d0 : d0 + cd, h0 : h0 + ch, w0 : w0 + cw]


def center_crop_3d(volume: np.ndarray, crop_shape: Sequence[int]) -> np.ndarray:
    d, h, w = volume.shape
    cd, ch, cw = crop_shape
    if cd > d or ch > h or cw > w:
        return volume

    d0 = (d - cd) // 2
    h0 = (h - ch) // 2
    w0 = (w - cw) // 2
    return volume[d0 : d0 + cd, h0 : h0 + ch, w0 : w0 + cw]


@dataclass(frozen=True)
class TransformConfig:
    target_shape: Tuple[int, int, int]
    normalization: str = "zscore"
    random_crop_shape: Optional[Tuple[int, int, int]] = None
    random_flip: bool = False


class VolumeDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        sample_ids: Sequence[int],
        class_names: Sequence[str],
        transform_cfg: TransformConfig,
        train: bool,
        seed: int,
    ) -> None:
        sample_set = set(int(s) for s in sample_ids)
        self.df = (
            manifest_df.loc[manifest_df["sample_id"].isin(sample_set)]
            .sort_values("sample_id")
            .reset_index(drop=True)
        )
        self.class_names = list(class_names)
        self.transform_cfg = transform_cfg
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def _apply_transforms(self, vol: np.ndarray) -> np.ndarray:
        cfg = self.transform_cfg
        vol = resize_volume(vol, cfg.target_shape)

        if cfg.random_crop_shape is not None:
            if self.train:
                vol = random_crop_3d(vol, cfg.random_crop_shape, self.rng)
            else:
                vol = center_crop_3d(vol, cfg.random_crop_shape)

        if cfg.random_flip and self.train:
            if float(self.rng.uniform()) < 0.5:
                vol = np.flip(vol, axis=0).copy()
            if float(self.rng.uniform()) < 0.5:
                vol = np.flip(vol, axis=1).copy()
            if float(self.rng.uniform()) < 0.5:
                vol = np.flip(vol, axis=2).copy()

        if cfg.normalization == "zscore":
            vol = zscore_normalize(vol)

        return vol.astype(np.float32, copy=False)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        vol = load_volume(str(row["filepath"]))
        vol = self._apply_transforms(vol)
        x = torch.from_numpy(vol).unsqueeze(0)  # [1, D, H, W]
        y = torch.tensor(int(row["class_id"]), dtype=torch.long)
        return x, y


class ClipDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        sample_ids: Sequence[int],
        class_names: Sequence[str],
        transform_cfg: TransformConfig,
        clip_length: int,
        clip_stride: int,
        clip_window_stride: int,
        hrf_shift: int,
        train: bool,
        seed: int,
    ) -> None:
        sample_set = set(int(s) for s in sample_ids)
        self.df = (
            manifest_df.loc[manifest_df["sample_id"].isin(sample_set)]
            .sort_values(["subject_id", "run_id", "class_id", "vol_id"])
            .reset_index(drop=True)
        )
        self.class_names = list(class_names)
        self.transform_cfg = transform_cfg
        self.clip_length = int(clip_length)
        self.clip_stride = int(clip_stride)
        self.clip_window_stride = int(clip_window_stride)
        self.hrf_shift = int(hrf_shift)
        self.train = train
        self.rng = np.random.default_rng(seed)

        self.clips: List[Dict[str, object]] = self._build_clips()
        if not self.clips:
            raise ValueError(
                "ClipDataset produced zero clips. Check clip_length/stride/hrf_shift against available volume IDs."
            )

    def _build_clips(self) -> List[Dict[str, object]]:
        clips: List[Dict[str, object]] = []
        for (subject_id, run_id, class_id), g in self.df.groupby(["subject_id", "run_id", "class_id"]):
            vol_to_fp = {
                int(row.vol_id): str(row.filepath)
                for row in g.itertuples(index=False)
            }
            sorted_vols = sorted(vol_to_fp.keys())
            if not sorted_vols:
                continue

            for start_idx in range(0, len(sorted_vols), self.clip_window_stride):
                raw_start = sorted_vols[start_idx]
                shifted_start = raw_start + self.hrf_shift
                target_vols = [
                    shifted_start + t * self.clip_stride
                    for t in range(self.clip_length)
                ]
                if all(v in vol_to_fp for v in target_vols):
                    clips.append(
                        {
                            "subject_id": str(subject_id),
                            "run_id": int(run_id),
                            "class_id": int(class_id),
                            "vol_ids": target_vols,
                            "filepaths": [vol_to_fp[v] for v in target_vols],
                        }
                    )
        return clips

    def __len__(self) -> int:
        return len(self.clips)

    def _apply_transforms_with_shared_crop(self, volumes: List[np.ndarray]) -> List[np.ndarray]:
        cfg = self.transform_cfg
        resized = [resize_volume(v, cfg.target_shape) for v in volumes]

        if cfg.random_crop_shape is not None:
            if self.train:
                d, h, w = resized[0].shape
                cd, ch, cw = cfg.random_crop_shape
                if cd <= d and ch <= h and cw <= w:
                    d0 = int(self.rng.integers(0, d - cd + 1))
                    h0 = int(self.rng.integers(0, h - ch + 1))
                    w0 = int(self.rng.integers(0, w - cw + 1))
                    resized = [
                        v[d0 : d0 + cd, h0 : h0 + ch, w0 : w0 + cw]
                        for v in resized
                    ]
            else:
                resized = [center_crop_3d(v, cfg.random_crop_shape) for v in resized]

        if cfg.random_flip and self.train:
            flip_axes = []
            for axis in (0, 1, 2):
                if float(self.rng.uniform()) < 0.5:
                    flip_axes.append(axis)
            if flip_axes:
                resized = [np.flip(v, axis=tuple(flip_axes)).copy() for v in resized]

        if cfg.normalization == "zscore":
            resized = [zscore_normalize(v) for v in resized]

        return [v.astype(np.float32, copy=False) for v in resized]

    def __getitem__(self, idx: int):
        clip = self.clips[idx]
        volumes = [load_volume(fp) for fp in clip["filepaths"]]
        volumes = self._apply_transforms_with_shared_crop(volumes)

        # [T, 1, D, H, W]
        x = torch.stack([torch.from_numpy(v).unsqueeze(0) for v in volumes], dim=0)
        y = torch.tensor(int(clip["class_id"]), dtype=torch.long)
        return x, y
