#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fmri_pipeline.data.datasets import ClipDataset, TransformConfig
from fmri_pipeline.data.manifest import build_manifest
from fmri_pipeline.utils.metrics import compute_classification_metrics


CLASS_NAMES = [
    "Left leg movements",
    "Right leg movements",
    "Forearm movements",
    "Upper arm movements",
]


def find_dataset_root(slug: str) -> Path | None:
    direct = Path("/kaggle/input") / slug
    if direct.exists():
        return direct
    datasets_root = Path("/kaggle/input/datasets")
    if datasets_root.exists():
        matches = sorted(p for p in datasets_root.rglob(slug) if p.is_dir())
        if matches:
            return matches[0]
    matches = sorted(p for p in Path("/kaggle/input").rglob(slug) if p.is_dir())
    return matches[0] if matches else None


def resolve_batch_roots(batch_slugs: Sequence[str]) -> List[Path]:
    roots: List[Path] = []
    missing: List[str] = []
    for slug in batch_slugs:
        root = find_dataset_root(slug)
        if root is None:
            missing.append(slug)
        else:
            roots.append(root)
    if missing:
        raise FileNotFoundError(f"Missing mounted batch datasets: {missing}")
    return roots


def make_clip_dataset(
    manifest_df,
    sample_ids: Sequence[int],
    target_shape: Sequence[int],
    clip_length: int,
    clip_stride: int,
    clip_window_stride: int,
    hrf_shift: int,
    seed: int,
) -> ClipDataset:
    transform = TransformConfig(
        target_shape=tuple(int(v) for v in target_shape),
        normalization="zscore",
        random_crop_shape=None,
        random_flip=False,
    )
    return ClipDataset(
        manifest_df=manifest_df,
        sample_ids=sample_ids,
        class_names=CLASS_NAMES,
        transform_cfg=transform,
        clip_length=clip_length,
        clip_stride=clip_stride,
        clip_window_stride=clip_window_stride,
        hrf_shift=hrf_shift,
        train=False,
        seed=seed,
    )


def extract_features(dataset: ClipDataset) -> tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    records: List[Dict[str, object]] = []
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        clip = dataset.clips[idx]
        xs.append(x.mean(dim=0).numpy().reshape(-1))
        ys.append(int(y.item()))
        records.append(
            {
                "idx": int(idx),
                "subject_id": str(clip["subject_id"]),
                "run_id": int(clip["run_id"]),
                "class_id": int(clip["class_id"]),
                "vol_ids": [int(v) for v in clip["vol_ids"]],
            }
        )
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), records


def nearest_centroid_predict(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray) -> np.ndarray:
    centroids = []
    for class_id in range(len(CLASS_NAMES)):
        mask = y_train == class_id
        if not np.any(mask):
            raise ValueError(f"Training split is missing class_id={class_id}")
        centroids.append(x_train[mask].mean(axis=0))
    centroid_arr = np.stack(centroids, axis=0)
    dist = ((x_val[:, None, :] - centroid_arr[None, :, :]) ** 2).mean(axis=2)
    return dist.argmin(axis=1).astype(np.int64)


def score_split(name: str, x: np.ndarray, y: np.ndarray, train_idx: Sequence[int], val_idx: Sequence[int]) -> dict:
    train_arr = np.asarray(train_idx, dtype=np.int64)
    val_arr = np.asarray(val_idx, dtype=np.int64)
    pred = nearest_centroid_predict(x[train_arr], y[train_arr], x[val_arr])
    metrics, cm = compute_classification_metrics(y[val_arr], pred, None, CLASS_NAMES)
    return {
        "name": name,
        "train_count": int(len(train_arr)),
        "val_count": int(len(val_arr)),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
    }


def within_group_leave_one_out(x: np.ndarray, y: np.ndarray, records: List[Dict[str, object]]) -> dict:
    grouped: Dict[tuple[str, int], List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        grouped[(str(rec["subject_id"]), int(rec["run_id"]))].append(i)

    y_true: List[int] = []
    y_pred: List[int] = []
    group_scores = []
    for (subject_id, run_id), idxs in sorted(grouped.items()):
        preds_for_group: List[int] = []
        true_for_group: List[int] = []
        for val_i in idxs:
            train_idx = [i for i in idxs if i != val_i]
            pred = nearest_centroid_predict(x[train_idx], y[train_idx], x[[val_i]])[0]
            preds_for_group.append(int(pred))
            true_for_group.append(int(y[val_i]))
        metrics, cm = compute_classification_metrics(true_for_group, preds_for_group, None, CLASS_NAMES)
        group_scores.append(
            {
                "subject_id": subject_id,
                "run_id": int(run_id),
                "clip_count": int(len(idxs)),
                "metrics": metrics,
                "confusion_matrix": cm.tolist(),
            }
        )
        y_true.extend(true_for_group)
        y_pred.extend(preds_for_group)

    overall_metrics, overall_cm = compute_classification_metrics(y_true, y_pred, None, CLASS_NAMES)
    return {
        "name": "within_subject_run_leave_one_clip_out",
        "group_count": int(len(group_scores)),
        "clip_count": int(len(y_true)),
        "metrics": overall_metrics,
        "confusion_matrix": overall_cm.tolist(),
        "group_scores": group_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze corrected clip feature transfer.")
    parser.add_argument("--batch-slugs", nargs="+", default=[f"thesis-batch-{i:02d}" for i in range(1, 8)])
    parser.add_argument("--max-subjects", type=int, default=8)
    parser.add_argument("--val-subject-count", type=int, default=2)
    parser.add_argument("--val-run-id", type=int, default=6)
    parser.add_argument("--target-shape", nargs=3, type=int, default=[24, 24, 24])
    parser.add_argument("--clip-length", type=int, default=6)
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument("--clip-window-stride", type=int, default=1)
    parser.add_argument("--hrf-shift", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="/kaggle/working/clip_feature_transfer")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = resolve_batch_roots(args.batch_slugs)
    manifest_df, qc = build_manifest(roots, CLASS_NAMES)
    subjects = sorted(manifest_df["subject_id"].unique().tolist())[: int(args.max_subjects)]
    if len(subjects) <= int(args.val_subject_count):
        raise ValueError("Need more selected subjects than validation subjects")

    selected_df = manifest_df.loc[manifest_df["subject_id"].isin(subjects)]
    sample_ids = selected_df["sample_id"].astype(int).tolist()
    dataset = make_clip_dataset(
        manifest_df,
        sample_ids=sample_ids,
        target_shape=args.target_shape,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        clip_window_stride=args.clip_window_stride,
        hrf_shift=args.hrf_shift,
        seed=args.seed,
    )
    print(f"selected_subjects={subjects} clip_count={len(dataset)}", flush=True)

    x, y, records = extract_features(dataset)
    np.save(out_dir / "features.npy", x)
    np.save(out_dir / "labels.npy", y)
    (out_dir / "records.json").write_text(json.dumps(records, indent=2))

    rec_subjects = np.asarray([str(r["subject_id"]) for r in records])
    rec_runs = np.asarray([int(r["run_id"]) for r in records])
    val_subjects = subjects[-int(args.val_subject_count) :]
    train_subjects = subjects[: -int(args.val_subject_count)]

    all_idx = np.arange(len(records), dtype=np.int64)
    run_train_idx = all_idx[rec_runs != int(args.val_run_id)]
    run_val_idx = all_idx[rec_runs == int(args.val_run_id)]
    subject_train_idx = all_idx[np.isin(rec_subjects, train_subjects)]
    subject_val_idx = all_idx[np.isin(rec_subjects, val_subjects)]

    splits = [
        score_split("train_eval_all_selected", x, y, all_idx, all_idx),
        score_split("same_subject_run_holdout", x, y, run_train_idx, run_val_idx),
        score_split("subject_holdout", x, y, subject_train_idx, subject_val_idx),
    ]

    summary = {
        "class_names": CLASS_NAMES,
        "qc": qc,
        "target_shape": list(args.target_shape),
        "clip_length": int(args.clip_length),
        "clip_count": int(len(records)),
        "feature_dim": int(x.shape[1]),
        "selected_subjects": subjects,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "val_run_id": int(args.val_run_id),
        "within_run_leave_one_clip_out": within_group_leave_one_out(x, y, records),
        "splits": splits,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
