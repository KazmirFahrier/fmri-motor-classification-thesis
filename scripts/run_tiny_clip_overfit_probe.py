#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def resolve_batch_roots(batch_slugs: List[str]) -> List[Path]:
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


class TinyClipCNN3D(nn.Module):
    """Small no-normalization clip model for eval-mode overfit sanity checks."""

    def __init__(self, num_classes: int = 4, base_channels: int = 8) -> None:
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        self.frame_encoder = nn.Sequential(
            nn.Conv3d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 1, D, H, W]
        b, t, c, d, h, w = x.shape
        frames = x.reshape(b * t, c, d, h, w)
        feat = self.frame_encoder(frames).flatten(1).reshape(b, t, -1)
        return self.fc(feat.mean(dim=1))


def make_overfit_clip_dataset(
    manifest_df,
    subject_id: str,
    run_id: int,
    target_shape: List[int],
    clip_length: int,
    clip_stride: int,
    clip_window_stride: int,
    hrf_shift: int,
    train: bool,
    seed: int,
) -> ClipDataset:
    selected = manifest_df.loc[
        (manifest_df["subject_id"] == subject_id)
        & (manifest_df["run_id"].astype(int) == int(run_id))
    ]
    if selected.empty:
        raise ValueError(f"No samples found for {subject_id} run {run_id}")
    transform = TransformConfig(
        target_shape=tuple(int(v) for v in target_shape),
        normalization="zscore",
        random_crop_shape=None,
        random_flip=False,
    )
    return ClipDataset(
        manifest_df=manifest_df,
        sample_ids=selected["sample_id"].astype(int).tolist(),
        class_names=CLASS_NAMES,
        transform_cfg=transform,
        clip_length=clip_length,
        clip_stride=clip_stride,
        clip_window_stride=clip_window_stride,
        hrf_shift=hrf_shift,
        train=train,
        seed=seed,
    )


def evaluate_model(model, loader, device) -> dict:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)
            total_loss += float(loss.item()) * int(yb.numel())
            total += int(yb.numel())
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy())

    metrics, cm = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=np.asarray(y_prob, dtype=np.float32),
        class_names=CLASS_NAMES,
    )
    metrics["loss"] = float(total_loss / max(total, 1))
    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_pred": np.asarray(y_pred, dtype=np.int64),
    }


def extract_clip_mean_features(dataset: ClipDataset) -> tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        # Average time, then flatten the normalized spatial image.
        xs.append(x.mean(dim=0).numpy().reshape(-1))
        ys.append(int(y.item()))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def nearest_centroid_probe(x: np.ndarray, y: np.ndarray) -> dict:
    centroids = []
    for class_id in range(len(CLASS_NAMES)):
        centroids.append(x[y == class_id].mean(axis=0))
    centroids_arr = np.stack(centroids, axis=0)
    dist = ((x[:, None, :] - centroids_arr[None, :, :]) ** 2).mean(axis=2)
    pred = dist.argmin(axis=1)
    metrics, cm = compute_classification_metrics(y, pred, None, CLASS_NAMES)

    loo_pred: List[int] = []
    for i in range(len(y)):
        loo_centroids = []
        for class_id in range(len(CLASS_NAMES)):
            mask = y == class_id
            if y[i] == class_id:
                mask = mask.copy()
                mask[i] = False
            loo_centroids.append(x[mask].mean(axis=0))
        loo_centroids_arr = np.stack(loo_centroids, axis=0)
        loo_dist = ((x[i][None, :] - loo_centroids_arr) ** 2).mean(axis=1)
        loo_pred.append(int(loo_dist.argmin()))
    loo_metrics, loo_cm = compute_classification_metrics(y, loo_pred, None, CLASS_NAMES)

    return {
        "train_eval_metrics": metrics,
        "train_eval_confusion_matrix": cm.tolist(),
        "leave_one_clip_out_metrics": loo_metrics,
        "leave_one_clip_out_confusion_matrix": loo_cm.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny corrected-clip overfit probe.")
    parser.add_argument("--batch-slugs", nargs="+", default=[f"thesis-batch-{i:02d}" for i in range(1, 8)])
    parser.add_argument("--subject-id", default="sub-01")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--target-shape", nargs=3, type=int, default=[24, 24, 24])
    parser.add_argument("--clip-length", type=int, default=6)
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument("--clip-window-stride", type=int, default=1)
    parser.add_argument("--hrf-shift", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--success-threshold", type=float, default=0.95)
    parser.add_argument("--no-error-on-fail", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="/kaggle/working/tiny_clip_overfit_probe")
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = resolve_batch_roots(args.batch_slugs)
    manifest_df, qc = build_manifest(roots, CLASS_NAMES)
    (out_dir / "index_qc.json").write_text(json.dumps(qc, indent=2))

    train_ds = make_overfit_clip_dataset(
        manifest_df,
        subject_id=args.subject_id,
        run_id=args.run_id,
        target_shape=args.target_shape,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        clip_window_stride=args.clip_window_stride,
        hrf_shift=args.hrf_shift,
        train=True,
        seed=args.seed,
    )
    eval_ds = make_overfit_clip_dataset(
        manifest_df,
        subject_id=args.subject_id,
        run_id=args.run_id,
        target_shape=args.target_shape,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        clip_window_stride=args.clip_window_stride,
        hrf_shift=args.hrf_shift,
        train=False,
        seed=args.seed + 1,
    )
    print(f"clip_count={len(train_ds)} target_shape={args.target_shape}", flush=True)

    x_feat, y_feat = extract_clip_mean_features(eval_ds)
    centroid = nearest_centroid_probe(x_feat, y_feat)
    (out_dir / "nearest_centroid_probe.json").write_text(json.dumps(centroid, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    model = TinyClipCNN3D(num_classes=len(CLASS_NAMES), base_channels=args.base_channels).to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    history = []
    best_eval = None
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(yb.numel())
            total += int(yb.numel())

        eval_out = evaluate_model(model, eval_loader, device)
        row = {
            "epoch": int(epoch),
            "train_loss": float(total_loss / max(total, 1)),
            "eval": eval_out["metrics"],
        }
        history.append(row)
        eval_acc = float(eval_out["metrics"]["top1_accuracy"])
        if best_eval is None or eval_acc > float(best_eval["eval"]["top1_accuracy"]):
            best_eval = row
            np.save(out_dir / "best_eval_confusion_matrix.npy", eval_out["confusion_matrix"])
            np.save(out_dir / "best_eval_y_true.npy", eval_out["y_true"])
            np.save(out_dir / "best_eval_y_pred.npy", eval_out["y_pred"])
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"eval_acc={eval_acc:.4f} eval_f1={eval_out['metrics']['macro_f1']:.4f}",
            flush=True,
        )
        if eval_acc >= args.success_threshold:
            break

    success = bool(best_eval and float(best_eval["eval"]["top1_accuracy"]) >= args.success_threshold)
    summary = {
        "success": success,
        "success_threshold": float(args.success_threshold),
        "subject_id": args.subject_id,
        "run_id": int(args.run_id),
        "target_shape": list(args.target_shape),
        "clip_length": int(args.clip_length),
        "clip_count": int(len(eval_ds)),
        "model": {
            "name": "TinyClipCNN3D",
            "base_channels": int(args.base_channels),
            "normalization_layers": "none",
        },
        "nearest_centroid_probe": centroid,
        "best_eval": best_eval,
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)

    if not success and not args.no_error_on_fail:
        raise SystemExit(
            f"Tiny clip overfit probe failed to reach eval accuracy >= {args.success_threshold:.4f}"
        )


if __name__ == "__main__":
    main()
