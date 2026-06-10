#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = [
    "Left leg movements",
    "Right leg movements",
    "Forearm movements",
    "Upper arm movements",
]

FILENAME_PATTERNS = [
    re.compile(r"^(?P<subject_id>sub-\d+)_run-(?P<run_id>\d+)_vol-(?P<vol_id>\d+)\.nii(?:\.gz)?$"),
    re.compile(r"^volume_(?P<subject_prefix>sub)_(?P<subject_num>\d+)_run_(?P<run_id>\d+)_(?P<vol_id>\d+)\.nii(?:\.gz)?$"),
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_file(path: Path) -> Tuple[str, int, int]:
    for pattern in FILENAME_PATTERNS:
        match = pattern.match(path.name)
        if not match:
            continue
        data = match.groupdict()
        if "subject_id" in data and data["subject_id"] is not None:
            subject_id = data["subject_id"]
        else:
            subject_id = f"{data['subject_prefix']}-{int(data['subject_num']):02d}"
        return subject_id, int(data["run_id"]), int(data["vol_id"])
    raise ValueError(f"Cannot parse NIfTI filename: {path.name}")


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


def iter_class_files(batch_roots: Iterable[Path]) -> Iterable[Tuple[str, Path]]:
    for root in batch_roots:
        for class_name in CLASS_NAMES:
            candidates = [root / class_name]
            candidates.extend(p for p in root.rglob(class_name) if p.is_dir())
            class_dirs = sorted({p.resolve() for p in candidates if p.exists()})
            for class_dir in class_dirs[:1]:
                for path in sorted(class_dir.glob("*.nii.gz")) + sorted(class_dir.glob("*.nii")):
                    yield class_name, path


def load_index(batch_slugs: List[str]) -> Dict[Tuple[str, int], Dict[str, List[Tuple[int, Path]]]]:
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

    index: Dict[Tuple[str, int], Dict[str, List[Tuple[int, Path]]]] = defaultdict(lambda: defaultdict(list))
    for class_name, path in iter_class_files(roots):
        subject_id, run_id, vol_id = parse_file(path)
        index[(subject_id, run_id)][class_name].append((vol_id, path))
    return index


def choose_complete_block(
    index: Dict[Tuple[str, int], Dict[str, List[Tuple[int, Path]]]],
    subject_id: str | None,
    run_id: int | None,
) -> Tuple[Tuple[str, int], Dict[str, List[Tuple[int, Path]]]]:
    keys = sorted(index.keys())
    if subject_id is not None:
        keys = [k for k in keys if k[0] == subject_id]
    if run_id is not None:
        keys = [k for k in keys if k[1] == run_id]

    for key in keys:
        block = index[key]
        if all(len(block.get(class_name, [])) >= 16 for class_name in CLASS_NAMES):
            return key, {class_name: sorted(block[class_name])[:16] for class_name in CLASS_NAMES}
    raise ValueError("Could not find a complete subject-run block with 16 volumes per class")


def load_volume(path: Path, target_shape: Tuple[int, int, int]) -> np.ndarray:
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={data.shape}: {path}")
    if tuple(data.shape) != target_shape:
        factors = [t / s for t, s in zip(target_shape, data.shape)]
        data = zoom(data, factors, order=1)
    mean = float(data.mean())
    std = float(data.std())
    data = (data - mean) / max(std, 1e-6)
    return data.astype(np.float32, copy=False)


def build_tensors(
    block: Dict[str, List[Tuple[int, Path]]],
    target_shape: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, object]]]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    samples: List[Dict[str, object]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        for vol_id, path in block[class_name]:
            xs.append(load_volume(path, target_shape))
            ys.append(class_id)
            samples.append({"class_name": class_name, "class_id": class_id, "vol_id": int(vol_id), "path": str(path)})
    x = torch.from_numpy(np.stack(xs, axis=0)).unsqueeze(1)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y, samples


class TinyCNN3D(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).flatten(1)
        return self.fc(x)


def train_overfit(
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> List[Dict[str, float]]:
    model = TinyCNN3D(num_classes=len(CLASS_NAMES)).to(device)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(yb.numel())
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += int(yb.numel())

        row = {
            "epoch": float(epoch),
            "loss": float(total_loss / max(total, 1)),
            "accuracy": float(total_correct / max(total, 1)),
        }
        history.append(row)
        print(f"epoch={epoch:03d} loss={row['loss']:.4f} acc={row['accuracy']:.4f}", flush=True)
        if row["accuracy"] >= 1.0:
            break
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny overfit sanity check on one subject-run.")
    parser.add_argument("--batch-slugs", nargs="+", default=[f"thesis-batch-{i:02d}" for i in range(1, 8)])
    parser.add_argument("--subject-id", default=None)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--target-shape", nargs=3, type=int, default=[32, 32, 32])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="/kaggle/working/tiny_overfit_sanity")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    index = load_index(args.batch_slugs)
    (subject_id, run_id), block = choose_complete_block(index, args.subject_id, args.run_id)
    print(f"selected_subject={subject_id} selected_run={run_id}", flush=True)

    x, y, samples = build_tensors(block, tuple(args.target_shape))
    class_counts = {class_name: int((y == idx).sum().item()) for idx, class_name in enumerate(CLASS_NAMES)}
    print(f"tensor_shape={tuple(x.shape)} class_counts={class_counts}", flush=True)

    history = train_overfit(x, y, args.epochs, args.batch_size, args.lr, device)
    final = history[-1]
    success = bool(final["accuracy"] >= 0.95)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "success": success,
        "criterion": "success if final training accuracy >= 0.95 on 64-sample one-run block",
        "selected_subject": subject_id,
        "selected_run": run_id,
        "target_shape": list(args.target_shape),
        "num_samples": int(y.numel()),
        "class_counts": class_counts,
        "device": str(device),
        "final": final,
        "history": history,
        "samples": samples,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", json.dumps({k: summary[k] for k in ["success", "selected_subject", "selected_run", "final"]}), flush=True)
    if not success:
        raise SystemExit("Tiny overfit sanity check failed to reach >=0.95 training accuracy")


if __name__ == "__main__":
    main()
