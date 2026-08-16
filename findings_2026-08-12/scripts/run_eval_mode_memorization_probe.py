#!/usr/bin/env python3
"""Eval-mode memorization gate for the neural lane.

The legacy subject-wise result (`0.2500` accuracy, `0.1000` macro F1, `0.0` MCC,
ROC-AUC `0.498`, `best_epoch: 0`) is the exact arithmetic signature of a constant
single-class output, not of a model that trained and failed to generalise. A model
that cannot memorise a handful of samples is evidence about the training path, not
about architectures, so it cannot be reported as an architecture negative control.

This probe applies the standard sanity gate: train on a small balanced block with
train == validation and require accuracy `1.00` **in eval mode**, which is the mode
every reported metric is computed in. Train-mode and eval-mode accuracy are reported
separately at every epoch, so a train/eval mismatch is visible rather than hidden
behind a single number.

Inputs come from the frozen `(48, 8, 13824)` event sequences reshaped to
`[B, T, C, 24, 24, 24]`, so the neural lane is probed on exactly the representation
that supports the frozen linear results.

The `--norm` sweep isolates the suspected cause: BatchNorm running statistics
estimated from tiny fMRI batches diverge from the batch statistics used during
training, so a model that fits in train mode collapses in eval mode.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# This module lives in a findings folder rather than the repository's own
# scripts/ directory, so the shared primitives are resolved explicitly.
REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (REPO_ROOT / "src", REPO_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from fmri_pipeline.models.temporal_resnet3d import TemporalResNet3D
from run_balanced_event_assignment import center_by_subject_run


CLASS_COUNT = 4


class FlattenLinear(nn.Module):
    """Lower rung of the capacity ladder: a linear map on the time-averaged volume."""

    def __init__(self, feature_count: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feature_count, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=1).flatten(1))


class FlattenMLP(nn.Module):
    def __init__(self, feature_count: int, num_classes: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_count, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.mean(dim=1).flatten(1))


class ShallowCNN(nn.Module):
    """One strided conv and a global pool, with per-sample normalization only.

    This isolates whether the failure is 3D convolution as such or the depth and
    repeated downsampling of the ResNet stack on a 24^3 grid.
    """

    def __init__(self, num_classes: int, channels: int = 16, groups: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )
        self.fc = nn.Linear(channels * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, d, h, w = x.shape
        pooled = self.features(x.reshape(b * t, c, d, h, w)).flatten(1)
        return self.fc(pooled.reshape(b, t, -1).mean(dim=1))


class SpatialCNN(nn.Module):
    """Convolutional, but location-preserving.

    `temporal_resnet3d` downsamples a 24^3 input to 1^3 and then global-average-
    pools, which is translation-invariant. Somatotopic decoding is about *where*
    activation sits, so that pooling discards the discriminative variable. This
    variant keeps a coarse spatial map and flattens it instead of pooling, so
    location survives to the classifier.
    """

    def __init__(self, num_classes: int, channels: int = 16, groups: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, channels * 2),
            nn.GELU(),
        )
        self.grid = 6
        self.fc = nn.Linear(channels * 2 * self.grid**3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, d, h, w = x.shape
        feat = self.features(x.reshape(b * t, c, d, h, w))
        feat = torch.nn.functional.adaptive_avg_pool3d(
            feat, (self.grid, self.grid, self.grid)
        ).flatten(1)
        return self.fc(feat.reshape(b, t, -1).mean(dim=1))


def build_model(
    architecture: str,
    norm: str,
    clip_length: int,
    feature_count: int,
    dropout: float,
    base_channels: int,
) -> nn.Module:
    if architecture == "spatial_cnn":
        return SpatialCNN(CLASS_COUNT, channels=base_channels)
    if architecture == "linear":
        return FlattenLinear(feature_count, CLASS_COUNT)
    if architecture == "mlp":
        return FlattenMLP(feature_count, CLASS_COUNT)
    if architecture == "shallow_cnn":
        return ShallowCNN(CLASS_COUNT, channels=base_channels)
    if architecture == "temporal_resnet3d":
        return TemporalResNet3D(
            in_channels=1,
            num_classes=CLASS_COUNT,
            base_channels=base_channels,
            dropout=dropout,
            hidden_dim=128,
            num_layers=1,
            num_heads=4,
            max_clip_length=clip_length,
            norm=norm,
        )
    raise ValueError(f"Unknown architecture {architecture}.")


def select_balanced_block(
    y: np.ndarray,
    per_class: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = []
    for class_idx in range(CLASS_COUNT):
        candidates = np.flatnonzero(y == class_idx)
        if len(candidates) < per_class:
            raise ValueError(
                f"Class {class_idx} has {len(candidates)} events, need {per_class}."
            )
        chosen.append(rng.choice(candidates, size=per_class, replace=False))
    return np.sort(np.concatenate(chosen))


def accuracy_in_mode(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    training: bool,
    batch_size: int,
) -> tuple[float, int]:
    """Accuracy with the model forced into train or eval mode.

    Returns accuracy and the number of distinct predicted classes, which exposes
    constant single-class collapse directly.
    """
    was_training = model.training
    model.train(training)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(x[start : start + batch_size])
            predictions.append(logits.argmax(dim=1))
    model.train(was_training)
    pred = torch.cat(predictions)
    return (
        float((pred == y).float().mean().item()),
        int(torch.unique(pred).numel()),
    )


def run_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    norm: str,
    epochs: int,
    lr: float,
    batch_size: int,
    dropout: float,
    base_channels: int,
    seed: int,
    device: torch.device,
    architecture: str = "temporal_resnet3d",
) -> dict:
    torch.manual_seed(seed)
    model = build_model(
        architecture,
        norm,
        int(x.shape[1]),
        int(np.prod(x.shape[2:])),
        dropout,
        base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    history = []
    order = torch.arange(len(x), device=device)
    for epoch in range(epochs):
        model.train()
        shuffled = order[torch.randperm(len(x), device=device)]
        epoch_loss = 0.0
        for start in range(0, len(shuffled), batch_size):
            idx = shuffled[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(idx)

        train_acc, train_classes = accuracy_in_mode(model, x, y, True, batch_size)
        eval_acc, eval_classes = accuracy_in_mode(model, x, y, False, batch_size)
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / len(x),
                "train_mode_accuracy": train_acc,
                "eval_mode_accuracy": eval_acc,
                "eval_distinct_predicted_classes": eval_classes,
                "train_distinct_predicted_classes": train_classes,
            }
        )
        print(
            f"  norm={norm} epoch={epoch:03d} loss={epoch_loss / len(x):.4f} "
            f"train_mode={train_acc:.4f} eval_mode={eval_acc:.4f} "
            f"eval_classes={eval_classes}",
            flush=True,
        )
        if eval_acc >= 1.0:
            break

    best_eval = max(row["eval_mode_accuracy"] for row in history)
    best_train = max(row["train_mode_accuracy"] for row in history)
    return {
        "architecture": architecture,
        "norm": norm,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "epochs_run": len(history),
        "best_train_mode_accuracy": best_train,
        "best_eval_mode_accuracy": best_eval,
        "final_eval_distinct_classes": history[-1]["eval_distinct_predicted_classes"],
        "gate_passed": bool(best_eval >= 1.0),
        "train_eval_gap": float(best_train - best_eval),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Require eval-mode memorization before any neural claim is reported."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--sequence-key", default="offset_3_length_8_sequence")
    parser.add_argument("--per-class", type=int, default=6, help="6 x 4 classes = 24 events.")
    parser.add_argument("--norms", nargs="+", default=["batch", "group", "instance"])
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["temporal_resnet3d"],
        choices=["linear", "mlp", "shallow_cnn", "spatial_cnn", "temporal_resnet3d"],
        help="Capacity ladder. --norms applies only to temporal_resnet3d.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument(
        "--center-by-run",
        action="store_true",
        help=(
            "Apply the frozen pipeline's unlabeled subject-run centering before "
            "probing. Without it the discriminative between-event variation is a "
            "small fraction of each volume's total variance."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    path = Path(args.checkpoint_dir) / f"{args.subject}.npz"
    with np.load(path, allow_pickle=False) as data:
        sequence = data[args.sequence_key].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        records = json.loads(str(data["records_json"]))

    if args.center_by_run:
        for lag in range(sequence.shape[1]):
            sequence[:, lag] = center_by_subject_run(sequence[:, lag], records)

    grid = args.grid
    if sequence.shape[2] != grid**3:
        raise ValueError(
            f"Feature count {sequence.shape[2]} is not {grid}^3; pass a matching --grid."
        )

    block = select_balanced_block(labels, args.per_class, args.seed)
    volumes = sequence[block].reshape(len(block), sequence.shape[1], 1, grid, grid, grid)
    x = torch.from_numpy(volumes).to(device)
    y = torch.from_numpy(labels[block]).to(device)

    print(
        f"probe subject={args.subject} events={len(block)} "
        f"shape={tuple(x.shape)} device={device.type}",
        flush=True,
    )

    results = [
        run_probe(
            x,
            y,
            norm,
            args.epochs,
            args.lr,
            args.batch_size,
            args.dropout,
            args.base_channels,
            args.seed,
            device,
            architecture,
        )
        for architecture in args.architectures
        for norm in (args.norms if architecture == "temporal_resnet3d" else ["none"])
    ]

    payload = {
        "subject": args.subject,
        "sequence_key": args.sequence_key,
        "event_count": int(len(block)),
        "input_shape": list(x.shape),
        "device": device.type,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "base_channels": args.base_channels,
        "protocol": "train == validation memorization gate, accuracy required in eval mode",
        "results": results,
        "gate_summary": {
            f"{row['architecture']}|norm={row['norm']}": {
                "best_train_mode_accuracy": row["best_train_mode_accuracy"],
                "best_eval_mode_accuracy": row["best_eval_mode_accuracy"],
                "train_eval_gap": row["train_eval_gap"],
                "gate_passed": row["gate_passed"],
            }
            for row in results
        },
        "note": (
            "A neural architecture claim may only be reported for a normalization "
            "setting whose gate_passed is true. Settings that fail this gate are "
            "evidence about the training path, not about architectures."
        ),
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["gate_summary"], indent=2))


if __name__ == "__main__":
    main()
