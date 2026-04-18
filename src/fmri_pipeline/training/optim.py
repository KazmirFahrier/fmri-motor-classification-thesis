from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, optimizer_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = str(optimizer_cfg["name"]).lower()
    lr = float(optimizer_cfg["lr"])
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer.name: {optimizer_cfg['name']}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: Dict[str, Any],
    total_epochs: int,
    base_lr: float,
) -> torch.optim.lr_scheduler._LRScheduler:
    name = str(scheduler_cfg["name"]).lower()
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler.name: {scheduler_cfg['name']}")

    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
    min_lr = float(scheduler_cfg.get("min_lr", 0.0))
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))

        progress_num = epoch - warmup_epochs
        progress_den = max(1, total_epochs - warmup_epochs)
        progress = min(max(progress_num / progress_den, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_criterion(training_cfg: Dict[str, Any]) -> nn.Module:
    label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
