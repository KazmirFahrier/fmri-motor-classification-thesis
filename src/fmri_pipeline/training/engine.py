from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fmri_pipeline.utils.metrics import compute_classification_metrics


def _to_device(batch, device: torch.device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_names: List[str],
    amp: bool,
    grad_accum_steps: int,
    max_grad_norm: float | None,
) -> Dict[str, object]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))

    running_loss = 0.0
    n_samples = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(dataloader, desc="train", leave=False)

    for step, batch in enumerate(progress, start=1):
        x, y = _to_device(batch, device)

        with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)
            loss_for_backward = loss / grad_accum_steps

        scaler.scale(loss_for_backward).backward()

        if step % grad_accum_steps == 0:
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        probs = torch.softmax(logits.detach(), dim=1)
        preds = torch.argmax(probs, dim=1)

        batch_size = y.shape[0]
        running_loss += float(loss.item()) * batch_size
        n_samples += batch_size
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy())

        progress.set_postfix(loss=f"{running_loss / max(1, n_samples):.4f}")

    # Flush pending gradients if steps were not divisible.
    if len(dataloader) % grad_accum_steps != 0:
        if max_grad_norm is not None and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    metrics, cm = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=np.asarray(y_prob, dtype=np.float32),
        class_names=class_names,
    )
    metrics["loss"] = float(running_loss / max(1, n_samples))

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_pred": np.asarray(y_pred, dtype=np.int64),
        "y_prob": np.asarray(y_prob, dtype=np.float32),
    }


def evaluate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    class_names: List[str],
    amp: bool,
    desc: str = "val",
) -> Dict[str, object]:
    model.eval()

    running_loss = 0.0
    n_samples = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []

    progress = tqdm(dataloader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in progress:
            x, y = _to_device(batch, device)

            with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            batch_size = y.shape[0]
            running_loss += float(loss.item()) * batch_size
            n_samples += batch_size
            y_true.extend(y.detach().cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy())

    metrics, cm = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=np.asarray(y_prob, dtype=np.float32),
        class_names=class_names,
    )
    metrics["loss"] = float(running_loss / max(1, n_samples))

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_pred": np.asarray(y_pred, dtype=np.int64),
        "y_prob": np.asarray(y_prob, dtype=np.float32),
    }
