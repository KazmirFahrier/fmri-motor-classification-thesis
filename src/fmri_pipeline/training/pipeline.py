from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from fmri_pipeline.data.datasets import ClipDataset, TransformConfig, VolumeDataset
from fmri_pipeline.models import build_model
from fmri_pipeline.training.engine import evaluate_one_epoch, train_one_epoch
from fmri_pipeline.training.optim import build_criterion, build_optimizer, build_scheduler
from fmri_pipeline.utils.device import resolve_device
from fmri_pipeline.utils.io import append_jsonl, build_run_manifest, ensure_dir, write_json


def _save_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _build_transform_cfg(config: Dict[str, Any], train: bool) -> TransformConfig:
    augment = config["loader"].get("augment", {})
    random_crop = augment.get("random_crop")
    random_crop_shape = tuple(int(v) for v in random_crop) if random_crop else None

    return TransformConfig(
        target_shape=tuple(int(v) for v in config["input"]["target_shape"]),
        normalization=str(config["input"]["normalization"]),
        random_crop_shape=random_crop_shape,
        random_flip=bool(augment.get("random_flip", False)) if train else False,
    )


def _truncate(ids: List[int], max_count: int | None) -> List[int]:
    if max_count is None:
        return ids
    return ids[: max(0, int(max_count))]


def _build_datasets(
    config: Dict[str, Any],
    manifest_df: pd.DataFrame,
    train_sample_ids: Sequence[int],
    val_sample_ids: Sequence[int],
    seed: int,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
):
    task = str(config["task"]).lower()
    class_names = config["classes"]["names"]

    train_ids = _truncate([int(x) for x in train_sample_ids], max_train_samples)
    val_ids = _truncate([int(x) for x in val_sample_ids], max_val_samples)

    train_transform = _build_transform_cfg(config, train=True)
    val_transform = _build_transform_cfg(config, train=False)

    if task == "volume":
        train_ds = VolumeDataset(
            manifest_df,
            train_ids,
            class_names,
            transform_cfg=train_transform,
            train=True,
            seed=seed,
        )
        val_ds = VolumeDataset(
            manifest_df,
            val_ids,
            class_names,
            transform_cfg=val_transform,
            train=False,
            seed=seed + 1,
        )
        return train_ds, val_ds

    if task == "clip":
        train_ds = ClipDataset(
            manifest_df,
            train_ids,
            class_names,
            transform_cfg=train_transform,
            clip_length=int(config["loader"]["clip_length"]),
            clip_stride=int(config["loader"]["clip_stride"]),
            clip_window_stride=int(config["loader"]["clip_window_stride"]),
            hrf_shift=int(config["input"]["hrf_shift"]),
            train=True,
            seed=seed,
        )
        val_ds = ClipDataset(
            manifest_df,
            val_ids,
            class_names,
            transform_cfg=val_transform,
            clip_length=int(config["loader"]["clip_length"]),
            clip_stride=int(config["loader"]["clip_stride"]),
            clip_window_stride=int(config["loader"]["clip_window_stride"]),
            hrf_shift=int(config["input"]["hrf_shift"]),
            train=False,
            seed=seed + 1,
        )
        return train_ds, val_ds

    raise ValueError(f"Unsupported task: {config['task']}")


def train_fold(
    *,
    config: Dict[str, Any],
    manifest_df: pd.DataFrame,
    train_sample_ids: Sequence[int],
    val_sample_ids: Sequence[int],
    run_dir: str | Path,
    run_name: str,
    fold_name: str,
    config_path: str,
    logger,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    resume_from: str | Path | None = None,
) -> Dict[str, Any]:
    run_dir = ensure_dir(run_dir)

    local_cfg = copy.deepcopy(config)
    seed = int(local_cfg["seed"])
    class_names: List[str] = list(local_cfg["classes"]["names"])

    manifest = build_run_manifest(
        run_name=run_name,
        config_path=config_path,
        config_data=local_cfg,
        extra={
            "fold_name": fold_name,
            "num_train_samples": len(train_sample_ids),
            "num_val_samples": len(val_sample_ids),
        },
    )
    write_json(manifest, run_dir / "run_manifest.json")

    train_ds, val_ds = _build_datasets(
        local_cfg,
        manifest_df,
        train_sample_ids,
        val_sample_ids,
        seed=seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )

    loader_cfg = local_cfg["loader"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(loader_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(loader_cfg["num_workers"]),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(loader_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(loader_cfg["num_workers"]),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=False,
    )

    device = resolve_device(logger)
    model = build_model(local_cfg["model"]).to(device)

    criterion = build_criterion(local_cfg["training"])
    optimizer = build_optimizer(model, local_cfg["optimizer"])
    scheduler = build_scheduler(
        optimizer,
        local_cfg["scheduler"],
        total_epochs=int(local_cfg["training"]["epochs"]),
        base_lr=float(local_cfg["optimizer"]["lr"]),
    )

    monitor_key = str(local_cfg["training"]["monitor"])
    monitor_mode = str(local_cfg["training"]["monitor_mode"])
    early_patience = int(local_cfg["training"]["early_stopping_patience"])

    best_score = -math.inf if monitor_mode == "max" else math.inf
    best_epoch = -1
    epochs_without_improve = 0
    best_ckpt_path = run_dir / "best_model.pt"
    last_ckpt_path = run_dir / "last_checkpoint.pt"
    start_epoch = 0

    resume_path: Path | None = None
    if resume_from is not None:
        candidate = Path(resume_from).expanduser().resolve()
        if candidate.exists():
            resume_path = candidate
        else:
            raise FileNotFoundError(f"Requested resume checkpoint not found: {candidate}")
    elif last_ckpt_path.exists():
        resume_path = last_ckpt_path

    metrics_jsonl = run_dir / "metrics.jsonl"
    if metrics_jsonl.exists() and resume_path is None:
        metrics_jsonl.unlink()

    amp = bool(local_cfg["training"].get("amp", True))
    grad_accum = int(local_cfg["training"].get("gradient_accumulation_steps", 1))
    max_grad_norm = local_cfg["training"].get("max_grad_norm")
    max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

    if resume_path is not None:
        resume_ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        start_epoch = int(resume_ckpt["epoch"]) + 1
        best_score = float(resume_ckpt.get("best_score", best_score))
        best_epoch = int(resume_ckpt.get("best_epoch", best_epoch))
        epochs_without_improve = int(
            resume_ckpt.get("epochs_without_improve", epochs_without_improve)
        )
        logger.info(
            "[%s] resuming from %s at epoch=%d best_epoch=%d best_%s=%.4f",
            fold_name,
            resume_path,
            start_epoch,
            best_epoch,
            monitor_key,
            best_score,
        )

    for epoch in range(start_epoch, int(local_cfg["training"]["epochs"])):
        train_out = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            class_names=class_names,
            amp=amp,
            grad_accum_steps=grad_accum,
            max_grad_norm=max_grad_norm,
        )
        val_out = evaluate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            amp=amp,
            desc="val",
        )

        scheduler.step()

        train_metrics = train_out["metrics"]
        val_metrics = val_out["metrics"]
        current_score = float(val_metrics[monitor_key])

        improved = (
            current_score > best_score
            if monitor_mode == "max"
            else current_score < best_score
        )

        if improved:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config": local_cfg["model"],
                    "full_config": local_cfg,
                    "class_names": class_names,
                    "monitor_key": monitor_key,
                    "monitor_value": current_score,
                    "fold_name": fold_name,
                    "run_name": run_name,
                },
                best_ckpt_path,
            )
            np.save(run_dir / "best_val_confusion_matrix.npy", val_out["confusion_matrix"])
            _save_confusion_matrix(
                val_out["confusion_matrix"],
                class_names,
                run_dir / "best_val_confusion_matrix.pdf",
                title=f"{fold_name} Best Validation Confusion Matrix",
            )
        else:
            epochs_without_improve += 1

        log_row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
            "improved": improved,
            "monitor_key": monitor_key,
            "monitor_value": current_score,
        }
        append_jsonl(log_row, metrics_jsonl)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "model_config": local_cfg["model"],
                "full_config": local_cfg,
                "class_names": class_names,
                "monitor_key": monitor_key,
                "monitor_value": current_score,
                "best_score": float(best_score),
                "best_epoch": int(best_epoch),
                "epochs_without_improve": int(epochs_without_improve),
                "fold_name": fold_name,
                "run_name": run_name,
            },
            last_ckpt_path,
        )

        logger.info(
            "[%s] epoch=%d train_loss=%.4f train_macro_f1=%.4f val_loss=%.4f val_macro_f1=%.4f",
            fold_name,
            epoch,
            float(train_metrics["loss"]),
            float(train_metrics["macro_f1"]),
            float(val_metrics["loss"]),
            float(val_metrics["macro_f1"]),
        )

        if epochs_without_improve >= early_patience:
            logger.info("[%s] early stopping at epoch=%d", fold_name, epoch)
            break

    if best_epoch < 0:
        raise RuntimeError(f"No checkpoint was saved for {fold_name}; training failed.")

    # Reload and evaluate best checkpoint one more time for final summary.
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    final_val = evaluate_one_epoch(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        amp=amp,
        desc="val-final",
    )

    summary = {
        "run_name": run_name,
        "fold_name": fold_name,
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "best_epoch": int(best_epoch),
        "best_monitor_key": monitor_key,
        "best_monitor_value": float(best_score),
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
        "val_metrics": final_val["metrics"],
        "checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
    }

    write_json(summary, run_dir / "summary.json")
    np.save(run_dir / "final_val_y_true.npy", final_val["y_true"])
    np.save(run_dir / "final_val_y_pred.npy", final_val["y_pred"])
    np.save(run_dir / "final_val_y_prob.npy", final_val["y_prob"])
    np.save(run_dir / "final_val_confusion_matrix.npy", final_val["confusion_matrix"])
    _save_confusion_matrix(
        final_val["confusion_matrix"],
        class_names,
        run_dir / "final_val_confusion_matrix.pdf",
        title=f"{fold_name} Final Validation Confusion Matrix",
    )

    return summary
