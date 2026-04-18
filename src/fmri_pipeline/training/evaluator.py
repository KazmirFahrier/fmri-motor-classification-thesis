from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

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
from fmri_pipeline.training.engine import evaluate_one_epoch
from fmri_pipeline.training.optim import build_criterion
from fmri_pipeline.utils.device import resolve_device
from fmri_pipeline.utils.io import write_json
from fmri_pipeline.utils.metrics import compute_bootstrap_bundle


def _build_transform_cfg(config: Dict[str, Any]) -> TransformConfig:
    return TransformConfig(
        target_shape=tuple(int(v) for v in config["input"]["target_shape"]),
        normalization=str(config["input"]["normalization"]),
        random_crop_shape=None,
        random_flip=False,
    )


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


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    manifest_df: pd.DataFrame,
    sample_ids: Sequence[int],
    out_dir: str | Path,
    split_name: str,
) -> Dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["full_config"]
    model_cfg = checkpoint["model_config"]
    class_names = checkpoint["class_names"]

    criterion = build_criterion(config["training"])

    transform_cfg = _build_transform_cfg(config)
    task = str(config["task"]).lower()
    if task == "volume":
        dataset = VolumeDataset(
            manifest_df,
            sample_ids,
            class_names,
            transform_cfg=transform_cfg,
            train=False,
            seed=int(config["seed"]),
        )
    elif task == "clip":
        dataset = ClipDataset(
            manifest_df,
            sample_ids,
            class_names,
            transform_cfg=transform_cfg,
            clip_length=int(config["loader"]["clip_length"]),
            clip_stride=int(config["loader"]["clip_stride"]),
            clip_window_stride=int(config["loader"]["clip_window_stride"]),
            hrf_shift=int(config["input"]["hrf_shift"]),
            train=False,
            seed=int(config["seed"]),
        )
    else:
        raise ValueError(f"Unsupported task: {config['task']}")

    loader = DataLoader(
        dataset,
        batch_size=int(config["loader"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["loader"]["num_workers"]),
        pin_memory=bool(config["loader"].get("pin_memory", True)),
    )

    device = resolve_device()
    model = build_model(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_out = evaluate_one_epoch(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        amp=bool(config["training"].get("amp", True)),
        desc=split_name,
    )

    y_true = eval_out["y_true"]
    y_pred = eval_out["y_pred"]

    bootstrap_ci = compute_bootstrap_bundle(
        y_true=y_true,
        y_pred=y_pred,
        n_bootstrap=int(config["evaluation"]["bootstrap_samples"]),
        alpha=float(config["evaluation"]["ci_alpha"]),
        seed=int(config["seed"]),
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "split_name": split_name,
        "num_samples": int(len(dataset)),
        "metrics": eval_out["metrics"],
        "bootstrap_ci": bootstrap_ci,
    }

    write_json(summary, out_path / "summary.json")
    np.save(out_path / "y_true.npy", y_true)
    np.save(out_path / "y_pred.npy", y_pred)
    np.save(out_path / "y_prob.npy", eval_out["y_prob"])
    np.save(out_path / "confusion_matrix.npy", eval_out["confusion_matrix"])
    _save_confusion_matrix(
        eval_out["confusion_matrix"],
        class_names,
        out_path / "confusion_matrix.pdf",
        title=f"{split_name} Confusion Matrix",
    )

    return summary
