from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from fmri_pipeline.models import build_model
from fmri_pipeline.training.pipeline import train_fold
from fmri_pipeline.utils.log_utils import setup_logger


def test_temporal_resnet3d_groupnorm_forward():
    model = build_model(
        {
            "name": "temporal_resnet3d",
            "in_channels": 1,
            "num_classes": 4,
            "base_channels": 4,
            "dropout": 0.0,
            "norm": "group",
            "group_norm_groups": 2,
            "temporal": {
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 4,
                "max_clip_length": 6,
            },
        }
    )
    x = torch.randn(2, 6, 1, 16, 16, 16)
    y = model(x)
    assert y.shape == (2, 4)


def test_smoke_train_fold(tmp_path: Path):
    class_names = [
        "Left leg movements",
        "Right leg movements",
        "Forearm movements",
        "Upper arm movements",
    ]

    rows = []
    sample_id = 0
    subjects = ["sub-01", "sub-02"]
    for subject in subjects:
        for class_id, class_name in enumerate(class_names):
            class_dir = tmp_path / "batch_01" / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for vol_id in [1, 2]:
                fp = class_dir / f"{subject}_run-1_vol-{vol_id:03d}.nii.gz"
                data = np.random.randn(10, 10, 10).astype(np.float32)
                nib.save(nib.Nifti1Image(data, np.eye(4)), str(fp))
                rows.append(
                    {
                        "sample_id": sample_id,
                        "filepath": str(fp),
                        "class_name": class_name,
                        "class_id": class_id,
                        "subject_id": subject,
                        "run_id": 1,
                        "vol_id": vol_id,
                        "batch_id": "batch_01",
                        "exists": True,
                    }
                )
                sample_id += 1

    manifest_df = pd.DataFrame(rows)
    train_ids = manifest_df.loc[manifest_df["subject_id"] == "sub-01", "sample_id"].tolist()
    val_ids = manifest_df.loc[manifest_df["subject_id"] == "sub-02", "sample_id"].tolist()

    config = {
        "task": "volume",
        "classes": {"names": class_names},
        "input": {"target_shape": [8, 8, 8], "normalization": "zscore", "hrf_shift": 2},
        "loader": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "clip_lengths": [4],
            "clip_length": 4,
            "clip_stride": 1,
            "clip_window_stride": 1,
            "augment": {"random_flip": False},
        },
        "model": {
            "name": "resnet3d",
            "in_channels": 1,
            "num_classes": 4,
            "base_channels": 8,
            "dropout": 0.1,
        },
        "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
        "scheduler": {"name": "cosine", "warmup_epochs": 0, "min_lr": 1e-6},
        "training": {
            "epochs": 1,
            "gradient_accumulation_steps": 1,
            "early_stopping_patience": 2,
            "monitor": "macro_f1",
            "monitor_mode": "max",
            "amp": False,
            "max_grad_norm": 1.0,
            "label_smoothing": 0.0,
            "deterministic": True,
        },
        "evaluation": {
            "bootstrap_samples": 10,
            "ci_alpha": 0.95,
            "metrics": [
                "top1_accuracy",
                "balanced_accuracy",
                "macro_f1",
                "mcc",
                "roc_auc_ovr_macro",
                "pr_auc_macro",
            ],
        },
        "seed": 42,
        "paths": {"artifacts_root": "artifacts", "cache_root": "cache"},
    }

    out_dir = tmp_path / "run"
    logger = setup_logger("test_smoke", out_dir / "test.log")

    summary = train_fold(
        config=config,
        manifest_df=manifest_df,
        train_sample_ids=train_ids,
        val_sample_ids=val_ids,
        run_dir=out_dir,
        run_name="smoke",
        fold_name="smoke_fold",
        config_path="tests",
        logger=logger,
    )

    assert "checkpoint" in summary
    assert Path(summary["checkpoint"]).exists()
