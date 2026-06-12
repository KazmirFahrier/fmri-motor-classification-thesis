from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .cnn_transformer_ablation import CNNTransformerAblation
from .resnet3d import ResNet3DClassifier
from .temporal_resnet3d import TemporalResNet3D


def build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    name = str(model_cfg["name"]).lower()
    common_kwargs = {
        "in_channels": int(model_cfg["in_channels"]),
        "num_classes": int(model_cfg["num_classes"]),
        "base_channels": int(model_cfg.get("base_channels", 32)),
        "dropout": float(model_cfg.get("dropout", 0.3)),
        "norm": str(model_cfg.get("norm", "batch")),
        "group_norm_groups": int(model_cfg.get("group_norm_groups", 4)),
    }

    if name == "resnet3d":
        return ResNet3DClassifier(**common_kwargs)

    if name == "temporal_resnet3d":
        temporal_cfg = model_cfg.get("temporal", {})
        return TemporalResNet3D(
            **common_kwargs,
            hidden_dim=int(temporal_cfg.get("hidden_dim", 256)),
            num_layers=int(temporal_cfg.get("num_layers", 2)),
            num_heads=int(temporal_cfg.get("num_heads", 4)),
            max_clip_length=int(temporal_cfg.get("max_clip_length", 16)),
        )

    if name == "cnn_transformer_ablation":
        return CNNTransformerAblation(
            in_channels=common_kwargs["in_channels"],
            num_classes=common_kwargs["num_classes"],
            dropout=common_kwargs["dropout"],
            d_model=int(model_cfg.get("d_model", 128)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            num_heads=int(model_cfg.get("num_heads", 4)),
        )

    raise ValueError(f"Unsupported model.name: {model_cfg['name']}")


__all__ = [
    "build_model",
    "CNNTransformerAblation",
    "ResNet3DClassifier",
    "TemporalResNet3D",
]
