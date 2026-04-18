from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTransformerAblation(nn.Module):
    """Legacy-style CNN+Transformer ablation model.

    This is intentionally retained for ablation experiments and is not the default model.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.25,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv3d(64, d_model, kernel_size=3, stride=2, padding=1)

        self.pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 6:
            # For clip input, flatten temporal by mean to keep ablation comparable.
            x = x.mean(dim=1)

        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # [B, C, D, H, W] -> sequence tokens
        b, c, d, h, w = x.shape
        x = x.view(b, c, d * h * w).transpose(1, 2)  # [B, T, C]
        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.dropout(x)
        logits = self.fc(x)
        return logits
