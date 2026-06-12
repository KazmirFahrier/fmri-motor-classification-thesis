from __future__ import annotations

import torch
import torch.nn as nn

from .resnet3d import ResNet3DEncoder


class TemporalResNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        dropout: float = 0.3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_clip_length: int = 16,
        norm: str = "batch",
        group_norm_groups: int = 4,
    ) -> None:
        super().__init__()
        self.frame_encoder = ResNet3DEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            norm=norm,
            group_norm_groups=group_norm_groups,
        )
        self.proj = nn.Linear(self.frame_encoder.out_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_clip_length, hidden_dim))

        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: [B, T, C, D, H, W]
        if x.dim() == 5:
            # Fallback: treat single volume as clip length 1.
            x = x.unsqueeze(1)
        if x.dim() != 6:
            raise ValueError(f"Expected input dims [B,T,C,D,H,W], got {tuple(x.shape)}")

        b, t, c, d, h, w = x.shape
        x = x.reshape(b * t, c, d, h, w)
        feat = self.frame_encoder(x)  # [B*T, F]
        feat = self.proj(feat).reshape(b, t, -1)  # [B, T, H]

        if t > self.pos_embed.shape[1]:
            raise ValueError(
                f"Clip length {t} exceeds max_clip_length {self.pos_embed.shape[1]}"
            )
        feat = feat + self.pos_embed[:, :t, :]

        # Explicit temporal modeling before pooling.
        conv_in = feat.transpose(1, 2)  # [B, H, T]
        conv_out = self.temporal_conv(conv_in).transpose(1, 2)  # [B, T, H]
        feat = feat + conv_out

        feat = self.temporal_encoder(feat)

        weights = torch.softmax(self.attn_pool(feat), dim=1)  # [B, T, 1]
        pooled = torch.sum(weights * feat, dim=1)

        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits
