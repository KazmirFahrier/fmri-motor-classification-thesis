from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn


def _make_norm(norm: str, channels: int, group_norm_groups: int) -> nn.Module:
    norm = norm.lower()
    if norm == "batch":
        return nn.BatchNorm3d(channels)
    if norm == "group":
        groups = min(int(group_norm_groups), int(channels))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(num_groups=groups, num_channels=channels)
    if norm == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    raise ValueError(f"Unsupported 3D normalization: {norm}")


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_factory: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        norm_factory = norm_factory or (lambda channels: nn.BatchNorm3d(channels))
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = norm_factory(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = norm_factory(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_factory(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet3DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        layers: List[int] | None = None,
        norm: str = "batch",
        group_norm_groups: int = 4,
    ) -> None:
        super().__init__()
        layers = layers or [2, 2, 2, 2]
        self.norm_factory = lambda channels: _make_norm(norm, channels, group_norm_groups)

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            self.norm_factory(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.inplanes = base_channels
        self.layer1 = self._make_layer(base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2)

        self.out_dim = base_channels * 8
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock3D(self.inplanes, out_channels, stride=stride, norm_factory=self.norm_factory)]
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.inplanes, out_channels, stride=1, norm_factory=self.norm_factory))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)


class ResNet3DClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        dropout: float = 0.3,
        norm: str = "batch",
        group_norm_groups: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = ResNet3DEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            norm=norm,
            group_norm_groups=group_norm_groups,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.encoder.out_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
