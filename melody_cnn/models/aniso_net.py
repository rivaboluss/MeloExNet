"""Anisotropic Conv + bar/scale feats + FiLM key conditioning."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from melody_cnn.models.coordconv import FiLM


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        padding = spatial_kernel // 2
        self.spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        wa = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * wa
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        ws = torch.sigmoid(self.spatial(torch.cat([avg_s, max_s], dim=1)))
        return x * ws


class AnisoBlock(nn.Module):
    def __init__(self, channels: int, k_pitch: int = 5, k_time: int = 15, dropout: float = 0.1):
        super().__init__()
        pad_a = (k_pitch // 2, k_time // 2)
        pad_b = (k_time // 2, k_pitch // 2)
        self.conv1 = nn.Conv2d(channels, channels, (k_pitch, k_time), padding=pad_a)
        self.conv2 = nn.Conv2d(channels, channels, (k_time, k_pitch), padding=pad_b)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.attn = CBAM(channels)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = F.gelu(self.bn2(self.conv2(h)))
        h = self.attn(h)
        return x + h


class AnisoCoordMelodyNet(nn.Module):
    """
    Input roll: [B, in_ch, P, T]
      default in_ch=9: state, in_scale, is_bar, skyline, dist_below,
      dist_above, pitch_y, bar_norm, pos_in_bar
    key_vec: [B, 14] tonic one-hot + mode one-hot → FiLM after stem
    Output: [B, num_classes, P, T]
    """

    def __init__(
        self,
        in_channels: int = 9,
        base_channels: int = 32,
        num_classes: int = 2,
        k_pitch: int = 5,
        k_time: int = 15,
        num_blocks: int = 4,
        dropout: float = 0.1,
        cond_dim: int = 14,
        **_kwargs,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=(k_pitch, k_time),
                padding=(k_pitch // 2, k_time // 2),
            ),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        self.film = FiLM(cond_dim, base_channels)
        self.blocks = nn.ModuleList(
            [
                AnisoBlock(base_channels, k_pitch=k_pitch, k_time=k_time, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.film_mid = FiLM(cond_dim, base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(base_channels, num_classes, kernel_size=1),
        )

    def forward(self, roll: torch.Tensor, key_vec: torch.Tensor) -> torch.Tensor:
        x = self.stem(roll)
        x = self.film(x, key_vec)
        mid = len(self.blocks) // 2
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == mid:
                x = self.film_mid(x, key_vec)
        return self.head(x)
