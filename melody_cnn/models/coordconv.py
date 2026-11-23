"""Coordinate / conditioning helpers."""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise linear modulation from a condition vector."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B,C,P,T], cond: [B,cond_dim]
        gb = self.to_gb(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta
