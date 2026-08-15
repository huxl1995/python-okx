"""MLP baseline: flatten window and regress future steps."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """Fully-connected network over flattened input windows."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_features: int,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features

        input_dim = seq_len * num_features
        output_dim = pred_len * num_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        flat = x.reshape(batch, -1)
        out = self.net(flat)
        return out.view(batch, self.pred_len, self.num_features)
