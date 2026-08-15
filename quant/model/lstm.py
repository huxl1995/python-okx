"""LSTM encoder for multivariate time-series forecasting."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM encodes the input window and projects to future steps."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, pred_len * num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        out = self.head(h_n[-1])
        return out.view(-1, self.pred_len, self.num_features)
