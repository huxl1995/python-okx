"""DLinear time-series model for price prediction."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class MovingAvgDecomp(nn.Module):
    """Decompose series into trend and seasonal components."""

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        front = x[:, 0:1, :].repeat(1, self.kernel_size // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class DLinearModel(nn.Module):
    """DLinear with NLinear normalization for crypto OHLCV features."""

    def __init__(self, seq_len: int, pred_len: int, num_features: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.decomposition = MovingAvgDecomp(kernel_size=5)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_last = x[:, -1:, :]
        x = x - seq_last
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)
        out = (seasonal_output + trend_output).permute(0, 2, 1)
        return out + seq_last


class SequenceDataset(Dataset):
    """Sliding-window dataset for multivariate time series."""

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int) -> None:
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index : index + self.seq_len]
        y = self.data[index + self.seq_len : index + self.seq_len + self.pred_len]
        return x, y
