"""Shared types and dataset for forecast models."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ForecastModel(Protocol):
    seq_len: int
    pred_len: int
    num_features: int

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def eval(self) -> nn.Module: ...


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
