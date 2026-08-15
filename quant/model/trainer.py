"""Model training and persistence."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quant.model.dlinear import DLinearModel, SequenceDataset

logger = logging.getLogger(__name__)


def load_model(
    path: str | Path,
    seq_len: int = 30,
    pred_len: int = 5,
    num_features: int = 11,
    map_location: str | None = None,
) -> DLinearModel:
    device = map_location or "cpu"
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        seq_len = checkpoint.get("seq_len", seq_len)
        pred_len = checkpoint.get("pred_len", pred_len)
        num_features = checkpoint.get("num_features", num_features)
    else:
        state_dict = checkpoint

    model = DLinearModel(seq_len=seq_len, pred_len=pred_len, num_features=num_features)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def train_and_save(
    data: np.ndarray,
    save_path: str | Path,
    seq_len: int = 30,
    pred_len: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 5e-5,
) -> DLinearModel:
    if data.ndim != 2:
        raise ValueError("Training data must be 2D (samples, features)")
    if len(data) < seq_len + pred_len:
        raise ValueError(f"Need at least {seq_len + pred_len} rows for training")

    num_features = data.shape[1]
    dataset = SequenceDataset(data, seq_len, pred_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DLinearModel(seq_len=seq_len, pred_len=pred_len, num_features=num_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(loader), 1)
        logger.info("Epoch [%s/%s] loss=%.6f", epoch + 1, epochs, avg_loss)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seq_len": seq_len,
            "pred_len": pred_len,
            "num_features": num_features,
        },
        save_path,
    )
    model.eval()
    logger.info("Model saved to %s", save_path)
    return model
