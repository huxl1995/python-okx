"""Model inference."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from quant.config import QuantConfig
from quant.data.features import FEATURE_COLUMNS, PRICE_KEYS, preprocess_klines, restore_predictions
from quant.data.klines import KlineFetcher
from quant.model.trainer import load_model, train_and_save

logger = logging.getLogger(__name__)


def predict_scaled(model: nn.Module, window: np.ndarray) -> np.ndarray:
    tensor = torch.tensor(window, dtype=torch.float32)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(tensor)
    return output.numpy()[0]


class PricePredictor:
    """End-to-end kline fetch, train, and predict pipeline."""

    def __init__(self, config: QuantConfig, fetcher: KlineFetcher | None = None) -> None:
        self.config = config
        self.fetcher = fetcher or KlineFetcher(config)

    def _prepare_data(self) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        raw = self.fetcher.fetch(
            symbol=self.config.symbol,
            interval=self.config.interval,
            limit=self.config.kline_limit,
        )
        raw_df, feature_df = preprocess_klines(raw, self.config.window_size)
        min_rows = self.config.seq_len + self.config.pred_len + 10
        if len(feature_df) < min_rows:
            raise ValueError(
                f"Insufficient data after preprocessing: {len(feature_df)} rows, need {min_rows}"
            )
        matrix = feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        return raw_df, feature_df, matrix

    def train(self, epochs: int | None = None, lr: float | None = None) -> Path:
        _, _, matrix = self._prepare_data()
        train_data = matrix[: -self.config.seq_len]
        train_and_save(
            data=train_data,
            save_path=self.config.model_path,
            model_type=self.config.model_type,
            seq_len=self.config.seq_len,
            pred_len=self.config.pred_len,
            epochs=epochs or self.config.epochs,
            batch_size=self.config.batch_size,
            lr=lr or self.config.lr,
            hidden_size=self.config.hidden_size,
        )
        return self.config.model_path

    def predict_future(self) -> dict[str, np.ndarray]:
        raw_df, feature_df, matrix = self._prepare_data()
        if len(feature_df) < self.config.seq_len:
            raise ValueError("Not enough rows for inference window")

        input_window = matrix[-self.config.seq_len :]
        model = load_model(
            self.config.model_path,
            model_type=self.config.model_type,
            seq_len=self.config.seq_len,
            pred_len=self.config.pred_len,
            hidden_size=self.config.hidden_size,
        )
        scaled_pred = predict_scaled(model, input_window)
        restored = restore_predictions(
            scaled_pred,
            raw_df,
            self.config.window_size,
            len(raw_df),
            price_keys=PRICE_KEYS,
        )
        logger.info(
            "Prediction model=%s steps=%s",
            self.config.model_type,
            self.config.pred_len,
        )
        for i in range(self.config.pred_len):
            logger.info(
                "Pred step %s | close=%.6f",
                i + 1,
                restored["close"][i],
            )
        return restored
