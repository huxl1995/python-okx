"""Trading signal generation from model predictions."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from quant.exchange.client import TradeAction

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    action: TradeAction
    current_price: float
    predicted_price: float
    expected_return: float


class MLStrategy:
    """
    Compare predicted future close against current price.
    BUY if expected return exceeds threshold, SELL if below negative threshold.
    """

    def __init__(self, threshold: float = 0.001) -> None:
        self.threshold = threshold

    def generate(
        self,
        current_price: float,
        restored: dict[str, np.ndarray],
    ) -> SignalResult:
        predicted = float(restored["close"][-1])
        expected_return = (predicted - current_price) / current_price

        logger.info(
            "Signal inputs: current=%.6f predicted=%.6f return=%.4f%% threshold=%.4f%%",
            current_price,
            predicted,
            expected_return * 100,
            self.threshold * 100,
        )

        if expected_return > self.threshold:
            action: TradeAction = "BUY"
        elif expected_return < -self.threshold:
            action = "SELL"
        else:
            action = "HOLD"

        return SignalResult(
            action=action,
            current_price=current_price,
            predicted_price=predicted,
            expected_return=expected_return,
        )
