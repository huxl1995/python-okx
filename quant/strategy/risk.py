"""Stop-loss and take-profit risk management."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

logger = logging.getLogger(__name__)

ExitReason = Literal["STOP_LOSS", "TAKE_PROFIT"]


@dataclass
class RiskLevels:
    stop_loss_price: float
    take_profit_price: float


@dataclass
class RiskCheckResult:
    triggered: bool
    reason: Optional[ExitReason] = None
    pnl_pct: float = 0.0


class RiskManager:
    """Compute and monitor stop-loss / take-profit levels for long positions."""

    def __init__(self, stop_loss_pct: float, take_profit_pct: float) -> None:
        if stop_loss_pct <= 0 or take_profit_pct <= 0:
            raise ValueError("stop_loss_pct and take_profit_pct must be positive")
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def levels_for_long(self, entry_price: float) -> RiskLevels:
        return RiskLevels(
            stop_loss_price=entry_price * (1 - self.stop_loss_pct),
            take_profit_price=entry_price * (1 + self.take_profit_pct),
        )

    def check_long_position(
        self,
        current_price: float,
        entry_price: float,
        levels: RiskLevels,
    ) -> RiskCheckResult:
        pnl_pct = (current_price - entry_price) / entry_price

        if current_price <= levels.stop_loss_price:
            logger.warning(
                "Stop-loss triggered: price=%.6f <= sl=%.6f (entry=%.6f, pnl=%.4f%%)",
                current_price,
                levels.stop_loss_price,
                entry_price,
                pnl_pct * 100,
            )
            return RiskCheckResult(triggered=True, reason="STOP_LOSS", pnl_pct=pnl_pct)

        if current_price >= levels.take_profit_price:
            logger.info(
                "Take-profit triggered: price=%.6f >= tp=%.6f (entry=%.6f, pnl=%.4f%%)",
                current_price,
                levels.take_profit_price,
                entry_price,
                pnl_pct * 100,
            )
            return RiskCheckResult(triggered=True, reason="TAKE_PROFIT", pnl_pct=pnl_pct)

        return RiskCheckResult(triggered=False, pnl_pct=pnl_pct)
