"""Trading engine: orchestrates train -> predict -> trade -> optimize loop."""
from __future__ import annotations

import logging
import time

from quant.config import QuantConfig
from quant.data.klines import KlineFetcher
from quant.exchange.client import BinanceTrader, TradeAction
from quant.model.predictor import PricePredictor
from quant.optimizer.feedback import FeedbackOptimizer
from quant.storage.trades import Position, TradeRecord, TradeStore
from quant.strategy.risk import RiskLevels, RiskManager
from quant.strategy.signal import MLStrategy

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main quant trading orchestrator."""

    def __init__(self, config: QuantConfig) -> None:
        self.config = config
        self.fetcher = KlineFetcher(config)
        self.trader = BinanceTrader(config)
        self.store = TradeStore(
            config.trade_log_path,
            config.state_path,
            config.position_path,
        )
        self.optimizer = FeedbackOptimizer(config, self.store, self.fetcher)
        self.predictor = PricePredictor(config, self.fetcher)
        self.risk = RiskManager(config.stop_loss_pct, config.take_profit_pct)
        self._cycle = 0

    def _strategy(self) -> MLStrategy:
        state = self.store.load_state(self.optimizer._default_state())
        return MLStrategy(threshold=state.signal_threshold)

    def _log_trade(
        self,
        action: TradeAction,
        current_price: float,
        *,
        reason: str = "SIGNAL",
        predicted_price: float = 0.0,
        expected_return: float = 0.0,
        threshold: float = 0.0,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
        entry_price: float | None = None,
        exit_price: float | None = None,
        realized_return: float | None = None,
        evaluated: bool = False,
    ) -> None:
        entry = entry_price if entry_price is not None else current_price
        record = TradeRecord(
            cycle_id=self._cycle,
            symbol=self.config.symbol,
            interval=self.config.interval,
            action=action,
            quantity=self.config.quantity,
            entry_price=entry,
            predicted_price=predicted_price,
            expected_return=expected_return,
            threshold=threshold,
            reason=reason,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            exit_price=exit_price,
            realized_return=realized_return,
            direction_correct=(realized_return or 0) > 0 if realized_return is not None else None,
            evaluated=evaluated,
        )
        self.store.append(record)

    def _open_long(
        self,
        current_price: float,
        signal,
        strategy: MLStrategy,
        levels,
    ) -> None:
        self.trader.execute("BUY", self.config.symbol, self.config.quantity)
        position = Position(
            symbol=self.config.symbol,
            side="LONG",
            entry_price=current_price,
            quantity=self.config.quantity,
            stop_loss_price=levels.stop_loss_price,
            take_profit_price=levels.take_profit_price,
            entry_cycle_id=self._cycle,
        )
        self.store.save_position(position)
        self._log_trade(
            "BUY",
            current_price,
            reason="SIGNAL",
            predicted_price=signal.predicted_price,
            expected_return=signal.expected_return,
            threshold=strategy.threshold,
            stop_loss_price=levels.stop_loss_price,
            take_profit_price=levels.take_profit_price,
        )
        logger.info(
            "Opened LONG @ %.6f | SL=%.6f (%.2f%%) TP=%.6f (%.2f%%)",
            current_price,
            levels.stop_loss_price,
            self.config.stop_loss_pct * 100,
            levels.take_profit_price,
            self.config.take_profit_pct * 100,
        )

    def _close_long(
        self,
        current_price: float,
        position: Position,
        reason: str,
    ) -> None:
        pnl = (current_price - position.entry_price) / position.entry_price
        self.trader.execute("SELL", self.config.symbol, position.quantity)
        self._log_trade(
            "SELL",
            current_price,
            reason=reason,
            entry_price=position.entry_price,
            stop_loss_price=position.stop_loss_price,
            take_profit_price=position.take_profit_price,
            exit_price=current_price,
            realized_return=pnl,
            evaluated=True,
        )
        self.store.clear_position()
        logger.info(
            "Closed LONG (%s) entry=%.6f exit=%.6f pnl=%.4f%%",
            reason,
            position.entry_price,
            current_price,
            pnl * 100,
        )

    def _check_sl_tp(self, current_price: float, position: Position) -> bool:
        """Return True if position was closed by SL/TP."""
        if not self.config.enable_sl_tp:
            return False

        levels = RiskLevels(
            stop_loss_price=position.stop_loss_price,
            take_profit_price=position.take_profit_price,
        )
        result = self.risk.check_long_position(current_price, position.entry_price, levels)

        if not result.triggered or result.reason is None:
            logger.info(
                "Position open: entry=%.6f price=%.6f pnl=%.4f%% sl=%.6f tp=%.6f",
                position.entry_price,
                current_price,
                result.pnl_pct * 100,
                position.stop_loss_price,
                position.take_profit_price,
            )
            return False

        self._close_long(current_price, position, result.reason)
        return True

    def run_once(self) -> None:
        self._cycle += 1
        logger.info("========== Cycle %s ==========", self._cycle)

        tune = self.optimizer.run_cycle_updates()
        if tune.message:
            logger.info("Optimizer: %s (retrained=%s)", tune.message, tune.retrained)

        current_price = self.fetcher.latest_close(
            self.config.symbol, self.config.interval
        )
        position = self.store.load_position()

        # 1. Check stop-loss / take-profit before new signals
        if position is not None:
            if self._check_sl_tp(current_price, position):
                return

        strategy = self._strategy()
        restored = self.predictor.predict_future()
        signal = strategy.generate(current_price, restored)

        # 2. Handle signals with position awareness (spot: long only)
        if position is not None:
            if signal.action == "SELL":
                logger.info(
                    "Model SELL signal while in position (expected return %.4f%%)",
                    signal.expected_return * 100,
                )
                self._close_long(current_price, position, "SIGNAL_EXIT")
            else:
                logger.info(
                    "HOLD position (signal=%s, pnl=%.4f%%)",
                    signal.action,
                    (current_price - position.entry_price) / position.entry_price * 100,
                )
            return

        # 3. Flat: only open on BUY; ignore SELL (no short on spot)
        if signal.action == "BUY":
            levels = self.risk.levels_for_long(current_price)
            logger.info(
                "Decision: BUY (expected return %.4f%%)",
                signal.expected_return * 100,
            )
            self._open_long(current_price, signal, strategy, levels)
        elif signal.action == "SELL":
            logger.info("SELL signal ignored — no open position (spot long-only)")
            self._log_trade(
                "HOLD",
                current_price,
                reason="SIGNAL",
                predicted_price=signal.predicted_price,
                expected_return=signal.expected_return,
                threshold=strategy.threshold,
            )
        else:
            logger.info("Decision: HOLD")
            self._log_trade(
                "HOLD",
                current_price,
                reason="SIGNAL",
                predicted_price=signal.predicted_price,
                expected_return=signal.expected_return,
                threshold=strategy.threshold,
            )

    def run_loop(self) -> None:
        sleep_sec = self.config.interval_seconds()
        logger.info(
            "Starting quant engine: symbol=%s interval=%s dry_run=%s "
            "sl=%.2f%% tp=%.2f%% sleep=%ss",
            self.config.symbol,
            self.config.interval,
            self.config.dry_run,
            self.config.stop_loss_pct * 100,
            self.config.take_profit_pct * 100,
            sleep_sec,
        )

        cycles = 0
        while True:
            try:
                self.run_once()
            except Exception as exc:
                logger.error("Cycle failed: %s", exc, exc_info=True)

            cycles += 1
            if self.config.max_cycles > 0 and cycles >= self.config.max_cycles:
                logger.info("Completed %s cycles, stopping.", cycles)
                break

            logger.info("Sleeping %ss until next cycle...", sleep_sec)
            time.sleep(sleep_sec)
