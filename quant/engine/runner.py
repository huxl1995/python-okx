"""Trading engine: orchestrates train -> predict -> trade -> optimize loop."""
from __future__ import annotations

import logging
import time

from quant.config import QuantConfig
from quant.data.klines import KlineFetcher
from quant.exchange.client import BinanceTrader
from quant.model.predictor import PricePredictor
from quant.optimizer.feedback import FeedbackOptimizer
from quant.storage.trades import TradeRecord, TradeStore
from quant.strategy.signal import MLStrategy

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main quant trading orchestrator."""

    def __init__(self, config: QuantConfig) -> None:
        self.config = config
        self.fetcher = KlineFetcher(config)
        self.trader = BinanceTrader(config)
        self.store = TradeStore(config.trade_log_path, config.state_path)
        self.optimizer = FeedbackOptimizer(config, self.store, self.fetcher)
        self.predictor = PricePredictor(config, self.fetcher)
        self._cycle = 0

    def _strategy(self) -> MLStrategy:
        state = self.store.load_state(self.optimizer._default_state())
        return MLStrategy(threshold=state.signal_threshold)

    def run_once(self) -> None:
        self._cycle += 1
        logger.info("========== Cycle %s ==========", self._cycle)

        # 1. Evaluate past trades and tune hyperparameters
        tune = self.optimizer.run_cycle_updates()
        if tune.message:
            logger.info("Optimizer: %s (retrained=%s)", tune.message, tune.retrained)

        strategy = self._strategy()

        # 2. Predict future prices with freshly trained model
        restored = self.predictor.predict_future()
        current_price = self.fetcher.latest_close(
            self.config.symbol, self.config.interval
        )

        # 3. Generate signal and execute
        signal = strategy.generate(current_price, restored)
        logger.info("Decision: %s (expected return %.4f%%)", signal.action, signal.expected_return * 100)
        self.trader.execute(signal.action, self.config.symbol, self.config.quantity)

        # 4. Log trade for later evaluation
        record = TradeRecord(
            cycle_id=self._cycle,
            symbol=self.config.symbol,
            interval=self.config.interval,
            action=signal.action,
            quantity=self.config.quantity,
            entry_price=current_price,
            predicted_price=signal.predicted_price,
            expected_return=signal.expected_return,
            threshold=strategy.threshold,
        )
        self.store.append(record)

    def run_loop(self) -> None:
        sleep_sec = self.config.interval_seconds()
        logger.info(
            "Starting quant engine: symbol=%s interval=%s dry_run=%s sleep=%ss",
            self.config.symbol,
            self.config.interval,
            self.config.dry_run,
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
