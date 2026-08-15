"""Feedback loop: evaluate past trades and tune strategy/model hyperparameters."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from quant.config import QuantConfig
from quant.data.klines import KlineFetcher
from quant.model.predictor import PricePredictor
from quant.storage.trades import OptimizerState, TradeRecord, TradeStore

logger = logging.getLogger(__name__)


@dataclass
class TuneResult:
    state: OptimizerState
    retrained: bool
    message: str


class FeedbackOptimizer:
    """
    1. Evaluate open trades once enough bars have passed (pred_len horizon).
    2. Adjust signal threshold and learning rate based on rolling win rate.
    3. Retrain model with latest market data after each cycle.
    """

    def __init__(
        self,
        config: QuantConfig,
        store: TradeStore,
        fetcher: KlineFetcher | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.fetcher = fetcher or KlineFetcher(config)
        self.predictor = PricePredictor(config, self.fetcher)

    def _default_state(self) -> OptimizerState:
        return OptimizerState(
            signal_threshold=self.config.signal_threshold,
            lr=self.config.lr,
            epochs=self.config.epochs,
        )

    def evaluate_pending_trades(self) -> int:
        """Mark trades whose prediction horizon has elapsed."""
        records = self.store.load_all()
        if not records:
            return 0

        try:
            current_price = self.fetcher.latest_close(
                self.config.symbol, self.config.interval
            )
        except Exception as exc:
            logger.warning("Cannot evaluate trades: %s", exc)
            return 0

        updated = 0
        for record in records:
            if record.evaluated or record.action == "HOLD":
                continue
            # Entry BUY records are evaluated when the position closes.
            if record.action == "BUY":
                continue
            record.exit_price = current_price
            if record.action == "SELL":
                record.realized_return = (record.entry_price - current_price) / record.entry_price
                record.direction_correct = current_price < record.entry_price
            record.evaluated = True
            updated += 1
            logger.info(
                "Evaluated trade cycle=%s action=%s return=%.4f%% correct=%s",
                record.cycle_id,
                record.action,
                (record.realized_return or 0) * 100,
                record.direction_correct,
            )

        if updated:
            self.store.save_all(records)
        return updated

    def _compute_metrics(self, records: list[TradeRecord]) -> tuple[float, float, int]:
        closed = [r for r in records if r.realized_return is not None]
        if not closed:
            return 0.0, 0.0, 0
        wins = sum(1 for r in closed if (r.realized_return or 0) > 0)
        win_rate = wins / len(closed)
        avg_return = sum(r.realized_return or 0 for r in closed) / len(closed)
        return win_rate, avg_return, len(closed)

    def tune(self) -> TuneResult:
        state = self.store.load_state(self._default_state())
        records = self.store.load_all()
        win_rate, avg_return, n_eval = self._compute_metrics(records)
        state.win_rate = win_rate
        state.avg_return = avg_return
        state.evaluated_trades = n_eval
        state.total_trades = len(records)

        message_parts: list[str] = []
        if n_eval >= self.config.min_trades_for_tune:
            if win_rate < self.config.target_win_rate:
                old_threshold = state.signal_threshold
                state.signal_threshold = min(
                    old_threshold + self.config.threshold_step,
                    self.config.max_threshold,
                )
                state.lr = max(state.lr * self.config.lr_decay, self.config.min_lr)
                state.last_tune_reason = (
                    f"win_rate {win_rate:.2%} < target {self.config.target_win_rate:.2%}"
                )
                message_parts.append(
                    f"threshold {old_threshold:.4f} -> {state.signal_threshold:.4f}, "
                    f"lr -> {state.lr:.2e}"
                )
            else:
                old_threshold = state.signal_threshold
                state.signal_threshold = max(
                    old_threshold - self.config.threshold_step * 0.5,
                    self.config.signal_threshold * 0.5,
                )
                state.last_tune_reason = f"win_rate {win_rate:.2%} healthy"
                message_parts.append(f"threshold relaxed {old_threshold:.4f} -> {state.signal_threshold:.4f}")

        self.store.save_state(state)
        return TuneResult(state=state, retrained=False, message="; ".join(message_parts) or "no tune")

    def retrain(self, epochs: int | None = None, lr: float | None = None) -> None:
        state = self.store.load_state(self._default_state())
        self.predictor.train(epochs=epochs or state.epochs, lr=lr or state.lr)

    def run_cycle_updates(self) -> TuneResult:
        """Evaluate, tune hyperparameters, then retrain model."""
        self.evaluate_pending_trades()
        tune_result = self.tune()
        try:
            self.retrain(
                epochs=tune_result.state.epochs,
                lr=tune_result.state.lr,
            )
            tune_result.retrained = True
        except Exception as exc:
            logger.error("Retrain failed: %s", exc, exc_info=True)
            tune_result.message += f"; retrain failed: {exc}"
        return tune_result
