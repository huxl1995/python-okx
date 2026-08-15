"""Persist trade records and optimizer state."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    cycle_id: int
    symbol: str
    interval: str
    action: str
    quantity: str
    entry_price: float
    predicted_price: float
    expected_return: float
    threshold: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    exit_price: Optional[float] = None
    realized_return: Optional[float] = None
    direction_correct: Optional[bool] = None
    evaluated: bool = False


@dataclass
class OptimizerState:
    signal_threshold: float
    lr: float
    epochs: int
    total_trades: int = 0
    evaluated_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    last_tune_reason: str = ""


class TradeStore:
    """Append-only JSONL trade log with evaluation support."""

    def __init__(self, log_path: Path, state_path: Path) -> None:
        self.log_path = log_path
        self.state_path = state_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: TradeRecord) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def load_all(self) -> list[TradeRecord]:
        if not self.log_path.exists():
            return []
        records: list[TradeRecord] = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(TradeRecord(**json.loads(line)))
        return records

    def save_all(self, records: list[TradeRecord]) -> None:
        with self.log_path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def load_state(self, defaults: OptimizerState) -> OptimizerState:
        if not self.state_path.exists():
            return defaults
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        return OptimizerState(**{**asdict(defaults), **data})

    def save_state(self, state: OptimizerState) -> None:
        self.state_path.write_text(
            json.dumps(asdict(state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def pending_evaluations(self) -> list[tuple[int, TradeRecord]]:
        records = self.load_all()
        return [(i, r) for i, r in enumerate(records) if not r.evaluated and r.action != "HOLD"]
