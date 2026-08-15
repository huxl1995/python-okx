"""Configuration loaded from environment variables and defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

QUANT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = QUANT_ROOT.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(QUANT_ROOT / ".env")


@dataclass
class QuantConfig:
    """Runtime configuration for the quant trading system."""

    # Binance API
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("API_SECRET", ""))
    base_path: str = field(
        default_factory=lambda: os.getenv(
            "BASE_PATH", "https://api.binance.com"
        )
    )

    # Trading
    symbol: str = "BNBUSDT"
    interval: str = "1h"
    quantity: str = "0.01"
    dry_run: bool = True

    # Model
    kline_limit: int = 1000
    window_size: int = 30
    seq_len: int = 30
    pred_len: int = 5
    epochs: int = 50
    batch_size: int = 32
    lr: float = 5e-5

    # Strategy
    signal_threshold: float = 0.001

    # Optimizer
    min_trades_for_tune: int = 5
    target_win_rate: float = 0.45
    threshold_step: float = 0.0005
    max_threshold: float = 0.01
    lr_decay: float = 0.9
    min_lr: float = 1e-6

    # Loop
    sleep_seconds: int | None = None
    max_cycles: int = 0

    # Paths
    data_dir: Path = field(default_factory=lambda: QUANT_ROOT / "data_store")
    model_dir: Path = field(default_factory=lambda: QUANT_ROOT / "models")
    _model_path_override: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        if self._model_path_override is not None:
            return self._model_path_override
        return self.model_dir / f"{self.symbol}_{self.interval}_model.pt"

    @property
    def trade_log_path(self) -> Path:
        return self.data_dir / "trades.jsonl"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "optimizer_state.json"

    def interval_seconds(self) -> int:
        mapping = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "8h": 28800,
            "12h": 43200,
            "1d": 86400,
        }
        if self.sleep_seconds is not None:
            return self.sleep_seconds
        return mapping.get(self.interval, 3600)


def load_config(**overrides) -> QuantConfig:
    """Build config with optional CLI overrides."""
    cfg = QuantConfig()
    for key, value in overrides.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
