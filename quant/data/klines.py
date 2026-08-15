"""Fetch OHLCV klines from Binance spot API."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from binance_sdk_spot.spot import ConfigurationRestAPI, SPOT_REST_API_PROD_URL, Spot

if TYPE_CHECKING:
    from quant.config import QuantConfig

logger = logging.getLogger(__name__)

KLINES_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


class KlineFetcher:
    """Binance kline data fetcher."""

    def __init__(self, config: QuantConfig) -> None:
        rest_cfg = ConfigurationRestAPI(
            api_key=config.api_key,
            api_secret=config.api_secret,
            base_path=config.base_path or SPOT_REST_API_PROD_URL,
        )
        self._client = Spot(config_rest_api=rest_cfg)

    def fetch(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Return standardized kline DataFrame."""
        try:
            response = self._client.rest_api.klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
            )
            raw = response.data()
            logger.info("Fetched %s klines for %s %s", len(raw), symbol, interval)

            df = pd.DataFrame(raw, columns=KLINES_COLUMNS)
            df["date"] = pd.to_datetime(df["open_time"], unit="ms")
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df[["date", "open", "high", "low", "close", "volume"]].reset_index(
                drop=True
            )
        except Exception as exc:
            logger.error("fetch_klines failed: %s", exc)
            raise

    def latest_close(self, symbol: str, interval: str) -> float:
        df = self.fetch(symbol=symbol, interval=interval, limit=1)
        return float(df["close"].iloc[-1])
