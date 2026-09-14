import logging
import os
from typing import Optional

import pandas as pd

from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
    ConfigurationRestAPI,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
)
from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
    KlineCandlestickDataIntervalEnum,
)

from binanace.config import get_binanace_restAPI

logging.basicConfig(level=logging.INFO)

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
# Initialize DerivativesTradingUsdsFutures client
client = DerivativesTradingUsdsFutures(config_rest_api=get_binanace_restAPI())



def fetch_klines(
    symbol: str = "BNBUSDT",
    interval: str = KlineCandlestickDataIntervalEnum["INTERVAL_1m"].value,
    limit: int = 1000,
    start_time = None,
    end_time = None,
) -> pd.DataFrame:
    """
    从 Binance 获取 K 线数据并转换为标准 DataFrame。

    Returns:
        包含 date, open, high, low, close, volume 列的 DataFrame
    """
    try:
        response = client.rest_api.kline_candlestick_data(
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time
        )

        rate_limits = response.rate_limits
        logging.info("fetch_klines() rate limits: %s", rate_limits)

        raw = response.data()
        logging.info("fetch_klines() fetched %s rows", len(raw))

        df = pd.DataFrame(raw, columns=KLINES_COLUMNS)
        df["date"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    except Exception as e:
        logging.error("fetch_klines() error: %s", e)
        raise


def klines(
    symbol: str = "BNBUSDT",
    interval: str = KlineCandlestickDataIntervalEnum["INTERVAL_1m"].value,
    limit: int = 1000,
) -> Optional[pd.DataFrame]:
    """兼容旧接口，返回标准化后的 K 线 DataFrame。"""
    return fetch_klines(symbol=symbol, interval=interval, limit=limit)


if __name__ == "__main__":
    data = fetch_klines(interval=KlineCandlestickDataIntervalEnum.INTERVAL_1d)
    logging.info("klines sample:\n%s", data.tail())
