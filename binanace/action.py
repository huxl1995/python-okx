"""Binance 现货交易操作。"""
import logging
import os
from typing import Literal, Optional

from binance_sdk_spot.rest_api.models import NewOrderSideEnum, NewOrderTypeEnum
from binance_sdk_spot.spot import ConfigurationRestAPI, SPOT_REST_API_PROD_URL, Spot

from binanace.klines import fetch_klines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

configuration_rest_api = ConfigurationRestAPI(
    api_key=os.getenv("API_KEY", ""),
    api_secret=os.getenv("API_SECRET", ""),
    base_path=os.getenv("BASE_PATH", SPOT_REST_API_PROD_URL),
)

client = Spot(config_rest_api=configuration_rest_api)

TradeAction = Literal["BUY", "SELL", "HOLD"]


def get_current_price(symbol: str, interval: str = "1m") -> float:
    """获取最新 K 线收盘价作为当前价格。"""
    df = fetch_klines(symbol=symbol, interval=interval, limit=1)
    return float(df["close"].iloc[-1])


def decide_action(
    current_close: float,
    restored: dict,
    threshold: float = 0.001,
) -> TradeAction:
    """
    根据预测的未来收盘价与当前价比较，决定买卖方向。

    使用预测序列最后一根 K 线的 close 与当前 close 比较。
    threshold 为最小涨跌幅比例，低于该阈值则 HOLD。
    """
    pred_close = float(restored["close"][-1])
    change = (pred_close - current_close) / current_close
    logger.info(
        "Price compare: current=%.6f pred=%.6f change=%.4f%%",
        current_close,
        pred_close,
        change * 100,
    )
    if change > threshold:
        return "BUY"
    if change < -threshold:
        return "SELL"
    return "HOLD"


def _place_market_order(symbol: str, side: NewOrderSideEnum, quantity: str):
    response = client.rest_api.new_order(
        symbol=symbol,
        side=side,
        type=NewOrderTypeEnum.MARKET,
        quantity=quantity,
    )
    logger.info("Order placed: side=%s quantity=%s response=%s", side, quantity, response.data())
    return response


def buy(symbol: str, quantity: str, dry_run: bool = False):
    """市价买入。"""
    if dry_run:
        logger.info("[DRY RUN] BUY %s quantity=%s", symbol, quantity)
        return None
    return _place_market_order(symbol, NewOrderSideEnum.BUY, quantity)


def sell(symbol: str, quantity: str, dry_run: bool = False):
    """市价卖出。"""
    if dry_run:
        logger.info("[DRY RUN] SELL %s quantity=%s", symbol, quantity)
        return None
    return _place_market_order(symbol, NewOrderSideEnum.SELL, quantity)


def execute_trade(
    action: TradeAction,
    symbol: str,
    quantity: str,
    dry_run: bool = False,
) -> Optional[object]:
    """根据决策执行交易，HOLD 时不操作。"""
    if action == "BUY":
        return buy(symbol, quantity, dry_run=dry_run)
    if action == "SELL":
        return sell(symbol, quantity, dry_run=dry_run)
    logger.info("HOLD - no trade executed")
    return None
