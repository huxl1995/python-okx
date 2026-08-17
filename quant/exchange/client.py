"""Binance spot trading client."""
from __future__ import annotations

import logging
from typing import Literal, Optional

from binance_sdk_spot.rest_api.models import NewOrderSideEnum, NewOrderTypeEnum
from binance_sdk_spot.spot import ConfigurationRestAPI, SPOT_REST_API_PROD_URL, Spot

from quant.config import QuantConfig

logger = logging.getLogger(__name__)

TradeAction = Literal["BUY", "SELL", "HOLD"]


class BinanceTrader:
    """Execute spot market orders on Binance."""

    def __init__(self, config: QuantConfig) -> None:
        rest_cfg = ConfigurationRestAPI(
            api_key=config.api_key,
            api_secret=config.api_secret,
            base_path=config.base_path or SPOT_REST_API_PROD_URL,
        )
        self._client = Spot(config_rest_api=rest_cfg)
        self._dry_run = config.dry_run

    def place_market_order(self, symbol: str, side: TradeAction, quantity: str):
        if side == "HOLD":
            logger.info("HOLD - no order placed")
            return None

        side_enum = NewOrderSideEnum.BUY if side == "BUY" else NewOrderSideEnum.SELL
        if self._dry_run:
            logger.info("[DRY RUN] %s %s quantity=%s", side, symbol, quantity)
            return {"dry_run": True, "side": side, "symbol": symbol, "quantity": quantity}
        balances = self._client.rest_api.get_account(omit_zero_balances=True,recv_window=10000).data().balances
        is_new_order=False
        for balance in balances:
            if balance.asset==symbol:
                if balance.free<quantity and side=='BUY':
                    is_new_order=True
                elif balance.free>=quantity and side!='BUY':
                    is_new_order=True
        if not is_new_order:
            logger.info("[BALANCE IS NOT] %s %s quantity=%s", side, symbol, quantity)
            return {"dry_run": False, "side": side, "symbol": symbol, "quantity": quantity}
        response = self._client.rest_api.new_order(
            symbol=symbol,
            side=side_enum,
            type=NewOrderTypeEnum.MARKET,
            quantity=quantity,
        )
        payload = response.data()
        logger.info("Order executed: %s %s qty=%s -> %s", side, symbol, quantity, payload)
        return payload

    def execute(self, action: TradeAction, symbol: str, quantity: str) -> Optional[object]:
        if action == "HOLD":
            logger.info("Strategy signal HOLD")
            return None
        return self.place_market_order(symbol, action, quantity)
