import os
import logging
from pathlib import Path

from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
    ConfigurationRestAPI,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
)
from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
    NewOrderSideEnum,
)
from dotenv import load_dotenv

from binanace.config import get_binanace_restAPI
# Configure logging
logging.basicConfig(level=logging.INFO)


# Initialize DerivativesTradingUsdsFutures client
client = DerivativesTradingUsdsFutures(config_rest_api=get_binanace_restAPI())

def new_order(symbol,side,type):
    try:
        response = client.rest_api.new_order(
            symbol=symbol,
            side=NewOrderSideEnum[side].value,
            type=type,
        )

        rate_limits = response.rate_limits
        logging.info(f"new_order() rate limits: {rate_limits}")

        data = response.data()
        logging.info(f"new_order() response: {data}")
    except Exception as e:
        logging.error(f"new_order() error: {e}")
if __name__=="__main__":
    new_order("BTCUSDT","BUY","MARKET")