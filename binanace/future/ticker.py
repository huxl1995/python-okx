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
    NewOrderSideEnum, NewAlgoOrderSideEnum,
)
from dotenv import load_dotenv

from binanace.config import get_binanace_restAPI
# Configure logging
logging.basicConfig(level=logging.INFO)


# Initialize DerivativesTradingUsdsFutures client
client = DerivativesTradingUsdsFutures(config_rest_api=get_binanace_restAPI())

def mark_price(symbol):
    try:
        response = client.rest_api.mark_price(symbol)

        rate_limits = response.rate_limits
        logging.info(f"mark_price() rate limits: {rate_limits}")

        data = response.data()
        logging.info(f"mark_price() response: {data}")
    except Exception as e:
        logging.error(f"mark_price() error: {e}")

def symbol_price_ticker_v2(symbol):
    try:
        response = client.rest_api.symbol_price_ticker_v2(symbol)

        rate_limits = response.rate_limits
        logging.info(f"price() rate limits: {rate_limits}")

        data = response.data()
        return data.actual_instance.price
        logging.info(f"price() response: {data}")
    except Exception as e:
        logging.error(f"price() error: {e}")
if __name__ == "__main__":
    symbol_price_ticker_v2("BTCUSDT")