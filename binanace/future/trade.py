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

def new_market_order(symbol,side,quant):
    try:
        response = client.rest_api.new_order(
            symbol=symbol,
            side=NewOrderSideEnum[side].value,
            type="MARKET",
            quantity=quant
        )

        rate_limits = response.rate_limits
        logging.info(f"new_order() rate limits: {rate_limits}")

        data = response.data()
        logging.info(f"new_order() response: {data}")
    except Exception as e:
        logging.error(f"new_order() error: {e}")
def new_limit_order(symbol,side,quant,price):
    try:
        response = client.rest_api.new_order(
            symbol=symbol,
            side=NewOrderSideEnum[side].value,
            type="LIMIT",
            quantity=quant,
            price=price,
            time_in_force="GTC"
        )

        rate_limits = response.rate_limits
        logging.info(f"new_order() rate limits: {rate_limits}")

        data = response.data()
        logging.info(f"new_order() response: {data}")
    except Exception as e:
        logging.error(f"new_order() error: {e}")
def cancel_all_open_orders(symbol):
    try:
        response = client.rest_api.cancel_all_open_orders(
            symbol=symbol,
        )

        rate_limits = response.rate_limits
        logging.info(f"cancel_all_open_orders() rate limits: {rate_limits}")

        data = response.data()
        logging.info(f"cancel_all_open_orders() response: {data}")
    except Exception as e:
        logging.error(f"cancel_all_open_orders() error: {e}")
def new_algo_order(symbol,side,trigger_price):
    try:
        response = client.rest_api.new_algo_order(
            algo_type="CONDITIONAL",
            symbol=symbol,
            side=NewAlgoOrderSideEnum[side].value,
            type="STOP_MARKET",
            working_type="MARK_PRICE",
            close_position="true",
            trigger_price=trigger_price,
            time_in_force="GTC"
        )

        rate_limits = response.rate_limits
        logging.info(f"new_algo_order() rate limits: {rate_limits}")

        data = response.data()
        logging.info(f"new_algo_order() response: {data}")
    except Exception as e:
        logging.error(f"new_algo_order() error: {e}")
def cancel_all_algo_open_orders(symbol):
    try:
        response = client.rest_api.cancel_all_algo_open_orders(
            symbol=symbol,
        )

        rate_limits = response.rate_limits
        logging.info(f"cancel_all_algo_open_orders() rate limits: {rate_limits}")

        data = response.data()
        logging.info(f"cancel_all_algo_open_orders() response: {data}")
    except Exception as e:
        logging.error(f"cancel_all_algo_open_orders() error: {e}")
def trailing_stop_market(symbol):
    response = client.rest_api.position_information_v3(symbol)
    data=response.data()
    print(1)
if __name__=="__main__":
    #new_order("BTCUSDT","BUY","MARKET",1)
    #new_market_order("BTCUSDT","SELL",0.01)
    #new_algo_order("BTCUSDT","SELL",81000)
    trailing_stop_market("BTCUSDT")
