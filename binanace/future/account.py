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



def account_information_v3():
    try:
        response = client.rest_api.account_information_v3()

        rate_limits = response.rate_limits
        logging.info(f"account_information_v3() rate limits: {rate_limits}")

        data = response.data()
        logging.info(f"account_information_v3() response: {data}")
    except Exception as e:
        logging.error(f"account_information_v3() error: {e}")
def get_position_amt(symbol):
    response = client.rest_api.account_information_v3()
    data = response.data()
    for position in data.positions:
        if position.symbol==symbol:
            return float(position.position_amt)
    return 0
if __name__ == "__main__":
    print(get_position_amt('BTCUSDT'))