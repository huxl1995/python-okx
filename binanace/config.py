"""
Test configuration module - loads API credentials from environment variables.

Usage:
    from test.config import get_api_credentials
    
    api_key, api_secret, passphrase, flag = get_api_credentials()
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
    ConfigurationRestAPI,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
)
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent.parent / 'API.env')
def get_binanace_restAPI():
    # Create configuration for the REST API
    return ConfigurationRestAPI(
        api_key=os.getenv("API_KEY", ""),
        api_secret=os.getenv("API_SECRET", ""),
        base_path=os.getenv(
            "BASE_PATH", DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
        ),
    )

