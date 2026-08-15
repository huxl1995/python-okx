import os

from binance_common.configuration import ConfigurationRestAPI
from binance_common.constants import SPOT_REST_API_PROD_URL
from binance_sdk_spot import Spot
from binance_sdk_spot.rest_api.models import NewOrderSideEnum, NewOrderTypeEnum

configuration_rest_api = ConfigurationRestAPI(
    api_key=os.getenv("API_KEY", ""),
    api_secret=os.getenv("API_SECRET", ""),
    base_path=os.getenv("BASE_PATH", SPOT_REST_API_PROD_URL),
)

client = Spot(config_rest_api=configuration_rest_api)

def buy(symbol,num):
    client.rest_api.new_order(symbol=symbol,side=NewOrderSideEnum.BUY,type=NewOrderTypeEnum.LIMIT,quantity=num,price='61000')