import os

from binance_sdk_spot.spot import Spot, ConfigurationRestAPI

from config import get_binance_api_credentials
apiKey,privateKey=get_binance_api_credentials()
configuration_rest_api = ConfigurationRestAPI(
    private_key=apiKey,
    private_key_passphrase=privateKey,
)
spot_client = Spot(config_rest_api=configuration_rest_api)

account_info = spot_client.rest_api.get_account()