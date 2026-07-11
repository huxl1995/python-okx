from time import sleep

from dateutil.relativedelta import relativedelta
import pandas as pd
from dline.config import get_api_credentials
from dline.data import namedHistoryCandleSticks, getEffectiveHistoryCandleSticks
from dline.util import subtract_months
from okx import Account
from okx.MarketData import MarketAPI
from datetime import datetime

ms_timestamp = 1719163560000  # 毫秒级时间戳
dt_obj = datetime.fromtimestamp(ms_timestamp / 1000)
print(dt_obj.strftime("%Y-%m-%d %H:%M:%S"))
# 输出：2024-06-23 11:46:00

def testAccount():
    api_key, api_secret_key, passphrase, flag = get_api_credentials()
    accountAPI = Account.AccountAPI(api_key, api_secret_key, passphrase, flag=flag)
    accountBanance=accountAPI.get_account_balance()
    print(1)
def testMarket():
    api_key, api_secret_key, passphrase, flag = get_api_credentials()
    markey=MarketAPI(api_key,api_secret_key,passphrase,None,flag)
    print(1)
def testIndex():
    api_key, api_secret_key, passphrase, flag = get_api_credentials()
    markey=MarketAPI(api_key,api_secret_key,passphrase,None,flag)
    index=markey.get_index_tickers(quoteCcy='USDT',instId='BTC-USDT')
    time=datetime.fromtimestamp((float)(index['data'][0]['ts'])/1000)
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
def testHistory():
    api_key, api_secret_key, passphrase, flag = get_api_credentials()
    markey = MarketAPI(api_key, api_secret_key, passphrase, None, flag)
    beforetime=datetime(2022,5,26,18,00,00,0)
    num=0
    for i in range(10):
        aftertime=beforetime + relativedelta(months=6)
        data = markey.get_history_candlesticks(instId='BTC-USDT', before=int(beforetime.timestamp())*1000, after=int(aftertime.timestamp())*1000 ,bar='1D',limit=300)
        # data = markey.get_history_candlesticks(instId='BTC-USDT', after=int(aftertime.timestamp())*1000)
        beforetime=aftertime
        sleep(0.1)
        with open("BTC-USDT.txt", "a", encoding='utf-8') as f:
            for li in reversed(data['data']):
                f.write(str(num))
                f.write(',')
                num=num+1
                for it in li:
                    f.write(it)
                    f.write(',')
                f.write('\n')
    oriPath="BTC-USDT.txt"
    data=namedHistoryCandleSticks(pd.read_csv(oriPath))
    data=getEffectiveHistoryCandleSticks(data)
    data.to_csv('BTC_USDT.csv',header=True)