from time import sleep

from okx import Account
from okx.MarketData import MarketAPI
from test.config import get_api_credentials
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
    aftertime=datetime(2026,6,24,18,00,00,0)
    aftertimestamp=(int)(aftertime.timestamp())
    num=0
    for i in range(200):
        beforetimestamp=aftertimestamp-3600*301
        data = markey.get_history_candlesticks(instId='BTC-USDT', after=aftertimestamp*1000, before=beforetimestamp*1000, bar='1H',limit=300)
        aftertimestamp=beforetimestamp+3600
        sleep(0.1)
        with open("BTC-USDT.txt", "a", encoding='utf-8') as f:
            for li in data['data']:
                f.write(str(num))
                f.write(',')
                num=num+1
                for it in li:
                    f.write(it)
                    f.write(',')
                f.write('\n')