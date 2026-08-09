from time import sleep

from dateutil.relativedelta import relativedelta
import pandas as pd
from dline.config import get_api_credentials
from dline.data import namedHistoryCandleSticks, getEffectiveHistoryCandleSticks
from dline.stand import rollingZScoreStand
from dline.util import subtract_months
from okx import Account
from okx.MarketData import MarketAPI
from datetime import datetime
from dline.data import *
import matplotlib.pyplot as plt
import numpy as np
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
def testPic():
    oriPath="BTC_USDT.csv"
    data=pandaLoadData(oriPath)
    data["closeRolling_Mean"] = (
        data['close'].rolling(window=30, closed="left").mean()
    )
    data.drop(data.index[0:30])
    for i in range(0,len(data)):
        if data['close'][i]>2*data['closeRolling_Mean'][i] or data['close'][i]<0.5*data['closeRolling_Mean'][i]:
            data.drop(index=i,inplace=True)
    data.reset_index(drop=True)
    s = data['close']
    t = data.index
    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='About as simple as it gets, folks')
    ax.grid()

    fig.savefig("test.png")
    plt.show()
def testReturnOriValue():
    oriPath="BTC_USDT.csv"
    data=pandaLoadData(oriPath)
    rollingZScoreStand(data,30,'close')
    data["closeRolling_Mean"] = (
        data['close'].rolling(window=30, closed="left").mean()
    )
    data["closeRolling_Std"] = data["close"].rolling(window=30, closed="left").std()
    print("calculate value is ",data['closeScaled'][50]*(data['closeRolling_Std'][50]+1e-8)+data['closeRolling_Mean'][50])
    print("actual value is ",data['close'][50])
    data.drop(data.index[0:30],inplace=True)
    data.reset_index(inplace=True)
    print("1")

