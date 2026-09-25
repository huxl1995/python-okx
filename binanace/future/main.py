import logging
import os
from time import sleep
from datetime import datetime,timedelta

import pandas as pd


from binanace.config import get_binanace_restAPI

logging.basicConfig(level=logging.INFO)

KLINES_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]
if __name__ == "__main__":
    df=pd.read_csv("klines_24h.csv")
    drop_index=[]
    for i in range(len(df)):
        if(i>0 and df.iloc[i,1]==df.iloc[i-1,1]):
            drop_index.append(i)
    df=df.drop(index=drop_index)
    dfa=df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    dfa.to_csv("klines_24h_all.csv")
    print(drop_index)