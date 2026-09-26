"""
示例：使用 klines.py 获取 Binance K 线，训练 DLinearForStock 并预测未来价格。

运行方式（在项目根目录）:
    ./venv/bin/python binanace/example.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime,timedelta

from binanace.example import LIMIT

ROOT = Path(__file__).resolve().parent.parent
DLINE_DIR = ROOT / "dline"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DLINE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dline.app import load_model, predict, train_and_save
from dline.stand import CSNStand, LOGZSCOREStand, Type, restorePredictions, rollingZScoreStand
from binanace.future.future_klines import fetch_klines
# ---------- 参数配置 ----------
SYMBOL = "BTCUSDT"
INTERVAL = "4h"       # K 线周期: 1m, 5m, 1h, 1d 等
LIMIT = 150          # 拉取条数（Binance 单次最多 1000）
SEQ_LEN = 15
WINDOW_SIZE = SEQ_LEN      # 滚动 Z-Score 窗口# 输入序列长度（用过去 30 根 K 线）
PRED_LEN = 5          # 预测未来 5 根 K 线
EPOCHS = 50
PER_EPOCHS=5
TRADE_FEE_RATE=0.0005
STOP_MARKET_RATE=0.05
LABEL='future'
MODEL_NAME=str(LIMIT)+"_"+INTERVAL+"_"+str(EPOCHS)+"_"+str(PER_EPOCHS)+"_"+str(SEQ_LEN)+"_"+str(PRED_LEN)+"_"+str(TRADE_FEE_RATE)+"_"+str(STOP_MARKET_RATE)+"_"+LABEL+"_STOP_MARKET"+"model.pt"
MODEL_PATH = Path(__file__).parent / MODEL_NAME
FEATURE_COLUMNS = [
    "openScaled", "highScaled", "lowScaled", "closeScaled",
    "dateMonthSin", "dateMonthCos",
    "dateDaySin", "dateDayCos",
    "dateHourSin", "dateHourCos",
    "volumeLogScaled",
]
PRICE_KEYS = {"open": 0, "high": 1, "low": 2, "close": 3}

def simBTC():
    # 1. 从 Binance 拉取 K 线
    print(f"拉取 {SYMBOL} {INTERVAL} K 线，limit={LIMIT} ...")
    start_time=datetime(2025,1,19,20,0,0)
    begin_time=start_time-timedelta(hours=LIMIT*4)
    kline_df = fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT,end_time=int(start_time.timestamp())*1000)
    print(kline_df.tail(3))
    kline_df1=pd.read_csv("../binanace/future/klines_4h_all.csv")
    kline_df=pd.concat([kline_df,kline_df1],ignore_index=True)
    df_market=pd.read_csv("../binanace/future/klines_1m_all.csv")
    market_price_dict={}
    for i in range(len(df_market)):
        market_price_dict[convert_datetime(df_market['date'][i])]=i
    # 2. 特征标准化（与 dline/example.py 相同流程）
    kline_df["date"] = pd.to_datetime(kline_df["date"])
    for key in ("open", "high", "low", "close"):
        rollingZScoreStand(kline_df, WINDOW_SIZE, key)
    CSNStand(kline_df, Type.MONTH, "date")
    CSNStand(kline_df, Type.DAY, "date")
    CSNStand(kline_df, Type.HOUR, "date")
    LOGZSCOREStand(kline_df, WINDOW_SIZE, "volume")

    # 前 WINDOW_SIZE 行因滚动窗口不足会产生 NaN，丢弃
    kline_df.drop(kline_df.index[0:WINDOW_SIZE], inplace=True)
    kline_df.reset_index(drop=True, inplace=True)
    begin_index=LIMIT
    data_np = kline_df[FEATURE_COLUMNS][0:begin_index-WINDOW_SIZE].to_numpy(dtype=np.float64)

    # 3. 划分训练集与预测输入
    train_data = data_np

    print(f"训练样本数: {len(train_data)}, 输入窗口: {train_data.shape}")

    # 4. 训练 DLinearForStock 并保存
    train_and_save(
        data=train_data,
        save_path=str(MODEL_PATH),
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        epochs=EPOCHS,
    )
    money=0
    clear_money=0
    num=0
    quant=0
    money_list=[]
    money_list.append({"date":start_time,"money":0,"clear_money":0})
    stop_price=0
    while start_time<datetime(2026,9,16,22,0,0):
        begin_index+=1
        loop_kline_df=query_klines(kline_df,start_time-timedelta(hours=4*120),start_time)
        data_np = loop_kline_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        # 3. 划分训练集与预测输入
        train_data = data_np
        # 4. 训练 DLinearForStock 并保存
        train_and_save(
            data=train_data,
            save_path=str(MODEL_PATH),
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            epochs=PER_EPOCHS,
            log=False
        )
        preData=data_np[-SEQ_LEN:]
        # 5. 加载模型并预测
        model = load_model(str(MODEL_PATH))
        scaled_pred = predict(preData, model)

        # 6. 将预测结果还原为真实 OHLC 价格
        start_index = len(loop_kline_df)  # 预测的是未来数据，从 raw_data 末尾开始
        restored = restorePredictions(
            scaled_pred,
            loop_kline_df,
            WINDOW_SIZE,
            start_index,
            priceKeys=PRICE_KEYS,
        )

        change_quant=state(restored,quant,loop_kline_df['close'].to_numpy()[-1])
        quant=quant+change_quant
        money -= change_quant * loop_kline_df['close'].to_numpy()[-1]
        if change_quant<0:
            clear_money -= change_quant * (1 - TRADE_FEE_RATE) * loop_kline_df['close'].to_numpy()[-1]
            stop_price=loop_kline_df['close'].to_numpy()[-1]*(1+STOP_MARKET_RATE)
        elif change_quant>0:
            clear_money -= change_quant * (1 + TRADE_FEE_RATE) * loop_kline_df['close'].to_numpy()[-1]
            stop_price=loop_kline_df['close'].to_numpy()[-1]*(1-STOP_MARKET_RATE)
        print(f"num is {num},time is {loop_kline_df['date'].to_numpy()[-1]},state is {state},money is {money},clear_money is {clear_money},quant is {quant},actual clse is {loop_kline_df['close'].to_numpy()[-1]}")
        market_time=start_time+timedelta(minutes=1)
        while quant!=0 and market_time<start_time+timedelta(hours=4):
            market_price=query_price(market_price_dict,df_market,market_time)
            if quant <0:
                if market_price>=stop_price:
                    clear_money -= quant * (1 - TRADE_FEE_RATE) * stop_price
                    quant=0
                    break
                else:
                    if market_price*(1+STOP_MARKET_RATE)<stop_price:
                        stop_price=market_price*(1+STOP_MARKET_RATE)
            elif quant>0:
                if market_price<=stop_price:
                    clear_money += quant * (1 - TRADE_FEE_RATE) * stop_price
                    quant=0
                    break
                else:
                    if market_price*(1+STOP_MARKET_RATE)>stop_price:
                        stop_price=market_price*(1+STOP_MARKET_RATE)
            market_time=market_time+timedelta(minutes=1)
        start_time=start_time+timedelta(hours=4)
        money_list.append({"date":start_time,"money":money+quant*loop_kline_df['close'].to_numpy()[-1],"clear_money":clear_money+quant*loop_kline_df['close'].to_numpy()[-1]})
        num+=1
    money_df=pd.DataFrame(money_list)
    money_df.to_csv(f"./{LIMIT}_{INTERVAL}_{EPOCHS}_{PER_EPOCHS}_{SEQ_LEN}_{PRED_LEN}_{TRADE_FEE_RATE}_{LABEL}_STOP_MARKET_money.csv")
def state(restored,quant,close_price):
    min_close=min(restored['close'][0],restored['close'][1],restored['close'][2],restored['close'][3],restored['close'][4])
    max_close=max(restored['close'][0],restored['close'][1],restored['close'][2],restored['close'][3],restored['close'][4])
    if close_price>=max_close:
        if quant>=0:
            return -1
        else:
            return 0
    elif close_price<=min_close:
        if quant<=0:
            return 1
        else:
            return 0
    else:
        return 0
def stop_market(df, quant, stop_price):
    if quant>0:
        for i in range(len(df)):
            if df['close'][i]<=stop_price:
                return 0,quant*stop_price,stop_price
            elif df['close'][i]*(1-STOP_MARKET_RATE)>stop_price:
                stop_price=df['close'][i]*(1-STOP_MARKET_RATE)
    elif quant<0:
        for i in range(len(df)):
            if df['close'][i]>=stop_price:
                return 0,quant*stop_price,stop_price
            elif df['close'][i]*(1+STOP_MARKET_RATE)<stop_price:
                stop_price=df['close'][i]*(1+STOP_MARKET_RATE)
    return quant,quant*stop_price,stop_price
def query(df,start_time,end_time):
    start_index=0
    end_index=0
    for i in len(df):
        if df['date'][i]==start_time:
            start_index=i
        elif df['date'][i]==end_time:
            end_index=i
    return df[start_index:end_index].copy()
def query_price(price_dict,df,query_time):
    return df['close'][price_dict[query_time]]
def convert_datetime(string):
    return datetime.strptime(string,"%Y-%m-%d %H:%M:%S")
def query_klines(df,start_time,end_time):
    start_index=df[df['date']==str(start_time)].index.values[0]
    end_index=df[df['date']==str(end_time)].index.values[0]
    return df[start_index:end_index]
if __name__ == "__main__":
    simBTC()
