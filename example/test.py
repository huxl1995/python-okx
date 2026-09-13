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
from binanace.klines import fetch_klines
# ---------- 参数配置 ----------
SYMBOL = "BTCUSDT"
INTERVAL = "4h"       # K 线周期: 1m, 5m, 1h, 1d 等
LIMIT = 150          # 拉取条数（Binance 单次最多 1000）
SEQ_LEN = 30
WINDOW_SIZE = SEQ_LEN      # 滚动 Z-Score 窗口# 输入序列长度（用过去 30 根 K 线）
PRED_LEN = 5          # 预测未来 5 根 K 线
EPOCHS = 5
PER_EPOCHS=50
TRADE_FEE_RATE=0.001

MODEL_NAME=str(LIMIT)+"_"+INTERVAL+"_"+str(EPOCHS)+"_"+str(PER_EPOCHS)+"_"+str(SEQ_LEN)+"_"+str(PRED_LEN)+"_"+str(TRADE_FEE_RATE)+"_"+"model.pt"
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
    end_time=datetime(2026,1,20,0,0,0)
    kline_df = fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT,end_time=int(end_time.timestamp())*1000)
    raw_data = kline_df.copy()
    print(kline_df.tail(3))

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

    data_np = kline_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

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
    money_list.append({"date":end_time,"money":0,"clear_money":0})
    while end_time<datetime.now():
        kline_df = fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT, end_time=int(end_time.timestamp()) * 1000)
        raw_data=kline_df.copy()
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

        data_np = kline_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

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
        start_index = len(raw_data)  # 预测的是未来数据，从 raw_data 末尾开始
        restored = restorePredictions(
            scaled_pred,
            raw_data,
            WINDOW_SIZE,
            start_index,
            priceKeys=PRICE_KEYS,
        )

        change_quant=state(restored,quant,kline_df['close'].to_numpy()[-1])
        quant=quant+change_quant
        money -= change_quant * kline_df['close'].to_numpy()[-1]
        if change_quant<=0:
            clear_money -= change_quant * (1 - TRADE_FEE_RATE) * kline_df['close'].to_numpy()[-1]
        else:
            clear_money -= change_quant * (1 + TRADE_FEE_RATE) * kline_df['close'].to_numpy()[-1]
        print(f"num is {num},time is {kline_df['date'].to_numpy()[-1]},state is {state},money is {money},clear_money is {clear_money},quant is {quant},actual clse is {kline_df['close'].to_numpy()[-1]}")
        end_time=end_time+timedelta(hours=4)
        money_list.append({"date":end_time,"money":money+quant*kline_df['close'].to_numpy()[-1],"clear_money":clear_money+quant*kline_df['close'].to_numpy()[-1]})
        num+=1
    money_df=pd.DataFrame(money_list)
    money_df.to_csv(f"./{LIMIT}_{INTERVAL}_{EPOCHS}_{PER_EPOCHS}_{SEQ_LEN}_{PRED_LEN}_{TRADE_FEE_RATE}_money.csv")
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
if __name__ == "__main__":
    simBTC()
