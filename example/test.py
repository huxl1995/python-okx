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
INTERVAL = "1h"       # K 线周期: 1m, 5m, 1h, 1d 等
LIMIT = 1000          # 拉取条数（Binance 单次最多 1000）
WINDOW_SIZE = 30      # 滚动 Z-Score 窗口
SEQ_LEN = 30          # 输入序列长度（用过去 30 根 K 线）
PRED_LEN = 5          # 预测未来 5 根 K 线
EPOCHS = 50
MODEL_PATH = Path(__file__).parent / "model.pt"

FEATURE_COLUMNS = [
    "openScaled", "highScaled", "lowScaled", "closeScaled",
    "dateMonthSin", "dateMonthCos",
    "dateDaySin", "dateDayCos",
    "dateHourSin", "dateHourCos",
    "volumeLogScaled",
]
PRICE_KEYS = {"open": 0, "high": 1, "low": 2, "close": 3}

def simBTC(interval):
    # 1. 从 Binance 拉取 K 线
    print(f"拉取 {SYMBOL} {INTERVAL} K 线，limit={LIMIT} ...")
    end_time=datetime(2026,1,20,0,0,0)
    kline_df = fetch_klines(symbol='BTCUSDT', interval='1h', limit=1000,end_time=int(end_time.timestamp())*1000)
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
    have=False
    money=0
    num=0
    quant=0
    while end_time<datetime.now():
        kline_df = fetch_klines(symbol='BTCUSDT', interval='1h', limit=2*WINDOW_SIZE, end_time=int(end_time.timestamp())*1000)
        raw_data = kline_df.copy()
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
        # 5. 加载模型并预测
        model = load_model(str(MODEL_PATH))
        scaled_pred = predict(data_np, model)

        # 6. 将预测结果还原为真实 OHLC 价格
        start_index = len(raw_data)  # 预测的是未来数据，从 raw_data 末尾开始
        restored = restorePredictions(
            scaled_pred,
            raw_data,
            WINDOW_SIZE,
            start_index,
            priceKeys=PRICE_KEYS,
        )
        state="hold"
        if restored['close'][0]>kline_df['close'].to_numpy()[-1]:
            if quant<=0:
                quant+=1
                state = "buy"
                money-=kline_df['close'].to_numpy()[-1]
        else:
            if quant>0:
                state="sell"
                quant-=1
                money+=kline_df['close'].to_numpy()[-1]
        print(f"num is {num},state is {state},money is {money},quant is {quant},actual clse is {kline_df['close'].to_numpy()[-1]}")
        end_time=end_time+timedelta(hours=1)
if __name__ == "__main__":
    simBTC('1h')
