"""
示例：与 test.py 相同的训练/回测流程，
使用 RealtimeRollingZScore 做滚动 Z-Score 归一化，
使用 IncludedSelfZScoreRestorerFixed 将预测 Z-Score 逆归一化为价格。

运行方式（在项目根目录）:
    ./venv/bin/python example/run_realtime_zscore.py
"""
import sys
import time
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import schedule

from binanace.future.account import get_position_amt
from binanace.future.trade import cancel_all_open_orders, new_limit_order

ROOT = Path(__file__).resolve().parent.parent
DLINE_DIR = ROOT / "dline"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DLINE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dline.app import load_model, predict, train_and_save
from dline.stand import CSNStand, Type
from dline.realtimeRollingZScore import RealtimeRollingZScore
from dline.includedSelfZScoreRestorer import IncludedSelfZScoreRestorerFixed
from binanace.future.future_klines import fetch_klines
from decimal import Decimal, ROUND_FLOOR

# ---------- 参数配置 ----------
SYMBOL = "BTCUSDT"
INTERVAL = "12h"       # K 线周期: 1m, 5m, 1h, 1d 等
LIMIT = 150          # 拉取条数（Binance 单次最多 1000）
SEQ_LEN = 30
WINDOW_SIZE = SEQ_LEN      # 滚动 Z-Score 窗口# 输入序列长度（用过去 30 根 K 线）
PRED_LEN = 3          # 预测未来 5 根 K 线
EPOCHS = 100
PER_EPOCHS = 10
TRADE_FEE_RATE = 0.0005
STOP_MARKET_RATE = 0.04
PROFIT_MARKET_RATE=0.02
LABEL = "future_rtz"
MODEL_NAME = (
    str(LIMIT) + "_" + INTERVAL + "_" + str(EPOCHS) + "_" + str(PER_EPOCHS)
    + "_" + str(SEQ_LEN) + "_" + str(PRED_LEN) + "_" + str(TRADE_FEE_RATE)
    + "_" + LABEL + "_" + "model.pt"
)
MODEL_PATH = Path(__file__).parent / MODEL_NAME


FEATURE_COLUMNS = [
    "openScaled", "highScaled", "lowScaled", "closeScaled",
    "volumeLogScaled"
]
PRICE_KEYS = {"open": 0, "high": 1, "low": 2, "close": 3}

# 全局变量定义
MONEY = 0
CLEAR_MONEY = 0
PRICE=0

def apply_realtime_rolling_zscore(df, window_size, key):
    """用 RealtimeRollingZScore 对 df[key] 做包含自身的滚动 Z-Score，写入 keyScaled。"""
    rt = RealtimeRollingZScore(window=window_size)
    scaled = []
    for value in df[key].to_numpy():
        zscore = rt.update(float(value))
        scaled.append(np.nan if zscore is None else zscore)
    df[key + "Scaled"] = scaled
    return df


def apply_log_realtime_zscore(df, window_size, key):
    """log1p 后再用 RealtimeRollingZScore，写入 keyLogScaled（对应原 LOGZSCOREStand）。"""
    rt = RealtimeRollingZScore(window=window_size)
    scaled = []
    for value in np.log1p(df[key].to_numpy(dtype=np.float64)):
        zscore = rt.update(float(value))
        scaled.append(np.nan if zscore is None else zscore)
    df[key + "LogScaled"] = scaled
    return df


def stand_features(kline_df):
    kline_df["date"] = pd.to_datetime(kline_df["date"])
    for key in ("open", "high", "low", "close"):
        apply_realtime_rolling_zscore(kline_df, WINDOW_SIZE, key)
    CSNStand(kline_df, Type.MONTH, "date")
    CSNStand(kline_df, Type.DAY, "date")
    CSNStand(kline_df, Type.HOUR, "date")
    apply_log_realtime_zscore(kline_df, WINDOW_SIZE, "volume")
    kline_df.drop(kline_df.index[0:WINDOW_SIZE], inplace=True)
    kline_df.reset_index(drop=True, inplace=True)
    return kline_df


def restore_predictions_included_self(predictions, raw_df, window_size, price_keys):
    pred_len = predictions.shape[0]
    result = {key: np.zeros(pred_len) for key in price_keys}
    restorers = {}
    for key in price_keys:
        restorer = IncludedSelfZScoreRestorerFixed(window=window_size)
        history = raw_df[key].to_numpy(dtype=np.float64)[-(window_size - 1):]
        for price in history:
            restorer.push_history(float(price))
        restorers[key] = restorer

    for day in range(pred_len):
        for key, feat_idx in price_keys.items():
            zscore = float(predictions[day, feat_idx])
            try:
                ori_value = restorers[key].restore_today_value(zscore)
            except ValueError:
                ori_value = None
            if ori_value is None:
                ori_value = restorers[key].history_window[-1]
            result[key][day] = ori_value
            restorers[key].push_history(float(ori_value))
    return result


def dojob():
    global MONEY, CLEAR_MONEY  # 提到函数最顶部声明
    # 将核心持仓量、仓位数据初始化为 Decimal 确保全局运算一致
    kline_df = fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT, end_time=int(datetime.now().timestamp()) * 1000)
    raw_data = kline_df.copy()
    stand_features(kline_df)
    data_np = kline_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    train_data = data_np
    train_and_save(
        data=train_data,
        save_path=str(MODEL_PATH),
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        epochs=PER_EPOCHS,
        log=False,
        lr=5e-6,
    )
    preData = data_np[-SEQ_LEN:]
    model = load_model(str(MODEL_PATH))
    scaled_pred = predict(preData, model)
    restored = restore_predictions_included_self(
        scaled_pred,
        raw_data,
        WINDOW_SIZE,
        PRICE_KEYS,
    )
    cancel_all_open_orders(SYMBOL)
    quant = get_position_amt(SYMBOL)    # 获取到的变动数量统一强制转为 Decimal
    change_quant = Decimal(str(get_ori_change_quant(kline_df, quant, restored)))
    if change_quant < 0:
        new_limit_order(SYMBOL, "SELL", -1*change_quant, kline_df['close'].to_numpy()[-1])
        print(f"do job once, side is SELL")
    elif change_quant > 0:
        new_limit_order(SYMBOL, "BUY", change_quant, kline_df['close'].to_numpy()[-1])
        print(f"do job once, side is BUY")
    else:
        print(f"do job once, side is HOLD")


def get_ori_change_quant(kline_df, quant, restored):
    return state3(restored, quant, float(kline_df["close"].to_numpy()[-1]))


def state3(restored, quant, close_price):
    global MONEY  # 显式声明全局变量
    quant = Decimal(str(quant))
    # 1. 修复切片问题，取预测未来的前3个周期的最值
    min_close = min(restored["close"][0], restored["close"][1], restored["close"][2])
    max_close = max(restored["close"][0], restored["close"][1], restored["close"][2])

    # 2. 核心逻辑重构
    # 2. 核心逻辑重构
    if close_price > max_close:
        if quant > 0:
            return -1 * quant
        elif quant == 0:
            return (-1 * Decimal('2000') / Decimal(str(close_price))).quantize(Decimal('0.001'), ROUND_FLOOR)
        else:
            return Decimal('0')

    elif close_price < min_close:
        if quant == 0:
            return (Decimal('2000') / Decimal(str(close_price))).quantize(Decimal('0.001'), ROUND_FLOOR)
        elif quant < 0:
            return -1 * quant
        else:
            return Decimal('0')
    else:
        # 3. 修复原代码兜底直接清仓
        # /加仓的 Bug：在区间内波动时，保持仓位不动
        return Decimal('0')

#
# def state_clear(restored, quant, close_price):
#     global CLEAR_MONEY  # 显式声明全局变量
#     quant = Decimal(str(quant))
#
#     # 1. 修复切片问题
#     min_close = min(restored["close"][0], restored["close"][1], restored["close"][2])
#     max_close = max(restored["close"][0], restored["close"][1], restored["close"][2])
#
#     # 2. 核心逻辑重构
#     if close_price > max_close:
#         if quant > 0:
#             return -1 * quant
#         elif quant==0:
#             return (-1*Decimal('100000') / Decimal(str(close_price))).quantize(Decimal('0.0001'), ROUND_FLOOR)
#         else:
#             return Decimal('0')
#
#     elif close_price < min_close:
#         if quant == 0:
#             return (Decimal('100000') / Decimal(str(close_price))).quantize(Decimal('0.0001'), ROUND_FLOOR)
#         elif quant<0:
#             return -1*quant
#         else:
#             return Decimal('0')
#     else:
#         return Decimal('0')

def stop_market(kline_df, quant):
    min_price = float(kline_df["low"].to_numpy()[-1])
    max_price = float(kline_df["high"].to_numpy()[-1])
    if quant<0 and Decimal(max_price)>PRICE*Decimal(1+STOP_MARKET_RATE):
        return quant*-1,PRICE*Decimal(1+STOP_MARKET_RATE)
    elif quant>0 and Decimal(min_price)<PRICE*Decimal(1-STOP_MARKET_RATE):
        return quant*-1,PRICE*Decimal(1-STOP_MARKET_RATE)
    elif quant<0 and Decimal(min_price)<=PRICE*Decimal(1-PROFIT_MARKET_RATE):
        return quant*-1,PRICE*Decimal(1-PROFIT_MARKET_RATE)
    elif quant>0 and Decimal(max_price)>=PRICE*Decimal(1+PROFIT_MARKET_RATE):
        return quant*-1,PRICE*Decimal(1+PROFIT_MARKET_RATE)
    else:
        return 0,0
if __name__ == "__main__":
    schedule.every().day.at("23:59").do(dojob)
    schedule.every().day.at("11:59").do(dojob)
    while True:
        schedule.run_pending()
        time.sleep(1)