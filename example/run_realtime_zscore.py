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
STOP_MARKET_RATE = 1
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


def simBTC():
    global MONEY, CLEAR_MONEY  # 提到函数最顶部声明

    # 1. 从 Binance 拉取 K 线
    print(f"拉取 {SYMBOL} {INTERVAL} K 线，limit={LIMIT} ...")
    end_time = datetime(2025, 1, 20, 0, 0, 0)
    kline_df = fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT, end_time=int(end_time.timestamp()) * 1000)
    raw_data = kline_df.copy()
    print(kline_df.tail(3))

    # 2. 特征标准化
    stand_features(kline_df)
    data_np = kline_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
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
    init_model_name = MODEL_NAME + ".init"
    shutil.copy(MODEL_PATH, Path(__file__).parent / init_model_name)

    # 将核心持仓量、仓位数据初始化为 Decimal 确保全局运算一致
    num = 0
    quant = Decimal('0')
    money_list = []
    control_history = []
    money_list.append({"date": end_time, "money": 0, "clear_money": 0})

    while end_time < datetime.now():
        time.sleep(0.1)
        kline_df = fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT, end_time=int(end_time.timestamp()) * 1000)
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
            log=False
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
        global PRICE
        # 获取到的变动数量统一强制转为 Decimal
        change_quant = Decimal(str(get_ori_change_quant(kline_df, quant, restored)))
        current_close = Decimal(str(kline_df["close"].to_numpy()[-1]))
        stop_change_quant,stop_price=stop_market(kline_df,quant)
        if stop_change_quant!=0:
            quant=quant+stop_change_quant
            MONEY = MONEY - stop_change_quant * stop_price
            PRICE=stop_price
            if stop_change_quant <= 0:
                CLEAR_MONEY = CLEAR_MONEY - stop_change_quant * (Decimal('1') - Decimal(str(TRADE_FEE_RATE))) * stop_price
            else:
                CLEAR_MONEY = CLEAR_MONEY - stop_change_quant * (Decimal('1') + Decimal(str(TRADE_FEE_RATE))) * stop_price
        elif change_quant != 0:
            quant = quant + change_quant
            PRICE=current_close
            MONEY = MONEY - change_quant * current_close
            # 修复原逻辑中 clear_change_quant 误加到常规 change_quant 的逻辑隐患
            if change_quant <= 0:
                CLEAR_MONEY = CLEAR_MONEY - change_quant * (Decimal('1') - Decimal(str(TRADE_FEE_RATE))) * current_close
            else:
                CLEAR_MONEY = CLEAR_MONEY - change_quant * (Decimal('1') + Decimal(str(TRADE_FEE_RATE))) * current_close
        money_str = str(MONEY)
        clear_money_str = str(CLEAR_MONEY)
        print(
            f"num is {num},time is {kline_df['date'].to_numpy()[-1]},state is {change_quant},"
            f"money is {money_str},quant is {quant},clear state is {change_quant},clear_money is {clear_money_str},clear quant is {quant},"
            f"actual clse is {current_close}"
        )
        control_history.append({
            "date": end_time,
            "change": float(change_quant),
            "open": kline_df["open"].to_numpy()[-1],
            "high": kline_df["high"].to_numpy()[-1],
            "low": kline_df["low"].to_numpy()[-1],
            "close": kline_df["close"].to_numpy()[-1],
        })
        # 修复 Bug: 计算总资产时统一使用 Decimal 乘加，最后转回 str
        money_list.append({
            "date": end_time,
            "money": str(MONEY + quant * current_close),
            "clear_money": str(CLEAR_MONEY + quant * current_close), # 修改原逻辑中误用常规 quant 的问题
        })
        end_time = end_time + timedelta(hours=12)
        num += 1

    control_history = pd.DataFrame(control_history)
    control_history.to_csv(
        f"./{LIMIT}_{INTERVAL}_{EPOCHS}_{PER_EPOCHS}_{SEQ_LEN}_{PRED_LEN}_{TRADE_FEE_RATE}_{LABEL}_change_history.csv"
    )
    money_df = pd.DataFrame(money_list)
    money_df.to_csv(
        f"./{LIMIT}_{INTERVAL}_{EPOCHS}_{PER_EPOCHS}_{SEQ_LEN}_{PRED_LEN}_{TRADE_FEE_RATE}_{LABEL}_money.csv"
    )


# def get_change_quant(kline_df, restored):
#     quant = Decimal(str(ASSERT.quant))
#     price = Decimal(str(ASSERT.price))
#     current_close = Decimal(str(kline_df["close"].to_numpy()[-1]))
#     stop_rate = Decimal(str(STOP_MARKET_RATE))
#
#     if quant > 0 and price * (Decimal('1') - stop_rate) >= current_close:
#         change_quant = -1 * quant
#     elif quant < 0 and price * (Decimal('1') + stop_rate) <= current_close:
#         change_quant = -1 * quant
#     else:
#         change_quant = state3(restored, quant, float(current_close))
#     return change_quant, quant


def get_ori_change_quant(kline_df, quant, restored):
    return state3(restored, quant, float(kline_df["close"].to_numpy()[-1]))

#
# def state(restored, quant, close_price):
#     min_close = min(restored["close"][0], restored["close"][1], restored["close"][2], restored["close"][3], restored["close"][4])
#     max_close = max(restored["close"][0], restored["close"][1], restored["close"][2], restored["close"][3], restored["close"][4])
#     if close_price >= max_close:
#         if quant >= 0:
#             return -1
#         else:
#             return 0
#     elif close_price <= min_close:
#         if quant <= 0:
#             return 1
#         else:
#             return 0
#     else:
#         return 0


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
            return (-1 * Decimal('100000') / Decimal(str(close_price))).quantize(Decimal('0.0001'), ROUND_FLOOR)
        else:
            return Decimal('0')

    elif close_price < min_close:
        if quant == 0:
            return (Decimal('100000') / Decimal(str(close_price))).quantize(Decimal('0.0001'), ROUND_FLOOR)
        elif quant < 0:
            return -1 * quant
        else:
            return Decimal('0')
    else:
        # 3. 修复原代码兜底直接清仓/加仓的 Bug：在区间内波动时，保持仓位不动
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
    else:
        return 0,0
if __name__=='__main__':
    simBTC()