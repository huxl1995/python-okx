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

from dline.data_stand import stand_features, restore_predictions_included_self

ROOT = Path(__file__).resolve().parent.parent
DLINE_DIR = ROOT / "dline"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DLINE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dline.app import load_model, predict, train_and_save
from binanace.future.future_klines import fetch_klines
from decimal import Decimal, ROUND_FLOOR
import random
import torch
# ---------- 参数配置 ----------
SYMBOL = "BTCUSDT"
INTERVAL = "1d"       # K 线周期: 1m, 5m, 1h, 1d 等
LIMIT = 30          # 拉取条数（Binance 单次最多 1000）
SEQ_LEN = 6
WINDOW_SIZE = SEQ_LEN      # 滚动 Z-Score 窗口# 输入序列长度（用过去 30 根 K 线）
PRED_LEN = 3          # 预测未来 5 根 K 线
EPOCHS = 30
PER_EPOCHS = 5
TRADE_FEE_RATE = 0.0005
STOP_MARKET_RATE = 0.02
PROFIT_MARKET_RATE=0.01
SEED=35

LABEL = "future_quick_trade"
MODEL_NAME = (
    str(LIMIT) + "_"+str(SEED)+"_" + INTERVAL + "_" + str(EPOCHS) + "_" + str(PER_EPOCHS)
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
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
def simBTC():
    global MONEY, CLEAR_MONEY  # 提到函数最顶部声明

    # 1. 从 Binance 拉取 K 线
    print(f"拉取 {SYMBOL} {INTERVAL} K 线，limit={LIMIT} ...")
    end_time = datetime(2025, 1, 20, 8, 0, 0)
    kline_df = fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT, end_time=int(end_time.timestamp()) * 1000)
    raw_data = kline_df.copy()
    print(kline_df.tail(3))

    # 2. 特征标准化
    stand_features(kline_df,WINDOW_SIZE)
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
    end_time = datetime(2025, 1, 20, 8, 0, 0)
    money_list.append({"date": end_time, "money": 0, "clear_money": 0})
    while end_time < datetime.now():
        time.sleep(0.1)
        kline_df = fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT, end_time=int(end_time.timestamp()) * 1000)
        raw_data = kline_df.copy()
        stand_features(kline_df,WINDOW_SIZE)
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
        if change_quant != 0:
            quant = quant + change_quant
            PRICE=current_close
            MONEY = MONEY - change_quant * current_close
            # 修复原逻辑中 clear_change_quant 误加到常规 change_quant 的逻辑隐患
            if change_quant <= 0:
                CLEAR_MONEY = CLEAR_MONEY - change_quant * (Decimal('1') - Decimal(str(TRADE_FEE_RATE))) * current_close
            else:
                CLEAR_MONEY = CLEAR_MONEY - change_quant * (Decimal('1') + Decimal(str(TRADE_FEE_RATE))) * current_close
        # quant,change_money=quick_trade(quant,PRICE,PROFIT_MARKET_RATE,STOP_MARKET_RATE,end_time)
        # MONEY=MONEY+Decimal(change_money)
        # CLEAR_MONEY=CLEAR_MONEY+Decimal(change_money*Decimal(1-TRADE_FEE_RATE))
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
        end_time = end_time + timedelta(hours=24)
        num += 1

    control_history = pd.DataFrame(control_history)
    control_history.to_csv(
        f"./{LIMIT}_{SEED}_{INTERVAL}_{EPOCHS}_{PER_EPOCHS}_{SEQ_LEN}_{PRED_LEN}_{TRADE_FEE_RATE}_{LABEL}_change_history.csv"
    )
    money_df = pd.DataFrame(money_list)
    money_df.to_csv(
        f"./{LIMIT}_{SEED}_{INTERVAL}_{EPOCHS}_{PER_EPOCHS}_{SEQ_LEN}_{PRED_LEN}_{TRADE_FEE_RATE}_{LABEL}_money.csv"
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
    elif quant<0 and Decimal(min_price)<=PRICE*Decimal(1-PROFIT_MARKET_RATE):
        return quant*-1,PRICE*Decimal(1-PROFIT_MARKET_RATE)
    elif quant>0 and Decimal(max_price)>=PRICE*Decimal(1+PROFIT_MARKET_RATE):
        return quant*-1,PRICE*Decimal(1+PROFIT_MARKET_RATE)
    else:
        return 0,0
def quick_trade(quant,price,take_profit_rate,stop_lose_rate,start_time):
    if quant ==0:
        return 0,0
    start_time=start_time+timedelta(hours=24)
    end_time=start_time+timedelta(hours=24)
    end_time=end_time-timedelta(seconds=60)
    kline=fetch_klines(symbol=SYMBOL,interval='1m',limit=1500,start_time=int(start_time.timestamp())*1000,end_time=int(end_time.timestamp())*1000)
    if quant>0:
        take_profit_price=price*Decimal(1+take_profit_rate)
        stop_lose_price=price*Decimal(1-stop_lose_rate);
        for row in kline.itertuples():
            if row.low<=stop_lose_price:
                return 0,quant*stop_lose_price
            if row.high>=take_profit_price:
                return 0,quant*take_profit_price
        return quant,0
    else:
        take_profit_price=price*Decimal(1-take_profit_rate)
        stop_lose_price=price*Decimal(1+stop_lose_rate)
        for row in kline.itertuples():
            if row.high>=stop_lose_price:
                return 0,quant*stop_lose_price
            if row.low<=take_profit_price:
                return 0,quant*take_profit_price
        return quant,0
if __name__=='__main__':
    simBTC()