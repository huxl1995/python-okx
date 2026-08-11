from binanace.klines import fetch_klines
from dline.app import predict, load_model, train_and_save
from dline.stand import restorePredictions, LOGZSCOREStand, CSNStand, rollingZScoreStand, Type
import pandas as pd
import numpy as np
LIMIT = 1000          # 拉取条数（Binance 单次最多 1000）
FEATURE_COLUMNS = [
    "openScaled", "highScaled", "lowScaled", "closeScaled",
    "dateMonthSin", "dateMonthCos",
    "dateDaySin", "dateDayCos",
    "dateHourSin", "dateHourCos",
    "volumeLogScaled",
]
PRICE_KEYS = {"open": 0, "high": 1, "low": 2, "close": 3}
EPOCHS=50
def predictNext(symbol,interval,window_size,model_path):
    kline_df = fetch_klines(symbol=symbol, interval=interval, limit=window_size)
    raw_data = kline_df.copy()
    kline_df["date"] = pd.to_datetime(kline_df["date"])
    for key in ("open", "high", "low", "close"):
        rollingZScoreStand(kline_df, window_size, key)
    CSNStand(kline_df, Type.MONTH, "date")
    CSNStand(kline_df, Type.DAY, "date")
    CSNStand(kline_df, Type.HOUR, "date")
    LOGZSCOREStand(kline_df, window_size, "volume")

    # 前 WINDOW_SIZE 行因滚动窗口不足会产生 NaN，丢弃
    kline_df.drop(kline_df.index[0:window_size], inplace=True)
    kline_df.reset_index(drop=True, inplace=True)

    data_np = kline_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    # 5. 加载模型并预测
    model = load_model(str(model_path))
    scaled_pred = predict(data_np, model)
    # 6. 将预测结果还原为真实 OHLC 价格
    start_index = len(raw_data)  # 预测的是未来数据，从 raw_data 末尾开始
    restored = restorePredictions(
        scaled_pred,
        raw_data,
        window_size,
        start_index,
        priceKeys=PRICE_KEYS,
    )
    return restored

def train(symbol,interval,window_size,model_path,pre_len):
    kline_df = fetch_klines(symbol=symbol, interval=interval, limit=LIMIT)
    kline_df["date"] = pd.to_datetime(kline_df["date"])
    for key in ("open", "high", "low", "close"):
        rollingZScoreStand(kline_df, window_size, key)
    CSNStand(kline_df, Type.MONTH, "date")
    CSNStand(kline_df, Type.DAY, "date")
    CSNStand(kline_df, Type.HOUR, "date")
    LOGZSCOREStand(kline_df, window_size, "volume")

    # 前 WINDOW_SIZE 行因滚动窗口不足会产生 NaN，丢弃
    kline_df.drop(kline_df.index[0:window_size], inplace=True)
    kline_df.reset_index(drop=True, inplace=True)

    data_np = kline_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    train_and_save(
        data=data_np,
        save_path=str(model_path),
        seq_len=window_size,
        pred_len=pre_len,
        epochs=EPOCHS,
    )