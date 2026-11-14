import numpy as np
import pandas as pd

from dline.includedSelfZScoreRestorer import IncludedSelfZScoreRestorerFixed
from dline.realtimeRollingZScore import RealtimeRollingZScore
from dline.stand import CSNStand, Type


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


def stand_features(kline_df,window_size):
    kline_df["date"] = pd.to_datetime(kline_df["date"])
    for key in ("open", "high", "low", "close"):
        apply_realtime_rolling_zscore(kline_df, window_size, key)
    CSNStand(kline_df, Type.MONTH, "date")
    CSNStand(kline_df, Type.DAY, "date")
    CSNStand(kline_df, Type.HOUR, "date")
    apply_log_realtime_zscore(kline_df, window_size, "volume")
    kline_df.drop(kline_df.index[0:window_size], inplace=True)
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