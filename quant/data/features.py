"""Feature engineering for kline data."""
from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class TimeFeature(Enum):
    MONTH = 1
    DAY = 2
    HOUR = 3


FEATURE_COLUMNS = [
    "openScaled",
    "highScaled",
    "lowScaled",
    "closeScaled",
    "dateMonthSin",
    "dateMonthCos",
    "dateDaySin",
    "dateDayCos",
    "dateHourSin",
    "dateHourCos",
    "volumeLogScaled",
]

PRICE_KEYS = {"open": 0, "high": 1, "low": 2, "close": 3}


def _drop_warmup_rows(df: pd.DataFrame, key: str) -> None:
    df.drop(df[df[key + "Scaled"].isna()].index, inplace=True)


def rolling_zscore(df: pd.DataFrame, window_size: int, key: str) -> pd.DataFrame:
    """Rolling Z-Score with left-closed window to avoid look-ahead bias."""
    mean_col = key + "Rolling_Mean"
    std_col = key + "Rolling_Std"
    df[mean_col] = df[key].rolling(window=window_size, closed="left").mean()
    df[std_col] = df[key].rolling(window=window_size, closed="left").std()
    df.drop(df[df[std_col].isna()].index, inplace=True)
    df[key + "Scaled"] = (df[key] - df[mean_col]) / (df[std_col] + 1e-8)
    df.drop(columns=[mean_col, std_col], inplace=True)
    return df


def cyclic_time_features(df: pd.DataFrame, kind: TimeFeature, key: str = "date") -> pd.DataFrame:
    """Encode calendar features as sin/cos."""
    suffix = {TimeFeature.MONTH: "Month", TimeFeature.DAY: "Day", TimeFeature.HOUR: "Hour"}[kind]
    sin_col = key + suffix + "Sin"
    cos_col = key + suffix + "Cos"
    dates = pd.to_datetime(df[key])

    if kind == TimeFeature.MONTH:
        df[sin_col] = np.sin(2 * np.pi * dates.dt.month / 12.0)
        df[cos_col] = np.cos(2 * np.pi * dates.dt.month / 12.0)
    elif kind == TimeFeature.DAY:
        df[sin_col] = np.sin(2 * np.pi * dates.dt.dayofweek / 5.0)
        df[cos_col] = np.cos(2 * np.pi * dates.dt.dayofweek / 5.0)
    else:
        df[sin_col] = np.sin(2 * np.pi * dates.dt.hour / 24.0)
        df[cos_col] = np.cos(2 * np.pi * dates.dt.hour / 24.0)
    return df


def log_volume_zscore(df: pd.DataFrame, window_size: int, key: str = "volume") -> pd.DataFrame:
    log_key = key + "Log"
    df[log_key] = np.log1p(df[key])
    rolling_zscore(df, window_size, log_key)
    df.rename(columns={log_key + "Scaled": key + "LogScaled"}, inplace=True)
    df.drop(columns=[log_key], inplace=True)
    return df


def preprocess_klines(df: pd.DataFrame, window_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform raw klines into model features.

    Returns:
        (raw_df, feature_df) where feature_df is ready for DLinear training/inference.
    """
    raw_df = df.copy()
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    for price_col in ("open", "high", "low", "close"):
        rolling_zscore(data, window_size, price_col)

    cyclic_time_features(data, TimeFeature.MONTH)
    cyclic_time_features(data, TimeFeature.DAY)
    cyclic_time_features(data, TimeFeature.HOUR)
    log_volume_zscore(data, window_size)

    data.drop(data.index[0:window_size], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return raw_df, data


def _rolling_mean_std(series: pd.Series, index: int, window_size: int) -> tuple[float, float]:
    window = series.iloc[max(0, index - window_size) : index]
    return float(window.mean()), float(window.std())


def inverse_rolling_zscore(
    scaled_value: float, series: pd.Series, window_size: int, index: int
) -> float:
    mean, std = _rolling_mean_std(series, index, window_size)
    return scaled_value * (std + 1e-8) + mean


def restore_predictions(
    predictions: np.ndarray,
    raw_df: pd.DataFrame,
    window_size: int,
    start_index: int,
    price_keys: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Map scaled model outputs back to original price scale."""
    if price_keys is None:
        price_keys = PRICE_KEYS

    pred_len = predictions.shape[0]
    extended = {key: raw_df[key].tolist() for key in price_keys}
    result: dict[str, np.ndarray] = {key: np.zeros(pred_len) for key in price_keys}

    for day in range(pred_len):
        index = start_index + day
        for key, feat_idx in price_keys.items():
            series = pd.Series(extended[key])
            value = inverse_rolling_zscore(predictions[day, feat_idx], series, window_size, index)
            result[key][day] = value
            extended[key].append(value)
    return result
