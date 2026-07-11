
import numpy as np
import pandas as pd
from enum import Enum
class Type(Enum):
    MONTH = 1
    DAY = 2
    HOUR = 3

def meanAndStd(column: pd.Series) -> tuple[float, float]:
    """计算 pandas 列的平均值和标准差"""
    return column.mean(), column.std()

def rollingZScoreStand(df, windowSize, key):
    # 3. 计算滚动的均值和标准差 (注意：closed='left' 严格防止时间穿越，即今天的数据不参与今天均值的计算)
    df[key+"Rolling_Mean"] = (
        df[key].rolling(window=windowSize, closed="left").mean()
    )
    df[key+"Rolling_Std"] = df[key].rolling(window=windowSize, closed="left").std()

    # 4. 执行滚动 Z-Score 计算
    # 加 1e-8 防止复牌或横盘时标准差为 0 导致除以 0 报错
    df[key+"Scaled"] = (df[key] - df[key+"Rolling_Mean"]) / (
            df[key+"Rolling_Std"] + 1e-8
    )
    df.drop(key+"Rolling_Mean",axis=1,inplace=True)
    df.drop(key+"Rolling_Std",axis=1,inplace=True)
    return df
def CSNStand(data,type,key):
    if type == Type.MONTH:
        data[key + "MonthSin"]=0.0
        data[key + "MonthCos"]=0.0
    elif type == Type.DAY:
        data[key + "DaySin"]=0.0
        data[key + "DayCos"]=0.0
    elif type == Type.HOUR:
        data[key + "HourSin"]=0.0
        data[key + "HourCos"]=0.0
    i=0
    for dataItem in data[key]:
        if type==Type.MONTH:
            data.loc[i,key+"MonthSin"] = np.sin(2 * np.pi * dataItem.month / 12.0)
            data.loc[i,key+"MonthCos"] = np.cos(2 * np.pi * dataItem.month / 12.0)
        elif type==Type.DAY:
            data.loc[i,key + "DaySin"] = np.sin(2 * np.pi * dataItem.dayofweek / 5.0)
            data.loc[i,key + "DayCos"] = np.cos(2 * np.pi * dataItem.dayofweek / 5.0)
        elif type==Type.HOUR:
            data.loc[i,key + "HourSin"] = np.sin(2 * np.pi * dataItem.hour / 5.0)
            data.loc[i,key + "HourCos"] = np.cos(2 * np.pi * dataItem.hour / 5.0)
        i+=1
    return data
def LOGZSCOREStand(data,windowSize,key):
    data[key+"Log"]=np.log1p(data[key])
    data=rollingZScoreStand(data,windowSize,key+"Log")
    data.drop(key+"Log",axis=1,inplace=True)
    return data

def _rollingMeanStd(series: pd.Series, index: int, windowSize: int) -> tuple[float, float]:
    window = series.iloc[max(0, index - windowSize):index]
    return window.mean(), window.std()

def inverseRollingZScore(scaledValue, series: pd.Series, windowSize: int, index: int) -> float:
    """将单个滚动 Z-Score 值还原为原始值"""
    mean, std = _rollingMeanStd(series, index, windowSize)
    return scaledValue * (std + 1e-8) + mean

def inverseLogRollingZScore(scaledValue, volumeSeries: pd.Series, windowSize: int, index: int) -> float:
    """将 log1p + 滚动 Z-Score 标准化后的值还原为原始成交量"""
    logSeries = np.log1p(volumeSeries)
    logValue = inverseRollingZScore(scaledValue, logSeries, windowSize, index)
    return np.expm1(logValue)

def restorePredictions(
    predictions: np.ndarray,
    rawDf: pd.DataFrame,
    windowSize: int,
    startIndex: int,
    priceKeys: dict[str,int] | None = None
) -> dict[str, np.ndarray]:
    """
    将模型预测结果从标准化空间还原为原始价格/成交量。

    Args:
        predictions: 模型输出，形状 (pred_len, num_features)
        rawDf: 包含原始 open/high/low/close/volume 列的 DataFrame（标准化前的完整数据）
        windowSize: 滚动 Z-Score 窗口大小，需与训练时一致
        startIndex: 第一条预测对应的 rawDf 行索引
        priceKeys: 需要还原的价格列名
        priceFeatureIndices: 价格在 predictions 中的特征索引
        volumeKey: 成交量列名，为 None 时不还原成交量
        volumeFeatureIndex: 成交量在 predictions 中的特征索引

    Returns:
        包含还原后 priceKeys 及 volume（如有）数组的字典，各数组形状 (pred_len,)
    """
    predLen = predictions.shape[0]
    extended = {key: rawDf[key].tolist() for key in priceKeys}
    result: dict[str, np.ndarray] = {key: np.zeros(predLen) for key in priceKeys}
    for day in range(predLen):
        index = startIndex + day
        for key, featIdx in priceKeys.items():
            series = pd.Series(extended[key])
            oriValue = inverseRollingZScore(predictions[day, featIdx], series, windowSize, index)
            result[key][day] = oriValue
            extended[key].append(oriValue)
    return result

def returnOriValue(data, windowSize, key, standValue):
    """将单个滚动 Z-Score 值还原为原始值（兼容旧接口）"""
    return inverseRollingZScore(standValue, data[key], windowSize, len(data))


if __name__=='__main__':
    # 1. 模拟单只股票 100 天的价格数据 (从 100元 涨到 250元 左右)
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=100, freq="B")
    price_trend = np.linspace(100, 250, 100) + np.random.normal(0, 5, size=100)
    df = pd.DataFrame({"Close": price_trend}, index=dates)

    # 2. 设定滚动窗口大小 (例如使用过去 20 个交易日作为参考背景)
    window_size = 20

    # 3. 计算滚动的均值和标准差 (注意：closed='left' 严格防止时间穿越，即今天的数据不参与今天均值的计算)
    df["Rolling_Mean"] = (
        df["Close"].rolling(window=window_size, closed="left").mean()
    )
    df["Rolling_Std"] = df["Close"].rolling(window=window_size, closed="left").std()

    # 4. 执行滚动 Z-Score 计算
    # 加 1e-8 防止复牌或横盘时标准差为 0 导致除以 0 报错
    df["Close_Scaled"] = (df["Close"] - df["Rolling_Mean"]) / (
        df["Rolling_Std"] + 1e-8
    )

    # 5. 查看结果
    print("--- 单股滚动 Z-Score 处理结果 ---")
    # 前 window_size 天因为历史数据不足，会是 NaN，属于正常现象，训练时直接 dropna() 剔除
    print(df.tail(5))
