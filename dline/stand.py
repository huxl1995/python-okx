import numpy as np
import pandas as pd
from enum import Enum
class Type(Enum):
    MONTH = 1
    DAY = 2
    HOUR = 3
def rollingZScoreStand(df, windowSize, key):
    # 3. 计算滚动的均值和标准差 (注意：closed='left' 严格防止时间穿越，即今天的数据不参与今天均值的计算)
    df["Rolling_Mean"] = (
        df[key].rolling(window=windowSize, closed="left").mean()
    )
    df["Rolling_Std"] = df[key].rolling(window=windowSize, closed="left").std()

    # 4. 执行滚动 Z-Score 计算
    # 加 1e-8 防止复牌或横盘时标准差为 0 导致除以 0 报错
    df[key+"Scaled"] = (df[key] - df["Rolling_Mean"]) / (
            df["Rolling_Std"] + 1e-8
    )
    df.drop("Rolling_Mean",axis=1,inplace=True)
    df.drop("Rolling_Std",axis=1,inplace=True)
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
