from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from binanace.future.future_klines import fetch_klines
from dline.data_stand import stand_features, restore_predictions_included_self
from dline.dlineModel import DLinearForStock, StockDataset

def load_model(path, seq_len=30, pred_len=5, num_features=4, map_location=None):
    """
    从文件加载训练好的 DLinearForStock 模型。

    支持两种保存格式：
    - 完整 checkpoint：{"state_dict": ..., "seq_len": ..., "pred_len": ..., "num_features": ...}
    - 仅 state_dict
    """
    device = map_location if map_location is not None else "cpu"
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        seq_len = checkpoint.get("seq_len", seq_len)
        pred_len = checkpoint.get("pred_len", pred_len)
        num_features = checkpoint.get("num_features", num_features)
    else:
        state_dict = checkpoint

    model = DLinearForStock(seq_len=seq_len, pred_len=pred_len, num_features=num_features)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(data, model):
    """
    使用已加载的模型对输入数据进行预测。

    Args:
        data: numpy 数组或 torch.Tensor，形状为 (seq_len, num_features) 或 (batch, seq_len, num_features)
        model: load_model 返回的 DLinearForStock 实例

    Returns:
        numpy 数组，形状为 (pred_len, num_features) 或 (batch, pred_len, num_features)
    """
    if isinstance(data, np.ndarray):
        tensor = torch.tensor(data, dtype=torch.float32)
    elif isinstance(data, torch.Tensor):
        tensor = data.float()
    else:
        raise TypeError("data 必须是 numpy.ndarray 或 torch.Tensor")

    single_sample = tensor.dim() == 2
    if single_sample:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 3:
        raise ValueError("data 形状应为 (seq_len, num_features) 或 (batch, seq_len, num_features)")

    if tensor.size(1) != model.seq_len:
        raise ValueError(f"输入序列长度应为 {model.seq_len}，实际为 {tensor.size(1)}")

    model.eval()
    with torch.no_grad():
        output = model(tensor)

    result = output.numpy()
    return result[0] if single_sample else result


def train_and_save(
    data,
    save_path,
    seq_len=30,
    pred_len=5,
    epochs=10,
    batch_size=32,
    lr=5e-5,
    log=True
):
    """
    训练 DLinearForStock 模型并保存到文件。

    Args:
        data: 训练数据，numpy 数组 (样本数, 特征数) 或 csv 文件路径
        save_path: 模型保存路径
        seq_len: 输入序列长度
        pred_len: 预测序列长度
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率

    Returns:
        训练完成的 DLinearForStock 模型
    """
    if isinstance(data, str):
        data = np.loadtxt(data, delimiter=",", dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("data 形状应为 (样本数, 特征数)")

    num_features = data.shape[1]
    if len(data) < seq_len + pred_len:
        raise ValueError(f"训练数据长度不足，至少需要 {seq_len + pred_len} 条样本")

    dataset = StockDataset(data, seq_len, pred_len)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DLinearForStock(seq_len=seq_len, pred_len=pred_len, num_features=num_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    if log:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "seq_len": seq_len,
            "pred_len": pred_len,
            "num_features": num_features,
        },
        save_path,
    )
    model.eval()
    return model
def train_only(
    data,
    seq_len=30,
    pred_len=5,
    epochs=10,
    batch_size=32,
    lr=5e-5,
    log=True
):
    """
    训练 DLinearForStock 模型并保存到文件。

    Args:
        data: 训练数据，numpy 数组 (样本数, 特征数) 或 csv 文件路径
        save_path: 模型保存路径
        seq_len: 输入序列长度
        pred_len: 预测序列长度
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率

    Returns:
        训练完成的 DLinearForStock 模型
    """
    if isinstance(data, str):
        data = np.loadtxt(data, delimiter=",", dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("data 形状应为 (样本数, 特征数)")

    num_features = data.shape[1]
    if len(data) < seq_len + pred_len:
        raise ValueError(f"训练数据长度不足，至少需要 {seq_len + pred_len} 条样本")

    dataset = StockDataset(data, seq_len, pred_len)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DLinearForStock(seq_len=seq_len, pred_len=pred_len, num_features=num_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    if log:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
    model.eval()
    return model

def train_only_with_model(model,
    data,
    seq_len=30,
    pred_len=5,
    epochs=10,
    batch_size=32,
    lr=5e-5,
    log=True
):
    """
    训练 DLinearForStock 模型并保存到文件。

    Args:
        data: 训练数据，numpy 数组 (样本数, 特征数) 或 csv 文件路径
        save_path: 模型保存路径
        seq_len: 输入序列长度
        pred_len: 预测序列长度
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率

    Returns:
        训练完成的 DLinearForStock 模型
    """
    if isinstance(data, str):
        data = np.loadtxt(data, delimiter=",", dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("data 形状应为 (样本数, 特征数)")
    if len(data) < seq_len + pred_len:
        raise ValueError(f"训练数据长度不足，至少需要 {seq_len + pred_len} 条样本")

    dataset = StockDataset(data, seq_len, pred_len)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    if log:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
    model.eval()
    return model

def calculate(data,model,windowSize):
    preData=data[len(data)-windowSize:-1]
    preDataNp=preData[['openScaled','highScaled','lowScaled','closeScaled','dateMonthSin','dateMonthCos','dateDaySin','dateDayCos','volumeLogScaled']].to_numpy()
    preResultNp=predict(preDataNp,model)
    df=pd.DataFrame(preResultNp,columns=['openScaled','highScaled','lowScaled','closeScaled'])
    dataRes=pd.concat(data,df)
    return dataRes

def rollMeanAndStd(data,key,windowSize):
    data[key+'RollingMean']=data[key].rolling(window=windowSize,closed='left').mean()
    data[key+'RollingStd']=data[key].rolling(window=windowSize,closed='left').std()
def fillActualValue(data,key,num,windowSize):
    for i in range(num):
        rollMeanAndStd(data, key, windowSize)
        data[key][len(data)-num+i]=data[key+'Scaled'][len(data)-num+i]*(data[key+'RollingStd'][len(data)-num+i]+ 1e-8)+data[key+'RollingMean'][len(data)-num+i]
    return data

SYMBOL = "BTCUSDT"
INTERVAL = "1d"       # K 线周期: 1m, 5m, 1h, 1d 等
LIMIT = 30          # 拉取条数（Binance 单次最多 1000）
SEQ_LEN = 6
WINDOW_SIZE = SEQ_LEN      # 滚动 Z-Score 窗口# 输入序列长度（用过去 30 根 K 线）
PRED_LEN = 3          # 预测未来 5 根 K 线
EPOCHS = 30
PER_EPOCHS = 5

FEATURE_COLUMNS = [
    "openScaled", "highScaled", "lowScaled", "closeScaled",
    "volumeLogScaled"
]
PRICE_KEYS = {"open": 0, "high": 1, "low": 2, "close": 3}
SEED=39
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if __name__=='__main__':
    end_time = datetime(2025, 1, 20, 8, 0, 0)
    num=0
    right_num=0
    error_num=0
    kline_df = fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT, end_time=int(end_time.timestamp()) * 1000)
    raw_data = kline_df.copy()
    stand_features(kline_df, WINDOW_SIZE)
    data_np = kline_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    train_data = data_np
    model = train_only(
        data=train_data,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        epochs=EPOCHS,
        log=True
    )
    kline_df=raw_data.copy()
    print("init data", kline_df.tail(3))
    while end_time < datetime.now():
        raw_data = kline_df.copy()
        stand_features(kline_df,WINDOW_SIZE)
        data_np = kline_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        train_data = data_np
        model = train_only_with_model(model=model,
            data=train_data,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            epochs=PER_EPOCHS,
            log=True
        )
        preData = data_np[-SEQ_LEN:]
        scaled_pred = predict(preData, model)
        restored = restore_predictions_included_self(
            scaled_pred,
            raw_data,
            WINDOW_SIZE,
            PRICE_KEYS,
        )
        close_price=kline_df["close"].to_numpy()[-1]
        min_close = min(restored["close"][0], restored["close"][1], restored["close"][2])
        max_close = max(restored["close"][0], restored["close"][1], restored["close"][2])
        label=0
        if close_price > max_close:
            label=-1
        elif close_price < min_close:
            label=1
        else:
            label=0
        for i in range(PRED_LEN):
            print(f'pre{i},{restored["open"][i]},{restored["high"][i]},{restored["low"][i]},{restored["close"][i]},')
        end_time=end_time+timedelta(hours=24)
        kline_df = fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT, end_time=int(end_time.timestamp()) * 1000)
        actal_label=0
        next_price=kline_df["close"].to_numpy()[-1]
        next_min_price=kline_df["low"].to_numpy()[-1]
        next_max_price=kline_df["high"].to_numpy()[-1]
        if label==1:
            if next_max_price>close_price*1.01:
                actal_label=1
            else:
                actal_label=-1
        elif label==-1:
            if next_min_price<close_price*0.99:
                actal_label=-1
            else:
                actal_label=1
        if label!=0 and label==actal_label:
            right_num+=1
        elif label!=0 and label!=actal_label:
            error_num+=1
        num+=1
    print(f'num is {num},right is {right_num},error is {error_num}')
    # 覆盖写入文本
    with open("output.txt", "a", encoding="utf-8") as f:
        f.write(f"seed:{SEED},SYMBOL:{SYMBOL},INTERNAL:{INTERVAL},LIMIT:{LIMIT},SEQ_LEN:{SEQ_LEN},PRE_LEN:{PRED_LEN},EPOCHS:{EPOCHS},num:{num},right:{right_num},error:{error_num}\n")
