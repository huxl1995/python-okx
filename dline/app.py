import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dline.myDlineModel import MyDLinearForStock
from dlineModel import DLinearForStock, StockDataset


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
    lr=0.001,
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
