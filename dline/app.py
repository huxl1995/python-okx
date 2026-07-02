import numpy as np
import torch

from Dline import DLinearForStock


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
