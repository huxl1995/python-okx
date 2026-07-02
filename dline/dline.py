import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MovingAvgDecomp(nn.Module):
    """
    移动平均分解层：用于将股票序列分解为趋势分量和季节性（残差）分量
    """

    def __init__(self, kernel_size):
        super(MovingAvgDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x 维度: [Batch, Seq_Len, Channels]
        # 对序列两端进行填充，保持移动平均后长度不变
        front = x[:, 0:1, :].repeat(1, self.kernel_size // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # 计算趋势分量
        trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        # 原始数据减去趋势，得到季节性/残差分量
        seasonal = x - trend
        return seasonal, trend


class DLinearForStock(nn.Module):
    """
    针对股票优化的 DLinear 模型
    """

    def __init__(self, seq_len, pred_len, num_features):
        super(DLinearForStock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 股票推荐使用较小的窗口，例如 5 或 9，捕捉短期趋势
        self.decomposition = MovingAvgDecomp(kernel_size=5)

        # 独立线性层分别处理两个分量
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Features]

        # --- 股票优化：NLinear 归一化操作 ---
        # 记录每个序列的最后一个值（基准价）
        seq_last = x[:, -1:, :]
        # 整体减去基准价，让输入序列在 0 附近波动，消除绝对价格漂移
        x = x - seq_last

        # --- 序列分解 ---
        seasonal_init, trend_init = self.decomposition(x)

        # 变换维度以适应 nn.Linear 的输入 (Batch, Features, Seq_Len)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # --- 线性层预测 ---
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        # --- 结果合并与反归一化 ---
        # 两个分量相加
        x_out = seasonal_output + trend_output  # [Batch, Features, Pred_Len]
        x_out = x_out.permute(0, 2, 1)  # [Batch, Pred_Len, Features]

        # 加上之前减去的基准价，还原到真实的股票价格维度
        x_out = x_out + seq_last

        return x_out
class StockDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        """
        data: numpy array, 形状为 (总天数, 特征数)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        # 输入窗口：[index : index + seq_len]
        x = self.data[index : index + self.seq_len]
        # 预测窗口：[index + seq_len : index + seq_len + pred_len]
        y = self.data[index + self.seq_len : index + self.seq_len + self.pred_len]
        return x, y