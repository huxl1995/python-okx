import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dline.dlineModel import MovingAvgDecomp


class MyDLinearForStock(nn.Module):
    """
    针对股票优化的 DLinear 模型
    """

    def __init__(self, seq_len, pred_len, num_features):
        super(MyDLinearForStock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 股票推荐使用较小的窗口，例如 5 或 9，捕捉短期趋势
        self.decomposition = MovingAvgDecomp(kernel_size=5)

        # 独立线性层分别处理两个分量
        self.Linear_Seasonal = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len*2),  # 第1层：输入 -> 256
            nn.ReLU(),  # 添加非线性激活函数
            nn.Linear(in_features=self.seq_len*2, out_features=self.seq_len),  # 第2层：256 -> 128
            nn.ReLU(),
            nn.Linear(in_features=self.seq_len, out_features=self.pred_len)  # 第3层：128 -> 输出 (如10分类)
        )
        self.Linear_Trend = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len*2),  # 第1层：输入 -> 256
            nn.ReLU(),  # 添加非线性激活函数
            nn.Linear(in_features=self.seq_len*2, out_features=self.seq_len),  # 第2层：256 -> 128
            nn.ReLU(),
            nn.Linear(in_features=self.seq_len, out_features=self.pred_len)  # 第3层：128 -> 输出 (如10分类)
        )

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