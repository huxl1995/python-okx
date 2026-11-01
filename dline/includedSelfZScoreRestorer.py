from collections import deque
import numpy as np


class IncludedSelfZScoreRestorerFixed:
    """【修正版】支持包含自身情况下的流式 Z-Score 完美还原器"""

    def __init__(self, window):
        self.window = window
        # 队列中保存纯历史数据，长度最大为 window - 1
        self.history_window = deque(maxlen=window - 1)

    def push_history(self, pure_value):
        """当一个周期完全结束后，将真实价格同步进历史窗口"""
        self.history_window.append(pure_value)

    def restore_today_value(self, zscore):
        """核心方法：利用当天的 Z-Score 和纯历史数据，百分之百精准还原当天价格"""
        N = self.window

        if len(self.history_window) < N - 1:
            return None

        # 1. 计算纯历史数据的统计指标
        hist_arr = np.array(self.history_window)
        mu_hist = np.mean(hist_arr)
        ss_hist = np.sum((hist_arr - mu_hist) ** 2)

        # 2. 修正后的分母计算
        denom = (N - 1) * ((N - 1) ** 2 - N * (zscore**2))

        if denom <= 0:
            raise ValueError(
                f"在当前窗口 {N} 下，Z-Score 数值 {zscore} 超出了数学边界。"
            )

        # 3. 严格对齐的代数逆向求解
        delta = zscore * N * np.sqrt(ss_hist / denom)
        restored_value = mu_hist + delta

        return restored_value

# 模拟系统或币安端算出的【包含自身】的标准 Z-Score
def calculate_included_zscore(current_val, history_list):
    full_list = list(history_list) + [current_val]
    arr = np.array(full_list)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)  # 币安及Pandas默认的样本标准差
    return (current_val - mean) / std if std != 0 else 0.0


# —— 测试开始 ——
#prices = [100.0, 102.0, 98.0, 105.0, 95.0]
prices = [82374.0, 82350.0, 82195.3, 82088.0, 82260.4,82598.8,81320.1]

W = 4
restorer = IncludedSelfZScoreRestorerFixed(window=W)

# 先用前 W-1 个数据初始化历史窗口
for p in prices[: W - 1]:
    restorer.push_history(p)

print("--- 修正后的 Z-Score 逆向还原测试 ---")
for today_price in prices[W - 1 :]:
    # 1. 计算包含自身的 Z-Score
    current_zscore = calculate_included_zscore(
        today_price, restorer.history_window
    )

    # 2. 用修正后的公式进行还原
    restored_p = restorer.restore_today_value(current_zscore)

    print(
        f"实际原始值: {today_price:<6} -> 算得Z-Score: {current_zscore:.4f} -> 逆向还原值: {restored_p:.1f}"
    )

    # 3. 步进更新历史窗口
    restorer.push_history(today_price)
