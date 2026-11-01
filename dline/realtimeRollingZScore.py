
from datetime import datetime,timedelta
from collections import deque
import numpy as np
from includedSelfZScoreRestorer import IncludedSelfZScoreRestorerFixed,calculate_included_zscore
from binanace.future.future_klines import fetch_klines


class RealtimeRollingZScore:
    """实盘流式数据滚动 Z-Score 计算与双向还原器"""

    def __init__(self, window):
        self.window = window
        self.data_window = deque(maxlen=window)
        # 缓存当前最新的均值和标准差，供还原函数使用
        self.current_mean = None
        self.current_std = None

    def update(self, new_value):
        """传入最新的一笔原始值，实时返回当前的 Z-Score

        如果窗口未满，返回 None
        """
        self.data_window.append(new_value)

        # 窗口数据尚未填满，无法计算
        if len(self.data_window) < self.window:
            return None

        # 计算并更新当前的均值和标准差
        arr = np.array(self.data_window)
        self.current_mean = np.mean(arr)
        self.current_std = np.std(arr, ddof=1)  # 样本标准差

        if self.current_std == 0:
            zscore = 0.0
        else:
            zscore = (new_value - self.current_mean) / self.current_std

        return zscore

    def restore_current(self, zscore):
        """场景 A：根据【当前最新窗口】的均值和标准差，将传入的 Z-Score 还原为原始值"""
        if self.current_mean is None or self.current_std is None:
            raise ValueError("窗口尚未填满，无历史均值或标准差数据，无法还原。")

        return (zscore * self.current_std) + self.current_mean

    @staticmethod
    def restore_with_params(zscore, mean, std):
        """场景 B：根据【外部指定/历史快照】的均值和标准差进行反向还原（静态方法）"""
        return (zscore * std) + mean

    def get_snapshot(self):
        """获取当前窗口的统计快照 (均值, 标准差)，方便外部日志记录或持久化"""
        return self.current_mean, self.current_std

#
# # 模拟 5 根价格流，窗口大小为 3
# prices = [100, 102, 98, 105, 95]
# W = 3
#
# rt_stat = RealtimeRollingZScore(window=W)
#
# print("--- 实盘流式【计算并同步还原】测试 ---")
# for p in prices:
#     zscore = rt_stat.update(p)
#
#     if zscore is not None:
#         # 使用类内置方法完美还原
#         restored_price = rt_stat.restore_current(zscore)
#
#         # 提取快照（如果你的模型预测了未来的Z-score，可以用这个快照推算未来的绝对价格）
#         current_mean, current_std = rt_stat.get_snapshot()
#
#         print(
#             f"原始输入: {p} -> 算得Z-Score: {zscore:.4f} -> 内部同步还原: {restored_price:.1f} (当前均值:{current_mean:.1f})"
#         )
#     else:
#         print(f"原始输入: {p} -> 窗口未满，等待数据...")
if __name__=="__main__":
    start_time=datetime(2026,10,15,0,0,0)
    end_time=start_time+timedelta(days=10)
    kline_df = fetch_klines(symbol="BTCUSDT", interval="4h",start_time=int(start_time.timestamp())*1000,end_time=int(end_time.timestamp())*1000)
    windows=5
    rt_stat = RealtimeRollingZScore(window=windows)
    in_stat=IncludedSelfZScoreRestorerFixed(window=windows)
    zscore_list=[]
    for i in range (windows-1):
        in_stat.push_history(kline_df['open'][i])
        rt_stat.update(kline_df['open'][i])
    for i in range(windows-1,len(kline_df['open'])):
        #zscore_value1=calculate_included_zscore(price,in_stat.history_window)
        zscore_value=rt_stat.update(kline_df['open'][i])
        orivalue=in_stat.restore_today_value(zscore_value)
        in_stat.push_history(kline_df['open'][i])
        try:
            print(
                f"实际原始值: {kline_df['open'][i]:<6} -> 算得Z-Score: {zscore_value:.4f} -> 逆向还原值: {orivalue:.1f}"
            )
        except TypeError as e:
            print("except")
        zscore_list.append(zscore_value)
    print(kline_df)