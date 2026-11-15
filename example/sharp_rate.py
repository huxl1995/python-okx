import pandas as pd
import numpy as np

# 1. 读取数据并按日期去重（保留每天最后一条记录）
df = pd.read_csv('30_1d_30_5_6_3_0.0005_future_quick_trade_money1.csv')
df = df.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)

# 2. 设置初始本金与年化天数（重要：请替换为你的实际回测初始本金）
initial_capital = 100000.0
annual_days = 365  # 全天候交易品种（加密货币/24h期货）用 365，股票用 252
risk_free_rate = 0.0  # 年化无风险利率

# 3. 还原每日账户总净值
df['equity'] = initial_capital + df['clear_money']

# 4. 计算每日收益率
df['daily_return'] = df['equity'].pct_change()

# 5. 计算年化收益率 (CAGR / 复合年化增长率)
total_days = len(df) - 1  # 实际经过的交易天数
total_return = (df['equity'].iloc[-1] / initial_capital) - 1
annual_return = (1 + total_return) ** (annual_days / total_days) - 1

# 6. 计算最大回撤 (Max Drawdown)
df['cum_max'] = df['equity'].cummax()  # 历史最高净值
df['drawdown'] = (df['equity'] - df['cum_max']) / df['cum_max']  # 回撤比例（负数）
max_drawdown = abs(df['drawdown'].min())  # 取绝对值作为最大回撤率

# 7. 计算年化夏普比率 (Sharpe Ratio)
daily_rf = risk_free_rate / annual_days
excess_return = df['daily_return'].dropna() - daily_rf
sharpe_ratio = np.sqrt(annual_days) * (excess_return.mean() / excess_return.std())

# 8. 计算卡玛比率 (Calmar Ratio)
calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else np.nan

# 打印绩效报告
print(f"回测周期: {df['date'].iloc[0]} 至 {df['date'].iloc[-1]} (共 {total_days} 天)")
print(f"累计净盈利: {df['clear_money'].iloc[-1]:.2f}")
print(f"累计收益率: {total_return:.2%}")
print(f"年化收益率: {annual_return:.2%}")
print(f"最大回撤 (Max Drawdown): {max_drawdown:.2%}")
print(f"夏普比率 (Sharpe Ratio): {sharpe_ratio:.4f}")
print(f"卡玛比率 (Calmar Ratio): {calmar_ratio:.4f}")
