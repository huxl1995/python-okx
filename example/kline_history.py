import pandas as pd
import plotly.graph_objects as go
import io

# 1. 载入你上传的数据（真实情况下替换为 df = pd.read_csv('你的数据文件.csv')）

df = pd.read_csv('150_4h_50_5_15_5_0.0005_future_change_history.csv')
df['date'] = pd.to_datetime(df['date'])
# 2. 计算均线数据
df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean()
df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()

# 3. 过滤信号点
buy_signals = df[df['change'] == 1]
sell_signals = df[df['change'] == -1]

# 4. 创建基础 K 线
fig = go.Figure(data=[go.Candlestick(
    x=df['date'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    increasing_line_color='red',
    increasing_fillcolor='red',
    decreasing_line_color='green',
    decreasing_fillcolor='green',
    name='K线'
)])

# 5. 叠加均线
fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], mode='lines', line=dict(color='orange', width=1.5), name='MA5'))
fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], mode='lines', line=dict(color='cyan', width=1.5), name='MA20'))

# 6. 叠加买卖信号
fig.add_trace(go.Scatter(x=buy_signals['date'], y=buy_signals['low'] * 0.995, mode='markers',
                         marker=dict(symbol='triangle-up', size=13, color='rgb(255, 50, 50)',
                                     line=dict(width=1, color='white')), name='买入'))
fig.add_trace(go.Scatter(x=sell_signals['date'], y=sell_signals['high'] * 1.005, mode='markers',
                         marker=dict(symbol='triangle-down', size=13, color='rgb(50, 255, 50)',
                                     line=dict(width=1, color='white')), name='卖出'))

# 7. 配置布局与纵轴自适应缩放
fig.update_layout(
    title={'text': "K线行情图 (纵轴随缩放自适应范围)", 'y': 0.95, 'x': 0.5, 'xanchor': 'center'},
    xaxis_title="时间 (Date)",
    yaxis_title="价格 (Price)",
    template='plotly_dark',
    hovermode='x unified',

    # 核心控制：配置 Y 轴在横轴放大时自动计算并匹配当前视野范围
    yaxis=dict(
        autorange=True,  # 随区域放大自动计算当前视野下的Y轴范围
        fixedrange=False  # 允许纵轴独立拉伸或缩放
    ),

    # 配置 X 轴以及底部的滑动导航条
    xaxis=dict(
        rangeslider=dict(
            visible=True,
            yaxis=dict(rangemode='match')  # 修正处：通过 match 让滑动条完美同步主图的自适应逻辑
        ),
        type='date'
    )
)

# 8. 渲染图表
fig.show()