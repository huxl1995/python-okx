import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv('30_1d_30_5_6_3_0.0005_future_quick_trade_money1.csv')
df1 = pd.read_csv('30_1d_30_5_6_3_0.0005_future_quick_trade_money1.csv')


def plot_with_hover(data, title):
    fig, ax = plt.subplots(figsize=(10, 5))

    # 绘制折线图，x轴使用数字索引以便精确计算鼠标位置
    x_idx = np.arange(len(data))
    ax.plot(x_idx, data['money'], label='money')
    ax.plot(x_idx, data['clear_money'], label='clear_money')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # 设置 X 轴刻度显示日期
    step = max(1, len(data) // 8)
    xticks = x_idx[::step]
    ax.set_xticks(xticks)
    ax.set_xticklabels(data['date'].iloc[xticks], rotation=15)

    # 初始化垂直参考线
    vline = ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, visible=False)

    # 初始化悬浮提示框
    annot = ax.annotate(
        "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gray", alpha=0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0")
    )
    annot.set_visible(False)

    # 鼠标移动事件处理函数
    def on_move(event):
        if event.inaxes == ax and event.xdata is not None:
            # 找到距离鼠标最近的 X 轴索引
            idx = int(round(event.xdata))
            if 0 <= idx < len(data):
                row = data.iloc[idx]

                # 更新参考线位置
                vline.set_xdata([idx, idx])
                vline.set_visible(True)

                # 更新提示框位置和文本内容
                annot.xy = (idx, row['money'])
                text = (
                    f"Date: {row['date']}\n"
                    f"Money: {row['money']:.2f}\n"
                    f"Clear Money: {row['clear_money']:.2f}"
                )
                annot.set_text(text)

                # 防止提示框超出图表右边界
                if idx > len(data) * 0.75:
                    annot.set_position((-140, 15))
                else:
                    annot.set_position((15, 15))

                annot.set_visible(True)
                fig.canvas.draw_idle()
        else:
            # 鼠标移出图表区域时隐藏
            if annot.get_visible() or vline.get_visible():
                annot.set_visible(False)
                vline.set_visible(False)
                fig.canvas.draw_idle()

    # 绑定事件
    fig.canvas.mpl_connect("motion_notify_event", on_move)


# 生成图表
plot_with_hover(df, 'money')
plot_with_hover(df1, 'money1')

plt.show()
