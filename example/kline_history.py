"""
读取 change_history CSV，绘制可缩放 K 线：买卖点、均线，X / Y 轴可独立缩放。

运行方式（在 example 目录或项目根目录）:
    python example/kline_history.py
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CSV_NAME = "150_4h_50_5_15_5_0.0005_future_rtz_change_history.csv"
HTML_NAME = "kline_chart.html"
MA_WINDOWS = (5, 10, 20, 60)
MA_COLORS = {
    5: "#f5a623",
    10: "#7ed321",
    20: "#4a90d9",
    60: "#c17ecf",
}


def zoom_mode_args(mode: str) -> dict:
    """通过 fixedrange 把框选/滚轮限制在单轴或双轴。"""
    x_fixed = mode == "y"
    y_fixed = mode == "x"
    dragmode = "pan" if mode == "pan" else "zoom"
    return {
        "dragmode": dragmode,
        "xaxis.fixedrange": x_fixed,
        "xaxis2.fixedrange": x_fixed,
        "yaxis.fixedrange": y_fixed,
        "yaxis2.fixedrange": y_fixed,
    }


def load_history(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    required = {"date", "change", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} 缺少列: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df["change"] = pd.to_numeric(df["change"], errors="coerce").fillna(0)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    df["position"] = df["change"].cumsum()
    for w in MA_WINDOWS:
        df[f"MA{w}"] = df["close"].rolling(window=w, min_periods=1).mean()
    return df.reset_index(drop=True)


def build_figure(df: pd.DataFrame) -> go.Figure:
    buys = df[df["change"] > 0]
    sells = df[df["change"] < 0]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22],
        subplot_titles=("K 线 / 均线 / 买卖点", "累计仓位 (change 累加)"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="#ef5350",
            increasing_fillcolor="#ef5350",
            decreasing_line_color="#26a69a",
            decreasing_fillcolor="#26a69a",
            name="K线",
            whiskerwidth=0.8,
        ),
        row=1,
        col=1,
    )

    for w in MA_WINDOWS:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[f"MA{w}"],
                mode="lines",
                line=dict(color=MA_COLORS[w], width=1.4),
                name=f"MA{w}",
                hovertemplate=f"MA{w}=%{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys["date"],
                y=buys["low"] * 0.997,
                mode="markers+text",
                text=buys["change"].map(lambda q: f"买{int(q) if float(q).is_integer() else q}"),
                textposition="bottom center",
                textfont=dict(size=10, color="#ef5350"),
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="#ef5350",
                    line=dict(width=1, color="white"),
                ),
                name="买入",
                hovertemplate=(
                    "买入 change=%{customdata[0]}<br>low=%{y:.2f}<br>%{x}<extra></extra>"
                ),
                customdata=buys[["change"]],
            ),
            row=1,
            col=1,
        )

    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells["date"],
                y=sells["high"] * 1.003,
                mode="markers+text",
                text=sells["change"].map(lambda q: f"卖{int(abs(q)) if float(q).is_integer() else abs(q)}"),
                textposition="top center",
                textfont=dict(size=10, color="#26a69a"),
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="#26a69a",
                    line=dict(width=1, color="white"),
                ),
                name="卖出",
                hovertemplate=(
                    "卖出 change=%{customdata[0]}<br>high=%{customdata[1]:.2f}<br>%{x}<extra></extra>"
                ),
                customdata=sells[["change", "high"]],
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["position"],
            mode="lines",
            line=dict(color="#90caf9", width=1.5),
            name="累计仓位",
            hovertemplate="仓位=%{y}<br>%{x}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    price_min = float(df["low"].min())
    price_max = float(df["high"].max())
    price_pad = (price_max - price_min) * 0.06 or price_max * 0.01
    pos_min = float(df["position"].min())
    pos_max = float(df["position"].max())
    pos_pad = (pos_max - pos_min) * 0.12 or 0.5

    fig.update_layout(
        title=dict(
            text=f"BTCUSDT 4h K线  |  买入 {len(buys)} 次  卖出 {len(sells)} 次  |  共 {len(df)} 根",
            x=0.5,
            xanchor="center",
        ),
        template="plotly_dark",
        hovermode="x unified",
        dragmode="zoom",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0, font=dict(size=11)),
        margin=dict(l=60, r=30, t=120, b=40),
        height=900,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                active=0,
                x=0,
                y=1.18,
                xanchor="left",
                yanchor="top",
                bgcolor="#333",
                bordercolor="#666",
                font=dict(color="#eee", size=12),
                buttons=[
                    dict(label="缩放 X", method="relayout", args=[zoom_mode_args("x")]),
                    dict(label="缩放 Y", method="relayout", args=[zoom_mode_args("y")]),
                    dict(label="缩放 XY", method="relayout", args=[zoom_mode_args("xy")]),
                    dict(label="平移", method="relayout", args=[zoom_mode_args("pan")]),
                ],
            )
        ],
        annotations=[
            dict(
                text="缩放 X：只改时间轴；缩放 Y：只改价格轴；滚轮与框选都受当前模式限制",
                xref="paper",
                yref="paper",
                x=1,
                y=1.14,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=11, color="#aaa"),
            )
        ],
    )
    fig.update_xaxes(
        type="date",
        fixedrange=False,
        rangeslider=dict(visible=False),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1天", step="day", stepmode="backward"),
                dict(count=7, label="1周", step="day", stepmode="backward"),
                dict(count=1, label="1月", step="month", stepmode="backward"),
                dict(count=3, label="3月", step="month", stepmode="backward"),
                dict(step="all", label="全部"),
            ],
            bgcolor="#222",
            activecolor="#555",
            font=dict(color="#ddd"),
        ),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        type="date",
        fixedrange=False,
        rangeslider=dict(visible=True, thickness=0.06),
        title_text="时间",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="价格",
        fixedrange=True,
        autorange=False,
        range=[price_min - price_pad, price_max + price_pad],
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="仓位",
        fixedrange=True,
        autorange=False,
        range=[pos_min - pos_pad, pos_max + pos_pad],
        row=2,
        col=1,
    )
    return fig


def main():
    csv_path = Path(__file__).resolve().parent / CSV_NAME
    df = load_history(csv_path)
    fig = build_figure(df)
    html_path = Path(__file__).resolve().parent / HTML_NAME
    plot_config = {
        "scrollZoom": True,
        "displaylogo": False,
        "doubleClick": "reset",
        "modeBarButtonsToAdd": ["drawline", "eraseshape"],
    }
    fig.write_html(
        str(html_path),
        include_plotlyjs=True,
        full_html=True,
        config=plot_config,
    )
    print(f"已写入 {html_path}")
    fig.show(config=plot_config)


if __name__ == "__main__":
    main()
