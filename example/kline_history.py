"""
读取 change_history CSV，绘制可缩放 K 线：买卖点、均线、视野内 Y 轴自适应。

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

Y_AUTOSCALE_JS = """
var gd = document.getElementById('{plot_id}');
function parseX(v) {
    if (v === undefined || v === null) return null;
    if (typeof v === 'number') return v;
    var t = Date.parse(v);
    return isNaN(t) ? null : t;
}
function visibleYRange() {
    var layout = gd.layout || {};
    var xr = (layout.xaxis && layout.xaxis.range) ? layout.xaxis.range : null;
    var x0 = xr ? parseX(xr[0]) : null;
    var x1 = xr ? parseX(xr[1]) : null;
    var lows = [], highs = [];
    (gd.data || []).forEach(function(tr) {
        if (!tr || tr.visible === false || tr.visible === 'legendonly') return;
        var xs = tr.x || [];
        if (tr.type === 'candlestick') {
            var lo = tr.low || [], hi = tr.high || [];
            for (var i = 0; i < xs.length; i++) {
                var xi = parseX(xs[i]);
                if (x0 !== null && (xi < x0 || xi > x1)) continue;
                if (lo[i] != null) lows.push(lo[i]);
                if (hi[i] != null) highs.push(hi[i]);
            }
        } else if (tr.yaxis !== 'y2') {
            var ys = tr.y || [];
            for (var j = 0; j < xs.length; j++) {
                var xj = parseX(xs[j]);
                if (x0 !== null && (xj < x0 || xj > x1)) continue;
                if (ys[j] != null) { lows.push(ys[j]); highs.push(ys[j]); }
            }
        }
    });
    if (!lows.length) return;
    var minY = Math.min.apply(null, lows);
    var maxY = Math.max.apply(null, highs);
    var pad = (maxY - minY) * 0.08 || maxY * 0.01;
    Plotly.relayout(gd, {'yaxis.range': [minY - pad, maxY + pad]});
}
gd.on('plotly_relayout', function(e) {
    if (!e) return;
    if (e['yaxis.range'] || e['yaxis.range[0]'] || e['yaxis.autorange']) return;
    if (e['xaxis.range'] || e['xaxis.range[0]'] || e['xaxis.autorange']
        || e['xaxis.rangeslider.range'] || e['xaxis.rangeslider.range[0]']) {
        visibleYRange();
    }
});
gd.on('plotly_legendclick', function() { setTimeout(visibleYRange, 0); });
"""


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

    fig.update_layout(
        title=dict(
            text=f"BTCUSDT 4h K线  |  买入 {len(buys)} 次  卖出 {len(sells)} 次  |  共 {len(df)} 根",
            x=0.5,
            xanchor="center",
        ),
        template="plotly_dark",
        hovermode="x unified",
        dragmode="zoom",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        margin=dict(l=60, r=30, t=80, b=40),
        height=860,
    )
    fig.update_xaxes(
        type="date",
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
    fig.update_xaxes(type="date", rangeslider=dict(visible=True, thickness=0.06), row=2, col=1)
    fig.update_yaxes(title_text="价格", fixedrange=False, autorange=True, row=1, col=1)
    fig.update_yaxes(title_text="仓位", fixedrange=False, autorange=True, row=2, col=1)
    fig.update_xaxes(title_text="时间", row=2, col=1)
    return fig


def main():
    csv_path = Path(__file__).resolve().parent / CSV_NAME
    df = load_history(csv_path)
    fig = build_figure(df)
    html_path = Path(__file__).resolve().parent / HTML_NAME
    fig.write_html(
        str(html_path),
        include_plotlyjs=True,
        full_html=True,
        post_script=Y_AUTOSCALE_JS,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        },
    )
    print(f"已写入 {html_path}")
    fig.show(config={"scrollZoom": True, "displaylogo": False})


if __name__ == "__main__":
    main()
