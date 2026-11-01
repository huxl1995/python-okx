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
    args = {
        "dragmode": dragmode,
        "xaxis.fixedrange": x_fixed,
        "xaxis2.fixedrange": x_fixed,
        "yaxis.fixedrange": y_fixed,
        "yaxis2.fixedrange": y_fixed,
    }
    if mode in ("y", "xy"):
        args["meta.yAuto"] = False
    return args


Y_AUTO_JS = """
var gd = document.getElementById('{plot_id}');
var applying = false;

function parseX(v) {
    if (v === undefined || v === null) return null;
    if (typeof v === 'number') return v;
    var t = Date.parse(v);
    return isNaN(t) ? null : t;
}
function yAutoOn() {
    var meta = (gd.layout && gd.layout.meta) || {};
    return meta.yAuto === true;
}
function isXChange(e) {
    if (!e) return false;
    if (e['meta.yAuto'] === true) return true;
    for (var k in e) {
        if (Object.prototype.hasOwnProperty.call(e, k) && k.indexOf('xaxis') === 0) return true;
    }
    return false;
}
function visibleMinMax(xs, lows, highs, x0, x1) {
    var minV = Infinity, maxV = -Infinity, n = 0;
    for (var i = 0; i < xs.length; i++) {
        var xi = parseX(xs[i]);
        if (x0 !== null && (xi < x0 || xi > x1)) continue;
        var lo = lows[i], hi = highs[i];
        if (lo == null && hi == null) continue;
        if (lo != null && lo < minV) minV = lo;
        if (hi != null && hi > maxV) maxV = hi;
        n++;
    }
    return n ? [minV, maxV] : null;
}
function fitY() {
    if (applying || !yAutoOn()) return;
    var layout = gd._fullLayout || gd.layout || {};
    var xr = (layout.xaxis && layout.xaxis.range) ? layout.xaxis.range : null;
    var x0 = xr ? parseX(xr[0]) : null;
    var x1 = xr ? parseX(xr[1]) : null;
    var candle = null, pos = null;
    (gd.data || []).forEach(function(tr) {
        if (tr && tr.type === 'candlestick') candle = tr;
        if (tr && tr.name === '累计仓位') pos = tr;
    });
    if (!candle) return;
    var price = visibleMinMax(candle.x, candle.low, candle.high, x0, x1);
    if (!price) return;
    var pad = (price[1] - price[0]) * 0.06 || Math.abs(price[1]) * 0.01;
    var updates = {
        'yaxis.autorange': false,
        'yaxis.range': [price[0] - pad, price[1] + pad]
    };
    if (pos) {
        var pr = visibleMinMax(pos.x, pos.y, pos.y, x0, x1);
        if (pr) {
            var pp = (pr[1] - pr[0]) * 0.12 || 0.5;
            updates['yaxis2.autorange'] = false;
            updates['yaxis2.range'] = [pr[0] - pp, pr[1] + pp];
        }
    }
    applying = true;
    Plotly.relayout(gd, updates).then(function() { applying = false; });
}
function onRelayout(e) {
    if (applying || !yAutoOn() || !isXChange(e)) return;
    fitY();
}
gd.on('plotly_relayout', onRelayout);
gd.on('plotly_relayouting', onRelayout);
setTimeout(fitY, 0);
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
        meta=dict(yAuto=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0, font=dict(size=11)),
        margin=dict(l=60, r=30, t=136, b=40),
        height=920,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                active=0,
                x=0,
                y=1.20,
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
            ),
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                active=0,
                x=0,
                y=1.12,
                xanchor="left",
                yanchor="top",
                bgcolor="#333",
                bordercolor="#666",
                font=dict(color="#eee", size=12),
                buttons=[
                    dict(
                        label="Y自适应 开",
                        method="relayout",
                        args=[{**zoom_mode_args("x"), "meta.yAuto": True}],
                    ),
                    dict(
                        label="Y自适应 关",
                        method="relayout",
                        args=[{"meta.yAuto": False}],
                    ),
                ],
            ),
        ],
        annotations=[
            dict(
                text="Y自适应：价格轴随可见K线高低自动调整；点「缩放 Y」会关闭自适应",
                xref="paper",
                yref="paper",
                x=1,
                y=1.12,
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
        post_script=Y_AUTO_JS,
        config=plot_config,
    )
    print(f"已写入 {html_path}")
    fig.show(config=plot_config)


if __name__ == "__main__":
    main()
