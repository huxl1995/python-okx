"""
自动化交易循环：获取 K 线 -> 训练模型 -> 预测未来 K 线 -> 执行交易 -> 等待后重复。

运行方式（在项目根目录）:
    ./venv/bin/python -m binanace.app
    ./venv/bin/python -m binanace.app --dry-run --symbol BNBUSDT --interval 1h
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from binanace.action import decide_action, execute_trade, get_current_price  # noqa: E402
from binanace.klines import fetch_klines  # noqa: E402
from dline.app import load_model, predict, train_and_save  # noqa: E402
from dline.stand import CSNStand, LOGZSCOREStand, Type, restorePredictions, rollingZScoreStand  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "openScaled",
    "highScaled",
    "lowScaled",
    "closeScaled",
    "dateMonthSin",
    "dateMonthCos",
    "dateDaySin",
    "dateDayCos",
    "dateHourSin",
    "dateHourCos",
    "volumeLogScaled",
]
PRICE_KEYS = {"open": 0, "high": 1, "low": 2, "close": 3}

INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}


def preprocess_klines(df: pd.DataFrame, window_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对原始 K 线做特征标准化，返回 (raw_df, feature_df)。"""
    raw_df = df.copy()
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    for key in ("open", "high", "low", "close"):
        rollingZScoreStand(data, window_size, key)
    CSNStand(data, Type.MONTH, "date")
    CSNStand(data, Type.DAY, "date")
    CSNStand(data, Type.HOUR, "date")
    LOGZSCOREStand(data, window_size, "volume")

    data.drop(data.index[0:window_size], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return raw_df, data


def train_model(
    symbol: str,
    interval: str,
    limit: int,
    window_size: int,
    seq_len: int,
    pred_len: int,
    model_path: Path,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 5e-5,
) -> None:
    """拉取前 n 条 K 线并训练模型。"""
    logger.info("Fetching %s klines for training: %s %s", limit, symbol, interval)
    raw_klines = fetch_klines(symbol=symbol, interval=interval, limit=limit)
    _, feature_df = preprocess_klines(raw_klines, window_size)

    min_rows = seq_len + pred_len + 10
    if len(feature_df) < min_rows:
        raise ValueError(
            f"数据量不足: 预处理后仅 {len(feature_df)} 条，至少需要 {min_rows} 条。"
            "请增大 --limit 或换更长周期 --interval。"
        )

    data_np = feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    train_data = data_np[:-seq_len]

    logger.info("Training on %s samples, seq_len=%s, pred_len=%s", len(train_data), seq_len, pred_len)
    train_and_save(
        data=train_data,
        save_path=str(model_path),
        seq_len=seq_len,
        pred_len=pred_len,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    logger.info("Model saved to %s", model_path)


def predict_future(
    symbol: str,
    interval: str,
    limit: int,
    window_size: int,
    seq_len: int,
    pred_len: int,
    model_path: Path,
) -> dict:
    """拉取最新 K 线，用已训练模型预测未来 pred_len 根 K 线。"""
    fetch_limit = max(limit, window_size + seq_len + 10)
    logger.info("Fetching %s klines for prediction: %s %s", fetch_limit, symbol, interval)
    raw_klines = fetch_klines(symbol=symbol, interval=interval, limit=fetch_limit)
    raw_df, feature_df = preprocess_klines(raw_klines, window_size)

    if len(feature_df) < seq_len:
        raise ValueError(f"预测数据不足: 需要至少 {seq_len} 条，实际 {len(feature_df)} 条")

    data_np = feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    input_window = data_np[-seq_len:]

    model = load_model(str(model_path))
    scaled_pred = predict(input_window, model)

    start_index = len(raw_df)
    restored = restorePredictions(
        scaled_pred,
        raw_df,
        window_size,
        start_index,
        priceKeys=PRICE_KEYS,
    )

    logger.info("=== Future %s-step prediction ===", pred_len)
    for i in range(pred_len):
        logger.info(
            "Step %s | open=%.6f high=%.6f low=%.6f close=%.6f",
            i + 1,
            restored["open"][i],
            restored["high"][i],
            restored["low"][i],
            restored["close"][i],
        )
    return restored


def run_cycle(
    symbol: str,
    interval: str,
    limit: int,
    window_size: int,
    seq_len: int,
    pred_len: int,
    model_path: Path,
    quantity: str,
    threshold: float,
    epochs: int,
    dry_run: bool,
) -> None:
    """执行一次完整循环：训练 -> 预测 -> 交易。"""
    train_model(
        symbol=symbol,
        interval=interval,
        limit=limit,
        window_size=window_size,
        seq_len=seq_len,
        pred_len=pred_len,
        model_path=model_path,
        epochs=epochs,
    )

    restored = predict_future(
        symbol=symbol,
        interval=interval,
        limit=limit,
        window_size=window_size,
        seq_len=seq_len,
        pred_len=pred_len,
        model_path=model_path,
    )

    current_close = get_current_price(symbol, interval)
    action = decide_action(current_close, restored, threshold=threshold)
    logger.info("Trade decision: %s", action)
    execute_trade(action, symbol, quantity, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description="Binance K线训练预测自动交易循环")
    parser.add_argument("--symbol", default="BNBUSDT", help="交易对")
    parser.add_argument("--interval", default="1h", help="K 线周期，如 1m, 5m, 1h")
    parser.add_argument("--limit", type=int, default=1000, help="每次拉取的 K 线条数（最大 1000）")
    parser.add_argument("--window-size", type=int, default=30, help="滚动 Z-Score 窗口")
    parser.add_argument("--seq-len", type=int, default=30, help="模型输入序列长度")
    parser.add_argument("--pred-len", type=int, default=5, help="预测未来 K 线根数")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--quantity", default="0.01", help="每次交易数量")
    parser.add_argument("--threshold", type=float, default=0.001, help="触发买卖的最小涨跌幅比例")
    parser.add_argument("--sleep", type=int, default=None, help="循环间隔秒数，默认与 K 线周期一致")
    parser.add_argument("--cycles", type=int, default=0, help="循环次数，0 表示无限循环")
    parser.add_argument("--model-path", default=None, help="模型保存路径")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不实际下单")
    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else Path(__file__).parent / f"{args.symbol}_model.pt"
    sleep_seconds = args.sleep if args.sleep is not None else INTERVAL_SECONDS.get(args.interval, 3600)

    logger.info(
        "Starting trading loop: symbol=%s interval=%s limit=%s sleep=%ss dry_run=%s",
        args.symbol,
        args.interval,
        args.limit,
        sleep_seconds,
        args.dry_run,
    )

    cycle_count = 0
    while True:
        cycle_count += 1
        logger.info("========== Cycle %s ==========", cycle_count)
        try:
            run_cycle(
                symbol=args.symbol,
                interval=args.interval,
                limit=args.limit,
                window_size=args.window_size,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                model_path=model_path,
                quantity=args.quantity,
                threshold=args.threshold,
                epochs=args.epochs,
                dry_run=args.dry_run,
            )
        except Exception as e:
            logger.error("Cycle %s failed: %s", cycle_count, e, exc_info=True)

        if args.cycles > 0 and cycle_count >= args.cycles:
            logger.info("Completed %s cycles, exiting.", args.cycles)
            break

        logger.info("Sleeping %s seconds until next cycle...", sleep_seconds)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
