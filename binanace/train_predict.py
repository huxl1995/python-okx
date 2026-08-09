"""
使用 Binance K 线数据训练 DLinear 模型并预测未来价格。

流程:
  1. klines.py 拉取 K 线
  2. stand.py 标准化特征
  3. app.py 训练 / 预测 / 还原价格
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DLINE_DIR = ROOT / "dline"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DLINE_DIR))

from app import load_model, predict, train_and_save  # noqa: E402
from klines import fetch_klines  # noqa: E402
from stand import CSNStand, LOGZSCOREStand, Type, restorePredictions, rollingZScoreStand  # noqa: E402

logging.basicConfig(level=logging.INFO)
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


def preprocess(df: pd.DataFrame, window_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对原始 K 线做滚动 Z-Score 与时间特征编码，返回 (raw_df, feature_df)。"""
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


def run(
    symbol: str = "BNBUSDT",
    interval: str = "1m",
    limit: int = 1000,
    seq_len: int = 30,
    pred_len: int = 5,
    window_size: int = 30,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 5e-5,
    model_path: Optional[str] = None,
    train_only: bool = False,
):
    if model_path is None:
        model_path = str(Path(__file__).parent / "model.pt")

    logger.info("Fetching klines: symbol=%s interval=%s limit=%s", symbol, interval, limit)
    raw_klines = fetch_klines(symbol=symbol, interval=interval, limit=limit)
    raw_df, feature_df = preprocess(raw_klines, window_size)

    min_rows = seq_len + pred_len + 10
    if len(feature_df) < min_rows:
        raise ValueError(
            f"数据量不足: 预处理后仅 {len(feature_df)} 条，至少需要 {min_rows} 条。"
            "请增大 limit 或换更长周期 interval。"
        )

    data_np = feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    train_data = data_np[:-seq_len]
    input_window = data_np[-seq_len:]

    logger.info("Training model on %s samples, seq_len=%s, pred_len=%s", len(train_data), seq_len, pred_len)
    train_and_save(
        data=train_data,
        save_path=model_path,
        seq_len=seq_len,
        pred_len=pred_len,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    if train_only:
        logger.info("Model saved to %s", model_path)
        return

    model = load_model(model_path)
    scaled_pred = predict(input_window, model)
    logger.info("Scaled prediction shape: %s", scaled_pred.shape)
    logger.info("Scaled prediction (first row): %s", scaled_pred[0])

    start_index = len(raw_df)
    restored = restorePredictions(
        scaled_pred,
        raw_df,
        window_size,
        start_index,
        priceKeys=PRICE_KEYS,
    )

    logger.info("=== Future %s-step prediction (restored prices) ===", pred_len)
    for i in range(pred_len):
        logger.info(
            "Step %s | open=%.4f high=%.4f low=%.4f close=%.4f",
            i + 1,
            restored["open"][i],
            restored["high"][i],
            restored["low"][i],
            restored["close"][i],
        )

    return {
        "raw_df": raw_df,
        "feature_df": feature_df,
        "scaled_pred": scaled_pred,
        "restored": restored,
        "model_path": model_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DLinear on Binance klines and predict future prices")
    parser.add_argument("--symbol", default="BNBUSDT")
    parser.add_argument("--interval", default="1m", help="Kline interval, e.g. 1m, 5m, 1h")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--pred-len", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()

    run(
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_path=args.model_path,
        train_only=args.train_only,
    )


if __name__ == "__main__":
    main()
