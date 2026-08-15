#!/usr/bin/env python3
"""
Binance quantitative trading CLI.

Examples (from project root):
    ./venv/bin/python -m quant.main --dry-run
    ./venv/bin/python -m quant.main --symbol BNBUSDT --interval 1h --cycles 1
    ./venv/bin/python -m quant.main train --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.config import load_config
from quant.engine.runner import TradingEngine
from quant.model.predictor import PricePredictor
from quant.model.registry import available_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binance quant trading system")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=("run", "train", "predict", "models"),
        help="Command (default: run)",
    )
    parser.add_argument(
        "--model",
        default="dlinear",
        choices=available_models(),
        help="Forecast model type (default: dlinear)",
    )
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for lstm/mlp")
    parser.add_argument("--symbol", default="BNBUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--pred-len", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--quantity", default="0.01")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--stop-loss", type=float, default=0.02, help="Stop-loss ratio, e.g. 0.02 = 2%%")
    parser.add_argument("--take-profit", type=float, default=0.03, help="Take-profit ratio, e.g. 0.03 = 3%%")
    parser.add_argument("--no-sl-tp", action="store_true", help="Disable stop-loss / take-profit")
    parser.add_argument("--sleep", type=int, default=None)
    parser.add_argument("--cycles", type=int, default=0, help="0 = infinite")
    parser.add_argument("--dry-run", action="store_true", help="Simulate orders")
    parser.add_argument("--live", action="store_true", help="Place real orders")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = build_parser().parse_args()
    command = args.command

    overrides = {
        "symbol": args.symbol,
        "interval": args.interval,
        "model_type": args.model,
        "hidden_size": args.hidden_size,
        "kline_limit": args.limit,
        "window_size": args.window_size,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }

    if command == "run":
        dry_run = not args.live
        if args.dry_run:
            dry_run = True
        overrides.update(
            {
                "quantity": args.quantity,
                "signal_threshold": args.threshold,
                "stop_loss_pct": args.stop_loss,
                "take_profit_pct": args.take_profit,
                "enable_sl_tp": not args.no_sl_tp,
                "sleep_seconds": args.sleep,
                "max_cycles": args.cycles,
                "dry_run": dry_run,
            }
        )

    if command == "models":
        print("Available forecast models:")
        for name in available_models():
            marker = " (default)" if name == "dlinear" else ""
            print(f"  - {name}{marker}")
        return

    cfg = load_config(**overrides)
    if args.model_path:
        cfg._model_path_override = Path(args.model_path)

    if command == "train":
        PricePredictor(cfg).train()
        return

    if command == "predict":
        restored = PricePredictor(cfg).predict_future()
        print("Predicted closes:", restored["close"].tolist())
        return

    engine = TradingEngine(cfg)
    if cfg.max_cycles == 1:
        engine.run_once()
    else:
        engine.run_loop()


if __name__ == "__main__":
    main()
