# Binance 量化交易系统

基于 DLinear 时序预测模型的币安现货自动交易程序，支持根据历史交易结果自动调参并重新训练模型。

## 架构

```
quant/
├── config.py           # 配置（API、模型、策略参数）
├── data/               # K 线获取与特征工程
├── exchange/           # 币安下单客户端
├── model/              # DLinear / LSTM / MLP 及模型注册表
│   ├── registry.py     # 模型选择与构建
│   ├── dlinear.py
│   ├── lstm.py
│   └── mlp.py
├── strategy/           # 买卖信号生成
├── optimizer/          # 交易反馈与超参优化
├── storage/            # 交易记录与优化器状态持久化
├── engine/runner.py    # 主交易循环
└── main.py             # CLI 入口
```

## 策略说明

1. **数据**：从 Binance 拉取 OHLCV K 线，做滚动 Z-Score 标准化 + 时间周期 sin/cos 编码。
2. **模型**：可选 DLinear / LSTM / MLP，通过 `--model` 切换；均预测未来 `pred_len` 根 K 线的标准化特征，再还原为价格。
3. **信号**：比较预测收盘价与当前价，涨跌幅超过 `threshold` 则 BUY/SELL，否则 HOLD。
4. **止损止盈**（现货做多）：
   - 开仓时根据入场价设置止损价（`-stop_loss%`）和止盈价（`+take_profit%`）；
   - 每轮循环优先检查当前价是否触发 SL/TP，触发则市价平仓；
   - 持仓期间模型发出 SELL 信号也会主动平仓；
   - 空仓时忽略 SELL 信号（现货不做空）。
5. **反馈优化**：
   - 记录每笔交易及预测方向；
   - 下一周期用实际价格评估盈亏与方向是否正确；
   - 胜率低于目标时提高 threshold、降低学习率；
   - 每周期用最新 K 线重新训练模型。

## 环境配置

在项目根目录或 `quant/` 下创建 `.env`：

```env
API_KEY=your_binance_api_key
API_SECRET=your_binance_api_secret
# BASE_PATH=https://testnet.binance.vision  # 可选：测试网
```

安装依赖（在项目 venv 中）：

```bash
pip install -r quant/requirements.txt
```

## 运行

**默认模拟交易（推荐先跑）：**

```bash
./venv/bin/python -m quant.main --dry-run --symbol BNBUSDT --interval 1h --cycles 1
```

**持续循环（模拟）：**

```bash
./venv/bin/python -m quant.main run --dry-run --symbol BNBUSDT --interval 1h
```

**仅训练模型：**

```bash
./venv/bin/python -m quant.main train --symbol BNBUSDT --interval 1h --epochs 50
```

**仅预测：**

```bash
./venv/bin/python -m quant.main predict --symbol BNBUSDT
```

**使用 LSTM 模型：**

```bash
./venv/bin/python -m quant.main train --model lstm --symbol BNBUSDT --hidden-size 128
./venv/bin/python -m quant.main --dry-run --model lstm --symbol BNBUSDT --interval 1h --cycles 1
```

**查看可用模型：**

```bash
./venv/bin/python -m quant.main models
```

**实盘（谨慎）：**

```bash
./venv/bin/python -m quant.main run --live --symbol BNBUSDT --quantity 0.01
```

## 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | dlinear | 预测模型：`dlinear` / `lstm` / `mlp` |
| `--hidden-size` | 64 | LSTM/MLP 隐藏层大小 |
| `--symbol` | BNBUSDT | 交易对 |
| `--interval` | 1h | K 线周期 |
| `--quantity` | 0.01 | 每笔数量 |
| `--threshold` | 0.001 | 触发买卖的最小预期涨跌幅 |
| `--stop-loss` | 0.02 | 止损比例（2%） |
| `--take-profit` | 0.03 | 止盈比例（3%） |
| `--no-sl-tp` | - | 关闭止损止盈 |
| `--epochs` | 50 | 每轮训练 epoch 数 |
| `--cycles` | 0 | 循环次数，0 为无限 |
| `--dry-run` | - | 模拟下单 |
| `--live` | - | 真实下单 |

## 数据文件

运行后会在 `quant/data_store/` 生成：

- `trades.jsonl` — 交易记录（含 `reason`: SIGNAL / STOP_LOSS / TAKE_PROFIT / SIGNAL_EXIT）
- `optimizer_state.json` — 优化器状态（threshold、lr、胜率等）
- `position.json` — 当前持仓及 SL/TP 价位

模型保存在 `quant/models/{SYMBOL}_{INTERVAL}_{MODEL}_model.pt`。

## 风险提示

本程序仅供学习与研究。加密货币交易存在高风险，实盘前请充分测试，并自行承担全部损失责任。
