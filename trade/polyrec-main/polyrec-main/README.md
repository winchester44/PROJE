# Polymarket BTC Dashboard (polyrec)

Real-time terminal dashboard for Polymarket BTC 15-minute UP/DOWN prediction markets. Aggregates price feeds from Chainlink oracle, Binance, and Polymarket orderbook data. Includes backtesting tools for trading strategy research.

## Features

- **Real-time Dashboard** (`dash.py`) - Terminal UI showing:
  - Chainlink BTC/USD oracle price (via Polymarket RTDS)
  - Binance BTCUSDT 1s kline price and volume
  - Polymarket orderbook depth for UP/DOWN markets
  - Technical indicators: returns, ATR, VWAP, volume spikes
  - Automatic CSV logging per 15-minute market

- **Backtesting Tools**:
  - `replicate_balance.py` - Balance replication strategy simulator
  - `fade_impulse_backtest.py` - Impulse fade strategy backtester
  - `visualize_fade_impulse.py` - Strategy visualization

## Requirements

- Python 3.10+
- Node.js (for Chainlink price feed via external script)
- Active internet connection (WebSocket streams)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/txbabaxyz/polyrec.git
cd polyrec
```

### 2. Create virtual environment

```bash
# Create venv
python3 -m venv .venv

# Activate venv (Linux/macOS)
source .venv/bin/activate

# Activate venv (Windows)
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. External dependency (Chainlink feed)

The dashboard uses an external Node.js script for Chainlink oracle data.
Make sure you have the Chainlink feed script available. By default, the dashboard looks for `./chainlink/btc-feed.js`. Modify the path in `dash.py` if needed:

```python
# Line 745 in dash.py
cmd = ["node", "./chainlink/btc-feed.js"]  # Change this path
```

## Usage

### Running the Dashboard

```bash
# Activate venv first
source .venv/bin/activate

# Run dashboard
python dash.py
```

The dashboard will:
1. Connect to Binance WebSocket for price data
2. Connect to Polymarket WebSocket for orderbook data
3. Start Chainlink price feed subprocess
4. Display real-time terminal dashboard
5. Log all data to `./logs/` directory (CSV per market)

Press `Ctrl+C` to stop.

### Running Backtests

```bash
# Balance replication strategy
python replicate_balance.py

# Fade impulse strategy (requires external polymarket_api module)
python fade_impulse_backtest.py
```

### Visualizing Strategy Results

```bash
python visualize_fade_impulse.py
```

## Project Structure

```
polyrec/
├── dash.py                 # Main real-time dashboard
├── replicate_balance.py    # Balance strategy backtester
├── fade_impulse_backtest.py # Impulse fade backtester
├── visualize_fade_impulse.py # Strategy visualization
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── logs/                   # Auto-created, stores CSV market logs
```

## Data Sources

| Source | Type | URL |
|--------|------|-----|
| Binance | WebSocket | `wss://stream.binance.com:9443/ws/btcusdt@kline_1s` |
| Polymarket Orderbook | WebSocket | `wss://ws-subscriptions-clob.polymarket.com/ws/market` |
| Polymarket Gamma API | REST | `https://gamma-api.polymarket.com` |
| Chainlink Oracle | External | Via Polymarket RTDS |

## CSV Log Format

Each 15-minute market generates a CSV file with 70+ columns:

- **Timestamps**: `timestamp_ms`, `timestamp_et`, `seconds_till_end`
- **Prices**: `oracle_btc_price`, `binance_btc_price`, `lag`
- **Returns**: `binance_ret1s_x100`, `binance_ret5s_x100`
- **Volume**: `binance_volume_1s`, `binance_volume_5s`, `binance_volma_30s`
- **Volatility**: `binance_atr_5s`, `binance_atr_30s`, `binance_rvol_30s`
- **Orderbook**: 5 levels of bids/asks for UP and DOWN markets
- **Analytics**: spread, imbalance, microprice, slope, eat-flow

## Configuration

Key parameters in `dash.py`:

```python
# Chainlink
CL_URL = "wss://ws-live-data.polymarket.com"
CL_SYMBOL = "btc/usd"

# Binance
BN_URL = "wss://stream.binance.com:9443/ws/btcusdt@kline_1s"

# Polymarket
PM_GAMMA_API = "https://gamma-api.polymarket.com"
PM_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Logging
LOG_DIR = "./logs"  # Change in DataLogger.__init__()
```

## Troubleshooting

### Dashboard shows no data

1. Check internet connection
2. Verify WebSocket URLs are accessible
3. Check if Chainlink feed script exists and works

### Chainlink price not updating

1. Ensure Node.js is installed: `node --version`
2. Check if feed script path is correct in `dash.py`
3. Run feed script manually to test: `node ./chainlink/btc-feed.js`

### Permission errors

```bash
chmod +x dash.py
chmod +x replicate_balance.py
```

## License

MIT License

## Disclaimer

This software is for research and educational purposes only. Trading cryptocurrency derivatives involves significant risk. Use at your own risk.
