"""
Terminal dashboard that aggregates:
- CL: Chainlink BTC/USD price via Polymarket RTDS.
- BN: Binance btcusdt 1s kline price + 1s/5s quote volumes.
- PM: Polymarket best asks for UP/DOWN (BTC 15m market).

Requirements:
  pip install requests websocket-client

Run:
  python dash.py
"""

import csv
import json
import re
import subprocess
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Optional
from zoneinfo import ZoneInfo

import requests
import websocket

# ---- Chainlink (CL) ----
CL_URL = "wss://ws-live-data.polymarket.com"
CL_SYMBOL = "btc/usd"

# ---- Binance (BN) ----
BN_URL = "wss://stream.binance.com:9443/ws/btcusdt@kline_1s"

# ---- Polymarket (PM) ----
PM_GAMMA_API = "https://gamma-api.polymarket.com"
PM_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


# --- Shared state
@dataclass
class ChainlinkState:
    price: Optional[float] = None
    ts: float = 0.0
    age: float = 0.0
    price_history: deque = field(default_factory=lambda: deque(maxlen=10), repr=False)  # (ts, price)
    ptb: Optional[float] = None  # Price To Beat (first price of current market)
    ptb_market_slot: int = 0  # Which 15m slot the PTB belongs to


@dataclass
class BinanceState:
    price: Optional[float] = None
    vol_1s: Optional[float] = None
    vol_5s: Optional[float] = None
    ts: float = 0.0
    age: float = 0.0
    _last_five: deque = field(default_factory=lambda: deque(maxlen=5), repr=False)
    price_history: deque = field(default_factory=lambda: deque(maxlen=10), repr=False)  # (ts, price)
    kline_history: deque = field(default_factory=lambda: deque(maxlen=35), repr=False)  # (ts, high, low, close)
    volume_history: deque = field(default_factory=lambda: deque(maxlen=35), repr=False)  # (ts, volume)
    price_volume_history: deque = field(default_factory=lambda: deque(maxlen=35), repr=False)  # (ts, price, volume)


@dataclass
class PMState:
    up_best: Optional[tuple[float, float]] = None  # price, size
    down_best: Optional[tuple[float, float]] = None
    up_bids: list = field(default_factory=list)  # [(price, size), ...]
    up_asks: list = field(default_factory=list)  # [(price, size), ...]
    down_bids: list = field(default_factory=list)  # [(price, size), ...]
    down_asks: list = field(default_factory=list)  # [(price, size), ...]
    ts: float = 0.0
    age: float = 0.0
    tokens: Optional[dict] = None
    # History for eat-flow tracking
    up_bid_depth_history: deque = field(default_factory=lambda: deque(maxlen=10), repr=False)
    up_ask_depth_history: deque = field(default_factory=lambda: deque(maxlen=10), repr=False)
    down_bid_depth_history: deque = field(default_factory=lambda: deque(maxlen=10), repr=False)
    down_ask_depth_history: deque = field(default_factory=lambda: deque(maxlen=10), repr=False)


state_cl = ChainlinkState()
state_bn = BinanceState()
state_pm = PMState()
lock_cl = threading.Lock()
lock_bn = threading.Lock()
lock_pm = threading.Lock()
stop_event = threading.Event()

# Global websocket references for cleanup
ws_binance = None
ws_polymarket = None
cl_process = None


# ---- Data Logger
class DataLogger:
    """Logs all dashboard data to CSV with automatic market rotation."""
    
    def __init__(self, output_dir="./logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.log_queue = Queue()
        self.current_market_slot = None
        self.current_slug = None
        self.csv_writer = None
        self.csv_file = None
        self.fieldnames = self._build_fieldnames()
        
        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()
        print(f"[LOGGER] Initialized. Output: {self.output_dir}")
    
    def _build_fieldnames(self):
        """Build list of all CSV column names."""
        fields = [
            'market_slug', 'timestamp_ms', 'timestamp_et',
            'time_till_end', 'seconds_till_end',
            'oracle_btc_price', 'binance_btc_price', 'lag',
            'binance_ret1s_x100', 'binance_ret5s_x100',
            'binance_volume_1s', 'binance_volume_5s',
            'binance_atr_5s', 'binance_atr_30s', 'binance_rvol_30s',
            'binance_volma_30s', 'binance_volume_spike', 'binance_vwap_30s', 'binance_p_vwap_5s', 'binance_p_vwap_30s',
            'lat_dir_raw_x1000', 'lat_dir_norm_x1000',
        ]
        
        # Add orderbook fields
        for i in range(1, 6):
            fields.extend([f'up_bid_{i}_price', f'up_bid_{i}_size'])
        for i in range(1, 6):
            fields.extend([f'up_ask_{i}_price', f'up_ask_{i}_size'])
        for i in range(1, 6):
            fields.extend([f'down_bid_{i}_price', f'down_bid_{i}_size'])
        for i in range(1, 6):
            fields.extend([f'down_ask_{i}_price', f'down_ask_{i}_size'])
        
        # Add depth metrics
        fields.extend(['pm_up_bid_depth5', 'pm_up_ask_depth5', 'pm_up_total_depth5',
                      'pm_down_bid_depth5', 'pm_down_ask_depth5', 'pm_down_total_depth5'])
        
        # Add orderbook analytics
        fields.extend(['pm_up_spread', 'pm_down_spread',
                      'pm_up_imbalance', 'pm_down_imbalance',
                      'pm_up_microprice', 'pm_down_microprice',
                      'pm_up_bid_slope', 'pm_up_ask_slope',
                      'pm_down_bid_slope', 'pm_down_ask_slope',
                      'pm_up_bid_eatflow', 'pm_up_ask_eatflow',
                      'pm_down_bid_eatflow', 'pm_down_ask_eatflow'])
        
        return fields
    
    def log_snapshot(self, trigger_source: str = "UNKNOWN"):
        """Collect snapshot of all data and queue for writing."""
        try:
            timestamp_ms = int(time.time() * 1000)
            
            # Get current market slug
            market_slug = current_btc_slug()
            
            # Convert to Eastern Time
            et_tz = ZoneInfo('America/New_York')
            dt_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=ZoneInfo('UTC'))
            dt_et = dt_utc.astimezone(et_tz)
            timestamp_et = dt_et.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            market_time_left, market_seconds_left = time_to_market_end()
            
            # Read all states (with locks)
            with lock_cl:
                cl_price = state_cl.price
            
            with lock_bn:
                bn_price = state_bn.price
                bn_v1 = state_bn.vol_1s
                bn_v5 = state_bn.vol_5s
                bn_ret1s = calculate_return(state_bn.price_history, 1.0)
                bn_ret5s = calculate_return(state_bn.price_history, 5.0)
                bn_atr5s = calculate_atr_full(state_bn.kline_history, 5.0)
                bn_atr30s = calculate_atr_full(state_bn.kline_history, 30.0)
                bn_rvol30s = calculate_rvol(state_bn.price_history, 30.0)
                bn_volma30s = calculate_volma(state_bn.volume_history, 30.0)
                bn_volume_spike = calculate_volume_spike(bn_v1, bn_volma30s)
                bn_vwap5s = calculate_vwap(state_bn.price_volume_history, 5.0)
                bn_vwap30s = calculate_vwap(state_bn.price_volume_history, 30.0)
                bn_vwap_dev5s = calculate_vwap_deviation(bn_price, bn_vwap5s)
                bn_vwap_dev30s = calculate_vwap_deviation(bn_price, bn_vwap30s)
                lat_dir_raw, lat_dir_norm = calculate_lat_dir(bn_price, cl_price, state_bn.price_history)
            
            with lock_pm:
                up_bids = list(state_pm.up_bids[:5])
                up_asks = list(state_pm.up_asks[:5])
                down_bids = list(state_pm.down_bids[:5])
                down_asks = list(state_pm.down_asks[:5])
                # Get depth history for eat-flow calculation
                up_bid_eatflow = calculate_eat_flow(state_pm.up_bid_depth_history, 5.0)
                up_ask_eatflow = calculate_eat_flow(state_pm.up_ask_depth_history, 5.0)
                down_bid_eatflow = calculate_eat_flow(state_pm.down_bid_depth_history, 5.0)
                down_ask_eatflow = calculate_eat_flow(state_pm.down_ask_depth_history, 5.0)
            
            # Calculate derived metrics
            lag = (cl_price - bn_price) if (cl_price and bn_price) else None
            
            up_bid_depth5 = calculate_depth(up_bids, 5)
            up_ask_depth5 = calculate_depth(up_asks, 5)
            down_bid_depth5 = calculate_depth(down_bids, 5)
            down_ask_depth5 = calculate_depth(down_asks, 5)
            
            # Calculate orderbook analytics
            up_spread = (up_asks[0][0] - up_bids[0][0]) if up_asks and up_bids else None
            down_spread = (down_asks[0][0] - down_bids[0][0]) if down_asks and down_bids else None
            up_imbalance = calculate_imbalance(up_bids, up_asks, 5)
            down_imbalance = calculate_imbalance(down_bids, down_asks, 5)
            up_microprice = calculate_microprice(up_bids, up_asks)
            down_microprice = calculate_microprice(down_bids, down_asks)
            up_bid_slope = calculate_orderbook_slope(up_bids, 5)
            up_ask_slope = calculate_orderbook_slope(up_asks, 5)
            down_bid_slope = calculate_orderbook_slope(down_bids, 5)
            down_ask_slope = calculate_orderbook_slope(down_asks, 5)
            
            # Build row dictionary
            row = {
                'market_slug': market_slug,
                'timestamp_ms': timestamp_ms,
                'timestamp_et': timestamp_et,
                'time_till_end': market_time_left,
                'seconds_till_end': market_seconds_left,
                'oracle_btc_price': cl_price,
                'binance_btc_price': bn_price,
                'lag': lag,
                'binance_ret1s_x100': bn_ret1s * 100 if bn_ret1s else None,
                'binance_ret5s_x100': bn_ret5s * 100 if bn_ret5s else None,
                'binance_volume_1s': bn_v1,
                'binance_volume_5s': bn_v5,
                'binance_atr_5s': bn_atr5s,
                'binance_atr_30s': bn_atr30s,
                'binance_rvol_30s': bn_rvol30s,
                'binance_volma_30s': bn_volma30s,
                'binance_volume_spike': bn_volume_spike,
                'binance_vwap_30s': bn_vwap30s,
                'binance_p_vwap_5s': bn_vwap_dev5s,
                'binance_p_vwap_30s': bn_vwap_dev30s,
                'lat_dir_raw_x1000': lat_dir_raw * 1000 if lat_dir_raw else None,
                'lat_dir_norm_x1000': lat_dir_norm * 1000 if lat_dir_norm else None,
            }
            
            # Add orderbook levels
            for i in range(5):
                row[f'up_bid_{i+1}_price'] = up_bids[i][0] if i < len(up_bids) else None
                row[f'up_bid_{i+1}_size'] = up_bids[i][1] if i < len(up_bids) else None
                row[f'up_ask_{i+1}_price'] = up_asks[i][0] if i < len(up_asks) else None
                row[f'up_ask_{i+1}_size'] = up_asks[i][1] if i < len(up_asks) else None
                row[f'down_bid_{i+1}_price'] = down_bids[i][0] if i < len(down_bids) else None
                row[f'down_bid_{i+1}_size'] = down_bids[i][1] if i < len(down_bids) else None
                row[f'down_ask_{i+1}_price'] = down_asks[i][0] if i < len(down_asks) else None
                row[f'down_ask_{i+1}_size'] = down_asks[i][1] if i < len(down_asks) else None
            
            # Add depth metrics
            row['pm_up_bid_depth5'] = up_bid_depth5
            row['pm_up_ask_depth5'] = up_ask_depth5
            row['pm_up_total_depth5'] = up_bid_depth5 + up_ask_depth5
            row['pm_down_bid_depth5'] = down_bid_depth5
            row['pm_down_ask_depth5'] = down_ask_depth5
            row['pm_down_total_depth5'] = down_bid_depth5 + down_ask_depth5
            
            # Add orderbook analytics
            row['pm_up_spread'] = up_spread
            row['pm_down_spread'] = down_spread
            row['pm_up_imbalance'] = up_imbalance
            row['pm_down_imbalance'] = down_imbalance
            row['pm_up_microprice'] = up_microprice
            row['pm_down_microprice'] = down_microprice
            row['pm_up_bid_slope'] = up_bid_slope
            row['pm_up_ask_slope'] = up_ask_slope
            row['pm_down_bid_slope'] = down_bid_slope
            row['pm_down_ask_slope'] = down_ask_slope
            row['pm_up_bid_eatflow'] = up_bid_eatflow
            row['pm_up_ask_eatflow'] = up_ask_eatflow
            row['pm_down_bid_eatflow'] = down_bid_eatflow
            row['pm_down_ask_eatflow'] = down_ask_eatflow
            
            # Queue for async writing
            self.log_queue.put(row)
            
        except Exception as e:
            print(f"\n[LOGGER] Error collecting snapshot: {e}")
    
    def _writer_worker(self):
        """Background thread that writes queued snapshots to CSV."""
        while not stop_event.is_set():
            try:
                # Get snapshot from queue (with timeout to check stop_event)
                try:
                    row = self.log_queue.get(timeout=1.0)
                except:
                    continue
                
                # Check if we need to rotate file (new market)
                market_slot = int(row['timestamp_ms'] / 1000) // 900
                if market_slot != self.current_market_slot:
                    self._rotate_file(market_slot)
                
                # Write row
                if self.csv_writer:
                    self.csv_writer.writerow(row)
                    self.csv_file.flush()  # Flush immediately to not lose data
                
            except Exception as e:
                print(f"\n[LOGGER] Error writing: {e}")
        
        # Cleanup on shutdown
        if self.csv_file:
            self.csv_file.close()
            print(f"\n[LOGGER] Closed log file")
    
    def _rotate_file(self, market_slot: int):
        """Create new CSV file for new market."""
        if self.csv_file:
            self.csv_file.close()
            print(f"\n[LOGGER] Closed previous market log")
        
        # Use full slug in filename
        market_slug = f"btc-updown-15m-{market_slot}"
        filename = self.output_dir / f"{market_slug}.csv"
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.csv_writer.writeheader()
        self.current_market_slot = market_slot
        self.current_slug = market_slug
        print(f"[LOGGER] Started logging to: {filename}")


# Global logger instance
logger = None


# ---- Helpers
def now() -> float:
    return time.time()


def fmt_price(val: Optional[float], digits=2) -> str:
    if val is None:
        return "—"
    return f"{val:.{digits}f}"


def fmt_age(ts: float) -> str:
    if ts == 0:
        return "—"
    return f"{now() - ts:4.1f}s"


def update_ptb(price: float, state: 'ChainlinkState') -> None:
    """Update PTB (Price To Beat) if new market started."""
    current_slot = int(now()) // 900
    
    # Check if market slot changed (new 15min period started)
    if state.ptb_market_slot != current_slot:
        # New market - reset PTB
        state.ptb = price
        state.ptb_market_slot = current_slot
    elif state.ptb is None:
        # PTB not set yet for current market - set it
        state.ptb = price
        state.ptb_market_slot = current_slot


def calculate_return(price_history: deque, seconds_ago: float) -> Optional[float]:
    """Calculate return (%) for price N seconds ago."""
    if not price_history or len(price_history) < 2:
        return None
    
    current_time = now()
    current_price = price_history[-1][1]  # last price
    
    # Find price closest to N seconds ago
    target_time = current_time - seconds_ago
    closest_price = None
    min_diff = float('inf')
    
    for ts, price in price_history:
        diff = abs(ts - target_time)
        if diff < min_diff:
            min_diff = diff
            closest_price = price
    
    if closest_price is None or closest_price == 0:
        return None
    
    return ((current_price - closest_price) / closest_price) * 100


def calculate_atr_simple(price_history: deque, seconds: float) -> Optional[float]:
    """Calculate simplified ATR for CL (average absolute price changes)."""
    if not price_history or len(price_history) < 2:
        return None
    
    current_time = now()
    cutoff_time = current_time - seconds
    
    # Filter prices within time window
    prices = [(ts, price) for ts, price in price_history if ts >= cutoff_time]
    if len(prices) < 2:
        return None
    
    # Calculate absolute changes
    changes = []
    for i in range(1, len(prices)):
        change = abs(prices[i][1] - prices[i-1][1])
        changes.append(change)
    
    return sum(changes) / len(changes) if changes else None


def calculate_atr_full(kline_history: deque, seconds: float) -> Optional[float]:
    """Calculate full ATR for BN using kline data."""
    if not kline_history or len(kline_history) < 2:
        return None
    
    current_time = now()
    cutoff_time = current_time - seconds
    
    # Filter klines within time window
    klines = [(ts, high, low, close) for ts, high, low, close in kline_history if ts >= cutoff_time]
    if len(klines) < 2:
        return None
    
    # Calculate True Range for each bar
    true_ranges = []
    for i in range(1, len(klines)):
        _, high, low, close = klines[i]
        prev_close = klines[i-1][3]
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)
    
    return sum(true_ranges) / len(true_ranges) if true_ranges else None


def calculate_rvol(price_history: deque, seconds: float) -> Optional[float]:
    """Calculate Realized Volatility (std of log returns) in %."""
    if not price_history or len(price_history) < 2:
        return None
    
    import math
    
    current_time = now()
    cutoff_time = current_time - seconds
    
    # Filter prices within time window
    prices = [(ts, price) for ts, price in price_history if ts >= cutoff_time]
    if len(prices) < 2:
        return None
    
    # Calculate log returns
    log_returns = []
    for i in range(1, len(prices)):
        if prices[i-1][1] > 0 and prices[i][1] > 0:
            log_ret = math.log(prices[i][1] / prices[i-1][1])
            log_returns.append(log_ret)
    
    if len(log_returns) < 2:
        return None
    
    # Calculate standard deviation
    mean = sum(log_returns) / len(log_returns)
    variance = sum((x - mean) ** 2 for x in log_returns) / len(log_returns)
    std_dev = math.sqrt(variance)
    
    # Convert to percentage
    return std_dev * 100


def calculate_volma(volume_history: deque, seconds: float) -> Optional[float]:
    """Calculate Volume Moving Average."""
    if not volume_history:
        return None
    
    current_time = now()
    cutoff_time = current_time - seconds
    
    # Filter volumes within time window
    volumes = [vol for ts, vol in volume_history if ts >= cutoff_time]
    if not volumes:
        return None
    
    return sum(volumes) / len(volumes)


def calculate_volume_spike(current_vol: Optional[float], volma: Optional[float]) -> Optional[float]:
    """Calculate Volume Spike ratio."""
    if current_vol is None or volma is None or volma == 0:
        return None
    return current_vol / volma


def calculate_vwap(price_volume_history: deque, seconds: float) -> Optional[float]:
    """Calculate Volume Weighted Average Price."""
    if not price_volume_history:
        return None
    
    current_time = now()
    cutoff_time = current_time - seconds
    
    # Filter within time window
    data = [(ts, price, vol) for ts, price, vol in price_volume_history if ts >= cutoff_time]
    if not data:
        return None
    
    total_pv = sum(price * vol for _, price, vol in data)
    total_vol = sum(vol for _, _, vol in data)
    
    if total_vol == 0:
        return None
    
    return total_pv / total_vol


def calculate_price_to_vwap(current_price: Optional[float], vwap: Optional[float]) -> Optional[float]:
    """Calculate Price to VWAP ratio."""
    if current_price is None or vwap is None or vwap == 0:
        return None
    return current_price / vwap


def calculate_vwap_deviation(current_price: Optional[float], vwap: Optional[float]) -> Optional[float]:
    """Calculate percentage deviation from VWAP."""
    if current_price is None or vwap is None or vwap == 0:
        return None
    return ((current_price / vwap) - 1) * 100


def calculate_lat_dir(bn_price: Optional[float], cl_price: Optional[float], 
                     bn_price_history: deque) -> tuple[Optional[float], Optional[float]]:
    """
    Calculate Lat Dir (Latency Direction) indicator.
    Returns: (lat_dir_raw, lat_dir_norm)
    
    lat_dir shows if Binance "leads" the move while Oracle lags behind.
    """
    import math
    
    if bn_price is None or cl_price is None:
        return None, None
    
    # Get BN price 1 second ago
    if not bn_price_history or len(bn_price_history) < 2:
        return None, None
    
    current_time = now()
    target_time = current_time - 1.0  # 1 second ago
    
    # Find closest price to 1s ago
    bn_price_1s_ago = None
    min_diff = float('inf')
    for ts, price in bn_price_history:
        diff = abs(ts - target_time)
        if diff < min_diff:
            min_diff = diff
            bn_price_1s_ago = price
    
    # Validations (per spec)
    if cl_price <= 0 or bn_price_1s_ago is None or bn_price_1s_ago <= 0:
        return None, None
    
    # Calculate ret1s (Binance return over 1 second, in fraction not %)
    ret1s = (bn_price - bn_price_1s_ago) / bn_price_1s_ago
    
    # Check if ret1s is finite
    if not math.isfinite(ret1s):
        return None, None
    
    # Calculate lag
    lag = bn_price - cl_price
    
    # Calculate lat_dir_raw and lat_dir_norm
    lat_dir_raw = lag * ret1s
    lat_dir_norm = (lag / cl_price) * ret1s
    
    return lat_dir_raw, lat_dir_norm


def current_btc_slug() -> str:
    current_slot = int(now()) // 900 * 900
    return f"btc-updown-15m-{current_slot}"


def time_to_market_end() -> tuple[str, int]:
    """Calculate time remaining until current 15m market ends.
    Returns: (formatted_time, total_seconds)
    """
    current_time = int(now())
    current_slot_start = (current_time // 900) * 900
    current_slot_end = current_slot_start + 900
    time_left_seconds = current_slot_end - current_time
    
    minutes = time_left_seconds // 60
    seconds = time_left_seconds % 60
    return f"{minutes}:{seconds:02d}", time_left_seconds


def fetch_pm_tokens() -> Optional[dict]:
    slug = current_btc_slug()
    resp = requests.get(f"{PM_GAMMA_API}/events?slug={slug}", timeout=10)
    resp.raise_for_status()
    events = resp.json()
    if not events:
        return None
    market = events[0]["markets"][0]
    clob_token_ids = market.get("clobTokenIds", [])
    outcomes = market.get("outcomes", [])
    if isinstance(clob_token_ids, str):
        clob_token_ids = json.loads(clob_token_ids)
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)
    up_idx = outcomes.index("Up") if "Up" in outcomes else 0
    down_idx = outcomes.index("Down") if "Down" in outcomes else 1
    return {"up": clob_token_ids[up_idx], "down": clob_token_ids[down_idx]}


def parse_pm_best_ask(asks_raw) -> Optional[tuple[float, float]]:
    asks = []
    for ask in asks_raw or []:
        if isinstance(ask, dict):
            price = float(ask.get("price", 0))
            size = float(ask.get("size", 0))
        else:
            price = float(ask[0])
            size = float(ask[1]) if len(ask) > 1 else 0.0
        if price > 0 and size > 0:
            asks.append((price, size))
    if not asks:
        return None
    asks.sort(key=lambda x: x[0])
    return asks[0]


def parse_pm_orderbook(orders_raw) -> list[tuple[float, float]]:
    """Parse bids or asks into sorted list of (price, size) tuples."""
    orders = []
    for order in orders_raw or []:
        if isinstance(order, dict):
            price = float(order.get("price", 0))
            size = float(order.get("size", 0))
        else:
            price = float(order[0])
            size = float(order[1]) if len(order) > 1 else 0.0
        if price > 0 and size > 0:
            orders.append((price, size))
    return orders


def calculate_depth(orders: list[tuple[float, float]], levels: int = 5) -> float:
    """Calculate total depth (sum of sizes) for top N levels."""
    if not orders:
        return 0.0
    return sum(size for _, size in orders[:levels])


def calculate_imbalance(bids: list, asks: list, levels: int = 5) -> Optional[float]:
    """Calculate orderbook imbalance: (bid_size - ask_size) / (bid_size + ask_size)."""
    if not bids or not asks:
        return None
    
    sum_bid = sum(size for _, size in bids[:levels])
    sum_ask = sum(size for _, size in asks[:levels])
    
    if sum_bid + sum_ask == 0:
        return None
    
    return (sum_bid - sum_ask) / (sum_bid + sum_ask)


def calculate_microprice(bids: list, asks: list) -> Optional[float]:
    """Calculate weighted mid price (microprice)."""
    if not bids or not asks:
        return None
    
    best_bid_price, best_bid_size = bids[0]
    best_ask_price, best_ask_size = asks[0]
    
    if best_bid_size + best_ask_size == 0:
        return None
    
    microprice = (best_ask_price * best_bid_size + best_bid_price * best_ask_size) / (best_bid_size + best_ask_size)
    return microprice


def calculate_orderbook_slope(orders: list, levels: int = 5) -> Optional[float]:
    """Calculate orderbook slope (depth concentration near best price)."""
    if not orders or len(orders) < levels:
        return None
    
    # Compare top 2 levels vs all 5 levels
    top2_depth = sum(size for _, size in orders[:2])
    top5_depth = sum(size for _, size in orders[:levels])
    
    if top5_depth == 0:
        return None
    
    # Higher ratio = more concentrated (steep wall), lower = more distributed
    return top2_depth / top5_depth


def calculate_eat_flow(depth_history: deque, seconds: float = 5.0) -> Optional[float]:
    """Calculate consumption rate (depth change per second)."""
    if not depth_history or len(depth_history) < 2:
        return None
    
    current_time = now()
    cutoff_time = current_time - seconds
    
    # Filter within time window
    recent = [(ts, depth) for ts, depth in depth_history if ts >= cutoff_time]
    if len(recent) < 2:
        return None
    
    # Calculate rate of change (depth decrease per second)
    time_span = recent[-1][0] - recent[0][0]
    if time_span == 0:
        return None
    
    depth_change = recent[-1][1] - recent[0][1]
    return depth_change / time_span  # Negative = being eaten


# ---- Workers
def chainlink_worker():
    """
    Uses the existing Node script to mirror its output and parse the price line
    `Price btc/usd: 89082 ...`.
    """
    global cl_process
    # Path to Chainlink feed script - adjust if needed
    cmd = ["node", "./chainlink/btc-feed.js"]
    while not stop_event.is_set():
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            cl_process = proc
            while not stop_event.is_set():
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                    continue
                cl_on_message(line)
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception as exc:
            print(f"\n[CL] exception: {exc}")
        time.sleep(3)
    if cl_process:
        try:
            cl_process.terminate()
            cl_process.wait(timeout=2)
        except Exception:
            cl_process.kill()


_cl_first_logged = False


def cl_on_message(message: str):
    try:
        if isinstance(message, (bytes, bytearray)):
            message = message.decode(errors="ignore")
        message = message.strip()
        if not message:
            return
        # Fast path: JSON message
        try:
            data = json.loads(message)
        except Exception:
            data = None

        if data is None:
            # Parse text output: "CL: 89082" or "Price btc/usd: 89082"
            m = re.search(r"CL:\s*([0-9.]+)", message, re.IGNORECASE)
            if not m:
                m = re.search(r"Price\s+btc/usd:\s*([0-9.]+)", message, re.IGNORECASE)
            if m:
                price = float(m.group(1))
                ts = now()
                with lock_cl:
                    state_cl.price = price
                    state_cl.ts = ts
                    state_cl.price_history.append((ts, price))
                    update_ptb(price, state_cl)
                
                # Log snapshot
                if logger:
                    logger.log_snapshot('CL')
            return

        channel = data.get("channel") or data.get("topic")
        payload = data.get("payload") or data.get("data") or {}
        symbol = (payload.get("symbol") or payload.get("pair") or payload.get("name") or "").lower()
        is_cl_channel = channel in ("crypto_prices_chainlink", "crypto_prices")
        if is_cl_channel:
            price_val = (
                payload.get("value")
                or payload.get("price")
                or payload.get("last")
                or payload.get("close")
                or payload.get("mark")
            )
            if price_val is None:
                return
            price = float(price_val)
            ts = now()
            with lock_cl:
                state_cl.price = price
                state_cl.ts = ts
                state_cl.price_history.append((ts, price))
                update_ptb(price, state_cl)
            
            # Log snapshot
            if logger:
                logger.log_snapshot('CL')
        else:
            # Log first unexpected message for debugging
            global _cl_first_logged
            if not _cl_first_logged:
                _cl_first_logged = True
                print(f"\n[CL] sample msg: {data}")
    except Exception as exc:
        print(f"\n[CL] parse error: {exc}")


def binance_worker():
    global ws_binance
    while not stop_event.is_set():
        try:
            ws = websocket.WebSocketApp(
                BN_URL,
                on_message=lambda ws, msg: bn_on_message(msg),
                on_error=lambda ws, err: print(f"\n[BN] error: {err}"),
                on_close=lambda ws, code, reason: None,
            )
            ws_binance = ws
            
            # Run in a way that respects stop_event
            def run_with_stop():
                while not stop_event.is_set():
                    ws.run_forever(ping_interval=20, ping_timeout=10, skip_utf8_validation=True)
                    if stop_event.is_set():
                        break
                    time.sleep(1)
            
            run_with_stop()
        except Exception as exc:
            if not stop_event.is_set():
                print(f"\n[BN] exception: {exc}")
        if not stop_event.is_set():
            time.sleep(3)


def bn_on_message(message: str):
    try:
        data = json.loads(message)
        if data.get("e") != "kline":
            return
        k = data.get("k", {})
        price = float(k.get("c", 0))
        high = float(k.get("h", 0))
        low = float(k.get("l", 0))
        vol_quote = float(k.get("q", 0))
        ts = now()
        with lock_bn:
            state_bn.price = price
            state_bn.vol_1s = vol_quote
            state_bn._last_five.append(vol_quote)
            state_bn.vol_5s = sum(state_bn._last_five)
            state_bn.ts = ts
            state_bn.price_history.append((ts, price))
            state_bn.kline_history.append((ts, high, low, price))
            state_bn.volume_history.append((ts, vol_quote))
            state_bn.price_volume_history.append((ts, price, vol_quote))
        
        # Log snapshot
        if logger:
            logger.log_snapshot('BN')
    except Exception as exc:
        print(f"\n[BN] parse error: {exc}")


def polymarket_worker():
    global ws_polymarket
    close_timer = None
    
    while not stop_event.is_set():
        # Get tokens for current market
        tokens = fetch_pm_tokens()
        if not tokens:
            print("[PM] failed to fetch tokens, retrying...")
            time.sleep(5)
            continue
        
        with lock_pm:
            state_pm.tokens = tokens
        
        # Calculate market end time + 2 seconds
        current_time = int(now())
        market_end_time = (current_time // 900) * 900 + 900 + 2
        time_until_end = market_end_time - current_time
        print(f"[PM] Connected to market, reconnecting in {time_until_end}s")
        
        try:
            ws = websocket.WebSocketApp(
                PM_WS_URL,
                on_message=lambda ws, msg: pm_on_message(msg, tokens),
                on_error=lambda ws, err: None,
                on_close=lambda ws, code, reason: None,
            )

            def on_open(ws):
                sub_msg = {"auth": {}, "type": "MARKET", "assets_ids": [tokens["up"], tokens["down"]]}
                ws.send(json.dumps(sub_msg))

            ws.on_open = on_open
            ws_polymarket = ws
            
            # Set timer to close WebSocket when market ends
            def close_and_reconnect():
                if ws and not stop_event.is_set():
                    print("\n[PM] Market ended, reconnecting to new market...")
                    ws.close()
            
            close_timer = threading.Timer(time_until_end, close_and_reconnect)
            close_timer.start()
            
            # Run WebSocket (blocks until closed by timer or error)
            ws.run_forever(ping_interval=20, ping_timeout=10, skip_utf8_validation=True)
            
            # Cancel timer if exited early (e.g., error or stop_event)
            if close_timer and close_timer.is_alive():
                close_timer.cancel()
            
        except Exception as exc:
            if not stop_event.is_set():
                print(f"\n[PM] exception: {exc}")
            if close_timer and close_timer.is_alive():
                close_timer.cancel()
            time.sleep(3)


def pm_on_message(message: str, tokens: dict):
    try:
        data = json.loads(message)
        if data.get("event_type") != "book":
            return
        
        # Parse full orderbook
        bids_raw = data.get("bids", [])
        asks_raw = data.get("asks", [])
        
        bids = parse_pm_orderbook(bids_raw)
        asks = parse_pm_orderbook(asks_raw)
        
        # Sort bids descending (highest first), asks ascending (lowest first)
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        
        # Get best ask for backward compatibility
        best = asks[0] if asks else None
        
        # Calculate depths for history tracking
        bid_depth = calculate_depth(bids, 5)
        ask_depth = calculate_depth(asks, 5)
        
        asset = data.get("asset_id", "")
        ts = now()
        
        with lock_pm:
            if asset == tokens["up"]:
                state_pm.up_best = best
                state_pm.up_bids = bids
                state_pm.up_asks = asks
                state_pm.up_bid_depth_history.append((ts, bid_depth))
                state_pm.up_ask_depth_history.append((ts, ask_depth))
            elif asset == tokens["down"]:
                state_pm.down_best = best
                state_pm.down_bids = bids
                state_pm.down_asks = asks
                state_pm.down_bid_depth_history.append((ts, bid_depth))
                state_pm.down_ask_depth_history.append((ts, ask_depth))
            state_pm.ts = ts
        
        # Log snapshot
        if logger:
            logger.log_snapshot('PM')
    except Exception as exc:
        print(f"\n[PM] parse error: {exc}")


# ---- Render loop
def render_loop():
    while not stop_event.is_set():
        ts = time.strftime("%H:%M:%S")
        with lock_bn:
            bn_v1 = state_bn.vol_1s
            bn_v5 = state_bn.vol_5s
            bn_ret1s = calculate_return(state_bn.price_history, 1.0)
            bn_ret5s = calculate_return(state_bn.price_history, 5.0)
            bn_atr5s = calculate_atr_full(state_bn.kline_history, 5.0)
            bn_atr30s = calculate_atr_full(state_bn.kline_history, 30.0)
            bn_rvol30s = calculate_rvol(state_bn.price_history, 30.0)
            # Volume metrics
            bn_volma30s = calculate_volma(state_bn.volume_history, 30.0)
            bn_volume_spike = calculate_volume_spike(bn_v1, bn_volma30s)
            bn_vwap5s = calculate_vwap(state_bn.price_volume_history, 5.0)
            bn_vwap30s = calculate_vwap(state_bn.price_volume_history, 30.0)
            bn_price = state_bn.price
            bn_vwap_dev5s = calculate_vwap_deviation(bn_price, bn_vwap5s)
            bn_vwap_dev30s = calculate_vwap_deviation(bn_price, bn_vwap30s)
        with lock_pm:
            up = state_pm.up_best
            down = state_pm.down_best
            up_bids = list(state_pm.up_bids)
            up_asks = list(state_pm.up_asks)
            down_bids = list(state_pm.down_bids)
            down_asks = list(state_pm.down_asks)
            # Orderbook metrics
            up_imbalance = calculate_imbalance(up_bids, up_asks, 5)
            down_imbalance = calculate_imbalance(down_bids, down_asks, 5)
            # Spread (ask - bid)
            up_spread = (up_asks[0][0] - up_bids[0][0]) if up_asks and up_bids else None
            down_spread = (down_asks[0][0] - down_bids[0][0]) if down_asks and down_bids else None
            up_microprice = calculate_microprice(up_bids, up_asks)
            down_microprice = calculate_microprice(down_bids, down_asks)
            up_bid_slope = calculate_orderbook_slope(up_bids, 5)
            up_ask_slope = calculate_orderbook_slope(up_asks, 5)
            down_bid_slope = calculate_orderbook_slope(down_bids, 5)
            down_ask_slope = calculate_orderbook_slope(down_asks, 5)
            # Eat-flow (consumption rate)
            up_bid_eatflow = calculate_eat_flow(state_pm.up_bid_depth_history, 5.0)
            up_ask_eatflow = calculate_eat_flow(state_pm.up_ask_depth_history, 5.0)
            down_bid_eatflow = calculate_eat_flow(state_pm.down_bid_depth_history, 5.0)
            down_ask_eatflow = calculate_eat_flow(state_pm.down_ask_depth_history, 5.0)

        up_str = f"{up[0]:.2f}" if up else "—"
        down_str = f"{down[0]:.2f}" if down else "—"
        
        # Calculate depth metrics
        up_bid_depth5 = calculate_depth(up_bids, 5)
        up_ask_depth5 = calculate_depth(up_asks, 5)
        up_total_depth5 = up_bid_depth5 + up_ask_depth5
        
        down_bid_depth5 = calculate_depth(down_bids, 5)
        down_ask_depth5 = calculate_depth(down_asks, 5)
        down_total_depth5 = down_bid_depth5 + down_ask_depth5
        
        # Format returns with fixed width (space for positive, minus for negative)
        # Multiply by 100 to make values more visible
        bn_ret1s_str = f"{bn_ret1s * 100: .2f}" if bn_ret1s is not None else "   —   "
        bn_ret5s_str = f"{bn_ret5s * 100: .2f}" if bn_ret5s is not None else "   —   "
        
        # Format volatility metrics (2 decimal places, no $ or %)
        bn_atr5s_str = f"{bn_atr5s:.2f}" if bn_atr5s is not None else "—"
        bn_atr30s_str = f"{bn_atr30s:.2f}" if bn_atr30s is not None else "—"
        bn_rvol30s_str = f"{bn_rvol30s:.4f}" if bn_rvol30s is not None else "—"
        
        # Format volume metrics (2 decimal places)
        bn_volma30s_str = f"{bn_volma30s:.2f}" if bn_volma30s is not None else "—"
        bn_volume_spike_str = f"{bn_volume_spike:.2f}" if bn_volume_spike is not None else "—"
        bn_vwap30s_str = f"{bn_vwap30s:.0f}" if bn_vwap30s is not None else "—"
        # Format VWAP deviations in % (with space for alignment like returns)
        bn_vwap_dev5s_str = f"{bn_vwap_dev5s: .2f}" if bn_vwap_dev5s is not None else "   —   "
        bn_vwap_dev30s_str = f"{bn_vwap_dev30s: .2f}" if bn_vwap_dev30s is not None else "   —   "
        
        # Format orderbook metrics (fixed width to prevent jumping)
        up_imb_str = f"{up_imbalance:>+7.3f}" if up_imbalance is not None else "   —   "
        down_imb_str = f"{down_imbalance:>+7.3f}" if down_imbalance is not None else "   —   "
        up_spread_str = f"{up_spread:>6.3f}" if up_spread is not None else "  —   "
        down_spread_str = f"{down_spread:>6.3f}" if down_spread is not None else "  —   "
        up_micro_str = f"{up_microprice:>6.3f}" if up_microprice is not None else "  —   "
        down_micro_str = f"{down_microprice:>6.3f}" if down_microprice is not None else "  —   "
        up_bid_slope_str = f"{up_bid_slope:>5.2f}" if up_bid_slope is not None else "  —  "
        up_ask_slope_str = f"{up_ask_slope:>5.2f}" if up_ask_slope is not None else "  —  "
        down_bid_slope_str = f"{down_bid_slope:>5.2f}" if down_bid_slope is not None else "  —  "
        down_ask_slope_str = f"{down_ask_slope:>5.2f}" if down_ask_slope is not None else "  —  "
        up_bid_eat_str = f"{up_bid_eatflow:>+6.1f}" if up_bid_eatflow is not None else "   —  "
        up_ask_eat_str = f"{up_ask_eatflow:>+6.1f}" if up_ask_eatflow is not None else "   —  "
        down_bid_eat_str = f"{down_bid_eatflow:>+6.1f}" if down_bid_eatflow is not None else "   —  "
        down_ask_eat_str = f"{down_ask_eatflow:>+6.1f}" if down_ask_eatflow is not None else "   —  "
        
        # Time to market end
        market_time_left, market_seconds_left = time_to_market_end()

        lines = [
            f"[{ts}] Dashboard │ ENDS: {market_time_left} ({market_seconds_left}s) │ PM_UP: {up_str:>5} │ PM_DN: {down_str:>5}",
            "═" * 100,
            f"RETURNS   │ Ret1s×100: {bn_ret1s_str:>7} │ Ret5s×100: {bn_ret5s_str:>7}",
            f"VOLATILITY│ ATR5s: {bn_atr5s_str:>6} │ ATR30s: {bn_atr30s_str:>6} │ RVol30s: {bn_rvol30s_str:>8}",
            f"VOLUME    │ V1s: {fmt_price(bn_v1, 0):>7} │ V5s: {fmt_price(bn_v5, 0):>7} │ VolMA30s: {bn_volma30s_str:>7} │ Spike: {bn_volume_spike_str:>5}",
            f"VWAP      │ VWAP30s: {bn_vwap30s_str:>6} │ P/VWAP5s: {bn_vwap_dev5s_str:>7} │ P/VWAP30s: {bn_vwap_dev30s_str:>7}",
            "",
            f"ORDERBOOK ANALYTICS:",
            f"IMBALANCE │ UP: {up_imb_str:>7} │ DN: {down_imb_str:>7}  (>0=bid pressure, <0=ask pressure)",
            f"SPREAD    │ UP: {up_spread_str:>6} │ DN: {down_spread_str:>6}  (ask-bid, taker cost)",
            f"MICROPRICE│ UP: {up_micro_str:>6} │ DN: {down_micro_str:>6}  (weighted mid)",
            f"OB SLOPE  │ UP: Bid:{up_bid_slope_str:>5} Ask:{up_ask_slope_str:>5} │ DN: Bid:{down_bid_slope_str:>5} Ask:{down_ask_slope_str:>5}  (>0.6=wall)",
            f"EAT-FLOW/s│ UP: Bid:{up_bid_eat_str:>6} Ask:{up_ask_eat_str:>6}",
            f"          │ DN: Bid:{down_bid_eat_str:>6} Ask:{down_ask_eat_str:>6}  (negative=eaten)",
            "",
        ]
        
        # Build ORDERBOOK block (left side)
        orderbook_block = []
        orderbook_block.append("POLYMARKET ORDERBOOK")
        orderbook_block.append("─" * 40)
        
        # Build UP market orderbook
        up_lines = []
        up_lines.append("UP │ BIDS:")
        for price, size in up_bids[:5]:
            up_lines.append(f"  {price:.2f}→{size:.0f}")
        if not up_bids:
            up_lines.append("  —")
        up_lines.append("UP │ ASKS:")
        for price, size in up_asks[:5]:
            up_lines.append(f"  {price:.2f}→{size:.0f}")
        if not up_asks:
            up_lines.append("  —")
        
        # Build DOWN market orderbook
        down_lines = []
        down_lines.append("DN │ BIDS:")
        for price, size in down_bids[:5]:
            down_lines.append(f"  {price:.2f}→{size:.0f}")
        if not down_bids:
            down_lines.append("  —")
        down_lines.append("DN │ ASKS:")
        for price, size in down_asks[:5]:
            down_lines.append(f"  {price:.2f}→{size:.0f}")
        if not down_asks:
            down_lines.append("  —")
        
        # Combine UP and DOWN markets side by side into orderbook_block
        max_len = max(len(up_lines), len(down_lines))
        for i in range(max_len):
            up_text = up_lines[i] if i < len(up_lines) else ""
            down_text = down_lines[i] if i < len(down_lines) else ""
            orderbook_block.append(f"{up_text:<20} {down_text}")
        
        # Build DEPTH METRICS block (right side)
        depth_block = []
        depth_block.append("DEPTH & ARBITRAGE")
        depth_block.append("─" * 40)
        
        # Calculate BID price sums for each level (UP + DOWN)
        bid_level_sums = []
        for i in range(5):
            up_price = up_bids[i][0] if i < len(up_bids) else 0.0
            down_price = down_bids[i][0] if i < len(down_bids) else 0.0
            level_sum = up_price + down_price
            edge = 1.0 - level_sum
            bid_level_sums.append((level_sum, edge))
        
        # Build depth metrics columns
        up_depth_lines = []
        up_depth_lines.append("UP│Bid:{:>4} Ask:{:>4}".format(int(up_bid_depth5), int(up_ask_depth5)))
        
        down_depth_lines = []
        down_depth_lines.append("DN│Bid:{:>4} Ask:{:>4}".format(int(down_bid_depth5), int(down_ask_depth5)))
        
        # Add BID level sums (compact format)
        down_depth_lines.append("")
        down_depth_lines.append("ARBS (UP+DN BID):")
        for i, (level_sum, edge) in enumerate(bid_level_sums, 1):
            edge_str = f"{edge:+.3f}" if edge != 0 else " 0.000"
            down_depth_lines.append(f"L{i}:{level_sum:.3f} {edge_str}")
        
        # Combine depth metrics side by side
        for i in range(max(len(up_depth_lines), len(down_depth_lines))):
            up_text = up_depth_lines[i] if i < len(up_depth_lines) else ""
            down_text = down_depth_lines[i] if i < len(down_depth_lines) else ""
            depth_block.append(f"{up_text:<20} {down_text}")
        
        # Combine orderbook_block and depth_block side by side
        max_block_len = max(len(orderbook_block), len(depth_block))
        for i in range(max_block_len):
            left = orderbook_block[i] if i < len(orderbook_block) else ""
            right = depth_block[i] if i < len(depth_block) else ""
            lines.append(f"{left:<45} {right}")
        
        lines.append("")
        lines.append("─" * 80 + " Ctrl+C to exit")
        
        # Use full terminal reset for reliable screen clearing
        sys.stdout.write("\033c" + "\n".join(lines))
        sys.stdout.flush()
        time.sleep(0.5)


def main():
    def handle_sigint(sig, frame):
        print("\nStopping...")
        stop_event.set()
        
        # Close all websockets and subprocess
        if ws_binance:
            try:
                ws_binance.close()
            except Exception:
                pass
        if ws_polymarket:
            try:
                ws_polymarket.close()
            except Exception:
                pass
        if cl_process:
            try:
                cl_process.terminate()
            except Exception:
                pass

    signal.signal(signal.SIGINT, handle_sigint)
    
    # Initialize data logger
    global logger
    logger = DataLogger(output_dir="./logs")

    threads = [
        threading.Thread(target=chainlink_worker, daemon=True),
        threading.Thread(target=binance_worker, daemon=True),
        threading.Thread(target=polymarket_worker, daemon=True),
        threading.Thread(target=render_loop, daemon=True),
    ]
    for t in threads:
        t.start()
    
    # Wait for stop event with timeout to allow clean exit
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    
    # Give threads a moment to cleanup
    time.sleep(1)
    print("Exited.")


if __name__ == "__main__":
    main()






