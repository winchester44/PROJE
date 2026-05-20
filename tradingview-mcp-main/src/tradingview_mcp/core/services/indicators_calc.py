"""
Technical Indicators Calculator — pure Python stdlib, zero dependencies.

All functions take a list of float closing prices (or OHLCV dicts)
and return computed indicator values.

Indicators:
  - EMA, SMA
  - RSI (Wilder's smoothing)
  - Bollinger Bands
  - MACD
  - ATR (Average True Range)
  - Supertrend
  - Donchian Channel
  - ADX (Average Directional Index)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


# ─── EMA ──────────────────────────────────────────────────────────────────────

def calc_ema(closes: list[float], period: int) -> list[Optional[float]]:
    """Exponential Moving Average using pandas for O(1) vectorized speed."""
    if not closes: return []
    s = pd.Series(closes)
    ema = s.ewm(span=period, adjust=False).mean()
    # Mask initial values with None to match original behavior
    result = ema.tolist()
    for i in range(min(period - 1, len(result))):
        result[i] = None
    return result


# ─── SMA ──────────────────────────────────────────────────────────────────────

def calc_sma(closes: list[float], period: int) -> list[Optional[float]]:
    """Simple Moving Average using pandas."""
    if not closes: return []
    s = pd.Series(closes)
    sma = s.rolling(window=period).mean()
    return sma.where(sma.notna(), None).tolist()


# ─── RSI ──────────────────────────────────────────────────────────────────────

def calc_rsi(closes: list[float], period: int = 14) -> list[Optional[float]]:
    """Relative Strength Index (Wilder's smoothing) using vectorized pandas."""
    if len(closes) < period + 1:
        return [None] * len(closes)
    
    delta = pd.Series(closes).diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    # Wilder's smoothing: alpha = 1/period
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    result = rsi.tolist()
    for i in range(min(period, len(result))):
        result[i] = None
    return result


# ─── Bollinger Bands ──────────────────────────────────────────────────────────

def calc_bollinger(
    closes: list[float], period: int = 20, std_mult: float = 2.0
) -> dict[str, list[Optional[float]]]:
    """Bollinger Bands using vectorized pandas."""
    if not closes: return {"upper": [], "middle": [], "lower": []}
    s = pd.Series(closes)
    middle = s.rolling(window=period).mean()
    std = s.rolling(window=period).std()
    upper = middle + (std_mult * std)
    lower = middle - (std_mult * std)
    
    return {
        "upper": upper.where(upper.notna(), None).tolist(),
        "middle": middle.where(middle.notna(), None).tolist(),
        "lower": lower.where(lower.notna(), None).tolist(),
    }


# ─── MACD ─────────────────────────────────────────────────────────────────────

def calc_macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, list[Optional[float]]]:
    """MACD using vectorized pandas."""
    if not closes: return {"macd": [], "signal": [], "histogram": []}
    s = pd.Series(closes)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    # Mask values before 'slow' to match logic
    mask = [None] * len(closes)
    res_macd = macd_line.tolist()
    res_sig = signal_line.tolist()
    res_hist = histogram.tolist()
    
    for i in range(min(slow - 1, len(closes))):
        res_macd[i] = None
        res_sig[i] = None
        res_hist[i] = None
        
    return {"macd": res_macd, "signal": res_sig, "histogram": res_hist}


# ─── ATR (Average True Range) ─────────────────────────────────────────────────

def calc_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> list[Optional[float]]:
    """Average True Range using vectorized pandas."""
    if len(closes) < 2: return [None] * len(closes)
    
    h, l, c = pd.Series(highs), pd.Series(lows), pd.Series(closes)
    prev_c = c.shift(1)
    
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    
    # Wilder's smoothing for ATR
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    result = atr.tolist()
    for i in range(min(period, len(result))):
        result[i] = None
    return result


# ─── Supertrend ───────────────────────────────────────────────────────────────

def calc_supertrend(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> dict[str, list]:
    """Supertrend indicator using vectorized logic where possible."""
    n = len(closes)
    atr = calc_atr(highs, lows, closes, atr_period)
    
    # ATR warmup
    atr_s = pd.Series(atr)
    
    hl2 = (pd.Series(highs) + pd.Series(lows)) / 2.0
    upper_band = hl2 + multiplier * atr_s
    lower_band = hl2 - multiplier * atr_s
    
    # Supertrend requires a loop because of recursive dependency on previous bands
    direction = [None] * n
    final_upper = [None] * n
    final_lower = [None] * n
    
    closes_array = np.array(closes)
    ub_array = upper_band.values
    lb_array = lower_band.values
    
    curr_direction = 1
    
    for i in range(1, n):
        if atr[i] is None: continue
        
        # Upper Band logic
        prev_up = final_upper[i-1] if final_upper[i-1] is not None else float('inf')
        if ub_array[i] < prev_up or closes_array[i-1] > prev_up:
            final_upper[i] = ub_array[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        # Lower Band logic
        prev_low = final_lower[i-1] if final_lower[i-1] is not None else float('-inf')
        if lb_array[i] > prev_low or closes_array[i-1] < prev_low:
            final_lower[i] = lb_array[i]
        else:
            final_lower[i] = final_lower[i-1]
            
        # Direction
        if curr_direction == 1:
            if closes_array[i] < final_lower[i]:
                curr_direction = -1
        else:
            if closes_array[i] > final_upper[i]:
                curr_direction = 1
        
        direction[i] = curr_direction
        
    return {"direction": direction, "upper": final_upper, "lower": final_lower}


# ─── Donchian Channel ─────────────────────────────────────────────────────────

def calc_donchian(
    highs: list[float], lows: list[float], period: int = 20
) -> dict[str, list[Optional[float]]]:
    """Donchian Channel using vectorized pandas."""
    h = pd.Series(highs)
    l = pd.Series(lows)
    
    upper = h.rolling(window=period).max()
    lower = l.rolling(window=period).min()
    middle = (upper + lower) / 2
    
    return {
        "upper": upper.where(upper.notna(), None).tolist(),
        "lower": lower.where(lower.notna(), None).tolist(),
        "middle": middle.where(middle.notna(), None).tolist(),
    }


# ─── ADX (Average Directional Index) ──────────────────────────────────────────

def calc_adx(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> list[Optional[float]]:
    """Average Directional Index (ADX) using vectorized pandas."""
    if len(closes) < period + 1: return [None] * len(closes)
    
    h, l, c = pd.Series(highs), pd.Series(lows), pd.Series(closes)
    
    # +DM and -DM
    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = pd.Series(plus_dm)
    minus_dm = pd.Series(minus_dm)
    
    # TR
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    
    # TR, +DM, -DM smoothing (Wilder's smoothing: alpha = 1 / period)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    
    # DX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs()
    
    # ADX
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    result = adx.tolist()
    # Mask initial values 
    for i in range(min(period * 2 - 1, len(result))):
        result[i] = None
    return result
