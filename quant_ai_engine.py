"""
quant_ai_engine.py
==================
Quant-AI Signal Engine — based on "151 Trading Strategies" (Kakushadze & Serur)

Implements 6 strategy modules and synthesises them into a single
structured JSON trading signal with risk management levels.

Strategies implemented:
  1. Price Momentum (Section 3.1)     — cumulative + risk-adjusted return ranking
  2. Dual Moving Average (Sec 3.12)   — SMA crossover, two-MA signal
  3. Triple Moving Average (Sec 3.13) — three-MA confirmation filter
  4. Support & Resistance / Pivot (Sec 3.14) — pivot-point price-action
  5. Donchian Channel (Sec 3.15)      — T-day high/low breakout/bounce
  6. ASO (Average Sentiment Oscillator) — intrabar + group bull/bear pressure
+ Naïve Bayes Sentiment (Sec 18.3 analogue) — FinBERT sentiment label

Output schema (strict JSON-compatible dict):
{
  "consensus_signal": "BUY | SELL | HOLD",
  "confidence_score": int 0-100,
  "primary_strategy_triggered": str,
  "market_regime": "Uptrend | Downtrend | Sideways | Highly Volatile",
  "analysis_summary": str,
  "strategy_breakdown": {
      "momentum": "Bullish | Bearish | Neutral",
      "price_action": "Near Support | Near Resistance | Middle",
      "sentiment": "Positive | Negative | Neutral"
  },
  "risk_management": {
      "suggested_stop_loss": float,
      "suggested_take_profit": float,
      "risk_reward_ratio": str,
      "risk_level": "Low | Medium | High"
  },
  # Extra detail fields (shown in UI subpanels):
  "_detail": { ... }
}
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 1. PRICE MOMENTUM  (Kakushadze & Serur §3.1)
# ──────────────────────────────────────────────────────────────────────────────

def _price_momentum_signal(df: pd.DataFrame, formation_months: int = 12, skip_months: int = 1):
    """
    Computes cumulative and risk-adjusted momentum signals.

    Returns:
        signal   : "Bullish" | "Bearish" | "Neutral"
        R_cum    : float  – cumulative return over formation period
        R_risadj : float  – risk-adjusted mean return
        detail   : dict
    """
    close = df["Close"].squeeze()
    if len(close) < (formation_months + skip_months) * 21:
        return "Neutral", 0.0, 0.0, {}

    # Convert months to trading days (≈21 per month)
    T = formation_months * 21
    S = skip_months * 21

    # Formation window: rows from [S : S+T] going backward from today
    # (index 0 = most recent; pandas iloc[-1] = most recent)
    series = close.values[::-1]          # reverse so index 0 = most recent

    if len(series) < S + T + 1:
        return "Neutral", 0.0, 0.0, {}

    price_now   = series[S]              # price at end of skip period
    price_start = series[S + T]         # price at start of formation period

    # Eq. (267) – cumulative return
    R_cum = (price_now / price_start) - 1.0

    # Monthly returns within formation period
    monthly_returns = []
    for m in range(formation_months):
        p0 = series[S + m * 21]
        p1 = series[S + (m + 1) * 21]
        if p1 > 0:
            monthly_returns.append((p0 / p1) - 1.0)

    if len(monthly_returns) < 2:
        return "Neutral", R_cum, 0.0, {}

    R_mean  = np.mean(monthly_returns)                         # Eq. (268)
    sigma_i = np.std(monthly_returns, ddof=1)                  # Eq. (270)
    R_risadj = R_mean / sigma_i if sigma_i > 1e-9 else 0.0    # Eq. (269)

    # Signal classification
    if R_cum > 0.05 and R_risadj > 0.5:
        signal = "Bullish"
    elif R_cum < -0.05 and R_risadj < -0.5:
        signal = "Bearish"
    else:
        signal = "Neutral"

    detail = {
        "cumulative_return_pct": round(R_cum * 100, 2),
        "risk_adjusted_return":  round(R_risadj, 3),
        "monthly_volatility_pct": round(sigma_i * 100, 2),
        "formation_months": formation_months,
        "skip_months": skip_months,
    }
    return signal, R_cum, R_risadj, detail


# ──────────────────────────────────────────────────────────────────────────────
# 2 & 3. MOVING AVERAGES  (Kakushadze & Serur §3.12 / §3.13)
# ──────────────────────────────────────────────────────────────────────────────

def _moving_average_signal(df: pd.DataFrame):
    """
    Two-MA crossover (§3.12) + Three-MA confirmation (§3.13).

    Uses pre-computed SMA columns from calculate_technical_indicators().

    Returns:
        signal   : "Bullish" | "Bearish" | "Neutral"
        regime   : "Uptrend" | "Downtrend" | "Sideways"
        detail   : dict
    """
    close = df["Close"].squeeze()

    # Prefer pre-computed SMAs; fall back to on-the-fly calculation
    def get_sma(col, window):
        if col in df.columns:
            v = df[col].iloc[-1]
            return float(v) if pd.notna(v) else float(close.rolling(window).mean().iloc[-1])
        return float(close.rolling(window).mean().iloc[-1])

    sma5  = get_sma("SMA_5",  5)   # fast  — may not exist, computed on-the-fly
    sma20 = get_sma("SMA_20", 20)
    sma50 = get_sma("SMA_50", 50)

    # Also compute SMA_5 independently if not in df
    if "SMA_5" not in df.columns:
        sma5 = float(close.rolling(5).mean().iloc[-1])

    price = float(close.iloc[-1])

    # Two-MA (§3.12): SMA_20 vs SMA_50
    two_ma_bull = sma20 > sma50
    two_ma_bear = sma20 < sma50

    # Three-MA (§3.13): SMA_5 > SMA_20 > SMA_50 → Long
    three_ma_bull = sma5 > sma20 > sma50
    three_ma_bear = sma5 < sma20 < sma50

    # Price above/below all MAs
    price_above_all = price > sma20 and price > sma50
    price_below_all = price < sma20 and price < sma50

    # Combined signal logic
    bull_count = sum([two_ma_bull, three_ma_bull, price_above_all])
    bear_count = sum([two_ma_bear, three_ma_bear, price_below_all])

    if bull_count >= 2:
        signal = "Bullish"
        regime = "Uptrend"
    elif bear_count >= 2:
        signal = "Bearish"
        regime = "Downtrend"
    else:
        signal = "Neutral"
        regime = "Sideways"

    detail = {
        "sma5":  round(sma5, 4),
        "sma20": round(sma20, 4),
        "sma50": round(sma50, 4),
        "price": round(price, 4),
        "two_ma_crossover":  "Golden Cross" if two_ma_bull else "Death Cross" if two_ma_bear else "Flat",
        "three_ma_alignment": "Full Bull" if three_ma_bull else "Full Bear" if three_ma_bear else "Mixed",
    }
    return signal, regime, detail


# ──────────────────────────────────────────────────────────────────────────────
# 4. SUPPORT & RESISTANCE / PIVOT POINTS  (Kakushadze & Serur §3.14)
# ──────────────────────────────────────────────────────────────────────────────

def _pivot_signal(df: pd.DataFrame):
    """
    Pivot-point price-action signal using previous day's High, Low, Close.
    Eqs. (325)–(328).

    Returns:
        signal      : "Near Support" | "Near Resistance" | "Middle"
        pivot_dict  : dict  with C, R, S, SL, TP
    """
    if len(df) < 2:
        mid_price = float(df["Close"].iloc[-1])
        return "Middle", {"C": mid_price, "R": mid_price, "S": mid_price,
                          "stop_loss": mid_price, "take_profit": mid_price}

    prev = df.iloc[-2]
    PH = float(prev["High"])
    PL = float(prev["Low"])
    PC = float(prev["Close"])

    C = (PH + PL + PC) / 3.0        # Eq. (325) – Pivot Centre
    R = 2.0 * C - PL                 # Eq. (326) – Resistance
    S = 2.0 * C - PH                 # Eq. (327) – Support

    price = float(df["Close"].iloc[-1])

    # Eq. (328) — price-action signal
    band = (R - S) * 0.15            # 15 % proximity band
    if price <= S + band:
        signal = "Near Support"
    elif price >= R - band:
        signal = "Near Resistance"
    else:
        signal = "Middle"

    pivot_dict = {
        "pivot_centre":    round(C, 4),
        "resistance":      round(R, 4),
        "support":         round(S, 4),
        "stop_loss":       round(S, 4),      # natural SL = Support level
        "take_profit":     round(R, 4),      # natural TP = Resistance level
        "current_price":   round(price, 4),
        "band_pct":        round((R - S) / C * 100, 2),
    }
    return signal, pivot_dict


# ──────────────────────────────────────────────────────────────────────────────
# 5. DONCHIAN CHANNEL  (Kakushadze & Serur §3.15)
# ──────────────────────────────────────────────────────────────────────────────

def _donchian_signal(df: pd.DataFrame, T: int = 20):
    """
    Donchian Channel breakout/bounce signal. Eqs. (329)–(331).

    Returns:
        signal         : "Bullish" | "Bearish" | "Neutral"
        channel_detail : dict
    """
    if len(df) < T + 1:
        p = float(df["Close"].iloc[-1])
        return "Neutral", {"B_up": p, "B_down": p, "channel_width_pct": 0.0}

    prices = df["Close"].squeeze().values

    B_up   = float(np.max(prices[-T-1:-1]))  # Eq. (329): max of last T closes (excl today)
    B_down = float(np.min(prices[-T-1:-1]))  # Eq. (330): min of last T closes (excl today)
    price  = float(prices[-1])

    width_pct = ((B_up - B_down) / B_down * 100) if B_down > 0 else 0.0

    # Breakout above ceiling → bullish; breakthrough below floor → bearish
    tol = (B_up - B_down) * 0.03            # 3 % tolerance
    if price >= B_up - tol:
        signal = "Bullish"
    elif price <= B_down + tol:
        signal = "Bearish"
    else:
        signal = "Neutral"

    channel_detail = {
        "donchian_upper":    round(B_up, 4),
        "donchian_lower":    round(B_down, 4),
        "channel_width_pct": round(width_pct, 2),
        "channel_period":    T,
        "price_position_pct": round((price - B_down) / (B_up - B_down) * 100, 1) if (B_up - B_down) > 0 else 50.0,
    }
    return signal, channel_detail


# ──────────────────────────────────────────────────────────────────────────────
# 6. NAÏVE BAYES SENTIMENT  (Kakushadze & Serur §18.3 analogue)
# ──────────────────────────────────────────────────────────────────────────────

def _sentiment_signal(sentiment_score: float):
    """
    Maps a continuous FinBERT sentiment score to a discrete Positive/Negative/Neutral label.
    Mirrors the Bernoulli Naïve Bayes idea: token polarity sum.

    Returns:
        label : "Positive" | "Negative" | "Neutral"
    """
    if sentiment_score > 0.10:
        return "Positive"
    elif sentiment_score < -0.10:
        return "Negative"
    else:
        return "Neutral"


# ──────────────────────────────────────────────────────────────────────────────
# 7. ASO (Average Sentiment Oscillator) — intrabar + group pressure
# ──────────────────────────────────────────────────────────────────────────────

def _aso_signal(df: pd.DataFrame):
    """
    Uses pre-computed ASO_Bulls / ASO_Bears columns.
    Returns signal and detail dict.
    """
    if 'ASO_Bulls' not in df.columns or 'ASO_Bears' not in df.columns:
        return "Neutral", {"aso_bulls": 0, "aso_bears": 0, "aso_diff": 0}

    bulls = float(df['ASO_Bulls'].iloc[-1]) if pd.notna(df['ASO_Bulls'].iloc[-1]) else 0.0
    bears = float(df['ASO_Bears'].iloc[-1]) if pd.notna(df['ASO_Bears'].iloc[-1]) else 0.0
    diff  = bulls - bears

    if diff > 2:
        signal = "Bullish"
    elif diff < -2:
        signal = "Bearish"
    else:
        signal = "Neutral"

    return signal, {"aso_bulls": round(bulls, 2), "aso_bears": round(bears, 2), "aso_diff": round(diff, 2)}


# ──────────────────────────────────────────────────────────────────────────────
# 8. BOLLINGER BANDS + RSI (Freqtrade)
# ──────────────────────────────────────────────────────────────────────────────

def _bollinger_rsi_signal(df: pd.DataFrame):
    """
    Freqtrade inspired strategy: Buy when price crosses below lower Bollinger Band and RSI < 30.
    Sell when price crosses above upper Bollinger Band and RSI > 70.
    """
    if 'BB_Lower' not in df.columns or 'BB_Upper' not in df.columns or 'RSI' not in df.columns:
        return "Neutral", {"bb_lower": 0, "bb_upper": 0, "rsi": 0}

    close = float(df['Close'].iloc[-1])
    bb_lower = float(df['BB_Lower'].iloc[-1]) if pd.notna(df['BB_Lower'].iloc[-1]) else 0.0
    bb_upper = float(df['BB_Upper'].iloc[-1]) if pd.notna(df['BB_Upper'].iloc[-1]) else 0.0
    rsi = float(df['RSI'].iloc[-1]) if pd.notna(df['RSI'].iloc[-1]) else 50.0

    if close < bb_lower and rsi < 30:
        signal = "Bullish"
    elif close > bb_upper and rsi > 70:
        signal = "Bearish"
    else:
        signal = "Neutral"

    return signal, {"bb_lower": round(bb_lower, 2), "bb_upper": round(bb_upper, 2), "rsi": round(rsi, 2)}


# ──────────────────────────────────────────────────────────────────────────────
# 9. MACD v2.4 — Adaptive MACD + Divergence Engine
# ──────────────────────────────────────────────────────────────────────────────

def _adaptive_macd_v24_signal(df: pd.DataFrame, profile: str = "Standart"):
    """
    Python port of MACD v2.4 Ultimate Edition.
    Profiles: 'Standart'(12-26-9), 'Fibonacci'(13-34-8), 'Balina'(20-50-10),
              'AgirCekim'(55-68-6), 'Scalp'(5-13-3)
    Detects: crossover, histogram zone, regular + hidden divergence (last 2 pivots).
    """
    profiles = {
        "Standart":   (12, 26, 9),
        "Fibonacci":  (13, 34, 8),
        "Balina":     (20, 50, 10),
        "AgirCekim":  (55, 68, 6),
        "Scalp":      (5,  13, 3),
    }
    fast, slow, sig_p = profiles.get(profile, (12, 26, 9))

    if len(df) < slow + sig_p + 5:
        return "Neutral", {}

    close = df["Close"].squeeze()

    ema_fast   = close.ewm(span=fast,  adjust=False).mean()
    ema_slow   = close.ewm(span=slow,  adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=sig_p, adjust=False).mean()
    histogram  = macd_line - signal_line

    m_cur  = float(macd_line.iloc[-1])
    m_prev = float(macd_line.iloc[-2])
    s_cur  = float(signal_line.iloc[-1])
    s_prev = float(signal_line.iloc[-2])
    h_cur  = float(histogram.iloc[-1])
    h_prev = float(histogram.iloc[-2])

    bullish_cross = (m_prev <= s_prev) and (m_cur > s_cur)
    bearish_cross = (m_prev >= s_prev) and (m_cur < s_cur)
    hist_zone = 2 if h_cur > 0 and h_cur > h_prev else \
                1 if h_cur > 0 and h_cur <= h_prev else \
               -1 if h_cur < 0 and h_cur > h_prev else -2

    # --- Divergence (last 50 bars, 2 pivots) ---
    n = min(50, len(df) - 1)
    price_slice  = df["Low"].iloc[-n:].values
    macd_slice   = macd_line.iloc[-n:].values
    high_slice   = df["High"].iloc[-n:].values
    macd_h_slice = macd_line.iloc[-n:].values

    def _find_pivots(arr, order=3):
        pivots = []
        for i in range(order, len(arr) - order):
            if all(arr[i] < arr[i-j] for j in range(1, order+1)) and \
               all(arr[i] < arr[i+j] for j in range(1, order+1)):
                pivots.append(i)
        return pivots

    def _find_peaks(arr, order=3):
        pivots = []
        for i in range(order, len(arr) - order):
            if all(arr[i] > arr[i-j] for j in range(1, order+1)) and \
               all(arr[i] > arr[i+j] for j in range(1, order+1)):
                pivots.append(i)
        return pivots

    bull_div = False
    bear_div = False
    hidden_bull = False
    hidden_bear = False

    lows  = _find_pivots(price_slice)
    highs = _find_peaks(high_slice)

    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if price_slice[i2] < price_slice[i1] and macd_slice[i2] > macd_slice[i1]:
            bull_div = True
        if price_slice[i2] > price_slice[i1] and macd_slice[i2] < macd_slice[i1]:
            hidden_bull = True

    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if high_slice[i2] > high_slice[i1] and macd_h_slice[i2] < macd_h_slice[i1]:
            bear_div = True
        if high_slice[i2] < high_slice[i1] and macd_h_slice[i2] > macd_h_slice[i1]:
            hidden_bear = True

    # --- Whisper (MTF proxy using hist_zone) ---
    if bullish_cross and hist_zone >= 1:
        signal = "Bullish"
    elif bearish_cross and hist_zone <= -1:
        signal = "Bearish"
    elif bull_div or hidden_bull:
        signal = "Bullish"
    elif bear_div or hidden_bear:
        signal = "Bearish"
    elif m_cur > s_cur and hist_zone >= 1:
        signal = "Bullish"
    elif m_cur < s_cur and hist_zone <= -1:
        signal = "Bearish"
    else:
        signal = "Neutral"

    detail = {
        "profile":       profile,
        "fast_slow_sig": f"{fast}-{slow}-{sig_p}",
        "macd":          round(m_cur, 4),
        "signal":        round(s_cur, 4),
        "histogram":     round(h_cur, 4),
        "hist_zone":     hist_zone,
        "bullish_cross": bullish_cross,
        "bearish_cross": bearish_cross,
        "bull_div":      bull_div,
        "bear_div":      bear_div,
        "hidden_bull":   hidden_bull,
        "hidden_bear":   hidden_bear,
    }
    return signal, detail


# ──────────────────────────────────────────────────────────────────────────────
# 10. StochRSI v2.4 — Jackpot + Fibonacci Time + Gann Harmonics
# ──────────────────────────────────────────────────────────────────────────────

def _stoch_rsi_v24_signal(df: pd.DataFrame):
    """
    Python port of StochRSI v2.4:
    • StochRSI K/D (RSI 14, Stoch 14, smooth 3-3)
    • Jackpot: K < 20 → aşırı satım fırsatı skoru
    • Fibonacci Time: bar sayısı kritik Fibo sayısına eşit mi?
    • Gann Square-of-Nine: fiyata yakın destek/direnç
    """
    if len(df) < 30:
        return "Neutral", {}

    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()

    # StochRSI hesaplama
    delta  = close.diff()
    up     = delta.clip(lower=0)
    down   = -delta.clip(upper=0)
    rs_up  = up.ewm(alpha=1/14, adjust=False).mean()
    rs_dn  = down.ewm(alpha=1/14, adjust=False).mean()
    rsi    = 100 - (100 / (1 + rs_up / rs_dn.replace(0, np.nan)))

    rsi_ll = rsi.rolling(14).min()
    rsi_hh = rsi.rolling(14).max()
    rng    = (rsi_hh - rsi_ll).replace(0, np.nan)
    stoch  = (rsi - rsi_ll) / rng * 100
    k_line = stoch.rolling(3).mean()
    d_line = k_line.rolling(3).mean()

    k = float(k_line.iloc[-1]) if pd.notna(k_line.iloc[-1]) else 50.0
    d = float(d_line.iloc[-1]) if pd.notna(d_line.iloc[-1]) else 50.0
    k_prev = float(k_line.iloc[-2]) if pd.notna(k_line.iloc[-2]) else 50.0
    d_prev = float(d_line.iloc[-2]) if pd.notna(d_line.iloc[-2]) else 50.0

    bull_cross = (k_prev <= d_prev) and (k > d)
    bear_cross = (k_prev >= d_prev) and (k < d)

    # Jackpot — K < 20 → oversold (aşırı satım)
    jackpot_score = 0
    if k < 20: jackpot_score += 2
    elif k < 30: jackpot_score += 1
    if k > 80: jackpot_score -= 2
    elif k > 70: jackpot_score -= 1

    # Fibonacci Time Cycles (son büyük pivottan kaç bar geçti?)
    fib_numbers = {13, 21, 34, 55, 89, 144, 233}
    n_bars = len(df)
    # Basit pivot: son 21 barlık en yüksek/düşük
    window = min(21, n_bars - 1)
    if window > 0:
        high_slice_w = high.iloc[-window:]
        low_slice_w  = low.iloc[-window:]
        ph_pos = int(high_slice_w.values.argmax())   # integer offset within slice
        pl_pos = int(low_slice_w.values.argmin())
        # Convert to offset from end of df
        bars_since_ph = window - 1 - ph_pos
        bars_since_pl = window - 1 - pl_pos
        bars_since = min(bars_since_ph, bars_since_pl)   # most recent pivot
    else:
        bars_since = 0
    is_fib_time = bars_since in fib_numbers

    # Gann Square-of-Nine
    price_now = float(close.iloc[-1])
    step = 0.5
    gann_root = np.sqrt(price_now)
    gann_sup  = (np.floor(gann_root / step) * step) ** 2
    gann_res  = (np.floor(gann_root / step) * step + step) ** 2
    near_gann_support  = abs(price_now - gann_sup) / (price_now + 1e-9) < 0.01
    near_gann_resist   = abs(price_now - gann_res) / (price_now + 1e-9) < 0.01

    # Sinyal
    if bull_cross and k < 30:
        signal = "Bullish"
    elif bear_cross and k > 70:
        signal = "Bearish"
    elif jackpot_score >= 2 and k > d:
        signal = "Bullish"
    elif jackpot_score <= -2 and k < d:
        signal = "Bearish"
    elif k > d and k > 50:
        signal = "Bullish"
    elif k < d and k < 50:
        signal = "Bearish"
    else:
        signal = "Neutral"

    # Eğer Fibonacci zaman + Gann destek üst üste gelirse sinyali güçlendir
    if is_fib_time and near_gann_support and signal == "Neutral":
        signal = "Bullish"   # Holy Trinity hint

    zone = "💀 Aşırı Satım" if k < 20 else "🐋 Uyanış" if k < 40 else \
           "⚖️ Karar" if k < 60 else "🚀 Ralli" if k < 80 else "🔥 Turbo"

    detail = {
        "k":               round(k, 2),
        "d":               round(d, 2),
        "zone":            zone,
        "bull_cross":      bull_cross,
        "bear_cross":      bear_cross,
        "jackpot_score":   jackpot_score,
        "bars_since_pivot": bars_since,
        "is_fib_time":     is_fib_time,
        "gann_support":    round(gann_sup, 4),
        "gann_resistance": round(gann_res, 4),
        "near_gann_support": near_gann_support,
        "near_gann_resist":  near_gann_resist,
    }
    return signal, detail


# ──────────────────────────────────────────────────────────────────────────────
# 11. MACD + EMA (Freqtrade — korundu)
# ──────────────────────────────────────────────────────────────────────────────

def _macd_ema_signal(df: pd.DataFrame):
    """
    Freqtrade inspired strategy: Buy when MACD crosses above MACD_Signal and close > EMA_200.
    Sell when MACD crosses below MACD_Signal and close < EMA_200.
    """
    if 'MACD' not in df.columns or 'MACD_Signal' not in df.columns:
        return "Neutral", {"macd": 0, "macd_signal": 0, "ema200": 0}

    close = float(df['Close'].iloc[-1])
    macd = float(df['MACD'].iloc[-1]) if pd.notna(df['MACD'].iloc[-1]) else 0.0
    macd_signal = float(df['MACD_Signal'].iloc[-1]) if pd.notna(df['MACD_Signal'].iloc[-1]) else 0.0
    macd_prev = float(df['MACD'].iloc[-2]) if len(df) > 1 and pd.notna(df['MACD'].iloc[-2]) else 0.0
    macd_signal_prev = float(df['MACD_Signal'].iloc[-2]) if len(df) > 1 and pd.notna(df['MACD_Signal'].iloc[-2]) else 0.0
    
    # We don't have EMA_200, so we use SMA_200 or 50 if 200 not available.
    ema200 = float(df['SMA_200'].iloc[-1]) if 'SMA_200' in df.columns and pd.notna(df['SMA_200'].iloc[-1]) else (float(df['SMA_50'].iloc[-1]) if 'SMA_50' in df.columns and pd.notna(df['SMA_50'].iloc[-1]) else close)

    macd_cross_up = (macd_prev <= macd_signal_prev) and (macd > macd_signal)
    macd_cross_down = (macd_prev >= macd_signal_prev) and (macd < macd_signal)

    if macd_cross_up and close > ema200:
        signal = "Bullish"
    elif macd_cross_down and close < ema200:
        signal = "Bearish"
    else:
        signal = "Neutral"

    return signal, {"macd": round(macd, 4), "macd_signal": round(macd_signal, 4), "ema200": round(ema200, 2)}


# ──────────────────────────────────────────────────────────────────────────────
# 12. WaveTrend MTF v5.0 PRO  (LazyBear port + ATR dynamic + divergence)
# ──────────────────────────────────────────────────────────────────────────────

def _wavetrend_v5_signal(df: pd.DataFrame, n1: int = 10, n2: int = 21,
                          ob1: float = 60, os1: float = -60):
    """
    Python port of WaveTrend MTF v5.0 PRO.
    Computes WT1/WT2 from hlc3, detects crossovers, zones, and
    simple pivot-based divergence (last 2 troughs / peaks).
    ATR-dynamic period scaling mirrors the Pine Script logic.
    """
    if len(df) < n2 + 10:
        return "Neutral", {}

    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    close = df["Close"].squeeze()
    hlc3  = (high + low + close) / 3

    # ATR-dynamic period scaling (mirror Pine clamped_ratio logic)
    atr_cur = df["ATR"].iloc[-1] if "ATR" in df.columns and pd.notna(df["ATR"].iloc[-1]) else float(close.diff().abs().rolling(14).mean().iloc[-1])
    atr_base = float(close.diff().abs().rolling(14).mean().rolling(50).mean().iloc[-1]) if len(df) >= 64 else atr_cur
    vol_ratio = atr_cur / atr_base if atr_base > 0 else 1.0
    clamped   = max(0.6, min(1.5, vol_ratio))
    dyn_n1 = max(1, round(n1 * clamped))
    dyn_n2 = max(1, round(n2 * clamped))

    # EMA helper
    def _ema(s, span):
        return s.ewm(span=span, adjust=False).mean()

    esa  = _ema(hlc3, dyn_n1)
    d    = _ema((hlc3 - esa).abs(), dyn_n1)
    ci   = (hlc3 - esa) / (0.015 * d.replace(0, np.nan))
    wt1  = _ema(ci, dyn_n2)
    wt2  = wt1.rolling(4).mean()

    w1 = float(wt1.iloc[-1]); w1p = float(wt1.iloc[-2])
    w2 = float(wt2.iloc[-1]); w2p = float(wt2.iloc[-2])

    cross_up   = (w1p <= w2p) and (w1 > w2)
    cross_down = (w1p >= w2p) and (w1 < w2)
    is_ob = w1 >= ob1
    is_os = w1 <= os1
    w1_up  = w1 > w1p

    # Simple divergence (last 50 bars, 2 pivots)
    bull_div = bear_div = False
    n = min(50, len(df) - 1)
    wt1_s  = wt1.iloc[-n:].values
    low_s  = low.iloc[-n:].values
    high_s = high.iloc[-n:].values

    def _lows(arr, order=3):
        return [i for i in range(order, len(arr)-order)
                if all(arr[i] < arr[i-j] for j in range(1,order+1))
                and all(arr[i] < arr[i+j] for j in range(1,order+1))]
    def _highs(arr, order=3):
        return [i for i in range(order, len(arr)-order)
                if all(arr[i] > arr[i-j] for j in range(1,order+1))
                and all(arr[i] > arr[i+j] for j in range(1,order+1))]

    lows_idx  = _lows(low_s)
    highs_idx = _highs(high_s)
    if len(lows_idx) >= 2:
        i1, i2 = lows_idx[-2], lows_idx[-1]
        if low_s[i2] < low_s[i1] and wt1_s[i2] > wt1_s[i1]:
            bull_div = True
    if len(highs_idx) >= 2:
        i1, i2 = highs_idx[-2], highs_idx[-1]
        if high_s[i2] > high_s[i1] and wt1_s[i2] < wt1_s[i1]:
            bear_div = True

    # Status label
    if cross_up and is_os:
        status = "DERIN DIP"
    elif cross_up and w1 < 0:
        status = "DIP KESISIM"
    elif cross_up:
        status = "TREND UP"
    elif cross_down and is_ob:
        status = "TAVAN SAT"
    elif cross_down and w1 < 0:
        status = "DIP SAT"
    elif cross_down:
        status = "KIRILIM DOWN"
    elif is_ob and not w1_up:
        status = "TAVAN DONUSU"
    elif is_os and w1_up:
        status = "DIP DONUSU"
    elif w1 > 0 and w1_up:
        status = "YUKSELIYOR"
    elif w1 < 0 and w1_up:
        status = "TOPARLIYOR"
    else:
        status = "ZAYIFLIYOR"

    # Signal
    if cross_up and (is_os or w1 < 0):
        signal = "Bullish"
    elif cross_down and (is_ob or w1 > 0):
        signal = "Bearish"
    elif bull_div:
        signal = "Bullish"
    elif bear_div:
        signal = "Bearish"
    elif w1 > w2 and w1 > 0:
        signal = "Bullish"
    elif w1 < w2 and w1 < 0:
        signal = "Bearish"
    else:
        signal = "Neutral"

    detail = {
        "wt1":        round(w1, 2),
        "wt2":        round(w2, 2),
        "dyn_n1":     dyn_n1,
        "dyn_n2":     dyn_n2,
        "vol_ratio":  round(vol_ratio, 3),
        "status":     status,
        "cross_up":   cross_up,
        "cross_down": cross_down,
        "is_ob":      is_ob,
        "is_os":      is_os,
        "bull_div":   bull_div,
        "bear_div":   bear_div,
    }
    return signal, detail


# ──────────────────────────────────────────────────────────────────────────────
# 13. Bollinger Hunter Pro v5.7  (Squeeze + Breakout + Walking + W/M + %B Div)
# ──────────────────────────────────────────────────────────────────────────────

def _bollinger_hunter_v57_signal(df: pd.DataFrame, length: int = 20,
                                   mult: float = 2.0, sq_len: int = 100,
                                   trend_len: int = 3, bo_limit: float = 10.0):
    """
    Python port of Bollinger Hunter Pro v5.7.
    Detects: Squeeze, Breakout, Walking-the-Bands, Mean Reversion,
             W/M double-bottom/top patterns, %B divergence, MA cross.
    Priority order (highest wins): %B Div > Breakout > Walking > MA Cross > W/M > Squeeze.
    """
    if len(df) < max(length, sq_len) + 10:
        return "Neutral", {}

    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    open_ = df["Open"].squeeze()

    # Bollinger Bands
    middle = close.rolling(length).mean()
    std    = close.rolling(length).std()
    upper  = middle + mult * std
    lower  = middle - mult * std
    bb_width  = ((upper - lower) / middle.replace(0, np.nan)) * 100
    percent_b = (close - lower) / (upper - lower).replace(0, np.nan)

    # Volatility filter (ATR)
    atr14 = df["ATR"].squeeze() if "ATR" in df.columns else close.diff().abs().rolling(14).mean()
    atr50 = atr14.rolling(50).mean()
    is_high_vol = float(atr14.iloc[-1]) > float(atr50.iloc[-1]) * 1.2

    dist_to_upper = max(0.0, (float(close.iloc[-1]) - float(upper.iloc[-1])) / float(upper.iloc[-1]) * 100)

    # 1. Squeeze
    is_squeeze = float(bb_width.iloc[-1]) <= float(bb_width.rolling(sq_len).min().iloc[-1]) * 1.10

    # 2. Breakout (strict: upper up AND lower down)
    bw_up   = float(bb_width.iloc[-1]) > float(bb_width.iloc[-2])
    up_up   = float(upper.iloc[-1])    > float(upper.iloc[-2])
    low_dn  = float(lower.iloc[-1])    < float(lower.iloc[-2])
    is_breakout = (float(close.iloc[-1]) > float(upper.iloc[-1])) and bw_up and up_up and low_dn

    # 3. Walking the Bands
    walk_series = (close >= upper).astype(int)
    is_walking = int(walk_series.iloc[-trend_len:].sum()) >= (trend_len - 1)

    # 4. Mean Reversion
    pb_now = float(percent_b.iloc[-1]) if pd.notna(percent_b.iloc[-1]) else 0.5
    is_rev_long  = (not is_squeeze) and (not is_breakout) and \
                   (float(low.iloc[-1]) <= float(lower.iloc[-1])) and \
                   (float(close.iloc[-1]) > float(open_.iloc[-1])) and pb_now > 0.1
    is_rev_short = (not is_squeeze) and (not is_breakout) and \
                   (float(high.iloc[-1]) >= float(upper.iloc[-1])) and \
                   (float(close.iloc[-1]) < float(open_.iloc[-1])) and pb_now < 0.9

    # 5. W / M Patterns (simplified: pivot outside band then pivot inside)
    order = 5
    n = len(df)

    def _find_pl(series_low, series_lower, ord_=5):
        """Returns index of most recent pivot low if it exists."""
        for i in range(ord_, min(n - ord_, 60)):
            idx = -(i + 1)
            if all(series_low.iloc[idx] < series_low.iloc[idx - j] for j in range(1, ord_+1)) and \
               all(series_low.iloc[idx] < series_low.iloc[idx + j] for j in range(1, ord_+1)):
                return idx, float(series_low.iloc[idx]) < float(series_lower.iloc[idx])
        return None, False

    def _find_ph(series_high, series_upper, ord_=5):
        for i in range(ord_, min(n - ord_, 60)):
            idx = -(i + 1)
            if all(series_high.iloc[idx] > series_high.iloc[idx - j] for j in range(1, ord_+1)) and \
               all(series_high.iloc[idx] > series_high.iloc[idx + j] for j in range(1, ord_+1)):
                return idx, float(series_high.iloc[idx]) > float(upper.iloc[idx])
        return None, False

    pl_idx, pl_outside = _find_pl(low, lower)
    ph_idx, ph_outside = _find_ph(high, upper)

    # W: previous low outside band, current low inside band
    is_w = pl_outside and (float(low.iloc[-1]) > float(lower.iloc[-1]))
    # M: previous high outside band, current high inside band
    is_m = ph_outside and (float(high.iloc[-1]) < float(upper.iloc[-1]))

    # 6. %B Divergence
    bear_div = False
    bull_div = False

    if ph_idx is not None:
        prev_ph_price = float(high.iloc[ph_idx])
        prev_pb       = float(percent_b.iloc[ph_idx]) if pd.notna(percent_b.iloc[ph_idx]) else 0.5
        cur_ph_price  = float(high.iloc[-1])
        cur_pb        = pb_now
        if cur_ph_price > prev_ph_price and cur_pb < prev_pb:
            bear_div = True

    if pl_idx is not None:
        prev_pl_price = float(low.iloc[pl_idx])
        prev_pb_pl    = float(percent_b.iloc[pl_idx]) if pd.notna(percent_b.iloc[pl_idx]) else 0.5
        cur_pl_price  = float(low.iloc[-1])
        if cur_pl_price < prev_pl_price and pb_now > prev_pb_pl:
            bull_div = True

    # 7. MA Cross
    is_ma_cross = (float(close.iloc[-2]) <= float(middle.iloc[-2])) and \
                  (float(close.iloc[-1]) > float(middle.iloc[-1]))

    # Priority signal
    if bear_div:
        signal = "Bearish"
        mode   = "BB %B Bear Div"
    elif bull_div:
        signal = "Bullish"
        mode   = "BB %B Bull Div"
    elif is_breakout and dist_to_upper <= bo_limit:
        signal = "Bullish"
        mode   = "BB Breakout"
    elif is_walking:
        signal = "Bullish"
        mode   = "BB Walking"
    elif is_ma_cross:
        signal = "Bullish"
        mode   = "BB MA Cross"
    elif is_w:
        signal = "Bullish"
        mode   = "BB W Pattern"
    elif is_m:
        signal = "Bearish"
        mode   = "BB M Pattern"
    elif is_rev_long:
        signal = "Bullish"
        mode   = "BB Mean Reversion Long"
    elif is_rev_short:
        signal = "Bearish"
        mode   = "BB Mean Reversion Short"
    elif is_squeeze:
        signal = "Neutral"
        mode   = "BB Squeeze"
    else:
        signal = "Neutral"
        mode   = "BB Ranging"

    detail = {
        "mode":          mode,
        "bb_width":      round(float(bb_width.iloc[-1]), 2),
        "percent_b":     round(pb_now, 3),
        "is_squeeze":    is_squeeze,
        "is_breakout":   is_breakout,
        "is_walking":    is_walking,
        "is_w":          is_w,
        "is_m":          is_m,
        "bear_div":      bear_div,
        "bull_div":      bull_div,
        "is_ma_cross":   is_ma_cross,
        "is_high_vol":   is_high_vol,
        "dist_upper_pct": round(dist_to_upper, 2),
    }
    return signal, detail


# ──────────────────────────────────────────────────────────────────────────────
# MASTER SYNTHESIS  — combines all 12 signals into a consensus
# ──────────────────────────────────────────────────────────────────────────────

def run_quant_ai_analysis(df: pd.DataFrame, sentiment_score: float = 0.0) -> dict:
    """
    Main entry point.  Runs all 6 strategy modules and produces the
    consensus JSON signal dict.

    Args:
        df               – OHLCV DataFrame with technical indicators (from calculate_technical_indicators)
        sentiment_score  – float in [-1, +1] from get_finbert_sentiment()

    Returns:
        result dict matching the Quant-AI JSON schema
    """
    if df is None or df.empty or len(df) < 55:
        return _error_signal("Yetersiz veri: En az 55 günlük OHLCV verisi gereklidir.")

    # ── Run each module ───────────────────────────────────────────────────────
    mom_signal,  R_cum, R_risadj, mom_detail  = _price_momentum_signal(df)
    ma_signal,   ma_regime, ma_detail         = _moving_average_signal(df)
    pivot_pa,    pivot_dict                   = _pivot_signal(df)
    don_signal,  don_detail                   = _donchian_signal(df)
    sent_label                                = _sentiment_signal(sentiment_score)
    aso_signal,  aso_detail                   = _aso_signal(df)
    bb_rsi_signal, bb_rsi_detail              = _bollinger_rsi_signal(df)
    macd_ema_signal, macd_ema_detail          = _macd_ema_signal(df)
    # ── v2.4 Yeni Modüller ────────────────────────────────────────────────────
    macd_v24_signal, macd_v24_detail          = _adaptive_macd_v24_signal(df)
    stochrsi_v24_signal, stochrsi_v24_detail  = _stoch_rsi_v24_signal(df)
    # ── v5.0 / v5.7 Yeni Modüller ───────────────────────────────────────────
    wavetrend_signal, wavetrend_detail        = _wavetrend_v5_signal(df)
    bbhunter_signal,  bbhunter_detail         = _bollinger_hunter_v57_signal(df)

    # ── ATR for volatility context ────────────────────────────────────────────
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and pd.notna(df["ATR"].iloc[-1]) else 0.0
    price = float(df["Close"].iloc[-1])
    atr_pct = (atr / price * 100) if price > 0 else 0.0

    # ── Convert signals to direction votes ───────────────────────────────────
    # Each strategy casts a vote: +1 (bull), -1 (bear), 0 (neutral)
    def _vote(sig, positive_labels, negative_labels):
        if sig in positive_labels: return +1
        if sig in negative_labels: return -1
        return 0

    vote_mom    = _vote(mom_signal,  ["Bullish"], ["Bearish"])
    vote_ma     = _vote(ma_signal,   ["Bullish"], ["Bearish"])
    vote_pivot  = _vote(pivot_pa,    ["Near Support"], ["Near Resistance"])
    vote_don    = _vote(don_signal,  ["Bullish"], ["Bearish"])
    vote_sent   = _vote(sent_label,  ["Positive"], ["Negative"])
    vote_aso    = _vote(aso_signal,   ["Bullish"], ["Bearish"])

    vote_bb_rsi       = _vote(bb_rsi_signal,       ["Bullish"], ["Bearish"])
    vote_macd_ema     = _vote(macd_ema_signal,     ["Bullish"], ["Bearish"])
    vote_macd_v24     = _vote(macd_v24_signal,     ["Bullish"], ["Bearish"])
    vote_stochrsi_v24 = _vote(stochrsi_v24_signal, ["Bullish"], ["Bearish"])
    vote_wavetrend    = _vote(wavetrend_signal,     ["Bullish"], ["Bearish"])
    vote_bbhunter     = _vote(bbhunter_signal,      ["Bullish"], ["Bearish"])

    votes = [vote_mom, vote_ma, vote_pivot, vote_don, vote_sent, vote_aso,
             vote_bb_rsi, vote_macd_ema, vote_macd_v24, vote_stochrsi_v24,
             vote_wavetrend, vote_bbhunter]
    total_score = sum(votes)
    bull_count  = sum(v > 0 for v in votes)
    bear_count  = sum(v < 0 for v in votes)
    neut_count  = sum(v == 0 for v in votes)

    # ── Consensus signal (12 strateji, eşik 6/12) ────────────────────────────
    if bull_count >= 6:
        consensus = "BUY"
    elif bear_count >= 6:
        consensus = "SELL"
    else:
        consensus = "HOLD"

    # ── Confidence score (0-100) based on signal confluence ──────────────────
    # Max confluence = 6 strategies agree → confidence 95
    # Bare majority (3 out of 6) → confidence ~47
    # Edge: if ATR% > 3% market is highly volatile → reduce confidence by 10
    n_strategies = len(votes)
    max_agreement = max(bull_count, bear_count)
    base_confidence = int(max_agreement / n_strategies * 95)
    if atr_pct > 3.0:
        base_confidence = max(10, base_confidence - 10)
    confidence_score = min(95, max(5, base_confidence))

    # ── Market regime ─────────────────────────────────────────────────────────
    if atr_pct > 3.0:
        market_regime = "Highly Volatile"
    else:
        market_regime = ma_regime  # derived from MA structure

    # ── Primary strategy (strongest contributor) ──────────────────────────────
    strategy_labels = {
        "momentum":      ("Price Momentum (§3.1)",                vote_mom),
        "two_ma":        ("Dual Moving Average (§3.12)",           vote_ma),
        "pivot":         ("Support & Resistance / Pivot (§3.14)",  vote_pivot),
        "donchian":      ("Donchian Channel (§3.15)",              vote_don),
        "sentiment":     ("Naïve Bayes Sentiment (§18.3)",         vote_sent),
        "aso":           ("ASO Sentiment Oscillator",               vote_aso),
        "bb_rsi":        ("Bollinger Bands + RSI",                  vote_bb_rsi),
        "macd_ema":      ("MACD + EMA (Freqtrade)",                vote_macd_ema),
        "macd_v24":      ("Adaptive MACD v2.4 + Divergence",       vote_macd_v24),
        "stochrsi_v24": ("StochRSI v2.4 (Jackpot + Fibo + Gann)", vote_stochrsi_v24),
        "wavetrend":     ("WaveTrend v5.0 PRO",                    vote_wavetrend),
        "bbhunter":      ("Bollinger Hunter v5.7",                  vote_bbhunter),
    }
    # Primary = strategy whose vote agrees with consensus AND has strongest backing
    consensus_vote = +1 if consensus == "BUY" else -1 if consensus == "SELL" else 0
    matching = [(name, label, v) for name, (label, v) in strategy_labels.items()
                if v == consensus_vote and consensus_vote != 0]
    if matching:
        primary_strategy = matching[0][1]
    else:
        # Fall back to whichever strategy is most extreme
        primary_strategy = "Signal Confluence Engine"

    # ── Stop-loss / Take-profit ────────────────────────────────────────────────
    # Use Pivot S/R as primary; Donchian channel as secondary
    sl_pivot = pivot_dict["stop_loss"]
    tp_pivot = pivot_dict["take_profit"]
    don_lower = don_detail["donchian_lower"]
    don_upper = don_detail["donchian_upper"]

    if consensus == "BUY":
        suggested_sl = round(min(sl_pivot, don_lower), 4)
        suggested_tp = round(max(tp_pivot, don_upper), 4)
    elif consensus == "SELL":
        suggested_sl = round(max(tp_pivot, don_upper), 4)
        suggested_tp = round(min(sl_pivot, don_lower), 4)
    else:  # HOLD
        suggested_sl = round(sl_pivot, 4)
        suggested_tp = round(tp_pivot, 4)

    # Risk-reward ratio
    risk   = abs(price - suggested_sl)
    reward = abs(suggested_tp - price)
    if risk > 1e-6:
        rr = round(reward / risk, 2)
        rr_str = f"1:{rr}"
    else:
        rr_str = "N/A"

    # Risk level based on ATR%
    if atr_pct < 1.5:
        risk_level = "Low"
    elif atr_pct < 3.0:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # ── Analysis summary ──────────────────────────────────────────────────────
    bull_strategies = [label for name, (label, v) in strategy_labels.items() if v > 0]
    bear_strategies = [label for name, (label, v) in strategy_labels.items() if v < 0]

    if consensus == "BUY":
        bullish_list = ", ".join(bull_strategies) if bull_strategies else "momentum faktörleri"
        summary = (f"On iki strateji modülünden {bull_count}'i yukarı yönlü sinyal üretiyor "
                   f"({bullish_list}). {market_regime} piyasa rejimi altında "
                   f"fiyat destek seviyesine yakın ve haber duygusu {sent_label.lower()}.")
    elif consensus == "SELL":
        bearish_list = ", ".join(bear_strategies) if bear_strategies else "momentum faktörleri"
        summary = (f"On iki strateji modülünden {bear_count}'i aşağı yönlü sinyal üretiyor "
                   f"({bearish_list}). {market_regime} piyasa rejimi altında "
                   f"fiyat direnç seviyesine yakın ve haber duygusu {sent_label.lower()}.")
    else:
        summary = (f"Stratejiler arasında güçlü bir uzlaşı bulunmuyor ({bull_count} yukarı, "
                   f"{bear_count} aşağı, {neut_count} nötr sinyal). "
                   f"Piyasa {market_regime} rejimine giriyor; pozisyon açmaktan kaçının.")

    # ── Build final result ────────────────────────────────────────────────────
    result = {
        "consensus_signal":          consensus,
        "confidence_score":          confidence_score,
        "primary_strategy_triggered": primary_strategy,
        "market_regime":             market_regime,
        "analysis_summary":          summary,
        "strategy_breakdown": {
            "momentum":      mom_signal,
            "price_action":  pivot_pa,
            "sentiment":     sent_label,
            "bb_rsi":        bb_rsi_signal,
            "macd_ema":      macd_ema_signal,
            "macd_v24":      macd_v24_signal,
            "stochrsi_v24": stochrsi_v24_signal,
            "wavetrend":     wavetrend_signal,
            "bbhunter":      bbhunter_signal,
        },
        "risk_management": {
            "suggested_stop_loss":  suggested_sl,
            "suggested_take_profit": suggested_tp,
            "risk_reward_ratio":    rr_str,
            "risk_level":           risk_level,
        },
        # ── Extra detail panels (used by UI subpanels) ────────────────────────
        "_detail": {
            "momentum":  mom_detail,
            "moving_avg": ma_detail,
            "pivot":     pivot_dict,
            "donchian":  don_detail,
            "aso":       aso_detail,
            "bb_rsi":    bb_rsi_detail,
            "macd_ema":  macd_ema_detail,
            "sentiment": {
                "raw_score": round(sentiment_score, 4),
                "label":     sent_label,
            },
            "macd_v24":      macd_v24_detail,
            "stochrsi_v24":  stochrsi_v24_detail,
            "wavetrend":     wavetrend_detail,
            "bbhunter":      bbhunter_detail,
            "votes": {
                "momentum":       vote_mom,
                "moving_avg":     vote_ma,
                "pivot":          vote_pivot,
                "donchian":       vote_don,
                "sentiment":      vote_sent,
                "aso":            vote_aso,
                "bb_rsi":         vote_bb_rsi,
                "macd_ema":       vote_macd_ema,
                "macd_v24":       vote_macd_v24,
                "stochrsi_v24":   vote_stochrsi_v24,
                "wavetrend":      vote_wavetrend,
                "bbhunter":       vote_bbhunter,
                "total_bull":     bull_count,
                "total_bear":     bear_count,
                "total_neutral":  neut_count,
            },
            "atr_pct": round(atr_pct, 2),
            "current_price": round(price, 4),
        }
    }
    return result


def _error_signal(msg: str) -> dict:
    """Returns a safe HOLD signal when insufficient data."""
    return {
        "consensus_signal": "HOLD",
        "confidence_score": 0,
        "primary_strategy_triggered": "Insufficient Data",
        "market_regime": "Sideways",
        "analysis_summary": msg,
        "strategy_breakdown": {
            "momentum":      "Neutral",
            "price_action":  "Middle",
            "sentiment":     "Neutral",
            "bb_rsi":        "Neutral",
            "macd_ema":      "Neutral",
            "macd_v24":      "Neutral",
            "stochrsi_v24": "Neutral",
            "wavetrend":     "Neutral",
            "bbhunter":      "Neutral",
        },
        "risk_management": {
            "suggested_stop_loss":  0.0,
            "suggested_take_profit": 0.0,
            "risk_reward_ratio":    "N/A",
            "risk_level":           "High",
        },
        "_detail": {}
    }
