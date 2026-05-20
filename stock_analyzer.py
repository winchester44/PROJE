import os
import re
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────────────────────────────────────

def _is_truthy_env(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_user_error(message, source="System"):
    return {
        "title": message,
        "label": "neutral",
        "polarity": 0.0,
        "link": "#",
        "source": source,
    }


def detect_market_type(ticker_symbol: str) -> str:
    """Ticker sembolünden piyasa tipini otomatik algılar: 'us' | 'bist' | 'crypto'"""
    t = ticker_symbol.upper()
    if '.IS' in t:
        return 'bist'
    crypto_keywords = ['-USD', '-USDT', '-EUR', '-BTC', 'BTC', 'ETH', 'BNB',
                       'SOL', 'XRP', 'ADA', 'DOGE', 'AVAX', 'MATIC']
    if any(k in t for k in crypto_keywords):
        return 'crypto'
    return 'us'


# ──────────────────────────────────────────────────────────────────────────────
# FINBERT PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_sentiment_pipeline():
    from transformers import pipeline
    model_name = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
    offline_mode = _is_truthy_env("FINBERT_OFFLINE_ONLY") or _is_truthy_env("TRANSFORMERS_OFFLINE")
    if offline_mode:
        model_path = os.getenv("FINBERT_LOCAL_PATH", "").strip()
        if model_path:
            return pipeline("sentiment-analysis", model=model_path, local_files_only=True)
        return pipeline("sentiment-analysis", model=model_name, local_files_only=True)
    return pipeline("sentiment-analysis", model=model_name)


# ──────────────────────────────────────────────────────────────────────────────
# VERİ ÇEKME
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=900)
def fetch_data(ticker_symbol, period="2y"):
    """Borsa veya Kripto verilerini ceker. BIST hisseleri icin otomatik .IS dener."""
    def _download_with_retry(sym, period, retries=3):
        """yfinance 1.2.0 intermittent TypeError icin retry wrapper."""
        import time
        for attempt in range(retries):
            try:
                df = yf.download(sym, period=period, interval="1d", progress=False)
                if df is not None:
                    return df
            except TypeError:
                if attempt < retries - 1:
                    time.sleep(0.5)
                else:
                    raise
        return pd.DataFrame()

    df = _download_with_retry(ticker_symbol, period)

    if df.empty and "." not in ticker_symbol:
        bist_symbol = ticker_symbol + ".IS"
        df = _download_with_retry(bist_symbol, period)
        if not df.empty:
            ticker_symbol = bist_symbol

    if df.empty:
        raise ValueError(f"'{ticker_symbol}' icin veri bulunamadi.")

    if isinstance(df.columns, pd.MultiIndex):
        if 'Ticker' in df.columns.names:
            df.columns = df.columns.droplevel('Ticker')
        else:
            df.columns = df.columns.droplevel(1)

    # Gun ici eksik/islenmemis yfinance verilerinde kapanis NaN olabiliyor
    if 'Close' in df.columns:
        df.dropna(subset=['Close'], inplace=True)

    return df, ticker_symbol


# ──────────────────────────────────────────────────────────────────────────────
# TEKNİK İNDİKATÖRLER (GENİŞLETİLMİŞ)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def calculate_technical_indicators(df):
    """
    Gelişmiş teknik indikatörler.
    Yeni: ADX, Stochastic, Volume Spike (Z-score), Gap Pct, Open-Close Pct,
          Return_3D, Return_10D, HighLow Range.
    """
    close  = df['Close']
    volume = df['Volume']
    high   = df['High']
    low    = df['Low']
    open_  = df['Open']

    # ── 1. Hareketli Ortalamalar ─────────────────────────────────────────────
    df['SMA_20']  = close.rolling(window=20).mean()
    df['SMA_50']  = close.rolling(window=50).mean()
    df['SMA_200'] = close.rolling(window=200).mean()
    df['EMA_12']  = close.ewm(span=12, adjust=False).mean()
    df['EMA_26']  = close.ewm(span=26, adjust=False).mean()

    # ── 2. MACD ──────────────────────────────────────────────────────────────
    df['MACD']        = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ── 3. RSI (Wilder alpha = 1/14) ─────────────────────────────────────────
    delta    = close.diff()
    up       = delta.clip(lower=0)
    down     = -1 * delta.clip(upper=0)
    # Fix #6: Wilder RSI doğru alpha = 1/14 (ewm span=14 ile eşdeğer)
    ema_up   = up.ewm(alpha=1/14, adjust=False).mean()
    ema_down = down.ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + ema_up / ema_down.replace(0, np.nan)))

    # ── 4. Bollinger Bandı ───────────────────────────────────────────────────
    df['BB_Middle'] = close.rolling(window=20).mean()
    std             = close.rolling(window=20).std()
    df['BB_Upper']  = df['BB_Middle'] + (std * 2)
    df['BB_Lower']  = df['BB_Middle'] - (std * 2)

    # ── 5. ATR ───────────────────────────────────────────────────────────────
    high_low   = high - low
    high_close = np.abs(high - close.shift())
    low_close  = np.abs(low - close.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR']  = true_range.rolling(14).mean()

    # ── 6. OBV & VWAP ────────────────────────────────────────────────────────
    direction   = np.sign(close.diff().fillna(0))
    df['OBV']   = (direction * volume).cumsum()
    typical     = (high + low + close) / 3
    df['VWAP']  = (typical * volume).cumsum() / volume.cumsum()

    # ── 7. Momentum / Getiri ─────────────────────────────────────────────────
    df['Return_1D']     = close.pct_change()
    df['Return_3D']     = close.pct_change(3)
    df['Return_5D']     = close.pct_change(5)
    df['Return_10D']    = close.pct_change(10)
    df['Return_20D']    = close.pct_change(20)
    df['Volatility_20'] = df['Return_1D'].rolling(20).std()
    df['Trend_Strength']   = (df['SMA_20'] - df['SMA_50']) / df['SMA_50'].replace(0, np.nan)
    df['Price_vs_SMA20']   = (close - df['SMA_20']) / df['SMA_20'].replace(0, np.nan)
    df['Price_vs_SMA50']   = (close - df['SMA_50']) / df['SMA_50'].replace(0, np.nan)
    bb_range               = (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)
    df['BB_Position']      = (close - df['BB_Lower']) / bb_range
    df['Volume_Ratio']     = volume / volume.rolling(20).mean().replace(0, np.nan)
    df['ATR_Pct']          = df['ATR'] / close.replace(0, np.nan)

    # ── 8. ADX (Average Directional Index — Wilder RMA) ─────────────────────
    # Fix #5: Gerçek Wilder smoothing (RMA) kullanılıyor; ewm(alpha=1/14) değil
    def _wilder_smooth(series: pd.Series, period: int = 14) -> pd.Series:
        """Wilder smoothing (RMA): s[i] = s[i-1] * (1 - 1/p) + val[i] * (1/p)"""
        result = np.full(len(series), np.nan)
        vals = series.values
        # İlk geçerli pencereyi bul
        first_valid = pd.Series(vals).first_valid_index()
        if first_valid is None:
            return pd.Series(result, index=series.index)
        start = first_valid + period
        if start > len(vals):
            return pd.Series(result, index=series.index)
        # Fix: İlk değer ORTALAMA olmalı (sum değil) — Wilder'ın orijinal tanımı
        result[start] = np.nanmean(vals[first_valid:start])
        alpha = 1.0 / period
        for i in range(start + 1, len(vals)):
            if np.isnan(vals[i]):
                result[i] = result[i - 1]
            else:
                result[i] = result[i - 1] * (1 - alpha) + vals[i] * alpha
        return pd.Series(result, index=series.index)

    high_chg  = high.diff()
    low_chg   = -low.diff()
    plus_dm   = high_chg.where((high_chg > low_chg) & (high_chg > 0), 0.0)
    minus_dm  = low_chg.where((low_chg > high_chg) & (low_chg > 0), 0.0)
    atr14     = _wilder_smooth(true_range, 14).replace(0, np.nan)
    plus_dm14 = _wilder_smooth(plus_dm, 14)
    minus_dm14= _wilder_smooth(minus_dm, 14)
    df['Plus_DI']  = 100 * plus_dm14  / atr14
    df['Minus_DI'] = 100 * minus_dm14 / atr14
    di_sum    = (df['Plus_DI'] + df['Minus_DI']).replace(0, np.nan)
    dx        = np.abs(df['Plus_DI'] - df['Minus_DI']) / di_sum * 100
    # Note: pass dx directly; _wilder_smooth carries prev value on NaN
    df['ADX'] = _wilder_smooth(dx, 14)

    # ── 9. Stochastic Oscillator (%K / %D) ───────────────────────────────────
    low14      = low.rolling(14).min()
    high14     = high.rolling(14).max()
    df['Stoch_K'] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # ── 10. Volume Spike (Z-score) ────────────────────────────────────────────
    vol_mean          = volume.rolling(20).mean()
    vol_std           = volume.rolling(20).std().replace(0, np.nan)
    df['Volume_Spike'] = (volume - vol_mean) / vol_std

    # ── 11. Gap & Open-Close İlişkisi ────────────────────────────────────────
    df['Gap_Pct']           = (open_ - close.shift(1)) / close.shift(1).replace(0, np.nan)
    df['OpenClose_Pct']     = (close - open_) / open_.replace(0, np.nan)
    df['HighLow_Range_Pct'] = (high - low) / open_.replace(0, np.nan)

    # ── 12. Yeni İndikatörler (Fix #12) ─────────────────────────────────────
    # Williams %R (14 periyot)
    low14_wr  = low.rolling(14).min()
    high14_wr = high.rolling(14).max()
    df['Williams_R'] = -100 * (high14_wr - close) / (high14_wr - low14_wr).replace(0, np.nan)

    # CCI — Commodity Channel Index (20 periyot)
    tp_cci        = (high + low + close) / 3
    tp_sma        = tp_cci.rolling(20).mean()
    tp_mad        = tp_cci.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df['CCI']     = (tp_cci - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

    # VWAP Sapması — günlük fiyatın kümülatif VWAP'tan sapması
    df['VWAP_Dev'] = (close - df['VWAP']) / df['VWAP'].replace(0, np.nan)

    # 52 Hafta Yüksek/Düşük Mesafesi (momentum faktörü)
    high52 = high.rolling(252).max()
    low52  = low.rolling(252).min()
    df['Dist_52W_High'] = (close - high52) / high52.replace(0, np.nan)   # negatif: tepeden uzaklık
    df['Dist_52W_Low']  = (close - low52)  / low52.replace(0, np.nan)    # pozitif: dipten uzaklık

    # Hacim Ağırlıklı RSI
    vol_weight    = volume / volume.rolling(14).mean().replace(0, np.nan)
    vw_up         = (delta.clip(lower=0) * vol_weight)
    vw_down       = (-delta.clip(upper=0) * vol_weight)
    vw_ema_up     = vw_up.ewm(alpha=1/14, adjust=False).mean()
    vw_ema_down   = vw_down.ewm(alpha=1/14, adjust=False).mean()
    df['Vol_RSI'] = 100 - (100 / (1 + vw_ema_up / vw_ema_down.replace(0, np.nan)))

    # ── 12. ASO (Average Sentiment Oscillator) ───────────────────────────────
    # Matriks Indicator Builder formülünden Python'a çevrildi.
    # İki katmanlı boğa/ayı baskısı ölçer: mum-içi (intrabar) + dönemsel (group)
    aso_length = 20

    # Mum-içi aralık
    intrarange = high - low
    # Fix #3: Doji mumda 0/1 yerine NaN kullan; anlamsız değer üretimi önlenir
    K1 = intrarange.where(intrarange > 0, np.nan)

    # Dönemsel (group) aralık
    grouplow   = low.rolling(aso_length).min()
    grouphigh  = high.rolling(aso_length).max()
    groupopen  = open_.shift(aso_length - 1)
    grouprange = grouphigh - grouplow
    K2 = grouprange.where(grouprange > 0, np.nan)

    # Boğa gücü
    intrabarbulls = (((close - low) + (high - open_)) / 2 * 100) / K1
    groupbulls    = (((close - grouplow) + (grouphigh - groupopen)) / 2 * 100) / K2

    # Ayı gücü
    intrabarbears = (((high - close) + (open_ - low)) / 2 * 100) / K1
    groupbears    = (((grouphigh - close) + (groupopen - grouplow)) / 2 * 100) / K2

    # Mode=0: İkisinin ortalaması (hem mum-içi hem dönemsel)
    temp_bulls = (intrabarbulls + groupbulls) / 2
    temp_bears = (intrabarbears + groupbears) / 2

    # SMA ile düzleştirme
    df['ASO_Bulls'] = temp_bulls.rolling(aso_length).mean()
    df['ASO_Bears'] = temp_bears.rolling(aso_length).mean()
    df['ASO_Diff']  = df['ASO_Bulls'] - df['ASO_Bears']

    # Cross sinyali: Bulls Bears'ı yukarı keserse +1, aşağı keserse -1
    aso_prev_diff = df['ASO_Diff'].shift(1)
    df['ASO_Cross'] = np.where(
        (aso_prev_diff <= 0) & (df['ASO_Diff'] > 0),  1,   # Bullish cross
        np.where(
            (aso_prev_diff >= 0) & (df['ASO_Diff'] < 0), -1,  # Bearish cross
            0
        )
    )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# GÖRELİ GÜÇ (RELATIVE STRENGTH VS BENCHMARK INDEX)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def add_relative_strength(df: pd.DataFrame, ticker_symbol: str) -> pd.DataFrame:
    """
    Hissenin benchmark (SPY / XU100.IS / BTC-USD) karşısındaki göreli gücünü hesaplar.
    Başarısız olursa RS sütunlarını 0 ile doldurur (sessizce).
    """
    t = ticker_symbol.upper()
    if '.IS' in t:
        bench_ticker = 'XU100.IS'
    elif any(k in t for k in ['-USD', '-USDT', 'BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'ADA']):
        bench_ticker = 'BTC-USD'
    else:
        bench_ticker = 'SPY'

    try:
        raw = None
        for _attempt in range(3):   # yfinance 1.2.0 intermittent TypeError icin retry
            try:
                raw = yf.download(bench_ticker, period='2y', interval='1d', progress=False)
                if raw is not None and not raw.empty:
                    break
            except (TypeError, Exception):
                import time; time.sleep(0.5)
        if raw is None or raw.empty:
            raise ValueError("Benchmark verisi alinamadi")
        if isinstance(raw.columns, pd.MultiIndex):
            if 'Ticker' in raw.columns.names:
                raw.columns = raw.columns.droplevel('Ticker')
            else:
                raw.columns = raw.columns.droplevel(1)
        bench_close       = raw['Close'].reindex(df.index, fill_value=np.nan)
        # Fix #1: Eksik benchmark gunlerini forward fill ile doldur
        bench_ret_20 = bench_close.pct_change(20).ffill()
        bench_ret_5  = bench_close.pct_change(5).ffill()
        df['RS_Index']       = df['Close'].pct_change(20) - bench_ret_20
        df['RS_Index_Short'] = df['Close'].pct_change(5)  - bench_ret_5
    except Exception:
        df['RS_Index']       = 0.0
        df['RS_Index_Short'] = 0.0
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MAKİNE ÖĞRENMESİ — WALK-FORWARD + PIYASA BAZLI MODEL
# ──────────────────────────────────────────────────────────────────────────────

# Piyasaya göre varsayılan parametreler
_MARKET_DEFAULTS = {
    'us':     {'horizon': 5, 'threshold': 0.02, 'n_est': 200, 'depth': 8},
    'bist':   {'horizon': 5, 'threshold': 0.03, 'n_est': 200, 'depth': 8},
    'crypto': {'horizon': 3, 'threshold': 0.03, 'n_est': 200, 'depth': 10},
}

# Her piyasaya özel feature ağırlık seti
_FEATURE_SETS = {
    'base': [
        'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'OBV',
        'Return_1D', 'Return_3D', 'Return_5D', 'Return_10D', 'Return_20D',
        'Volatility_20', 'Trend_Strength', 'Price_vs_SMA20', 'Price_vs_SMA50',
        'BB_Position', 'Volume_Ratio', 'ATR_Pct',
        'ADX', 'Stoch_K', 'Stoch_D',
        'Volume_Spike',
        'ASO_Bulls', 'ASO_Bears', 'ASO_Diff', 'ASO_Cross',  # ASO sentiment
        # Fix #12: Yeni indikatörler
        'Williams_R', 'CCI', 'VWAP_Dev',
        'Dist_52W_High', 'Dist_52W_Low',
        'Vol_RSI',
    ],
    'gap': ['Gap_Pct', 'OpenClose_Pct', 'HighLow_Range_Pct'],   # BIST & kripto için önemli
    'rs':  ['RS_Index', 'RS_Index_Short'],                        # Benchmark RS (varsa)
}


@st.cache_data(ttl=900)
def train_ml_model(df, horizon: int = 5, threshold: float = 0.02, market_type: str = 'us'):
    """
    RF + ExtraTrees walk-forward ensemble tahmini.
    """
    ml_df = df.copy()

    # ── Lag Features ekle (momentum hafızası için) ─────────────────────────
    _lag_source_cols = ['RSI', 'MACD', 'Volume_Ratio', 'ADX', 'Stoch_K', 'ATR_Pct']
    for col in _lag_source_cols:
        if col in ml_df.columns:
            for lag in [1, 3, 5]:
                ml_df[f'{col}_lag{lag}'] = ml_df[col].shift(lag)

    # ── Feature seti ─────────────────────────────────────────────────────────
    features = list(_FEATURE_SETS['base'])
    features += _FEATURE_SETS['gap']          # Tüm piyasalar için faydalı
    # RS feature varsa ekle (add_relative_strength çağrılmışsa bulunur)
    rs_available = [f for f in _FEATURE_SETS['rs'] if f in ml_df.columns]
    features += rs_available
    # Lag features ekle
    lag_features = [f'{col}_lag{lag}' for col in _lag_source_cols
                    for lag in [1, 3, 5] if f'{col}_lag{lag}' in ml_df.columns]
    features += lag_features

    # Yalnızca df'te var olan feature'ları kullan ve çok fazla NaN barındıranları ele (kısa seriler için, örn. Dist_52W_High)
    features = [f for f in features if f in ml_df.columns and ml_df[f].isna().sum() < 0.5 * len(ml_df)]

    ml_df = ml_df.dropna(subset=features).copy()

    if len(ml_df) < 60:
        raise ValueError("ML modeli için yeterli temiz veri yok. En az 60 satır gerekli.")

    # ── Hedef Değişken: horizon gün içinde threshold üstü getiri ─────────────
    # "Yarın artar mı?" → "N gün içinde %X+ hareket eder mi?"
    future_return = ml_df['Close'].shift(-horizon) / ml_df['Close'] - 1
    ml_df['Target'] = np.where(future_return > threshold, 1, 0)
    ml_df = ml_df.iloc[:-horizon]          # son horizon satır hedef bilinemiyor

    if len(ml_df) < 50:
        raise ValueError("ML eğitimi için yeterli geçmiş veri yok.")

    X = ml_df[features]
    y = ml_df['Target']

    if y.nunique() < 2:
        raise ValueError("ML modeli eğitilemiyor; tüm eğitim verisi tek yönlü.")

    # Son gün için tahmin verisi
    prediction_data = ml_df[features].dropna().iloc[-1].values.reshape(1, -1)

    # ── Model parametreleri (piyasaya göre) ──────────────────────────────────
    cfg = _MARKET_DEFAULTS.get(market_type, _MARKET_DEFAULTS['us'])
    n_est = cfg['n_est']
    depth = cfg['depth']

    def _make_rf():
        return RandomForestClassifier(
            n_estimators=n_est, max_depth=depth, min_samples_leaf=4,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1)

    def _make_et():
        return ExtraTreesClassifier(
            n_estimators=n_est, max_depth=depth, min_samples_leaf=4,
            class_weight='balanced', random_state=42, n_jobs=-1)

    def _make_lgbm():
        if not _LGBM_AVAILABLE:
            return None
        return lgb.LGBMClassifier(
            n_estimators=n_est, max_depth=depth, min_child_samples=4,
            class_weight='balanced', random_state=42, n_jobs=-1,
            verbosity=-1, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)

    # ── Walk-Forward Validation (rolling window) ──────────────────────────────
    if market_type == 'crypto':
        TRAIN_WINDOW = min(126, int(len(X) * 0.6))
    else:
        TRAIN_WINDOW = min(252, int(len(X) * 0.6))
    TEST_WINDOW  = max(10, min(21, int(len(X) * 0.1)))

    fold_scores = []
    def _safe_prob_up(model, X_input) -> np.ndarray:
        classes = list(model.classes_)
        probs   = model.predict_proba(X_input)
        if 1 not in classes:
            return np.zeros(probs.shape[0])
        if 0 not in classes:
            return np.ones(probs.shape[0])
        return probs[:, classes.index(1)]

    # ── Walk-Forward Validation ───────────────────────────────────────────────
    idx = TRAIN_WINDOW
    lgbm_model = _make_lgbm()

    while idx + TEST_WINDOW <= len(X):
        X_tr = X.iloc[idx - TRAIN_WINDOW : idx]
        y_tr = y.iloc[idx - TRAIN_WINDOW : idx]
        X_te = X.iloc[idx : idx + TEST_WINDOW]
        y_te = y.iloc[idx : idx + TEST_WINDOW]

        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            idx += TEST_WINDOW
            continue

        rf = _make_rf()
        et = _make_et()
        rf.fit(X_tr, y_tr)
        et.fit(X_tr, y_tr)

        if lgbm_model is not None:
            lgbm_fold = _make_lgbm()
            try:
                lgbm_fold.fit(X_tr, y_tr)
                lgbm_prob = _safe_prob_up(lgbm_fold, X_te)
                avg_prob  = (_safe_prob_up(rf, X_te) + _safe_prob_up(et, X_te) + lgbm_prob) / 3
            except Exception:
                avg_prob  = (_safe_prob_up(rf, X_te) + _safe_prob_up(et, X_te)) / 2
        else:
            avg_prob  = (_safe_prob_up(rf, X_te) + _safe_prob_up(et, X_te)) / 2

        ensemble_pred = (avg_prob >= 0.5).astype(int)
        fold_scores.append(accuracy_score(y_te, ensemble_pred))
        idx += TEST_WINDOW

    if not fold_scores:
        cv_splits = min(5, max(3, len(X) // 40))
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        for tr_idx, te_idx in tscv.split(X):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            if y_tr.nunique() < 2 or y_te.nunique() < 2:
                continue
            rf = _make_rf()
            et = _make_et()
            rf.fit(X_tr, y_tr)
            et.fit(X_tr, y_tr)

            if _LGBM_AVAILABLE:
                try:
                    lgbm_ts = _make_lgbm()
                    lgbm_ts.fit(X_tr, y_tr)
                    lgbm_prob = _safe_prob_up(lgbm_ts, X_te)
                    avg_prob  = (_safe_prob_up(rf, X_te) + _safe_prob_up(et, X_te) + lgbm_prob) / 3
                except Exception:
                    avg_prob = (_safe_prob_up(rf, X_te) + _safe_prob_up(et, X_te)) / 2
            else:
                avg_prob = (_safe_prob_up(rf, X_te) + _safe_prob_up(et, X_te)) / 2
            fold_scores.append(accuracy_score(y_te, (avg_prob >= 0.5).astype(int)))

    if not fold_scores:
        raise ValueError("ML çapraz doğrulama için yeterli sınıf çeşitliliği bulunamadı.")

    rf_final = _make_rf()
    et_final = _make_et()
    rf_final.fit(X, y)
    et_final.fit(X, y)

    rf_up = _safe_prob_up(rf_final, prediction_data)[0]
    et_up = _safe_prob_up(et_final, prediction_data)[0]

    lgbm_up = 0.0
    n_models = 2
    if _LGBM_AVAILABLE:
        try:
            lgbm_final = _make_lgbm()
            lgbm_final.fit(X, y)
            lgbm_up = _safe_prob_up(lgbm_final, prediction_data)[0]
            n_models = 3
        except Exception:
            pass

    up_probability = ((rf_up + et_up + lgbm_up) / n_models) * 100
    accuracy       = float(np.mean(fold_scores) * 100)
    return up_probability, accuracy


def get_risk_management(current_price, atr_value):
    """ATR tabanlı dinamik stop-loss ve take-profit hesaplar."""
    stop_loss   = current_price - (1.5 * atr_value)
    take_profit = current_price + (2.0 * atr_value)
    return stop_loss, take_profit


def fast_ml_filter(df, min_prob=52.0, market_type='us'):
    try:
        df = calculate_technical_indicators(df)
        ml_features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'OBV',
                       'ADX', 'Stoch_K', 'Stoch_D', 'Gap_Pct', 'OpenClose_Pct', 'Volume_Spike']
        available   = [f for f in ml_features if f in df.columns]
        df_clean    = df.dropna(subset=available)
        if df_clean.empty:
            return False, 0, 0, "Temiz teknik veri bulunamadi."
        last_rsi = df_clean['RSI'].iloc[-1]
        cfg       = _MARKET_DEFAULTS.get(market_type, _MARKET_DEFAULTS['us'])
        up_prob, _ = train_ml_model(df, horizon=cfg['horizon'], threshold=cfg['threshold'], market_type=market_type)
        return (True, up_prob, last_rsi, None) if up_prob >= min_prob else (False, up_prob, last_rsi, None)
    except Exception as exc:
        return False, 0, 0, str(exc)


def screen_bist_stocks(index_name='XU100', max_price=500.0, min_prob=52.0, pe_max=None, market_cap_min=None):
    """BIST endeks bileşenlerini canlı çeker ve gelişmiş filtrelerle tarar."""
    champions, scan_errors = [], []
    try:
        import borsapy as bp
        idx = bp.Index(index_name)
        bist_universe = idx.component_symbols
        if not bist_universe:
            bist_universe = ['THYAO', 'TUPRS', 'ASELS', 'KCHOL', 'GARAN', 'EREGL', 'ISCTR']
    except Exception as e:
        scan_errors.append(f"Endeks listesi alınamadı ({index_name}): {e}")
        bist_universe = ['THYAO', 'TUPRS', 'ASELS', 'KCHOL', 'GARAN', 'EREGL', 'ISCTR']

    for symbol in bist_universe:
        t = f"{symbol}.IS"
        try:
            borsapy_data = {"Son Temettü": "Bilinmiyor", "Yabancı Oranı": "%-"}
            try:
                bt = bp.Ticker(symbol)
                f_info = bt.fast_info
                last_price = f_info.get("last_price", 0)
                if last_price > max_price or last_price == 0: continue
                pe = f_info.get("pe_ratio")
                if pe_max and (pe is None or pe > pe_max or pe <= 0): continue
                mcap = f_info.get("market_cap", 0)
                if market_cap_min and mcap < (market_cap_min * 1e6): continue
                borsapy_data["Yabancı Oranı"] = f"%{f_info.get('foreign_ratio', '-'):.2f}" if f_info.get('foreign_ratio') else "%-"
                borsapy_data["Halka Açıklık"] = f"%{f_info.get('free_float', '-'):.2f}" if f_info.get('free_float') else "%-"
                borsapy_data["F/K"] = f"{pe:.2f}" if pe else "N/A"
                try:
                    divs = bt.dividends
                    if divs is not None and not divs.empty:
                        borsapy_data["Son Temettü"] = f"{divs['Amount'].iloc[0]:.2f} TL"
                    else: borsapy_data["Son Temettü"] = "Yok"
                except: pass
            except: pass
            hist, _ = fetch_data(t, period="2y")
            if hist.empty: continue
            is_champion, prob, rsi, err = fast_ml_filter(hist, min_prob=min_prob, market_type='bist')
            if is_champion:
                champions.append({"Ticker": t, "Prob": prob, "RSI": rsi, "Price": hist['Close'].iloc[-1], "Borsapy": borsapy_data})
            if err: scan_errors.append(f"{t}: {err}")
            if len(champions) >= 15: break
        except Exception as exc:
            scan_errors.append(f"{symbol}: {exc}")
    return champions, scan_errors


def screen_us_stocks(price_filter='Under $5', sector_filter='Any', min_prob=52.0, limit=50):
    try:
        from finvizfinance.screener.overview import Overview
        foverview = Overview()
        f_dict = {'Price': price_filter}
        if sector_filter != 'Any': f_dict['Sector'] = sector_filter
        foverview.set_filter(filters_dict=f_dict)
        df_fv   = foverview.screener_view(limit=limit, verbose=0)
        tickers = df_fv['Ticker'].head(limit).tolist()
        champions, scan_errors = [], []
        for t in tickers:
            try:
                hist, _ = fetch_data(t, period="2y")
                is_champion, prob, rsi, err = fast_ml_filter(hist, min_prob=min_prob, market_type='us')
                if is_champion:
                    champions.append({"Ticker": t, "Prob": prob, "RSI": rsi, "Price": hist['Close'].iloc[-1]})
                if err: scan_errors.append(f"{t}: {err}")
                if len(champions) >= 10: break
            except Exception as exc: scan_errors.append(f"{t}: {exc}")
        return champions, scan_errors
    except Exception as exc:
        return [], [f"Finviz Error: {exc}"]


def calculate_portfolio_optimization(tickers, period="1y"):
    data, valid_tickers = pd.DataFrame(), []
    for t in tickers:
        t_fmt = t.upper().strip()
        try:
            df, final_t = fetch_data(t_fmt, period=period)
            if not df.empty:
                data[final_t] = df['Close']
                valid_tickers.append(final_t)
        except Exception: continue
    if len(valid_tickers) < 2:
        return {"error": "Portföy analizi için en az 2 geçerli hisse/kripto sembolü gereklidir.", "valid_tickers": valid_tickers}
    returns      = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix   = returns.cov() * 252
    n_ports      = 5000
    results      = np.zeros((3, n_ports))
    weights_rec  = []
    for i in range(n_ports):
        w = np.random.random(len(valid_tickers))
        w /= np.sum(w)
        weights_rec.append(w)
        results[0, i] = np.sum(mean_returns * w)
        results[1, i] = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        results[2, i] = (results[0, i] - 0.02) / results[1, i]
    ms_idx = np.argmax(results[2])
    mv_idx = np.argmin(results[1])
    return {
        "valid_tickers": valid_tickers, "results": results,
        "max_sharpe": {"return": results[0, ms_idx], "volatility": results[1, ms_idx], "sharpe": results[2, ms_idx], "weights": dict(zip(valid_tickers, weights_rec[ms_idx]))},
        "min_vol": {"return": results[0, mv_idx], "volatility": results[1, mv_idx], "sharpe": results[2, mv_idx], "weights": dict(zip(valid_tickers, weights_rec[mv_idx]))},
    }


def get_social_sentiment_mock(ticker, current_price, rsi):
    import random
    base_hype = (rsi / 100) * 80 + random.uniform(-10, 20)
    base_hype = max(10, min(95, base_hype))
    is_troll  = random.random() < 0.25
    trust     = random.uniform(20, 50) if is_troll else random.uniform(65, 95)
    trend     = "Aşırı Fomo (Greed)" if base_hype > 75 else "Yükselişte (Bullish)" if base_hype > 55 else "Korku (Fear)" if base_hype < 25 else "Düşüşte (Bearish)" if base_hype < 45 else "Nötr"
    pos_ph  = ["🚀 Aya uçuyoruz!", "Büyük kırılım geliyor hacalan", "İnanılmaz bir hacim var 🔥", "Dipten toplayın!"]
    neg_ph  = ["Rug pull geliyor kaçın", "Fiyatlandı, ben short giriyorum", "Çöküş başladı 📉", "Ölü kedi sıçraması (Dead cat bounce)"]
    troll_ph = ["Babam içeride çalışıyor kesin bilgi hemen alın!!", "Yarın garanti 1000x yapacak, evinizi arabayı satın", "CEO hapse girdi (yalan haber) hemen satın!!!"]
    posts = []
    for i in range(5):
        if is_troll and i < 2: body, label, color = random.choice(troll_ph), "SCAM/TROLL", "gray"
        elif base_hype > 50: body, label, color = random.choice(pos_ph), "BULLISH", "green"
        else: body, label, color = random.choice(neg_ph), "BEARISH", "red"
        posts.append({"user": f"kullanici_{random.randint(1000,9999)}", "body": f"{ticker} {body}", "label": label, "color": color})
    return {
        "hype_score": round(base_hype, 1),
        "trend": trend,
        "trust_score": round(trust, 1),
        "is_simulated": True,
        "posts": posts,
        "warning": (
            "🚨 DİKKAT: Verilerin çoğunda SPAM/TROLL bot manipülasyonu tespit edildi!" if is_troll
            else "✅ Sosyal medya paylaşımları büyük oranda doğal ve organik görünüyor."
        )
    }


def calculate_fibonacci_and_patterns(df):
    from scipy.signal import argrelextrema
    recent_df = df.iloc[-100:] if len(df) > 100 else df
    max_price = recent_df['High'].max()
    min_price = recent_df['Low'].min()
    diff      = max_price - min_price
    fib_levels = {"100% (Tepe)": max_price, "78.6%": max_price - 0.214 * diff, "61.8% (Altın Oran)": max_price - 0.382 * diff, "50.0%": max_price - 0.5 * diff, "38.2%": max_price - 0.618 * diff, "23.6%": max_price - 0.764 * diff, "0% (Dip)": min_price}
    prices, patterns = recent_df['Close'].values, []
    peaks, troughs = argrelextrema(prices, np.greater, order=5)[0], argrelextrema(prices, np.less, order=5)[0]
    if len(peaks) >= 2:
        if abs(prices[peaks[-2]] - prices[peaks[-1]]) / prices[peaks[-2]] < 0.02: patterns.append("⚠️ **Olası İkili Tepe Formasyonu**")
    if len(troughs) >= 2:
        if abs(prices[troughs[-2]] - prices[troughs[-1]]) / prices[troughs[-2]] < 0.02: patterns.append("📈 **Olası İkili Dip Formasyonu**")
    if not patterns: patterns.append("🔍 Keskin bir formasyon tespit edilmedi.")
    return fib_levels, patterns


def fetch_kap_news_rss(ticker):
    url = f"https://news.google.com/rss/search?q={ticker.replace('.IS', '')}+site:kap.org.tr&hl=tr&gl=TR&ceid=TR:tr"
    news_list = []
    try:
        r = requests.get(url, timeout=5)
        root = ET.fromstring(r.content)
        for item in root.findall('.//item')[:10]: news_list.append({"title": item.find('title').text, "link": item.find('link').text, "source": "KAP"})
    except: pass
    return news_list

def fetch_tradingview_news(ticker):
    url = f"https://news.google.com/rss/search?q={ticker.replace('.IS', '')}+site:tradingview.com&hl=tr&gl=TR&ceid=TR:tr"
    news_list = []
    try:
        r = requests.get(url, timeout=5)
        root = ET.fromstring(r.content)
        for item in root.findall('.//item')[:10]: news_list.append({"title": item.find('title').text, "link": item.find('link').text, "source": "TradingView"})
    except: pass
    return news_list

def fetch_midas_news(ticker):
    url = f"https://news.google.com/rss/search?q={ticker.replace('.IS', '')}+site:getmidas.com&hl=tr&gl=TR&ceid=TR:tr"
    news_list = []
    try:
        r = requests.get(url, timeout=5)
        root = ET.fromstring(r.content)
        for item in root.findall('.//item')[:10]: news_list.append({"title": item.find('title').text, "link": item.find('link').text, "source": "Midas"})
    except: pass
    return news_list

def get_finbert_sentiment(ticker_symbol):
    """FinBERT duygu analizi — paralel çeviri + toplu (batch) inference."""
    all_news = []
    if ".IS" in ticker_symbol: all_news.extend(fetch_kap_news_rss(ticker_symbol))
    all_news.extend(fetch_tradingview_news(ticker_symbol))
    all_news.extend(fetch_midas_news(ticker_symbol))
    try:
        y_news = yf.Ticker(ticker_symbol).news
        for item in y_news[:5]: all_news.append({"title": item['title'], "link": item['link'], "source": "Yahoo Finance"})
    except: pass
    if not all_news: return 0, []
    try:
        pipe = get_sentiment_pipeline()
        articles_to_process = all_news[:15]  # biraz daha geniş havuz al

        # ── Paralel çeviri (ThreadPoolExecutor) ──────────────────────────
        def _translate_one(article):
            """Tek bir başlığı çevirir; hata olursa None döner."""
            try:
                tr = GoogleTranslator(source="auto", target="en")
                translated = tr.translate(article['title'])
                if translated and len(translated.strip()) > 2:
                    return {**article, '_translated': translated}
            except Exception:
                pass
            return None

        translated_articles = []
        with ThreadPoolExecutor(max_workers=min(8, len(articles_to_process))) as executor:
            futures = {executor.submit(_translate_one, a): a for a in articles_to_process}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    translated_articles.append(result)

        if not translated_articles:
            return 0, []

        # ── Toplu (batch) FinBERT inference ──────────────────────────────
        translated_texts = [a['_translated'] for a in translated_articles]
        # HuggingFace pipeline batch çağrısı — tek seferde tüm listeyi işle
        batch_results = pipe(translated_texts, batch_size=len(translated_texts))

        total_score, analyzed_items = 0, []
        for article, res in zip(translated_articles, batch_results):
            pol = (1 if res['label'] == 'positive' else -1 if res['label'] == 'negative' else 0) * res['score']
            total_score += pol
            analyzed_items.append({
                "title": article['title'],
                "label": res['label'],
                "polarity": pol,
                "link": article['link'],
                "source": article['source']
            })

        return (total_score / len(analyzed_items)) if analyzed_items else 0, analyzed_items
    except: return 0, []

def screen_tefas_funds(fund_type="YAT", min_return_1y=50.0):
    """
    Screens TEFAS funds with a robust error handling for WAF blocks.
    Attempts to use borsapy first, falls back to a custom session fetcher.
    """
    import requests
    import pandas as pd
    
    def manual_tefas_fetch(f_type, min_ret):
        url = "https://www.tefas.gov.tr/api/DB/BindComparisonFundReturns"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://www.tefas.gov.tr/FonKarsilastirma.aspx",
            "Origin": "https://www.tefas.gov.tr",
            "Accept": "application/json, text/javascript, */*; q=0.01",
        }
        data = {
            "calismatipi": "2",
            "fontip": f_type,
            "sfontur": "Tümü",
            "kurucukod": "",
            "fongrup": "",
            "bastarih": "Başlangıç",
            "bittarih": "Bitiş",
            "fonturkod": "",
            "fonunvantip": "",
            "strperiod": "1,1,1,1,1,1,1",
            "islemdurum": "1",
        }
        try:
            # First establishing session cookies
            session = requests.Session()
            session.get("https://www.tefas.gov.tr/FonKarsilastirma.aspx", headers=headers, timeout=5, verify=False)
            r = session.post(url, data=data, headers=headers, timeout=10, verify=False)
            if r.status_code == 200:
                json_data = r.json()
                funds = json_data.get("data", [])
                if not funds: return None
                df = pd.DataFrame(funds)
                # Filtering
                if "GETIRI1Y" in df.columns:
                    df["GETIRI1Y"] = pd.to_numeric(df["GETIRI1Y"], errors='coerce').fillna(0)
                    df = df[df["GETIRI1Y"] >= min_ret]
                return df
            return None
        except: return None

    try:
        import borsapy as bp
        df = bp.screen_funds(fund_type=fund_type, min_return_1y=min_return_1y)
        
        if df is None or df.empty:
            # Fallback attempt
            df = manual_tefas_fetch(fund_type, min_return_1y)
            
        if df is None or df.empty:
            return pd.DataFrame(), "⚠️ TEFAS şu an yoğun güvenlik koruması (WAF) altında. Lütfen bir süre sonra tekrar deneyin veya tefas.gov.tr adresinden manuel kontrol edin."

        # Unified column mapping
        col_map = {
            "fund_code": "Fon Kodu", "FONKODU": "Fon Kodu",
            "name": "Fon Adı", "FONUNVAN": "Fon Adı",
            "return_1y": "1Y Getiri (%)", "GETIRI1Y": "1Y Getiri (%)"
        }
        df = df.rename(columns=lambda x: col_map.get(x, x))
        
        needed = ["Fon Kodu", "Fon Adı", "1Y Getiri (%)"]
        existing = [c for c in needed if c in df.columns]
        
        return df[existing].sort_values("1Y Getiri (%)", ascending=False), None
        
    except Exception as e:
        if "404" in str(e):
            return pd.DataFrame(), "🚫 TEFAS Erişimi Engellendi: Site şu an bot trafiğini reddediyor. (Genel bir TEFAS sorunudur, kütüphane güncellemesi bekleniyor)."
        return pd.DataFrame(), f"Sistem Hatası: {str(e)}"
