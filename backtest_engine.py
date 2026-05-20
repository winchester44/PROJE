"""
backtest_engine.py
==================
Gerçekçi Backtest Motoru — ATR Dinamik Slippage Destekli

Quant-AI sinyal motorunun (quant_ai_engine.py) ürettiği stratejileri
geçmiş OHLCV verisi üzerinde simüle eder.

Look-ahead bias önlemi:
  - Sinyaller günlük kapanış verisiyle hesaplanır (t günü).
  - İşlem (giriş/çıkış) t+1 gününün AÇILIŞ fiyatından yapılır.
  - Donchian/Pivot hesaplamalarında shift(1) ile bugünün verisi kullanılmaz.

Gerçekçilik önlemleri:
  - Her işlemde (alım VE satım ayrı ayrı) komisyon/friction cost uygulanır.
  - ATR tabanlı dinamik slippage: Slippage = ATR_Pct × k × Fiyat
  - Günlük sermaye eğrisi (equity curve) tutulur.
"""

import numpy as np
import pandas as pd


class BacktestEngine:
    """
    Vektörel Backtest Motoru — ATR Dinamik Slippage Destekli.

    Parametreler
    ------------
    df : pd.DataFrame
        'calculate_technical_indicators()' ile zenginleştirilmiş OHLCV verisi.
        Zorunlu sütunlar: Open, High, Low, Close, Volume, SMA_20, SMA_50, ATR
    initial_capital : float
        Başlangıç sermayesi (USD veya TL). Varsayılan: 10.000
    commission_pct : float
        İşlem başı komisyon oranı (0.001 = %0.1). Her alım VE satım için ayrı uygulanır.
    slippage_factor : float
        ATR_Pct çarpanı. Slippage = ATR_Pct * slippage_factor * Fiyat.
        Varsayılan 0.10 (ATR yüzdesinin %10'u kadar kayma). 0 ile devre dışı.
    """

    def __init__(self, df: pd.DataFrame, initial_capital: float = 10_000.0,
                 commission_pct: float = 0.001, slippage_factor: float = 0.10):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.commission = commission_pct
        self.slippage_factor = slippage_factor
        self.trailing_atr_mult = 2.5   # Daha genis stop = erken stop-out azaltir
        self.max_holding_days = 60      # Maksimum tutma suresi (gun)

    # ──────────────────────────────────────────────────────────────────────────
    # ATR DİNAMİK SLIPPAGE FONKSİYONU
    # ──────────────────────────────────────────────────────────────────────────
    def _apply_slippage(self, price: float, atr_pct: float, side: str) -> float:
        """
        Fiyata yöne bağlı dinamik kayma (slippage) uygular.

        Matematik:
            slippage_amount = ATR_Pct × slippage_factor × Fiyat

        Yön kuralları (look-ahead bias yok — ATR t gününe ait):
            BUY  → fiyat yukarı kayar (daha pahalıya alırsın)
            SELL / STOP → fiyat aşağı kayar (daha ucuza satarsın)

        Parameters
        ----------
        price : float     — Ham emir fiyatı
        atr_pct : float   — ATR / Close oranı (0-1 arası, ör. 0.025 = %2.5)
        side : str        — 'BUY' veya 'SELL'

        Returns
        -------
        float — Slippage uygulanmış gerçekçi dolum fiyatı
        """
        if self.slippage_factor <= 0 or not np.isfinite(atr_pct) or atr_pct <= 0:
            return price
        slip_amount = atr_pct * self.slippage_factor * price
        if side == 'BUY':
            return price + slip_amount  # Alışta yukarı kayma
        else:
            return price - slip_amount  # Satışta aşağı kayma


    # ──────────────────────────────────────────────────────────────────────────
    # İÇ FONKSİYON: Vektörel Sinyal Hesaplama (Look-ahead bias YOK)
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_signals_vectorized(self) -> pd.DataFrame:
        """
        Her gün için AL / SAT / BEKLE sinyalini vektörel (pandas) yöntemle üretir.
        quant_ai_engine'in aynı 5 strateji mantığını esas alır.
        """
        df = self.df.copy()
        close = df['Close']
        high  = df['High']
        low   = df['Low']

        # ── 1. Price Momentum (§3.1) ──────────────────────────────────────────
        # 12 aylık kümülatif getiri (252 işlem günü), 1 aylık atlama (21 gün) ile
        skip       = 21          # atlama süresi
        formation  = 252         # formasyon süresi

        # R_cum: t anındaki formasyon penceresi sonundaki → başındaki fiyat oranı
        # Bias önlemi: shift(skip) → t'yi değil t-21'i kullan
        past_close_end   = close.shift(skip)
        past_close_start = close.shift(skip + formation)
        df['R_cum_mom'] = past_close_end / past_close_start - 1.0

        # Risk-adjusted getiri: rolling aylık getiri (21 günlük) ortalaması / std
        monthly_ret      = close.pct_change(21)
        df['mom_sigma']  = monthly_ret.rolling(12).std()
        df['mom_mean']   = monthly_ret.rolling(12).mean()
        df['R_risadj']   = df['mom_mean'] / df['mom_sigma'].replace(0, np.nan)

        # Oy: +1 Yükselen, -1 Düşen, 0 Nötr
        mom_bull = (df['R_cum_mom'] > 0.05) & (df['R_risadj'] > 0.5)
        mom_bear = (df['R_cum_mom'] < -0.05) & (df['R_risadj'] < -0.5)
        df['vote_mom'] = np.where(mom_bull, 1, np.where(mom_bear, -1, 0))

        # ── 2. Dual & Triple Moving Average (§3.12 / §3.13) ──────────────────
        sma5  = close.rolling(5).mean()
        sma20 = df['SMA_20'] if 'SMA_20' in df.columns else close.rolling(20).mean()
        sma50 = df['SMA_50'] if 'SMA_50' in df.columns else close.rolling(50).mean()

        # İkili MA + Üçlü MA + Fiyat pozisyonu (3 alt koşuldan 2'si yeterliyse)
        two_bull  = (sma20 > sma50).astype(int)
        three_bull = ((sma5 > sma20) & (sma20 > sma50)).astype(int)
        price_bull = ((close > sma20) & (close > sma50)).astype(int)

        two_bear  = (sma20 < sma50).astype(int)
        three_bear = ((sma5 < sma20) & (sma20 < sma50)).astype(int)
        price_bear = ((close < sma20) & (close < sma50)).astype(int)

        bull_ma_count = two_bull + three_bull + price_bull
        bear_ma_count = two_bear + three_bear + price_bear

        df['vote_ma'] = np.where(bull_ma_count >= 2, 1,
                        np.where(bear_ma_count >= 2, -1, 0))

        # ── 3. Pivot Destek & Direnç (§3.14) ─────────────────────────────────
        # Bias önlemi: shift(1) → önceki günün H/L/C verisi
        PH = high.shift(1)
        PL = low.shift(1)
        PC = close.shift(1)

        C = (PH + PL + PC) / 3.0  # Pivot merkez  (Eq. 325)
        R = 2.0 * C - PL           # Direnç        (Eq. 326)
        S = 2.0 * C - PH           # Destek        (Eq. 327)

        band = (R - S) * 0.15      # Yakınlık bandı (%15)
        near_support    = close <= S + band
        near_resistance = close >= R - band

        # Desteğe yakın → potansiyel yukarı dönüş (+1)
        # Dirençe yakın  → potansiyel aşağı dönüş (-1)
        df['vote_pivot'] = np.where(near_support, 1,
                           np.where(near_resistance, -1, 0))
        df['pivot_S'] = S
        df['pivot_R'] = R

        # ── 4. Donchian Kanal (§3.15) ─────────────────────────────────────────
        T = 20  # kanal periyodu (işlem günü)
        # Bias önlemi: shift(1) → bugünün kapanışını kanaldan dışla
        B_up   = close.shift(1).rolling(T).max()  # Eq. (329)
        B_down = close.shift(1).rolling(T).min()   # Eq. (330)

        tol = (B_up - B_down) * 0.03  # %3 tolerans

        don_bull = close >= B_up - tol      # Kanal tavanını kırdı
        don_bear = close <= B_down + tol    # Kanal tabanını kırdı

        df['vote_don']    = np.where(don_bull, 1, np.where(don_bear, -1, 0))
        df['don_upper']   = B_up
        df['don_lower']   = B_down

        # ── 5. Sentiment: Backtest modunda nötr (§18.3 analogue) ─────────────
        # Gerçek zamanlı haber toplanamayacağı için geçmiş testlerde sabit 0
        df['vote_sent'] = 0

        # ── 6. ASO (Average Sentiment Oscillator) ────────────────────────────
        # Mum-içi ve dönemsel boğa/ayı baskısını ölçer.
        # Bulls, Bears'ı yukarı keserse → BUY oyu, aşağı keserse → SELL oyu
        if 'ASO_Bulls' in df.columns and 'ASO_Bears' in df.columns:
            aso_diff      = df['ASO_Bulls'] - df['ASO_Bears']
            aso_prev_diff = aso_diff.shift(1)
            # Sürekli cross yerine mevcut baskı yönünü de dikkate al
            aso_bull = (aso_diff > 0) & ((aso_prev_diff <= 0) | (aso_diff > 2))
            aso_bear = (aso_diff < 0) & ((aso_prev_diff >= 0) | (aso_diff < -2))
            df['vote_aso'] = np.where(aso_bull, 1, np.where(aso_bear, -1, 0))
        else:
            # ASO hesaplanmamışsa kendi başına hesapla
            aso_len     = 20
            intrarange  = high - low
            _K1         = intrarange.where(intrarange > 0, np.nan)  # Fix #3: NaN guard
            _grouplow   = low.rolling(aso_len).min()
            _grouphigh  = high.rolling(aso_len).max()
            _groupopen  = df['Open'].shift(aso_len - 1)
            _grouprange = _grouphigh - _grouplow
            _K2         = _grouprange.where(_grouprange > 0, np.nan)  # Fix #3: NaN guard

            _intra_bulls = (((close - low) + (high - df['Open'])) / 2 * 100) / _K1
            _group_bulls = (((close - _grouplow) + (_grouphigh - _groupopen)) / 2 * 100) / _K2
            _intra_bears = (((high - close) + (df['Open'] - low)) / 2 * 100) / _K1
            _group_bears = (((_grouphigh - close) + (_groupopen - _grouplow)) / 2 * 100) / _K2

            _aso_bulls = ((_intra_bulls + _group_bulls) / 2).rolling(aso_len).mean()
            _aso_bears = ((_intra_bears + _group_bears) / 2).rolling(aso_len).mean()

            _aso_diff      = _aso_bulls - _aso_bears
            _aso_prev_diff = _aso_diff.shift(1)
            aso_bull = (_aso_diff > 0) & ((_aso_prev_diff <= 0) | (_aso_diff > 2))
            aso_bear = (_aso_diff < 0) & ((_aso_prev_diff >= 0) | (_aso_diff < -2))
            df['vote_aso'] = np.where(aso_bull, 1, np.where(aso_bear, -1, 0))

        # ── 7. BOLLINGER BANDS + RSI (Freqtrade) ──────────────────────────────
        bb_window = 20
        df['BB_Mid'] = close.rolling(bb_window).mean()
        df['BB_Std'] = close.rolling(bb_window).std()
        df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])

        # Wilder RSI — stock_analyzer.py ile tutarlı (alpha=1/14 EWM)
        if 'RSI' not in df.columns:
            _delta   = close.diff()
            _up      = _delta.clip(lower=0)
            _down    = -1 * _delta.clip(upper=0)
            _ema_up  = _up.ewm(alpha=1/14, adjust=False).mean()
            _ema_dn  = _down.ewm(alpha=1/14, adjust=False).mean()
            df['RSI'] = 100 - (100 / (1 + _ema_up / _ema_dn.replace(0, np.nan)))

        # Freqtrade-like improvement: Volume confirmation
        vol_avg = df['Volume'].rolling(20).mean()
        vol_bull = df['Volume'] > vol_avg
        
        bb_rsi_bull = (close < df['BB_Lower']) & (df['RSI'] < 30) & vol_bull
        bb_rsi_bear = (close > df['BB_Upper']) & (df['RSI'] > 70)
        df['vote_bb_rsi'] = np.where(bb_rsi_bull, 1, np.where(bb_rsi_bear, -1, 0))

        # ── 8. MACD + EMA (Freqtrade) ─────────────────────────────────────────
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        macd_prev = df['MACD'].shift(1)
        macd_signal_prev = df['MACD_Signal'].shift(1)
        
        ema200 = df['SMA_200'] if 'SMA_200' in df.columns else (df['SMA_50'] if 'SMA_50' in df.columns else close)
        
        macd_cross_up = (macd_prev <= macd_signal_prev) & (df['MACD'] > df['MACD_Signal'])
        macd_cross_down = (macd_prev >= macd_signal_prev) & (df['MACD'] < df['MACD_Signal'])
        
        macd_ema_bull = macd_cross_up & (close > ema200) & (df['SMA_20'] > df['SMA_50']) & vol_bull
        macd_ema_bear = macd_cross_down & (close < ema200)
        df['vote_macd_ema'] = np.where(macd_ema_bull, 1, np.where(macd_ema_bear, -1, 0))

        # ── Konsensüs Sinyali (8 strateji) ────────────────────────────────────
        vote_cols  = ['vote_mom', 'vote_ma', 'vote_pivot', 'vote_don', 'vote_sent', 'vote_aso', 'vote_bb_rsi', 'vote_macd_ema']
        votes      = df[vote_cols]
        bull_total = (votes == 1).sum(axis=1)
        bear_total = (votes == -1).sum(axis=1)

        # >= 4 oy -> BUY / SELL (8 stratejiden en az 4 onay gerekli - kalite kontrolu)
        df['consensus_signal'] = np.where(
            bull_total >= 4, 'BUY',
            np.where(bear_total >= 4, 'SELL', 'HOLD')
        )

        # Güven skoru (artık 6 strateji üzerinden)
        atr     = df['ATR'] if 'ATR' in df.columns else (high - low).rolling(14).mean()
        atr_pct = (atr / close * 100).fillna(0)
        # ATR_Pct'yi sakla (slippage modeli ve güven hesabı için)
        df['atr_pct'] = (atr / close).fillna(0)  # 0-1 arası oran (slippage fonksiyonu için)
        max_agr = np.maximum(bull_total, bear_total)
        n_strategies = len(vote_cols)
        base_conf = (max_agr / n_strategies * 95).astype(int)
        penalty   = ((atr_pct > 3.0).astype(int) * 10)
        df['confidence'] = (base_conf - penalty).clip(5, 95)

        # Stop-Loss ve Take-Profit seviyeleri (Pivot + Donchian birlesimi)
        # Guvenlik: TP her zaman close'un ustunde, SL her zaman altinda olmali
        raw_sl = np.minimum(
            S.where(S.notna(), np.nan),
            B_down.where(B_down.notna(), np.nan)
        )
        raw_tp = np.maximum(
            R.where(R.notna(), np.nan),
            B_up.where(B_up.notna(), np.nan)
        )
        atr_series = df['ATR'] if 'ATR' in df.columns else (high - low).rolling(14).mean()
        # SL: close'un altinda olmali; yoksa ATR ile hesapla
        df['suggested_sl'] = np.where(
            raw_sl < close, raw_sl,
            close - 2.0 * atr_series
        )
        # TP: close'un ustunde olmali; yoksa ATR ile hesapla
        df['suggested_tp'] = np.where(
            raw_tp > close, raw_tp,
            close + 3.0 * atr_series
        )
        df['bull_total'] = bull_total
        df['bear_total'] = bear_total

        return df

    # ──────────────────────────────────────────────────────────────────────────
    # ANA FONKSİYON: Backtest'i Çalıştır
    # ──────────────────────────────────────────────────────────────────────────
    def run(self) -> dict:
        """
        Backtest'i çalıştırır ve sonuçları döndürür.

        Dönüş değeri (dict)
        -------------------
        equity_curve      : pd.Series   – Günlük sermaye değerleri
        total_return_pct  : float        – Toplam net getiri %
        hit_rate_pct      : float        – Kazanma oranı %
        max_drawdown_pct  : float        – Maksimum düşüş %
        sharpe_ratio      : float        – Yıllıklandırılmış Sharpe oranı
        total_trades      : int          – Gerçekleşen alım işlemi sayısı
        final_capital     : float        – Son sermaye değeri
        buy_and_hold_pct  : float        – Al-tut getirisi (karşılaştırma için)
        trade_log         : list[dict]   – Detaylı işlem kayıtları
        signals_df        : pd.DataFrame – Günlük sinyal tablosu
        """
        # Sinyalleri hesapla
        df = self._compute_signals_vectorized()

        # NaN içeren satırları at (başlangıç dönemindeki hesaplama boşlukları)
        df = df.dropna(subset=['consensus_signal', 'vote_mom', 'vote_ma']).copy()

        if len(df) < 10:
            raise ValueError("Backtest için yeterli veri yok. Daha uzun bir periyot seçin.")

        # ── Simülasyon Döngüsü ────────────────────────────────────────────────
        capital      = float(self.initial_capital)
        position     = 0.0        # Tutulan hisse adedi
        in_position  = False
        entry_price  = 0.0        # Giriş maliyeti (komisyon dahil)

        trade_log    = []

        # FIX #11: Equity curve başlangıcı = initial_capital (ilk alıştan önce).
        # Önceki kodda loop içindeki ilk append günün SONU fiyatıyla yapılıyordu,
        # bu 1 bar kaymaya yol açıyordu. Şimdi ilk eleman açıkça initial_capital.
        equity_list  = [float(self.initial_capital)]

        signals  = df['consensus_signal'].values
        opens    = df['Open'].values.astype(float)
        highs    = df['High'].values.astype(float)
        lows     = df['Low'].values.astype(float)
        closes   = df['Close'].values.astype(float)
        sl_vals  = df['suggested_sl'].values.astype(float)
        tp_vals  = df['suggested_tp'].values.astype(float)
        atr_vals = df['ATR'].values.astype(float) if 'ATR' in df.columns else np.zeros(len(df))
        atr_pct_vals = df['atr_pct'].values.astype(float)  # Slippage modeli için
        bear_totals = df['bear_total'].values.astype(int) if 'bear_total' in df.columns else np.zeros(len(df), dtype=int)
        dates    = df.index
        active_sl = np.nan
        active_tp = np.nan
        bars_held = 0
        total_slippage_cost = 0.0  # Toplam kayma maliyeti takibi

        for i in range(len(df) - 1):
            sig       = signals[i]
            next_open = opens[i + 1]
            curr_sl   = sl_vals[i]
            curr_tp   = tp_vals[i]
            next_high = highs[i + 1]
            next_low  = lows[i + 1]
            next_atr  = atr_vals[i + 1] if i + 1 < len(atr_vals) else 0.0
            curr_atr_pct = atr_pct_vals[i]  # t günü ATR_Pct (look-ahead yok)

            if in_position:
                bars_held += 1

                if np.isfinite(next_atr) and next_atr > 0:
                    trailing_sl = closes[i] - (next_atr * self.trailing_atr_mult)
                    active_sl = trailing_sl if not np.isfinite(active_sl) else max(active_sl, trailing_sl)

                exit_reason = None
                exit_price = None

                if np.isfinite(active_sl) and next_low <= active_sl:
                    exit_reason = 'STOP-LOSS'
                    exit_price = self._apply_slippage(active_sl, curr_atr_pct, 'SELL')
                elif np.isfinite(active_tp) and next_high >= active_tp:
                    exit_reason = 'TAKE-PROFIT'
                    exit_price = self._apply_slippage(active_tp, curr_atr_pct, 'SELL')
                elif sig == 'SELL':
                    exit_reason = 'SATIŞ'
                    exit_price = self._apply_slippage(next_open, curr_atr_pct, 'SELL')
                elif bars_held >= self.max_holding_days:
                    exit_reason = 'TIME-EXIT'
                    exit_price = self._apply_slippage(next_open, curr_atr_pct, 'SELL')

                if exit_reason is not None:
                    gross_recv = position * exit_price
                    capital = gross_recv * (1.0 - self.commission)
                    # Tek taraflı komisyon: girişte zaten düşeldü, çıkışta bir kez daha
                    # trade_ret = (exit net recv / entry net cost) - 1
                    entry_net_cost = entry_price * (1.0 + self.commission)
                    exit_net_recv  = exit_price  * (1.0 - self.commission)
                    trade_ret  = (exit_net_recv - entry_net_cost) / entry_net_cost * 100
                    # Slippage maliyet takibi
                    raw_exit = active_sl if exit_reason == 'STOP-LOSS' else (active_tp if exit_reason == 'TAKE-PROFIT' else next_open)
                    slip_amt = abs(exit_price - raw_exit) if np.isfinite(raw_exit) else 0.0
                    total_slippage_cost += slip_amt * position
                    position = 0.0
                    in_position = False
                    bars_held = 0

                    trade_log.append({
                        'işlem_no': len([t for t in trade_log if t['tip'] in ('SATIŞ', 'STOP-LOSS', 'TAKE-PROFIT', 'TIME-EXIT')]) + 1,
                        'tip': exit_reason,
                        'tarih': str(dates[i + 1].date()),
                        'fiyat': round(exit_price, 4),
                        'getiri_pct': round(trade_ret, 2),
                        'komisyon_pct': self.commission * 100,
                        'slippage': round(slip_amt, 4),
                    })
                    active_sl = np.nan
                    active_tp = np.nan

            if sig == 'BUY' and not in_position and capital > 0:
                # ATR Dinamik Slippage: Alış fiyatı yukarı kayar
                fill_price = self._apply_slippage(next_open, curr_atr_pct, 'BUY')
                slip_amt = fill_price - next_open
                total_slippage_cost += slip_amt * (capital / fill_price)
                gross_spend = capital
                net_spend = gross_spend * (1.0 - self.commission)
                position = net_spend / fill_price
                # Fix #2: Giriş fiyatı olarak slippage uygulanmış fiyat
                entry_price = fill_price
                capital = 0.0
                in_position = True
                bars_held = 0
                active_sl = curr_sl if np.isfinite(curr_sl) else np.nan
                active_tp = curr_tp if np.isfinite(curr_tp) else np.nan

                trade_log.append({
                    'işlem_no': len([t for t in trade_log if t['tip'] == 'ALIŞ']) + 1,
                    'tip': 'ALIŞ',
                    'tarih': str(dates[i + 1].date()),
                    'fiyat': round(fill_price, 4),
                    'ham_fiyat': round(next_open, 4),
                    'slippage': round(slip_amt, 4),
                    'komisyon_pct': self.commission * 100,
                    # Fix #8: NaN sl/tp değerlerini güvenli format
                    'sl': round(float(curr_sl), 4) if np.isfinite(curr_sl) else None,
                    'tp': round(float(curr_tp), 4) if np.isfinite(curr_tp) else None,
                })

            current_equity = capital + position * closes[i]
            equity_list.append(current_equity)

        # ── Dönem sonu: açık pozisyon varsa piyasa fiyatından kapat ──────────
        if in_position:
            final_close  = closes[-1]
            capital      = position * final_close * (1.0 - self.commission)
            # Fix #2: Correct two-sided commission — end-of-period close
            entry_cost_fc = entry_price * (1.0 + self.commission)
            exit_recv_fc  = final_close  * (1.0 - self.commission)
            trade_ret     = (exit_recv_fc - entry_cost_fc) / entry_cost_fc * 100
            position     = 0.0

            trade_log.append({
                'tip'         : 'OTOMATİK KAPANIŞ',
                'tarih'       : str(dates[-1].date()),
                'fiyat'       : round(final_close, 4),
                'getiri_pct'  : round(trade_ret, 2),
                'not'         : 'Test süresi doldu, pozisyon kapatıldı',
            })

        equity_list.append(capital)
        final_equity = capital

        # ── Equity Curve ─────────────────────────────────────────────────────
        # FIX #11: equity_list[0] = initial_capital (alıştan önce),
        # loop N-1 eleman, final append 1 eleman → toplam N+1 eleman.
        # Index: dates[0] öncesine "başlangıç" noktası ekliyoruz.
        equity_index = pd.DatetimeIndex(
            [dates[0] - pd.Timedelta(days=1)] + list(dates)
        )
        equity_series = pd.Series(equity_list, index=equity_index, name='equity')


        # ── Metrik Hesaplamaları ──────────────────────────────────────────────

        # 1. Toplam Getiri
        total_return_pct = (final_equity - self.initial_capital) / self.initial_capital * 100

        # 2. Hit Rate (Kazanma Oranı)
        sale_returns = [t['getiri_pct'] for t in trade_log
                        if t['tip'] in ('SATIŞ', 'STOP-LOSS', 'TAKE-PROFIT', 'TIME-EXIT', 'OTOMATİK KAPANIŞ')
                        and 'getiri_pct' in t]
        hit_rate_pct = (
            sum(1 for r in sale_returns if r > 0) / len(sale_returns) * 100
            if sale_returns else 0.0
        )

        # 3. Max Drawdown
        roll_max      = equity_series.cummax()
        drawdown      = (equity_series - roll_max) / roll_max * 100
        max_drawdown  = float(drawdown.min())

        # 4. Sharpe Ratio (yıllıklandırılmış, risk-free = 0 varsayımı)
        daily_returns  = equity_series.pct_change().dropna()
        if daily_returns.std() > 1e-9:
            sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        # 5. Sortino Ratio (yalnızca negatif getiriler üzerinden)
        neg_returns = daily_returns[daily_returns < 0]
        if len(neg_returns) > 0 and neg_returns.std() > 1e-9:
            sortino = float(daily_returns.mean() / neg_returns.std() * np.sqrt(252))
        else:
            sortino = 0.0

        # 6. Al-Tut (Buy & Hold) karşılaştırması
        bnh_return_pct = (closes[-1] - closes[0]) / closes[0] * 100

        # 7. İşlem sayıları
        n_buys  = len([t for t in trade_log if str(t['tip']).startswith('ALIŞ')])
        n_sells = len([t for t in trade_log if str(t['tip']) in ('SATIŞ', 'STOP-LOSS', 'TAKE-PROFIT', 'TIME-EXIT', 'OTOMATİK KAPANIŞ')])

        # ── Sinyal Tablosu (UI'da gösterilecek günlük tablo) ──────────────────
        signals_df = df[[
            'Open', 'High', 'Low', 'Close', 'Volume',
            'consensus_signal', 'confidence',
            'vote_mom', 'vote_ma', 'vote_pivot', 'vote_don', 'vote_aso', 'vote_bb_rsi', 'vote_macd_ema',
            'suggested_sl', 'suggested_tp'
        ]].copy()
        signals_df.columns = [
            'Açılış', 'Yüksek', 'Düşük', 'Kapanış', 'Hacim',
            'Sinyal', 'Güven%',
            'Mom.Oy', 'MA Oy', 'Pivot Oy', 'Don.Oy', 'ASO Oy', 'BB+RSI Oy', 'MACD+EMA Oy',
            'Stop-Loss', 'Take-Profit'
        ]

        return {
            # ── Ana Metrikler ──────────────────────────────────────────────────
            'equity_curve'     : equity_series,
            'total_return_pct' : round(total_return_pct, 2),
            'hit_rate_pct'     : round(hit_rate_pct, 1),
            'max_drawdown_pct' : round(max_drawdown, 2),
            'sharpe_ratio'     : round(sharpe, 2),
            'sortino_ratio'    : round(sortino, 2),
            'total_trades'     : n_buys,
            'total_sells'      : n_sells,
            'final_capital'    : round(final_equity, 2),
            'buy_and_hold_pct' : round(bnh_return_pct, 2),
            # ── Detay ─────────────────────────────────────────────────────────
            'trade_log'        : trade_log,
            'signals_df'       : signals_df,
            'initial_capital'  : self.initial_capital,
            'commission_pct'   : self.commission * 100,
            # ── Slippage Bilgisi ──────────────────────────────────────────────
            'slippage_factor'  : self.slippage_factor,
            'total_slippage_cost' : round(total_slippage_cost, 2),
        }
        
def run_vectorized_backtest(df, initial_capital=10000, commission_pct=0.001, slippage_factor=0.10):
    """BacktestEngine için kolaylaştırıcı fonksiyon."""
    engine = BacktestEngine(df, initial_capital=initial_capital, commission_pct=commission_pct, slippage_factor=slippage_factor)
    return engine.run()
