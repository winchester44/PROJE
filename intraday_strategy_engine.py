import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

class IntradayMomentumEngine:
    """
    Beat the Market: An Effective Intraday Momentum Strategy (SPY)
    Zarattini, Aziz, Barbon (2024)
    """
    
    def __init__(self, volatility_multiplier=1.0, target_vol=0.02, max_leverage=4.0):
        self.vm = volatility_multiplier
        self.target_vol = target_vol
        self.max_leverage = max_leverage

    def fetch_intraday_data(self, ticker, period="1mo", interval="1m"):
        """yFinance üzerinden intraday (1dk/5dk) veri çeker."""
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        return df

    def calculate_indicators(self, df):
        """Strateji için gerekli Noise Area ve VWAP hesaplamalarını yapar."""
        if df.empty: return df
        df = df.copy()
        
        # 1. VWAP (Intraday)
        df['price_vol'] = df['Close'] * df['Volume']
        df['cum_vol'] = df.groupby(df.index.date)['Volume'].cumsum()
        df['cum_price_vol'] = df.groupby(df.index.date)['price_vol'].cumsum()
        df['VWAP'] = df['cum_price_vol'] / df['cum_vol']

        # 2. Daily Volatility (Son 14 gün) - Pozisyon büyüklüğü için
        daily_close = df['Close'].resample('D').last().dropna()
        daily_returns = daily_close.pct_change()
        # Vektörel ffill ile dakikalık verilere yay
        df['daily_vol'] = daily_returns.rolling(window=14).std().reindex(df.index, method='ffill')

        # 3. Noise Area
        df['time'] = df.index.time
        df['date'] = df.index.date
        df['day_open'] = df.groupby('date')['Open'].transform('first')
        
        # Önceki gün kapanış (Overnight gap için)
        prev_close = daily_close.shift(1)
        df['prev_close'] = prev_close.reindex(df.index, method='ffill')
        
        # move_t,i = |Price / Open - 1|
        df['pct_move_from_open'] = np.abs(df['Close'] / df['day_open'] - 1)
        
        # Zaman bazlı sigma (Her saat dilimi için 14 günlük ortalama hareket)
        pivot_moves = df.pivot_table(index='date', columns='time', values='pct_move_from_open')
        rolling_sigma = pivot_moves.rolling(window=14).mean().shift(1) 
        
        # Sigmaları ana tabloya ekle
        df_sigma = rolling_sigma.melt(ignore_index=False, value_name='sigma').reset_index()
        df = df.reset_index().merge(df_sigma, on=['date', 'time'], how='left').set_index('Datetime')

        # 4. Dinamik Sınırlar (Boundaries)
        # UpperBound = max(Open_t, Close_t-1) * (1 + VM * sigma)
        df['adj_open_high'] = np.maximum(df['day_open'], df['prev_close'])
        df['adj_open_low'] = np.minimum(df['day_open'], df['prev_close'])
        
        df['upper_band'] = df['adj_open_high'] * (1 + self.vm * df['sigma'])
        df['lower_band'] = df['adj_open_low'] * (1 - self.vm * df['sigma'])
        
        return df

    def run_backtest(self, df, initial_balance=10000.0, commission=0.001):
        """Intraday simülasyonu — vektörleştirilmiş sinyal üretimi + minimal döngü."""
        df = self.calculate_indicators(df)
        df = df.dropna(subset=['sigma', 'daily_vol'])
        
        if df.empty:
            return None, "Yeterli geçmiş veri yok (Sigma hesaplanamadı)."

        # ── Numpy dizilerine dönüştür (row-by-row DataFrame erişimi yerine) ───
        n = len(df)
        times     = df.index
        closes    = df['Close'].values.astype(np.float64)
        uppers    = df['upper_band'].values.astype(np.float64)
        lowers    = df['lower_band'].values.astype(np.float64)
        daily_vols = df['daily_vol'].values.astype(np.float64)

        # Vektörel trailing stop hesaplaması
        long_stops  = np.maximum(uppers, df['VWAP'].values.astype(np.float64))
        short_stops = np.minimum(lowers, df['VWAP'].values.astype(np.float64))

        # Vektörel sinyal maskeleri (boolean)
        is_trade_window = np.isin(df.index.minute, [0, 30])
        hours = np.array([t.hour for t in times])
        minutes = np.array([t.minute for t in times])
        is_close = (hours == 16) & (minutes == 0)

        # Vektörel giriş sinyalleri
        long_entry_signal  = (closes > uppers) & is_trade_window & ~is_close
        short_entry_signal = (closes < lowers) & is_trade_window & ~is_close

        # Vektörel leverage hesaplaması
        safe_vols = np.where(daily_vols > 0, daily_vols, 1.0)
        leverages = np.minimum(self.max_leverage, self.target_vol / safe_vols)

        # ── State-machine döngüsü (sadece pozisyon geçişlerinde iş yapar) ────
        balance     = float(initial_balance)
        position    = 0.0
        entry_price = 0.0
        trade_log   = []
        equity_list = np.empty(n, dtype=np.float64)

        for i in range(n):
            cp = closes[i]

            # --- POZİSYON ÇIKIŞ ---
            if position > 0:  # Long'dayız
                if cp < long_stops[i] or is_close[i]:
                    reason = "🔴 STOP (Trailing)" if not is_close[i] else "🏁 GÜN SONU"
                    balance = position * cp * (1 - commission)
                    trade_log.append({
                        "Tarih": times[i].strftime("%H:%M %d.%m"),
                        "İşlem": reason,
                        "Fiyat": f"{cp:.2f}",
                        "Bakiye": f"${balance:,.2f}"
                    })
                    position = 0.0

            elif position < 0:  # Short'tayız
                if cp > short_stops[i] or is_close[i]:
                    reason = "🟢 STOP (Trailing)" if not is_close[i] else "🏁 GÜN SONU"
                    p_val = abs(position) * (entry_price + (entry_price - cp)) * (1 - commission)
                    balance = p_val
                    trade_log.append({
                        "Tarih": times[i].strftime("%H:%M %d.%m"),
                        "İşlem": reason,
                        "Fiyat": f"{cp:.2f}",
                        "Bakiye": f"${balance:,.2f}"
                    })
                    position = 0.0

            # --- POZİSYON GİRİŞ ---
            if position == 0 and not is_close[i]:
                if long_entry_signal[i]:  # LONG Giriş
                    lev = leverages[i]
                    shares = (balance * lev) / (cp * (1 + commission))
                    position = shares
                    entry_price = cp
                    trade_log.append({
                        "Tarih": times[i].strftime("%H:%M %d.%m"),
                        "İşlem": "🔵 LONG GİRİŞ",
                        "Fiyat": f"{cp:.2f}",
                        "Bakiye": "Pozisyonda"
                    })
                    balance = 0.0

                elif short_entry_signal[i]:  # SHORT Giriş
                    lev = leverages[i]
                    shares = (balance * lev) / (cp * (1 + commission))
                    position = -shares
                    entry_price = cp
                    trade_log.append({
                        "Tarih": times[i].strftime("%H:%M %d.%m"),
                        "İşlem": "🟠 SHORT GİRİŞ",
                        "Fiyat": f"{cp:.2f}",
                        "Bakiye": "Pozisyonda"
                    })
                    balance = 0.0

            # Equity curve takibi
            if position != 0:
                cur_val = abs(position) * cp if position > 0 else abs(position) * (entry_price + (entry_price - cp))
                equity_list[i] = cur_val
            else:
                equity_list[i] = balance

        return {
            "equity_curve": pd.Series(equity_list, index=df.index),
            "trade_log": trade_log,
            "final_balance": balance if balance > 0 else equity_list[-1]
        }, None
