"""
reliability_test.py
====================
FIX #13 — Model Güvenilirlik & Backtest Doğrulama Scripti

Amacı:
  - ML modelinin OOS (out-of-sample) gerçek doğruluğunu ölçmek
  - 2020 Covid çöküşü + 2022 bear market dönemlerinde ne kadar iyi çalıştığını test etmek
  - Backtest sonuçlarının 'Buy & Hold' ile karşılaştırmasını yapmak
  - Her hisse/kripto için özet rapor üretmek

Kullanım:
  python reliability_test.py                    # varsayılan semboller
  python reliability_test.py THYAO.IS BTC-USD   # özel semboller

Çıktı:
  - Konsola özet tablo
  - reliability_report.csv (detaylı sonuçlar)
"""

import sys
import io
import warnings
from datetime import datetime

# Ensure stdout and stderr support UTF-8 encoding even in Windows terminal
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# Proje modüllerini import et
try:
    from stock_analyzer import (
        calculate_technical_indicators,
        add_relative_strength,
        train_ml_model,
        detect_market_type,
    )
    from backtest_engine import BacktestEngine
except ImportError as e:
    print(f"[HATA] Modül import edilemedi: {e}")
    print("Bu scripti 'borsa site/' klasöründen çalıştırın.")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# YAPILANDIRMA
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SYMBOLS = [
    # BIST
    "THYAO.IS", "GARAN.IS", "ASELS.IS",
    # ABD
    "AAPL", "MSFT", "SPY",
    # Kripto
    "BTC-USD", "ETH-USD",
]

# Test dönemleri — kritik piyasa rejimleri
TEST_PERIODS = {
    "3_yil_tam":    {"start": "2021-01-01", "end": datetime.today().strftime("%Y-%m-%d")},
    "covid_cokus":  {"start": "2020-01-01", "end": "2020-12-31"},   # Covid çöküşü
    "bear_2022":    {"start": "2022-01-01", "end": "2022-12-31"},   # Fed faiz artışı bear market
    "boga_2023":    {"start": "2023-01-01", "end": "2023-12-31"},   # Toparlanma dönemi
}


# ──────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────────────────────────────────────

def fetch_period(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Belirli tarih aralığında veri çeker."""
    df = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(-1)
    df.dropna(subset=["Close"], inplace=True)
    return df


def buy_and_hold_return(df: pd.DataFrame) -> float:
    """Basit al-tut getirisi (%)."""
    if len(df) < 2:
        return 0.0
    return (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100


def ml_accuracy_for_period(df: pd.DataFrame, market_type: str) -> dict:
    """
    Belirli bir dönem için ML modelinin OOS doğruluğunu hesaplar.
    Dönem başındaki %60 eğitim, %40 test şeklinde bölünür.
    """
    if len(df) < 80:
        return {"accuracy": None, "up_prob": None, "error": "Yetersiz veri"}
    try:
        df = calculate_technical_indicators(df.copy())
        df = add_relative_strength(df, "SPY")  # basit benchmark

        cfg = {
            "us":     {"horizon": 5,  "threshold": 0.02},
            "bist":   {"horizon": 5,  "threshold": 0.03},
            "crypto": {"horizon": 3,  "threshold": 0.03},
        }.get(market_type, {"horizon": 5, "threshold": 0.02})

        up_prob, accuracy = train_ml_model(
            df,
            horizon=cfg["horizon"],
            threshold=cfg["threshold"],
            market_type=market_type,
        )
        return {"accuracy": round(accuracy, 2), "up_prob": round(up_prob, 2), "error": None}
    except Exception as exc:
        return {"accuracy": None, "up_prob": None, "error": str(exc)[:80]}


def backtest_for_period(df: pd.DataFrame) -> dict:
    """BacktestEngine ile tam strateji simülasyonu çalıştırır."""
    if len(df) < 60:
        return {"total_return_pct": None, "max_drawdown_pct": None,
                "sharpe": None, "n_trades": None, "error": "Yetersiz veri"}
    try:
        df_ind = calculate_technical_indicators(df.copy())
        engine = BacktestEngine(df_ind, initial_capital=10_000, commission_pct=0.001)
        result = engine.run()
        # BacktestEngine.run() metriklerı doğrudan üst seviyede döndürüyor (statistics alt anahtarı yok)
        return {
            "total_return_pct":  result.get("total_return_pct"),
            "max_drawdown_pct":  result.get("max_drawdown_pct"),
            "sharpe":            result.get("sharpe_ratio"),
            "n_trades":          result.get("total_trades"),
            "win_rate":          result.get("hit_rate_pct"),
            "error": None,
        }
    except Exception as exc:
        return {"total_return_pct": None, "max_drawdown_pct": None,
                "sharpe": None, "n_trades": None, "error": str(exc)[:80]}


# ──────────────────────────────────────────────────────────────────────────────
# ANA TEST DÖNGÜSÜ
# ──────────────────────────────────────────────────────────────────────────────

def run_reliability_test(symbols: list) -> pd.DataFrame:
    records = []

    print("\n" + "="*70)
    print("  Hisse Avcısı — MODEL GÜVENİLİRLİK TESTİ")
    print("="*70)
    print(f"  Test edilen semboller : {', '.join(symbols)}")
    print(f"  Test dönemleri        : {', '.join(TEST_PERIODS.keys())}")
    print("="*70 + "\n")

    for symbol in symbols:
        market_type = detect_market_type(symbol)
        print(f"\n▶ {symbol} ({market_type.upper()})")
        print("  " + "-"*60)

        for period_name, period_cfg in TEST_PERIODS.items():
            start = period_cfg["start"]
            end   = period_cfg["end"]

            print(f"  [{period_name}] {start} → {end}", end=" ... ")

            try:
                df = fetch_period(symbol, start, end)
                if df.empty or len(df) < 30:
                    print("VERİ YOK")
                    continue

                bah     = round(buy_and_hold_return(df), 2)
                ml_res  = ml_accuracy_for_period(df, market_type)
                bt_res  = backtest_for_period(df)

                accuracy  = ml_res["accuracy"]
                up_prob   = ml_res["up_prob"]
                bt_return = bt_res["total_return_pct"]
                bt_dd     = bt_res["max_drawdown_pct"]
                sharpe    = bt_res["sharpe"]
                n_trades  = bt_res["n_trades"]
                win_rate  = bt_res.get("win_rate")

                # Alfa: strateji getirisinin B&H'i ne kadar geçtiği
                alfa = round(bt_return - bah, 2) if bt_return is not None else None

                status = "✅" if (accuracy or 0) >= 55 else "⚠️" if accuracy else "❌"
                if bt_return is not None:
                    print(
                        f"{status}  "
                        f"ML Doğruluk: {accuracy or 'N/A'}%  |  "
                        f"B&H: {bah:+.1f}%  |  "
                        f"Strateji: {bt_return:+.1f}%"
                    )
                else:
                    print(
                        f"{status}  ML: {accuracy or 'N/A'}%  |  "
                        f"B&H: {bah:+.1f}%  |  "
                        f"Backtest: HATA"
                    )

                records.append({
                    "Sembol":           symbol,
                    "Piyasa":           market_type,
                    "Dönem":            period_name,
                    "Başlangıç":        start,
                    "Bitiş":            end,
                    "Veri_Gün":         len(df),
                    "ML_Doğruluk_%":    accuracy,
                    "ML_Yükseliş_%":    up_prob,
                    "BuyHold_Getiri_%": bah,
                    "Strateji_Getiri_%":bt_return,
                    "Alfa_%":           alfa,
                    "Max_DD_%":         bt_dd,
                    "Sharpe":           sharpe,
                    "İşlem_Sayısı":     n_trades,
                    "Kazanma_Oranı_%":  win_rate,
                    "ML_Hata":          ml_res["error"],
                    "BT_Hata":          bt_res["error"],
                })

            except Exception as exc:
                print(f"❌ HATA: {str(exc)[:60]}")

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# ÖZET RAPOR YAZICI
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("\n" + "="*70)
    print("  ÖZET SONUÇLAR")
    print("="*70)

    if df.empty:
        print("  Hiçbir test sonucu üretilemedi.")
        return

    # ML ortalama doğruluk
    valid_acc = df["ML_Doğruluk_%"].dropna()
    if not valid_acc.empty:
        print(f"\n  Ortalama ML Doğruluğu   : %{valid_acc.mean():.1f}")
        print(f"  Min / Maks ML Doğruluğu : %{valid_acc.min():.1f} / %{valid_acc.max():.1f}")

    # Strateji vs B&H karşılaştırması
    valid_alfa = df["Alfa_%"].dropna()
    if not valid_alfa.empty:
        positive_alfa = (valid_alfa > 0).sum()
        total         = len(valid_alfa)
        print(f"\n  Strateji > B&H oranı    : {positive_alfa}/{total} dönem (%{positive_alfa/total*100:.0f})")
        print(f"  Ortalama Alfa           : {valid_alfa.mean():+.1f}%")

    # Sharpe
    valid_sharpe = df["Sharpe"].dropna()
    if not valid_sharpe.empty:
        print(f"\n  Ortalama Sharpe Oranı   : {valid_sharpe.mean():.2f}")

    # Max Drawdown
    valid_dd = df["Max_DD_%"].dropna()
    if not valid_dd.empty:
        print(f"  Ortalama Max Drawdown   : %{valid_dd.mean():.1f}")
        print(f"  En Kötü Drawdown        : %{valid_dd.min():.1f}")

    # Covid ve Bear dönemleri özel analiz
    print("\n  --- Kritik Dönem Analizi ---")
    for dnem in ["covid_cokus", "bear_2022"]:
        subset = df[df["Dönem"] == dnem]
        if subset.empty:
            continue
        acc_mean = subset["ML_Doğruluk_%"].mean()
        alfa_mean = subset["Alfa_%"].mean()
        print(f"  {dnem:15s} → ML: %{acc_mean:.1f}  |  Alfa: {alfa_mean:+.1f}%")

    print("\n" + "="*70)


# ──────────────────────────────────────────────────────────────────────────────
# GİRİŞ NOKTASI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    symbols = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_SYMBOLS

    results_df = run_reliability_test(symbols)
    print_summary(results_df)

    # CSV'ye kaydet
    out_path = "reliability_report.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Detaylı rapor kaydedildi: {out_path}\n")
