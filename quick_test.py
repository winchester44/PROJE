"""Nihai dogrulama testi - tum duzeltmeler sonrasi tam pipeline."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf, pandas as pd
from stock_analyzer import calculate_technical_indicators, add_relative_strength, train_ml_model, _LGBM_AVAILABLE
from backtest_engine import BacktestEngine

print("=== BORSA CLAUDE v3 - NIHAI TEST ===")
print("LightGBM:", _LGBM_AVAILABLE)

results = []
for sym, mkt in [("AAPL","us"), ("THYAO.IS","bist"), ("BTC-USD","crypto")]:
    try:
        df = yf.download(sym, period="2y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")
        df.dropna(subset=["Close"], inplace=True)
        df = calculate_technical_indicators(df)
        df = add_relative_strength(df, sym)
        prob, acc = train_ml_model(df, horizon=5, threshold=0.02, market_type=mkt)
        engine = BacktestEngine(df, initial_capital=10000, commission_pct=0.001)
        res = engine.run()
        tr  = res["total_return_pct"]
        bnh = res["buy_and_hold_pct"]
        sr  = res["sharpe_ratio"]
        hr  = res["hit_rate_pct"]
        dd  = res["max_drawdown_pct"]
        nt  = res["total_trades"]
        alfa = tr - bnh
        results.append({"sym": sym, "acc": acc, "tr": tr, "bnh": bnh, "alfa": alfa, "sr": sr})
        status = "BASARILI" if alfa > 0 else "BASARISIZ"
        print(f"{sym} [{status}]: ML={acc:.0f}% | BT={tr:+.1f}% vs BH={bnh:+.1f}% | Alfa={alfa:+.1f}% | Sharpe={sr:.2f} | HR={hr:.0f}% | DD={dd:.1f}% | Trades={nt}")
    except Exception as e:
        import traceback
        print(f"{sym}: HATA - {e}")
        traceback.print_exc()

if results:
    print()
    print("=== OZET ===")
    avg_acc  = sum(r["acc"]  for r in results) / len(results)
    avg_alfa = sum(r["alfa"] for r in results) / len(results)
    avg_sr   = sum(r["sr"]   for r in results) / len(results)
    pos = sum(1 for r in results if r["alfa"] > 0)
    print(f"Ort ML Dogrulugu : {avg_acc:.1f}%")
    print(f"Ort Alfa         : {avg_alfa:+.1f}%")
    print(f"Ort Sharpe       : {avg_sr:.2f}")
    print(f"Strateji > B&H   : {pos}/{len(results)}")
