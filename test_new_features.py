"""
Yeni AI Hedge Fund Özelliklerini Test Scripti
================================================
Bu script, 6 gelişmiş özelliğin kurulumunu ve temel çalışmasını doğrular.
Her test bağımsız çalışır ve bağımlılık eksikse hata yerine uyarı verir.

Çalıştırma:
    python test_new_features.py
"""
import sys
import os
import io

# Ensure stdout and stderr support UTF-8 encoding even in Windows terminal
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HF_PATH = os.path.join(BASE_DIR, 'ai-hedge-fund-main')

# ai-hedge-fund-main'i Python path'ine ekle
if HF_PATH not in sys.path:
    sys.path.insert(0, HF_PATH)

# Test sonuçlarını takip et
results = []

def test_result(name, passed, detail=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    results.append((name, passed, detail))
    print(f"  {status} - {name}")
    if detail:
        print(f"         {detail}")

print("=" * 60)
print("🧪 AI Hedge Fund - Yeni Özellik Testleri")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# TEST 1: RAG Engine Import & Temel Çalışma
# ──────────────────────────────────────────────────────────────
print("\n📚 Test 1: RAG Engine")
try:
    from src.utils.rag_engine import RAGEngine, query_rag
    
    # RAG motorunu oluştur
    docs_dir = os.path.join(HF_PATH, "docs")
    rag = RAGEngine(docs_dir=docs_dir)
    
    test_result("RAG Engine import", True)
    
    # docs/ dizini var mı?
    if os.path.isdir(docs_dir):
        test_result("docs/ dizini mevcut", True, docs_dir)
    else:
        os.makedirs(docs_dir, exist_ok=True)
        test_result("docs/ dizini oluşturuldu", True, docs_dir)
    
    # Örnek bir TXT dosyası oluşturup RAG'ı test edelim
    sample_file = os.path.join(docs_dir, "test_document.txt")
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write("""
THYAO (Türk Hava Yolları) 2024 Yıllık Rapor Özeti
====================================================
Brüt Kâr Marjı: %23.5
Net Kâr: 42.7 Milyar TL
Toplam Gelir: 285 Milyar TL
Yolcu Sayısı: 83.4 Milyon
Filo Büyüklüğü: 435 Uçak

BIST 100 endeksinde %12.8 getiri sağlamıştır.
Temettü verimi %3.2 olarak açıklanmıştır.
Şirketin borç/özsermaye oranı 0.45'tir.
        """)
    test_result("Test dokümanı oluşturuldu", True, sample_file)
    
    # Ingest (vektör deposuna yükle)
    chunk_count = rag.ingest()
    test_result("RAG Ingest (vektör yükleme)", chunk_count > 0, f"{chunk_count} chunk yüklendi")
    
    # Query (sorgula)
    result = rag.query("THYAO brüt kâr marjı nedir?")
    has_result = len(result) > 0 and "THYAO" in result
    test_result("RAG Query (sorgulama)", has_result, result[:100] + "..." if len(result) > 100 else result)
    
    # has_documents kontrolü
    test_result("RAG has_documents()", rag.has_documents())
    
    # Temizlik: test dosyasını sil
    os.remove(sample_file)
    
except Exception as e:
    test_result("RAG Engine", False, str(e))

# ──────────────────────────────────────────────────────────────
# TEST 2: Multi-Agent Debate Engine
# ──────────────────────────────────────────────────────────────
print("\n🤝 Test 2: Multi-Agent Debate Engine")
try:
    from src.agents.debate_engine import (
        run_multi_agent_debate,
        GROWTH_AGENTS,
        VALUE_AGENTS,
        DebateArgument,
        DebateRoundOutput,
        DebateVerdict,
    )
    
    test_result("Debate Engine import", True)
    test_result("Growth Agents tanımlı", len(GROWTH_AGENTS) > 0, f"{len(GROWTH_AGENTS)} ajan: {GROWTH_AGENTS}")
    test_result("Value Agents tanımlı", len(VALUE_AGENTS) > 0, f"{len(VALUE_AGENTS)} ajan: {VALUE_AGENTS}")
    
    # Pydantic modelleri doğrula
    arg = DebateArgument(
        agent_name="test", signal="bullish", confidence=80,
        argument="Test argument", rebuttal="Test rebuttal"
    )
    test_result("DebateArgument modeli", arg.signal == "bullish")
    
    verdict = DebateVerdict(
        final_signal="neutral", final_confidence=60,
        consensus_summary="Test summary"
    )
    test_result("DebateVerdict modeli", verdict.final_signal == "neutral")
    
    print("\n  ℹ️  Not: Gerçek debate testi için LLM (OpenAI/Ollama) API bağlantısı gerekir.")
    print("       run_debate_analysis('AAPL') şeklinde çağırabilirsiniz.")
    
except Exception as e:
    test_result("Debate Engine", False, str(e))

# ──────────────────────────────────────────────────────────────
# TEST 3: Kronos + Sentiment Sinerjisi
# ──────────────────────────────────────────────────────────────
print("\n🧠 Test 3: Kronos + Sentiment Sinerjisi")
try:
    # ai_integrations.py'den fonksiyonları import et
    sys.path.insert(0, BASE_DIR)
    from ai_integrations import run_kronos_prediction, get_sentiment_enhanced_df
    
    test_result("Kronos + Sentiment import", True)
    
    # Sentiment enhanced DataFrame testi
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    test_df = pd.DataFrame({
        "Open": np.random.uniform(100, 110, 50),
        "High": np.random.uniform(110, 120, 50),
        "Low": np.random.uniform(90, 100, 50),
        "Close": np.random.uniform(100, 115, 50),
        "Volume": np.random.randint(1000000, 5000000, 50),
    }, index=dates)
    
    # Sentiment feature enjeksiyonu
    enhanced_df = get_sentiment_enhanced_df(test_df, sentiment_score=0.7)
    
    has_sentiment = "sentiment_score" in enhanced_df.columns
    has_interaction = "price_sentiment_interaction" in enhanced_df.columns
    
    test_result("Sentiment sütunu eklendi", has_sentiment, 
                f"sentiment_score = {enhanced_df['sentiment_score'].iloc[0]}")
    test_result("Price-Sentiment interaction", has_interaction)
    
    # run_kronos_prediction imzası kontrol (sentiment_score parametresi var mı?)
    import inspect
    sig = inspect.signature(run_kronos_prediction)
    has_sentiment_param = "sentiment_score" in sig.parameters
    test_result("Kronos sentiment_score parametresi", has_sentiment_param, str(sig))
    
    print("\n  ℹ️  Not: Gerçek Kronos tahmini için model indirilmesi gerekir (~100MB).")
    print("       run_kronos_prediction(df, sentiment_score=0.7) şeklinde çağırın.")
    
except Exception as e:
    test_result("Kronos + Sentiment", False, str(e))

# ──────────────────────────────────────────────────────────────
# TEST 4: Chain of Thought (CoT) Prompt Yapısı
# ──────────────────────────────────────────────────────────────
print("\n🧩 Test 4: Chain of Thought (CoT) Prompt")
try:
    import ast
    
    # fundamentals.py'de CoT var mı?
    fund_path = os.path.join(HF_PATH, "src", "agents", "fundamentals.py")
    with open(fund_path, "r", encoding="utf-8") as f:
        fund_content = f.read()
    
    has_cot_fund = "chain_of_thought" in fund_content and "step_1_macro" in fund_content
    test_result("Fundamentals CoT entegrasyonu", has_cot_fund)
    
    # portfolio_manager.py'de CoT var mı?
    pm_path = os.path.join(HF_PATH, "src", "agents", "portfolio_manager.py")
    with open(pm_path, "r", encoding="utf-8") as f:
        pm_content = f.read()
    
    has_cot_pm = "DECISION FRAMEWORK (Chain of Thought)" in pm_content
    has_macro = "Step 1: MACRO" in pm_content
    has_sector = "Step 2: SECTOR" in pm_content
    has_stock = "Step 3: STOCK" in pm_content
    has_action = "Step 4: ACTION" in pm_content
    
    test_result("Portfolio Manager CoT Framework", has_cot_pm)
    test_result("  Step 1: MACRO", has_macro)
    test_result("  Step 2: SECTOR", has_sector)
    test_result("  Step 3: STOCK", has_stock)
    test_result("  Step 4: ACTION", has_action)
    
except Exception as e:
    test_result("CoT Prompt", False, str(e))

# ──────────────────────────────────────────────────────────────
# TEST 5: Dinamik Ajan Ağırlıklandırma
# ──────────────────────────────────────────────────────────────
print("\n⚖️ Test 5: Dinamik Ajan Ağırlıklandırma")
try:
    from src.agents.portfolio_manager import _detect_market_regime, _apply_regime_weights
    
    test_result("Rejim fonksiyonları import", True)
    
    # Bull rejim testi
    bull_signals = {
        "AAPL": {
            "warren_buffett_agent": {"sig": "bullish", "conf": 80},
            "cathie_wood_agent": {"sig": "bullish", "conf": 70},
            "ben_graham_agent": {"sig": "bullish", "conf": 60},
        }
    }
    regime = _detect_market_regime(bull_signals)
    test_result("Bull rejim algılama", regime == "bull", f"Rejim: {regime}")
    
    # Bear rejim testi
    bear_signals = {
        "AAPL": {
            "warren_buffett_agent": {"sig": "bearish", "conf": 80},
            "cathie_wood_agent": {"sig": "bearish", "conf": 70},
            "ben_graham_agent": {"sig": "bearish", "conf": 60},
        }
    }
    regime = _detect_market_regime(bear_signals)
    test_result("Bear rejim algılama", regime == "bear", f"Rejim: {regime}")
    
    # Ağırlık uygulama testi (bull piyasada growth ajanları boost almalı)
    test_signals = {
        "AAPL": {
            "cathie_wood_agent": {"sig": "bullish", "conf": 60},
            "warren_buffett_agent": {"sig": "bullish", "conf": 60},
        }
    }
    weighted = _apply_regime_weights(test_signals, "bull")
    cw_conf = weighted["AAPL"]["cathie_wood_agent"]["conf"]
    wb_conf = weighted["AAPL"]["warren_buffett_agent"]["conf"]
    
    test_result("Bull: Growth boost (Cathie Wood)", cw_conf == 90, f"60 → {cw_conf} (1.5x)")
    test_result("Bull: Value dampen (Buffett)", wb_conf == 45, f"60 → {wb_conf} (0.75x)")
    
except Exception as e:
    test_result("Dinamik Ağırlıklandırma", False, str(e))

# ──────────────────────────────────────────────────────────────
# TEST 6: Backtest Komisyon & Slippage
# ──────────────────────────────────────────────────────────────
print("\n💰 Test 6: Backtest Komisyon & Slippage")
try:
    from src.backtesting.portfolio import Portfolio
    
    # Komisyon + slippage parametreleriyle Portfolio oluştur
    portfolio = Portfolio(
        tickers=["AAPL"],
        initial_cash=100000.0,
        margin_requirement=0.5,
        commission_pct=0.001,    # %0.1 komisyon
        slippage_pct=0.0005,     # %0.05 slippage
    )
    test_result("Portfolio + komisyon/slippage oluşturma", True)
    
    # Long buy testi
    initial_cash = portfolio.get_cash()
    bought = portfolio.apply_long_buy("AAPL", 100, 150.0)
    remaining_cash = portfolio.get_cash()
    
    # Slippage: 150 * 1.0005 = 150.075 → cost = 100 * 150.075 = 15007.50
    # Komisyon: 15007.50 * 0.001 = 15.0075
    # Toplam düşüş ≈ 15007.50 + 15.0075 = 15022.5075
    expected_cost_approx = 100 * 150 * 1.0005  # slippage dahil
    commission_approx = expected_cost_approx * 0.001
    
    test_result("Long Buy işlemi", bought == 100, f"{bought} hisse alındı")
    
    # Komisyon tahsil edildi mi?
    total_commission = portfolio.get_total_commissions()
    test_result("Komisyon tahsil edildi", total_commission > 0, f"Komisyon: ${total_commission:.4f}")
    
    # Slippage uygulandı mı?
    total_slippage = portfolio.get_total_slippage_cost()
    test_result("Slippage uygulandı", total_slippage > 0, f"Slippage maliyeti: ${total_slippage:.4f}")
    
    # Nakit doğru düştü mü?
    cash_drop = initial_cash - remaining_cash
    test_result("Nakit düşüşü doğru", cash_drop > 15000, 
                f"${initial_cash:.2f} → ${remaining_cash:.2f} (fark: ${cash_drop:.2f})")
    
    # Long sell testi
    sold = portfolio.apply_long_sell("AAPL", 50, 160.0)
    test_result("Long Sell işlemi", sold == 50, f"{sold} hisse satıldı")
    
    # Sell sonrası komisyon artmış olmalı
    new_commission = portfolio.get_total_commissions()
    test_result("Satış komisyonu", new_commission > total_commission, 
                f"Toplam komisyon: ${new_commission:.4f}")
    
except Exception as e:
    test_result("Backtest Komisyon", False, str(e))

# ──────────────────────────────────────────────────────────────
# SONUÇ ÖZETİ
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 TEST SONUÇ ÖZETİ")
print("=" * 60)

passed = sum(1 for _, p, _ in results if p)
failed = sum(1 for _, p, _ in results if not p)
total = len(results)

print(f"\n  Toplam: {total} test")
print(f"  ✅ Geçen: {passed}")
print(f"  ❌ Başarısız: {failed}")
print(f"  Başarı oranı: {passed/total*100:.1f}%")

if failed > 0:
    print("\n  Başarısız testler:")
    for name, passed_flag, detail in results:
        if not passed_flag:
            print(f"    ❌ {name}: {detail}")

print("\n" + "=" * 60)
print("🎯 Sonraki Adımlar:")
print("  1. LLM API ayarlayın (.env'de OPENAI_API_KEY veya Ollama)")
print("  2. docs/ dizinine bilanço PDF'leri koyun")
print("  3. Ana uygulamayı çalıştırın: streamlit run app.py")
print("=" * 60)
