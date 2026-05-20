import sys
import os
import io
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Ensure stdout and stderr support UTF-8 encoding even in Windows terminal
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# Modüllere erişmek için dizini ayarlayalım
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# .env dosyasını zorla yükle (mevcut boş değişkenleri ez)
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path, override=True)

# API Anahtarlarını .env'den çekiyoruz (OpenAI kotanız dolduğu için Google Gemini'ye geçiş yapabilirsiniz)
# os.environ["OPENAI_API_KEY"] = "sk-..."  # Kotası dolmuş anahtarı iptal ettik
if os.getenv("OPENAI_API_KEY") is None and os.getenv("GOOGLE_API_KEY") is None:
    print("UYARI: .env dosyasında OPENAI_API_KEY veya GOOGLE_API_KEY bulunamadı!")

from ai_integrations import run_rag_query, run_debate_analysis, run_kronos_prediction

def menu():
    print("\n" + "="*50)
    print("🤖 AI HEDGE FUND - YENİ ÖZELLİK DEMOSU")
    print("="*50)
    print("1) RAG Testi (PDF/Metin okuma)")
    print("2) Multi-Agent Debate Testi (Ajan Tartışması)")
    print("3) Kronos + Sentiment Testi")
    print("4) Çıkış")
    print("="*50)
    
    secim = input("Hangi özelliği test etmek istersin? (1/2/3/4): ")
    
    if secim == '1':
        print("\n📚 RAG (Belge Okuma) Testi Başlıyor...")
        print("Not: 'ai-hedge-fund-main/docs' klasöründeki belgeler okunuyor.")
        soru = input("Belgelerle ilgili ne sormak istersin? (Örn: THYAO brüt kar marjı nedir?): ")
        cevap = run_rag_query(soru)
        print("\nCevap:\n", cevap)
        
    elif secim == '2':
        print("\n🤝 Multi-Agent Debate Testi Başlıyor...")
        hisse = input("Hangi hisse için tartışma simülasyonu başlatılsın? (Örn: AAPL): ").upper()
        print(f"{hisse} için Warren Buffett ve Cathie Wood tartışıyor, lütfen bekleyin...")
        
        # Eğer API key kurulu değilse LLM çağrısı hata verebilir, onu yakalarız
        sonuc = run_debate_analysis(hisse)
        if "error" in sonuc:
            print("\n❌ Hata:", sonuc["error"])
            print("Not: Debate için .env dosyasında LLM API (OpenAI vs.) ayarlı olmalıdır.")
        else:
            print(f"\n✅ Karar: {sonuc.get('final_signal')} (Güven: %{sonuc.get('final_confidence')})")
            print("Özet:", sonuc.get('consensus_summary'))
            
    elif secim == '3':
        print("\n🧠 Kronos + Sentiment Sinerjisi Testi Başlıyor...")
        print("Sanal bir 50 günlük hisse verisi (OHLCV) oluşturuluyor...")
        
        # Sanal Veri Üretimi
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        df = pd.DataFrame({
            "Open": np.random.uniform(100, 110, 50),
            "High": np.random.uniform(110, 120, 50),
            "Low": np.random.uniform(90, 100, 50),
            "Close": np.random.uniform(100, 115, 50),
            "Volume": np.random.randint(1000000, 5000000, 50),
        }, index=dates)
        
        sentiment = float(input("Bu hisse için haber duygu skoru girin (-1.0 ile 1.0 arası, örn 0.7): "))
        print("Kronos modeli tahmini hesaplanıyor (ilk çalışmada model ineceği için biraz sürebilir)...")
        
        prob, acc, pct_change = run_kronos_prediction(df, horizon=5, sentiment_score=sentiment)
        if prob is not None:
            print(f"\n✅ 5 Günlük Tahmin Sonucu:")
            print(f"Yön Beklentisi Değişimi: %{pct_change*100:.2f}")
            print(f"Güven (Prob): %{prob:.2f}")
            print(f"Doğruluk (Acc): %{acc:.2f}")
        else:
            print("\n❌ Kronos hatası:", pct_change)
            
    elif secim == '4':
        print("\nÇıkış yapılıyor...")
        sys.exit(0)
    else:
        print("\nGeçersiz seçim, tekrar deneyin.")

if __name__ == "__main__":
    while True:
        menu()
