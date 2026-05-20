# 📈 Hisse Avcısı — Finansal Analiz ve Yapay Zeka Tabanlı İşlem Sistemi

Hisse Avcısı; BIST, ABD Borsaları ve Kripto para piyasalarında veri çekme, gelişmiş teknik analiz, makine öğrenmesi tahmin modelleri, çoklu ajanlı (multi-agent) karar mekanizmaları ve geriye dönük test (backtest) simülasyonları gerçekleştiren kapsamlı bir finansal yapay zeka platformudur.

Uygulama, modern ve dinamik bir kullanıcı arayüzü sunan **Streamlit** üzerinde çalışmaktadır.

---

## 🚀 Öne Çıkan Özellikler

### 1. Genişletilmiş Teknik Analiz & ASO (Average Sentiment Oscillator)
* **ASO Entegrasyonu:** Mum içi ve dönemsel boğa/ayı baskısını ölçen, kesişim sinyalleri üreten Matriks tabanlı ASO indikatörü.
* **Gelişmiş Göstergeler:** SMA, EMA, MACD, RSI (Wilder), ATR, Bollinger Bandı, OBV, VWAP, ADX, Williams %R, CCI ve VWAP Sapması.
* **Göreli Güç (Relative Strength):** Hisselerin endeks karşısındaki (SPY, XU100, BTC-USD) göreli performans analizi.
* **Hibrit Grafik Arayüzü (TradingView & Lightweight Charts):** ABD Hisseleri ve Kripto paralar için tüm çizim araçlarını barındıran gömülü **TradingView Advanced Chart** entegrasyonu; BIST hisseleri için ise lisans kısıtlamalarına takılmayan yerel **Lightweight Charts** otomatik yedekleme (fallback) motoru.

### 2. Yapay Zeka & Makine Öğrenmesi (Ensemble Model)
* **Hibrit Model Yapısı:** Random Forest, Extra Trees ve LightGBM sınıflandırıcılarını bir arada kullanan karar topluluğu.
* **Walk-Forward Validation:** Zaman serisi analizine uygun, kayan pencereli çapraz doğrulama ile yüksek model güvenilirliği.
* **Dinamik Filtreleme:** Kısa vadeli veri kümelerinde (örneğin 1 yıllık) aşırı eksik veri barındıran indikatörleri otomatik eleyerek eğitim başarısını koruyan akıllı yapı.

### 3. Çoklu Ajan Tartışma Motoru (Multi-Agent Debate Engine)
* Farklı yatırım ekollerinden gelen yapay zeka ajanları piyasa yönünü tartışır:
  * **Büyüme Ajanları (Growth):** Cathie Wood, Phil Fisher, Peter Lynch, Stanley Druckenmiller.
  * **Değer Ajanları (Value):** Warren Buffett, Charlie Munger, Ben Graham, Aswath Damodaran, Michael Burry, Nassim Taleb.
* **Rejim Duyarlı Ağırlıklandırma:** Boğa piyasasında büyüme odaklı ajanların, ayı piyasasında ise değer odaklı korumacı ajanların ağırlığı otomatik olarak artırılır.

### 4. RAG (Retrieval-Augmented Generation) & FinBERT Duygu Analizi
* Google Translate destekli paralel çeviri ve FinBERT modeli ile finansal haber analizleri.
* **RAG Motoru:** Bilanço PDF'leri, KAP bildirimleri ve haberleri vektör deposuna yükleyerek yapay zekaya kaynak göstererek sorgulama imkanı sunar.

### 5. Gelişmiş Backtest Simülasyonu
* **Gerçekçi Maliyetler:** İşlem başına %0.1 komisyon oranı ve %0.05 kayma (slippage) maliyeti dahil edilerek yapılmış gerçekçi portföy simülasyonu.
* **Metrikler:** Toplam getiri, Al-Tut (Buy & Hold) getirisi, Alfa, Sharpe oranı ve Maksimum Çekilme (Max Drawdown) hesaplaması.

---

## 📁 Proje Dosya Yapısı

* `app.py` - Ana Streamlit arayüzü ve dashboard ekranı.
* `stock_analyzer.py` - Teknik analiz hesaplamaları, veri çekme ve ML model eğitimi.
* `backtest_engine.py` - Portföy simülasyonu ve geriye dönük test motoru.
* `ai_integrations.py` - Kronos zaman serisi tahmin modeli ve duygu analizi sinerjisi.
* `reliability_test.py` - Model güvenilirliğini geçmiş kriz (Covid 2020, Bear 2022) dönemlerinde doğrulayan test scripti.
* `test_new_features.py` - RAG, Debate ve diğer 6 yeni özelliği doğrulayan test suite.
* `BASLAT.bat` - Uygulamayı Windows ortamında tek tıkla çalıştıran toplu iş dosyası.
* `.env.example` - Gerekli API anahtarları için şablon dosyası.

---

## 🛠️ Kurulum & Yapılandırma

### 1. Gereksinimler
Sisteminizde Python 3.10 veya üzeri bir sürümün kurulu olduğundan emin olun.

### 2. API Anahtarlarını Ayarlama
1. Proje ana dizinindeki `.env.example` dosyasının bir kopyasını oluşturun ve adını `.env` yapın.
2. `.env` dosyasını bir metin editöründe açarak aşağıdaki ilgili API anahtarlarını girin:

```env
FRED_API_KEY=your_fred_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
FINANCIAL_DATASETS_API_KEY=your_financial_datasets_api_key_here
```

> ⚠️ **UYARI:** `.env` dosyası gizli bilgilerinizi içerir. Bu dosya `.gitignore` listesine eklenmiştir, güvenliğiniz için GitHub'a **yüklemeyiniz**.

---

## 🚀 Çalıştırma

Windows işletim sisteminde projeyi başlatmak son derece kolaydır:

1. Klasördeki **`BASLAT.bat`** dosyasına çift tıklayın.
2. Bu betik sırasıyla şu işlemleri gerçekleştirecektir:
   * Python bağımlılıklarını (`requirements.txt`) otomatik olarak yükleyecek/güncelleyecektir.
   * Nginx proxy katmanını (varsa) kontrol edecek, yoksa otomatik olarak güvenli yerel Streamlit moduna geçecektir.
   * Tarayıcınızda otomatik olarak **`http://127.0.0.1:8501`** adresini açacaktır.

---

## ⚠️ Yasal Uyarı (Disclaimer)

Bu yazılım eğitim ve araştırma amacıyla geliştirilmiştir. Sistem tarafından üretilen tahminler, sinyaller, tartışma sonuçları ve analiz raporları **kesinlikle yatırım tavsiyesi niteliğinde değildir**. Gerçek para ile işlem yapmadan önce kendi araştırmanızı yapınız.
