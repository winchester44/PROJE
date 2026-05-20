import sys
import os
import warnings

# Transformers ve diğer kütüphanelerin terminali kirleten 'Accessing __path__' uyarılarını gizle
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_READ_TIMEOUT", "120")  # Zaman asimi suresini 120 saniyeye cikar
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message="Accessing `__path__` from")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Dashboard ve Polymarket dizinlerini Python Path'e ekle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dashboard_path = os.path.join(BASE_DIR, 'trade', 'Dashboard123-main', 'Dashboard123-main')
pm_path = os.path.join(BASE_DIR, 'trade', 'polymarket-paper-trader-main', 'polymarket-paper-trader-main')
mcp_path = os.path.join(BASE_DIR, 'tradingview-mcp-main', 'src')

for p in [dashboard_path, pm_path, mcp_path]:
    if p not in sys.path:
        sys.path.append(p)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

finbert_offline = os.getenv("FINBERT_OFFLINE_ONLY", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", finbert_offline)
os.environ.setdefault("HF_HUB_OFFLINE", finbert_offline)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from stock_analyzer import (
    detect_market_type,
    fetch_data,
    calculate_technical_indicators,
    get_finbert_sentiment,
    train_ml_model,
    get_risk_management,
    screen_us_stocks,
    screen_bist_stocks,
    screen_tefas_funds,
    add_relative_strength,
)
from backtest_engine import run_vectorized_backtest
from ai_integrations import run_kronos_prediction, run_trading_agents, run_ai_hedge_fund, check_and_push_hot_data

st.set_page_config(page_title="Finansal Yapay Zeka", layout="wide", initial_sidebar_state="expanded")

st.title("📈 Profesyonel Borsa & Kripto Analiz Platformu")
st.markdown("Bu sistem **Makine Öğrenmesi (Random Forest)**, **FinBERT (NLP)** ve **Gelişmiş İndikatörler** kullanarak piyasa verilerini analiz eder.")

# Sidebar
st.sidebar.header("Piyasa Modu")
mode = st.sidebar.radio("Platformu Seçin", [
    "Bireysel Hisse Analizi",
    "🧠 Quant-AI Sinyal Motoru",
    "📈 Backtest Simülatörü",
    "🤖 Gelişmiş AI Hisse Tarama",
    "💼 Portföy Optimizasyonu (Markowitz)",
    "🌍 Makro Ekonomi ve Global",
    "🤖 Otomatik Al-Sat & Paper Trading (Bot)",
    "📊 Performans Raporu (Quantstats)",
    "📱 Telegram Bot Ayarları"
])

if mode == "Bireysel Hisse Analizi":
    st.sidebar.header("Analiz Parametreleri")
    ticker_input = st.sidebar.text_input("Hisse Sembolü (Örn: AAPL, BTC-USD, THYAO)", "AAPL").upper()
    analyze_button = st.sidebar.button("Analizi Başlat", type="primary")

    if analyze_button:
        st.session_state['run_analysis'] = True
        st.session_state['active_ticker'] = ticker_input

    if st.session_state.get('run_analysis', False):
        active_ticker = st.session_state.get('active_ticker', ticker_input)
        try:
            with st.spinner(f"{active_ticker} için geçmiş veriler yükleniyor..."):
                df, final_ticker = fetch_data(active_ticker, period="2y")
                df = calculate_technical_indicators(df)
                df = add_relative_strength(df, final_ticker)   # RS vs benchmark

            _market_type  = detect_market_type(final_ticker)
            current_price = df['Close'].iloc[-1]
            
            # Sıcak Veri (Hot-Data) Kontrolü ve Bildirimi
            try:
                hot_res = check_and_push_hot_data(final_ticker)
                if hot_res.get('has_alert', False):
                    st.toast(hot_res.get('message', ''), icon="⚠️")
                    st.sidebar.warning(hot_res.get('message', ''))
            except Exception:
                pass
            
            # Tabs oluştur
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 ML Tahmini & Grafikler", 
                "📰 FinBERT Yapay Zeka", 
                "⚡ Gerçek Zamanlı (MCP)",
                "🧪 Backtesting & Risk",
                "🌐 Sosyal Medya Radarı",
                "📐 Formasyon AI & Fib",
                "📚 RAG (Belge Okuma)"
            ])

            # ── Aktif sekme takibi (lazy-load guard) ──────────────────────
            # Her sekme ilk açılışta otomatik yüklenir, ancak ağır
            # hesaplamalar session_state kilitleriyle korunur.
            if 'tab_loaded' not in st.session_state:
                st.session_state['tab_loaded'] = set()
            
            with tab1:
                # --- GRAFİK BÖLÜMÜ ---
                st.subheader(f"{final_ticker} - Profesyonel TradingView Grafiği")
                try:
                    from components.lightweight_chart import render_lightweight_chart
                    render_lightweight_chart(df, ticker_symbol=final_ticker, height=550)
                except Exception as e:
                    st.error(f"Grafik yüklenemedi: {e}")
                
                st.divider()
                col1, col2 = st.columns(2)
                
                # --- MAKİNE ÖĞRENMESİ BÖLÜMÜ ---
                with col1:
                    st.subheader("🤖 Yapay Zeka Tahmin Motoru")

                    # Piyasa tipine göre parametreler
                    from stock_analyzer import _MARKET_DEFAULTS
                    _cfg      = _MARKET_DEFAULTS.get(_market_type, _MARKET_DEFAULTS['us'])
                    _horizon  = _cfg['horizon']
                    _thresh   = _cfg['threshold']
                    _mkt_lbl  = {'us': '🇺🇸 ABD', 'bist': '🇹🇷 BIST', 'crypto': '₿ Kripto'}.get(_market_type, '🌐')

                    with st.spinner(f"Walk-forward model eğitiliyor ({_mkt_lbl} modu)..."):
                        up_prob, acc = train_ml_model(df, horizon=_horizon,
                                                      threshold=_thresh,
                                                      market_type=_market_type)

                    # --- 1) Model Güvenilirliği (Walk-Forward OOS) ---
                    st.metric("Walk-Forward OOS Doğruluğu", f"%{acc:.1f}")
                    if acc >= 60:
                        st.markdown("🟢 **Güvenilir model.** Walk-forward testlerde sinyal sağlam.")
                    elif acc >= 50:
                        st.markdown("🟡 **Orta seviye.** Yazı-turadan biraz iyi, garantisi yok.")
                    else:
                        st.markdown("🔴 **Dikkat!** Bu hisse için model tahmin gücü düşük.")
                    st.caption(
                        f"📖 Model {_mkt_lbl} modunda eğitildi. "
                        f"Hedef: **{_horizon} gün içinde %{int(_thresh*100)}+ yükseliş**. "
                        f"Walk-forward OOS (çıkış örneklem) doğruluğu: **%{acc:.1f}**. "
                        f"(%50 = rastgele, %60+ = iyi, %70+ = çok iyi) "
                        f"| Yeni özellikler: ADX, Stochastic, Gap, Volume Spike, RS vs Benchmark"
                    )

                    st.markdown("---")
                    
                    # --- KRONOS FOUNDATION MODEL INTEGRATION ---
                    if st.button("🚀 Kronos Deep Learning (Foundation Model) ile Tahmin Et", key="btn_kronos"):
                        with st.spinner("Kronos K-line modeli yükleniyor ve tahmin yapılıyor... (HuggingFace'den model iniyorsa uzun sürebilir)"):
                            # Sentiment destekli (Kronos + Sentiment Sinerjisi)
                            try:
                                sent_val, _ = get_finbert_sentiment(final_ticker)
                            except:
                                sent_val = 0.0
                            
                            k_prob, k_acc, k_pct = run_kronos_prediction(df, horizon=_horizon, sentiment_score=sent_val)
                            if k_prob is not None:
                                st.success(f"**Kronos Tahmini:** {_horizon} gün içinde beklenen getiri: %{k_pct*100:.2f}")
                                up_prob = k_prob  # Gostergeleri guncellemek icin ez
                            else:
                                st.error(f"Kronos tahmini basarisiz: {k_pct}")

                    st.markdown("---")

                    # --- 2) Hedef Yön Tahmini (5 gün / %2+ eşiği) ---
                    down_prob = 100 - up_prob
                    if up_prob > 60:
                        st.success(f"📈 **{_horizon} GÜNDE %{int(_thresh*100)}+ YÜKSELİŞ** — Olasılık: %{up_prob:.1f}")
                        st.markdown(
                            f"Model, önümüzdeki **{_horizon} gün** içinde bu hissenin "
                            f"**%{int(_thresh*100)}+** yükseleceğini öngörüyor. "
                            f"100 tahminden {up_prob:.0f}'inde eşik aşılırdı."
                        )
                    elif up_prob < 40:
                        st.error(f"📉 **{_horizon} GÜNDE DÜŞÜŞ / YATAY** — Yükseliş Olasılığı: %{up_prob:.1f}")
                        st.markdown(
                            f"Model, {_horizon} gün içinde %{int(_thresh*100)}+ yükseliş "
                            f"için yeterli momentum göremiyor. "
                            f"Yükselme şansı %{up_prob:.1f}, ayı/yatay şans %{down_prob:.1f}."
                        )
                    else:
                        st.warning(f"⚖️ **Kararsız — {_horizon}G Yükseliş: %{up_prob:.1f} / Düşüş/Yatay: %{down_prob:.1f}")
                        st.markdown(
                            f"{_horizon} günlük %{int(_thresh*100)}+ yükseliş için "
                            f"sinyal yeterince güçlü değil. "
                            f"Diğer sekmelerdeki göstergeleri de inceleyin."
                        )
                    st.caption(
                        f"📖 Model; ADX, Stochastic, Gap, Volume Spike, RSI, MACD, SMA, Bollinger "
                        f"ve benchmark'a göre göreli güç (RS) özelliklerini kullanarak "
                        f"{_horizon} günlük %{int(_thresh*100)}+ hareket olasılığını tahmin ediyor."
                    )
                        
                # --- İNDİKATÖR ÖZETLERİ ---
                with col2:
                    st.subheader("📊 Teknik Gösterge Özeti")
                    st.caption("Son işlem gününün kapanış değerleri baz alınmıştır.")
                    
                    rsi_val = df['RSI'].iloc[-1]
                    macd_val = df['MACD'].iloc[-1]
                    macd_signal = df['MACD_Signal'].iloc[-1]
                    macd_hist = macd_val - macd_signal
                    sma20 = df['SMA_20'].iloc[-1]
                    sma50 = df['SMA_50'].iloc[-1]
                    
                    col_i1, col_i2 = st.columns(2)
                    col_i1.metric("Son Fiyat", f"{current_price:.2f}")
                    col_i2.metric("RSI", f"{rsi_val:.1f}")
                    col_i1.metric("MACD", f"{macd_val:.4f}")
                    col_i2.metric("MACD Sinyal", f"{macd_signal:.4f}")
                    col_i1.metric("SMA 20", f"{sma20:.2f}")
                    col_i2.metric("Hacim (OBV)", f"{df['OBV'].iloc[-1]:,.0f}")
                    
                    st.markdown("---")
                    st.markdown("##### 🔍 Gösterge Yorumları")
                    
                    # RSI detaylı yorum
                    if rsi_val > 80:
                        st.error(f"🔴 **RSI = {rsi_val:.1f}** → Aşırı alım! Hisse çok pahalılaşmış, düzeltme riski yüksek.")
                    elif rsi_val > 70:
                        st.warning(f"⚠️ **RSI = {rsi_val:.1f}** → Aşırı alım bölgesine girdi. Dikkatli olun, geri çekilme gelebilir.")
                    elif rsi_val > 55:
                        st.info(f"📊 **RSI = {rsi_val:.1f}** → Normal-pozitif bölge. Alıcılar biraz baskın.")
                    elif rsi_val > 45:
                        st.info(f"⚖️ **RSI = {rsi_val:.1f}** → Nötr bölge. Ne alıcı ne satıcı baskın.")
                    elif rsi_val > 30:
                        st.info(f"📊 **RSI = {rsi_val:.1f}** → Normal-negatif bölge. Satıcılar biraz baskın.")
                    elif rsi_val > 20:
                        st.success(f"💡 **RSI = {rsi_val:.1f}** → Aşırı satım bölgesi! Hisse ucuzlamış, dönüş fırsatı olabilir.")
                    else:
                        st.success(f"🟢 **RSI = {rsi_val:.1f}** → Çok aşırı satım! Hisse çok ucuzlamış, güçlü toparlanma gelebilir.")
                    
                    # MACD detaylı yorum (değerlerle)
                    macd_hist_pct = abs(macd_hist / current_price * 100) if current_price > 0 else 0
                    if macd_hist > 0 and macd_hist_pct > 0.5:
                        st.success(f"📈 **MACD ({macd_val:.4f}) > Sinyal ({macd_signal:.4f})** → Fark: +{macd_hist:.4f}  \n"
                                   f"**Güçlü yukarı momentum.** Alım baskısı belirgin.")
                    elif macd_hist > 0:
                        st.info(f"📈 **MACD ({macd_val:.4f}) > Sinyal ({macd_signal:.4f})** → Fark: +{macd_hist:.4f}  \n"
                                f"**Hafif yukarı momentum.** Alım sinyali zayıf, teyit bekleyin.")
                    elif macd_hist < 0 and macd_hist_pct > 0.5:
                        st.error(f"📉 **MACD ({macd_val:.4f}) < Sinyal ({macd_signal:.4f})** → Fark: {macd_hist:.4f}  \n"
                                 f"**Güçlü aşağı momentum.** Satış baskısı belirgin.")
                    else:
                        st.warning(f"📉 **MACD ({macd_val:.4f}) ≈ Sinyal ({macd_signal:.4f})** → Fark: {macd_hist:.4f}  \n"
                                   f"**Hafif aşağı momentum.** Satış sinyali zayıf, yön değişebilir.")
                    
                    # Fiyat vs SMA pozisyonu
                    if current_price > sma20 and current_price > sma50:
                        st.success(f"📍 Fiyat ({current_price:.2f}) **hem SMA20 ({sma20:.2f}) hem SMA50 ({sma50:.2f}) üstünde** → Yükseliş trendi.")
                    elif current_price > sma20 and current_price < sma50:
                        st.warning(f"📍 Fiyat ({current_price:.2f}) **SMA20 ({sma20:.2f}) üstünde ama SMA50 ({sma50:.2f}) altında** → Kısa vadeli toparlanma, uzun vadeli trend hâlâ aşağı.")
                    elif current_price < sma20 and current_price > sma50:
                        st.warning(f"📍 Fiyat ({current_price:.2f}) **SMA20 ({sma20:.2f}) altında ama SMA50 ({sma50:.2f}) üstünde** → Kısa vadeli zayıflama, uzun vadeli trend hâlâ yukarı.")
                    else:
                        st.error(f"📍 Fiyat ({current_price:.2f}) **hem SMA20 ({sma20:.2f}) hem SMA50 ({sma50:.2f}) altında** → Düşüş trendi.")

                    # ASO (Average Sentiment Oscillator) Göstergesi
                    if 'ASO_Bulls' in df.columns and 'ASO_Bears' in df.columns:
                        st.markdown("---")
                        st.markdown("##### 🔮 ASO (Average Sentiment Oscillator)")
                        aso_bulls_val = df['ASO_Bulls'].iloc[-1]
                        aso_bears_val = df['ASO_Bears'].iloc[-1]
                        aso_diff_val  = df['ASO_Diff'].iloc[-1]
                        aso_cross_val = df['ASO_Cross'].iloc[-1]

                        col_aso1, col_aso2 = st.columns(2)
                        col_aso1.metric("🐂 ASO Boğa Gücü", f"{aso_bulls_val:.1f}")
                        col_aso2.metric("🐻 ASO Ayı Gücü", f"{aso_bears_val:.1f}")

                        if aso_cross_val == 1:
                            st.success(f"🔔 **BULLISH CROSS!** Boğa gücü ({aso_bulls_val:.1f}) ayı gücünü ({aso_bears_val:.1f}) yukarı kesti → ALIŞ sinyali.")
                        elif aso_cross_val == -1:
                            st.error(f"🔔 **BEARISH CROSS!** Ayı gücü ({aso_bears_val:.1f}) boğa gücünü ({aso_bulls_val:.1f}) yukarı kesti → SATIŞ sinyali.")
                        elif aso_diff_val > 5:
                            st.success(f"📈 **Boğalar baskın** (Fark: +{aso_diff_val:.1f}). Alıcı sentimenti güçlü.")
                        elif aso_diff_val > 0:
                            st.info(f"📊 **Hafif boğa baskısı** (Fark: +{aso_diff_val:.1f}). Alıcılar biraz önde.")
                        elif aso_diff_val > -5:
                            st.warning(f"📊 **Hafif ayı baskısı** (Fark: {aso_diff_val:.1f}). Satıcılar biraz önde.")
                        else:
                            st.error(f"📉 **Ayılar baskın** (Fark: {aso_diff_val:.1f}). Satıcı sentimenti güçlü.")

                        st.caption(
                            "📖 ASO, mum-içi (intrabar) ve dönemsel (group) alıcı/satıcı baskısını birleştirerek "
                            "piyasa duyarlılığını ölçer. Bulls > Bears → boğa hakimiyeti, tersi → ayı hakimiyeti."
                        )

                # ── ASO Osilatör Grafiği (Fiyat grafiği altına) ──────────────
                if 'ASO_Bulls' in df.columns and 'ASO_Bears' in df.columns:
                    st.divider()
                    st.subheader("🔮 ASO Osilatör Grafiği")

                    aso_chart_df = df[['ASO_Bulls', 'ASO_Bears', 'ASO_Diff']].dropna().iloc[-120:]

                    fig_aso = go.Figure()
                    fig_aso.add_trace(go.Scatter(
                        x=aso_chart_df.index, y=aso_chart_df['ASO_Bulls'],
                        name='🐂 Boğa Gücü', line=dict(color='#26a69a', width=2)
                    ))
                    fig_aso.add_trace(go.Scatter(
                        x=aso_chart_df.index, y=aso_chart_df['ASO_Bears'],
                        name='🐻 Ayı Gücü', line=dict(color='#ef5350', width=2)
                    ))
                    fig_aso.add_trace(go.Bar(
                        x=aso_chart_df.index, y=aso_chart_df['ASO_Diff'],
                        name='Fark (Bulls-Bears)',
                        marker_color=['rgba(38,166,154,0.4)' if v >= 0 else 'rgba(239,83,80,0.4)' for v in aso_chart_df['ASO_Diff']],
                    ))
                    fig_aso.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                    fig_aso.update_layout(
                        height=300,
                        template='plotly_dark',
                        margin=dict(l=0, r=0, t=10, b=0),
                        yaxis_title='ASO Değeri',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        barmode='overlay'
                    )
                    st.plotly_chart(fig_aso, use_container_width=True)
                    st.caption("Yeşil çizgi boğa gücü, kırmızı çizgi ayı gücü. Kesişim noktaları AL/SAT sinyallerini temsil eder.")

            with tab2:
                st.subheader(f"📰 FinBERT Kurumsal Duygu Analizi (NLP)")
                st.markdown("Hugging Face **ProsusAI/finbert** dil modeli, şirketin son güncel İngilizce haber başlıklarını okuyarak, insanların haberleri nasıl yorumlayacağını matematiksel olarak çıkarır.")

                # ── Lazy-load guard: FinBERT yalnızca buton ile aktifleşir ──
                _finbert_key = f'finbert_{final_ticker}'
                if _finbert_key not in st.session_state:
                    if st.button("📰 FinBERT Analizi Başlat", type="primary", key="btn_finbert_load"):
                        with st.spinner("Hugging Face modeline bağlanılıyor ve haberler yorumlanıyor..."):
                            sentiment_score, news = get_finbert_sentiment(final_ticker)
                        st.session_state[_finbert_key] = {'score': sentiment_score, 'news': news}
                    else:
                        st.info("💡 FinBERT haber analizi ağır bir işlemdir. Başlatmak için yukarıdaki butona tıklayın.")

                if _finbert_key in st.session_state:
                    sentiment_score = st.session_state[_finbert_key]['score']
                    news = st.session_state[_finbert_key]['news']

                    status_text = "Nötr"
                    status_color = "normal"
                    if sentiment_score > 0.1:
                        status_text = "Pozitif"
                    elif sentiment_score < -0.1:
                        status_text = "Negatif"
                        
                    st.metric("Ortalama Haber Skoru (-1 ile +1 arası)", f"{sentiment_score:.2f}", status_text)
                    
                    if news:
                        st.write("### Okunan Haberlerin Detayları:")
                        for idx, article in enumerate(news[:15]):
                            source_badge = f"📍 **Kaynak:** {article.get('source', 'Bilinmiyor')}"
                            with st.expander(f"{idx+1}. {article['title']}"):
                                color = "green" if article['label'] == 'positive' else "red" if article['label'] == 'negative' else "gray"
                                st.markdown(f"**Yapay Zeka Yorumu:** :{color}[**{article['label'].upper()}**] (Etki Gücü Puanı: {abs(article['polarity']):.2f})")
                                st.markdown(source_badge)
                                st.markdown(f"[Haberi Orijinal Kaynakta Oku]({article['link']})")
                    else:
                        st.info("İlgili şirket (veya filtre) için güncel haber veritabanlarında İngilizce şirket haberi bulunamadı.")
                    
                st.divider()
                st.subheader("🕵️‍♂️ Çoklu Yapay Zeka Ajanları (Multi-Agent Analizi)")
                st.markdown("TradingAgents (LangGraph) ve AI Hedge Fund (Persona tabanlı) ile hisseyi sanal bir şirkete analiz ettirebilirsiniz.")
                
                col_ai1, col_ai2 = st.columns(2)
                with col_ai1:
                    if st.button("👥 TradingAgents Şirketini Başlat", use_container_width=True):
                        with st.spinner("Analistler, Trader ve Risk Yöneticisi tartışıyor..."):
                            ta_res = run_trading_agents(final_ticker)
                        st.info("TradingAgents Kararı / Logu:")
                        st.text_area("TA Çıktısı", ta_res, height=300)
                        
                with col_ai2:
                    if st.button("🏦 AI Hedge Fund (Ünlü Yatırımcılar) Başlat", use_container_width=True):
                        with st.spinner("Warren Buffett, Cathie Wood ve Michael Burry analiz ediyor..."):
                            hf_res = run_ai_hedge_fund(final_ticker)
                        st.info("AI Hedge Fund Konsensüsü:")
                        st.text_area("HF Çıktısı", hf_res, height=300)
                        
                st.markdown("---")
                if st.button("🥊 Ajan Tartışması (Multi-Agent Debate) Başlat", use_container_width=True):
                    with st.spinner(f"{final_ticker} için gerçek veriler çekiliyor ve yapay zekalar tartışıyor... (1-2 dakika sürebilir)"):
                        from ai_integrations import run_debate_analysis
                        debate_res = run_debate_analysis(final_ticker)
                        
                        if "error" in debate_res:
                            st.error(f"Hata: {debate_res['error']}")
                        else:
                            # ── Chat Tabanlı Transcript Görselleştirmesi ─────────
                            st.markdown("#### 💬 Ajan Tartışması Kaydı")
                            
                            # Ajan avatar ve renk eşlemeleri
                            AGENT_PROFILES = {
                                "warren_buffett": {"name": "Warren Buffett", "avatar": "🧓", "style": "value"},
                                "buffett": {"name": "Warren Buffett", "avatar": "🧓", "style": "value"},
                                "cathie_wood": {"name": "Cathie Wood", "avatar": "🚀", "style": "growth"},
                                "wood": {"name": "Cathie Wood", "avatar": "🚀", "style": "growth"},
                                "michael_burry": {"name": "Michael Burry", "avatar": "🐻", "style": "contrarian"},
                                "burry": {"name": "Michael Burry", "avatar": "🐻", "style": "contrarian"},
                                "risk_manager": {"name": "Risk Yöneticisi", "avatar": "🛡️", "style": "risk"},
                                "moderator": {"name": "Moderatör", "avatar": "⚖️", "style": "moderator"},
                                "debate_verdict": {"name": "Jüri Kararı", "avatar": "🏛️", "style": "verdict"},
                            }
                            
                            transcript = debate_res.get('transcript', '')
                            if isinstance(transcript, list):
                                # Convert list of round dictionaries to string format expected by app.py
                                formatted_transcript = []
                                for entry in transcript:
                                    round_num = entry.get("round", 1)
                                    
                                    # Format Agent A
                                    agent_a = entry.get("agent_a", "")
                                    agent_a_name = agent_a.replace("_agent", "").replace("_", " ").title()
                                    agent_a_rev = entry.get("agent_a_revised", {})
                                    sig_a = agent_a_rev.get("signal", "neutral").upper()
                                    conf_a = agent_a_rev.get("confidence", 50)
                                    reason_a = agent_a_rev.get("reasoning", "")
                                    
                                    formatted_transcript.append(f"{agent_a_name}: [Round {round_num}] Signal: {sig_a} (Confidence: {conf_a}%)")
                                    formatted_transcript.append(reason_a)
                                    formatted_transcript.append("")
                                    
                                    # Format Agent B
                                    agent_b = entry.get("agent_b", "")
                                    agent_b_name = agent_b.replace("_agent", "").replace("_", " ").title()
                                    agent_b_rev = entry.get("agent_b_revised", {})
                                    sig_b = agent_b_rev.get("signal", "neutral").upper()
                                    conf_b = agent_b_rev.get("confidence", 50)
                                    reason_b = agent_b_rev.get("reasoning", "")
                                    
                                    formatted_transcript.append(f"{agent_b_name}: [Round {round_num}] Signal: {sig_b} (Confidence: {conf_b}%)")
                                    formatted_transcript.append(reason_b)
                                    formatted_transcript.append("")
                                
                                # Add the Consensus Summary
                                consensus_summary = debate_res.get("consensus_summary", "")
                                if consensus_summary:
                                    formatted_transcript.append(f"Jüri Kararı: {consensus_summary}")
                                
                                transcript = "\n".join(formatted_transcript)
                            
                            if transcript:
                                lines = transcript.strip().split('\n')
                                current_agent = None
                                current_text = []
                                
                                def _render_chat_bubble(agent_key, text):
                                    """Tek bir ajan mesajını chat baloncuğu olarak render eder."""
                                    profile = None
                                    for key, prof in AGENT_PROFILES.items():
                                        if key in agent_key.lower():
                                             profile = prof
                                             break
                                    if not profile:
                                        profile = {"name": agent_key.replace("_", " ").title(), "avatar": "🤖", "style": "default"}
                                    
                                    with st.chat_message(name=profile["name"], avatar=profile["avatar"]):
                                        st.markdown(f"**{profile['name']}**\n\n{text}")
                                
                                for line in lines:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    
                                    # "Agent Name:" veya "Agent Name Agent:" formatını yakala
                                    agent_match = None
                                    for key in AGENT_PROFILES:
                                        patterns = [f"{AGENT_PROFILES[key]['name']} Agent:", f"{AGENT_PROFILES[key]['name']}:", f"{key}:"]
                                        for pat in patterns:
                                            if line.lower().startswith(pat.lower()):
                                                agent_match = key
                                                line = line[len(pat):].strip()
                                                break
                                        if agent_match:
                                            break
                                    
                                    if agent_match:
                                        # Önceki ajanın birikmiş metnini render et
                                        if current_agent and current_text:
                                            _render_chat_bubble(current_agent, '\n\n'.join(current_text))
                                        current_agent = agent_match
                                        current_text = [line] if line else []
                                    else:
                                        # Devam satırı — mevcut ajana ekle
                                        if current_agent:
                                            current_text.append(line)
                                        else:
                                            # Ajan tanınmadıysa genel mesaj olarak göster
                                            with st.chat_message(name="Sistem", avatar="📋"):
                                                st.markdown(line)
                                
                                # Son ajanın metnini render et
                                if current_agent and current_text:
                                     _render_chat_bubble(current_agent, '\n\n'.join(current_text))
                            else:
                                st.info("Tartışma kaydı oluşturulamadı.")

                            # ── Konsensüs Verdict Badge (Konuşmanın Altına Yerleştirildi) ──
                            final_sig = debate_res.get('final_signal', 'Bilinmiyor').upper()
                            final_conf = debate_res.get('final_confidence', 0)
                            sig_emoji = "🟢" if final_sig in ("BUY", "BULLISH") else "🔴" if final_sig in ("SELL", "BEARISH") else "🟡"
                            sig_color = "green" if final_sig in ("BUY", "BULLISH") else "red" if final_sig in ("SELL", "BEARISH") else "orange"
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); border-left: 4px solid {sig_color}; 
                                        padding: 16px; border-radius: 8px; margin-top: 16px;">
                                <h3 style="margin:0; color: {sig_color};">{sig_emoji} Ortak Karar: {final_sig}</h3>
                                <p style="margin: 8px 0 0 0; color: #ccc;">Güven Seviyesi: <strong>%{final_conf}</strong></p>
                                <p style="margin: 4px 0 0 0; color: #aaa; font-size: 0.9em;">{debate_res.get('consensus_summary', '')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        
            with tab3:
                st.subheader("⚡ Gerçek Zamanlı Finansal Veriler (MCP)")
                st.markdown("Financial Datasets MCP sunucusu üzerinden kurumsal düzeyde gerçek zamanlı veriler.")
                
                import json
                def safe_json_parse(s):
                    if isinstance(s, dict): return s
                    try: return json.loads(s)
                    except: return {}

                def format_m(val):
                    if val is None or val == "N/A": return "N/A"
                    try:
                        v = float(val)
                        if abs(v) >= 1e12: return f"{v/1e12:.2f}T"
                        if abs(v) >= 1e9: return f"{v/1e9:.2f}B"
                        if abs(v) >= 1e6: return f"{v/1e6:.2f}M"
                        return f"{v:.2f}"
                    except: return str(val)

                if st.button("🔄 Verileri Güncelle (MCP)", key="btn_mcp_refresh"):
                    with st.spinner("Finansal veriler sunucudan çekiliyor..."):
                        from ai_integrations import get_real_time_summary
                        mcp_data = get_real_time_summary(final_ticker)
                        st.session_state[f'mcp_data_{final_ticker}'] = mcp_data
                
                if f'mcp_data_{final_ticker}' in st.session_state:
                    raw_data = st.session_state[f'mcp_data_{final_ticker}']
                    price_data = safe_json_parse(raw_data.get("price", "{}"))
                    metrics_data = safe_json_parse(raw_data.get("metrics", "{}"))
                    estimates_data = safe_json_parse(raw_data.get("estimates", "[]"))
                    
                    # 1. PRICE SECTION
                    if price_data:
                        st.markdown("#### 💰 Anlık Piyasa Fiyatı")
                        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                        p_col1.metric("Fiyat", f"${price_data.get('price', 0):.2f}")
                        p_col2.metric("Günlük Değişim ($)", f"{price_data.get('day_change', 0):+.2f}")
                        p_col3.metric("Günlük Değişim (%)", f"%{price_data.get('day_change_percent', 0):+.2f}")
                        p_col4.metric("Hacim", format_m(price_data.get('volume', 0)))
                    
                    st.divider()
                    
                    # 2. METRICS SECTION
                    m_tab1, m_tab2, m_tab3 = st.tabs(["📊 Değerleme", "📈 Büyüme & Karlılık", "🛡️ Sağlamlık"])
                    
                    with m_tab1:
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.write("**Piyasa Değeri:**", format_m(metrics_data.get("market_cap")))
                            st.write("**Enterprise Value:**", format_m(metrics_data.get("enterprise_value")))
                            st.write("**P/E Ratio (F/K):**", f"{metrics_data.get('pe_ratio', 'N/A')}")
                        with col_m2:
                            st.write("**P/S Ratio:**", f"{metrics_data.get('price_to_sales_ratio', 'N/A')}")
                            st.write("**P/B Ratio:**", f"{metrics_data.get('price_to_book_ratio', 'N/A')}")
                            st.write("**EV/EBITDA:**", f"{metrics_data.get('enterprise_value_to_ebitda_ratio', 'N/A')}")
                    
                    with m_tab2:
                        col_g1, col_g2 = st.columns(2)
                        with col_g1:
                            st.write("**Revenue Growth:**", f"%{metrics_data.get('revenue_growth', 0)*100:.2f}")
                            st.write("**Gross Margin:**", f"%{metrics_data.get('gross_margin', 0)*100:.2f}")
                        with col_g2:
                            st.write("**Net Margin:**", f"%{metrics_data.get('net_margin', 0)*100:.2f}")
                            st.write("**Return on Equity (ROE):**", f"%{metrics_data.get('return_on_equity', 0)*100:.2f}")

                    with m_tab3:
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.write("**Current Ratio:**", f"{metrics_data.get('current_ratio', 'N/A')}")
                            st.write("**Debt to Equity:**", f"{metrics_data.get('debt_to_equity_ratio', 'N/A')}")
                        with col_s2:
                            st.write("**Dividend Yield:**", f"%{metrics_data.get('dividend_yield', 0)*100:.2f}")
                            st.write("**Payout Ratio:**", f"%{metrics_data.get('payout_ratio', 0)*100:.2f}")

                    st.divider()
                    
                    # 3. ESTIMATES SECTION
                    if estimates_data:
                        st.markdown("#### 📈 Analist Beklentileri (Consensus)")
                        est_df = pd.DataFrame(estimates_data)
                        if not est_df.empty:
                            # Sütun isimlerini güzelleştir
                            rename_cols = {
                                "fiscal_period": "Dönem",
                                "period": "Tip",
                                "revenue": "Beklenen Gelir",
                                "earnings_per_share": "Beklenen EPS"
                            }
                            est_df = est_df.rename(columns=rename_cols)
                            # Geliri formatla
                            if 'Beklenen Gelir' in est_df.columns:
                                est_df['Beklenen Gelir'] = est_df['Beklenen Gelir'].apply(format_m)
                            st.dataframe(est_df, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # 4. INSIDER TRADES
                    if st.button("📑 Son Insider İşlemlerini Getir", key="btn_insider"):
                        with st.spinner("Insider işlemleri taranıyor..."):
                            from ai_integrations import run_financial_mcp_query
                            insider_raw = run_financial_mcp_query("get_insider_trades", {"ticker": final_ticker})
                            insider_data = safe_json_parse(insider_raw.get("result", "[]"))
                            
                            if insider_data and isinstance(insider_data, list):
                                st.write("### Insider İşlemleri:")
                                i_df = pd.DataFrame(insider_data)
                                st.dataframe(i_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("Insider işlemi bulunamadı veya veri formatı uyumsuz.")
                else:
                    st.info("Verileri görüntülemek için yukarıdaki butona tıklayın.")

            with tab4:
                st.subheader("🧪 Profesyonel Backtesting Motoru")
                st.markdown("6 farklı profesyonel strateji ile geriye dönük test. Komisyon ve kayma (slippage) dahil gerçekçi simülasyon.")
                
                try:
                    from tradingview_mcp.core.services.backtest_service import (
                        run_backtest, compare_strategies, detect_market_regime,
                        check_parameter_stability, multi_asset_backtest
                    )
                    
                    # 1. Market Regime Analysis
                    with st.spinner("Piyasa Rejimi (Trend/Yatay) hesaplanıyor..."):
                        regime_data = detect_market_regime(final_ticker)
                    
                    if "error" not in regime_data:
                        st.info(f"📊 **Piyasa Analizi (ADX):** Şuan **{regime_data['regime']}** hakim. {regime_data['description']}")
                        st.success(f"💡 **Tavsiye Edilen Stratejiler:** {regime_data['suggested_strategies']}")
                    
                    st.divider()
                    
                    # 2. Test Ayarları (Komisyon / Slipaj)
                    st.markdown("#### ⚙️ Gerçekçi Test Ayarları")
                    set_col1, set_col2 = st.columns(2)
                    with set_col1:
                        user_comm = st.number_input("İşlem Komisyonu (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05)
                    with set_col2:
                        user_slip = st.number_input("Slipaj / Kayma (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05)
                        
                    st.divider()
                    
                    @st.cache_data
                    def get_cached_backtest(ticker, strat, period, comm, slip):
                        return run_backtest(ticker, strat, period=period, commission_pct=comm, slippage_pct=slip, include_equity_curve=True, include_trade_log=True)
                    
                    @st.cache_data
                    def get_cached_comparison(ticker, period, comm, slip):
                        return compare_strategies(ticker, period=period, commission_pct=comm, slippage_pct=slip)
                    
                    bt_col1, bt_col2 = st.columns(2)
                    with bt_col1:
                        bt_mode = st.radio("Test Modu", ["Tek Strateji", "6 Strateji Karşılaştır", "Sepet (Multi-Asset) Test"], horizontal=True, key="bt_mode_radio")
                    with bt_col2:
                        bt_period = st.selectbox("Dönem", ["6mo", "1y", "2y"], index=1, format_func=lambda x: {"6mo": "6 Ay", "1y": "1 Yıl", "2y": "2 Yıl"}[x], key="bt_period_select")
                    
                    strategy_names = {
                        "rsi": "RSI (Aşırı Alım/Satım)",
                        "bollinger": "Bollinger Bandı",
                        "macd": "MACD Çaprazlama",
                        "ema_cross": "EMA 20/50 Golden Cross",
                        "supertrend": "Supertrend (ATR Trend)",
                        "donchian": "Donchian Kanalı (Turtle)"
                    }
                    
                    if bt_mode == "6 Strateji Karşılaştır":
                        if st.button("🚀 6 Stratejiyi Karşılaştır", type="primary", key="btn_compare"):
                            with st.spinner(f"{final_ticker} için stratejiler test ediliyor..."):
                                cmp = get_cached_comparison(final_ticker, period=bt_period, comm=user_comm, slip=user_slip)
                            st.session_state['bt_compare_result'] = cmp
                            st.session_state['bt_compare_ticker'] = final_ticker
                        
                        # Sonuçları session_state'den göster
                        if 'bt_compare_result' in st.session_state and st.session_state.get('bt_compare_ticker') == final_ticker:
                            cmp = st.session_state['bt_compare_result']
                            if 'error' in cmp:
                                st.error(cmp['error'])
                            else:
                                st.success(f"📊 {cmp['candles_analyzed']} mum analiz edildi ({cmp['date_from']} → {cmp['date_to']})")
                                
                                # Sonuç tablosu
                                for s in cmp['ranking']:
                                    emoji = "🥇" if s['rank'] == 1 else "🥈" if s['rank'] == 2 else "🥉" if s['rank'] == 3 else "  "
                                    with st.expander(f"{emoji} #{s['rank']} {strategy_names.get(s['strategy'], s['strategy_label'])} — Getiri: %{s['total_return_pct']}", expanded=(s['rank'] <= 3)):
                                        m1, m2, m3, m4 = st.columns(4)
                                        m1.metric("Toplam Getiri", f"%{s['total_return_pct']}")
                                        m2.metric("Sharpe Oranı", f"{s['sharpe_ratio']}")
                                        m3.metric("Win Rate", f"%{s['win_rate_pct']}")
                                        m4.metric("İşlem Sayısı", f"{s['total_trades']}")
                                        
                                        m5, m6, m7 = st.columns(3)
                                        m5.metric("Max Drawdown", f"%{s['max_drawdown_pct']}")
                                        m6.metric("Profit Factor", f"{s['profit_factor']}")
                                        m7.metric("Sortino Oranı", f"{s.get('sortino_ratio', 0)}")
                                
                                st.divider()
                                st.metric("📈 Buy & Hold (Al ve Tut) Getirisi", f"%{cmp['buy_and_hold_return_pct']}")
                                st.caption("Buy & Hold = Hiçbir strateji kullanmadan başta alıp sona kadar tutmak.")
                                
                                if cmp['ranking'][0]['total_return_pct'] > cmp['buy_and_hold_return_pct']:
                                    st.success(f"🏆 En iyi strateji ({strategy_names.get(cmp['winner'], cmp['winner'])}) Buy & Hold'dan daha iyi performans gösterdi!")
                                else:
                                    st.info("📊 Hiçbir strateji basit Buy & Hold'u geçemedi. Bazen en iyi strateji hiçbir şey yapmamaktır.")
                    
                    elif bt_mode == "Tek Strateji":
                        selected_strategy = st.selectbox("Strateji Seç", list(strategy_names.keys()), format_func=lambda x: strategy_names[x], key="bt_strategy_select")
                        
                        if st.button("🧪 Backtest Çalıştır", type="primary", key="btn_single_bt"):
                            with st.spinner(f"{final_ticker} için {strategy_names[selected_strategy]} test ediliyor..."):
                                result = get_cached_backtest(final_ticker, selected_strategy, period=bt_period, comm=user_comm, slip=user_slip)
                            st.session_state['bt_single_result'] = result
                            st.session_state['bt_single_ticker'] = final_ticker
                        
                        # Sonuçları session_state'den göster
                        if 'bt_single_result' in st.session_state and st.session_state.get('bt_single_ticker') == final_ticker:
                            result = st.session_state['bt_single_result']
                            if 'error' in result:
                                st.error(result['error'])
                            else:
                                st.success(f"📊 {result['candles_analyzed']} mum analiz edildi ({result['date_from']} → {result['date_to']})")
                                
                                # Warnings & Confidence
                                if result.get('min_trade_warning'):
                                    st.warning(f"⚠️ {result['min_trade_warning']} - Bu strateji çok az işleme girmiş.")
                                
                                conf = result.get('confidence_score_label', '')
                                color = "green" if "Yeşil" in conf else "orange" if "Sarı" in conf else "red"
                                st.markdown(f"**Sistem Güven Skoru:** :{color}[**{conf}**]")
                                
                                # Ana metrikler
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("Toplam Getiri", f"%{result['total_return_pct']}")
                                m2.metric("Son Bakiye", f"${result['final_capital']:,.2f}")
                                m3.metric("Win Rate", f"%{result['win_rate_pct']}")
                                m4.metric("İşlem Sayısı", f"{result['total_trades']}")
                                
                                m5, m6, m7, m8 = st.columns(4)
                                m5.metric("Sharpe Oranı", f"{result['sharpe_ratio']}")
                                m6.metric("Sortino Oranı", f"{result['sortino_ratio']}")
                                m7.metric("Max Drawdown", f"%{result['max_drawdown_pct']}")
                                m8.metric("Ort. İşlem Süresi (Gün)", f"{result['avg_trade_duration']}")
                                
                                m9, m10, m11, m12 = st.columns(4)
                                m9.metric("Profit Factor", f"{result['profit_factor']}")
                                m10.metric("Maks Ardışık Kazanç", f"{result.get('max_consecutive_wins', 0)}")
                                m11.metric("Maks Ardışık Kayıp", f"{result.get('max_consecutive_losses', 0)}")
                                m12.metric("Buy & Hold", f"%{result['buy_and_hold_return_pct']}")
                                
                                vs_bnh = result['vs_buy_and_hold_pct']
                                if vs_bnh > 0:
                                    st.success(f"🏆 Bu strateji Buy & Hold'dan **%{vs_bnh}** daha iyi!")
                                elif vs_bnh < 0:
                                    st.warning(f"📉 Bu strateji Buy & Hold'dan **%{abs(vs_bnh)}** daha kötü.")
                                    
                                # Parametre Overfitting Kontrolü
                                st.markdown("##### 🔍 Parametre Stabilite Analizi (Overfitting Kontrolü)")
                                if st.button("Overfit Kontrolü Yap", key="btn_overfit"):
                                    with st.spinner("Varyasyon testleri koşturuluyor..."):
                                        stb = check_parameter_stability(final_ticker, selected_strategy, period=bt_period, commission_pct=user_comm, slippage_pct=user_slip)
                                        if "error" in stb:
                                            st.error(stb["error"])
                                        else:
                                            st.info(f"**Stabilite Skoru:** {stb['stability_score']} / 100")
                                            st.write(f"**Sonuç:** {stb['warning']}")
                                            st.caption(f"Örneklem Varyansı: {stb['std_deviation']} | {stb['variations_tested']} farklı versiyon test edildi.")
                                
                                # Equity Curve
                                if 'equity_curve' in result and len(result['equity_curve']) > 1:
                                    st.markdown("---")
                                    st.markdown("##### 📈 Portföy Değeri (Equity Curve)")
                                    eq_data = result['equity_curve']
                                    eq_df = pd.DataFrame(eq_data[1:])  # skip 'start'
                                    if not eq_df.empty:
                                        fig_eq = go.Figure()
                                        fig_eq.add_trace(go.Scatter(x=eq_df['date'], y=eq_df['equity'], fill='tozeroy', fillcolor='rgba(38,166,154,0.15)', line=dict(color='#26a69a', width=2), name='Portföy'))
                                        fig_eq.update_layout(height=300, template='plotly_dark', margin=dict(l=0,r=0,t=10,b=0), yaxis_title='Portföy ($)')
                                        st.plotly_chart(fig_eq, use_container_width=True)
                                
                                # Trade Log
                                if 'trade_log' in result and result['trade_log']:
                                    st.markdown("##### 📋 İşlem Geçmişi")
                                    log_df = pd.DataFrame(result['trade_log'])
                                    display_cols = ['trade_no', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 'holding_days', 'return_pct', 'capital_after']
                                    col_names = {'trade_no': '#', 'entry_date': 'Giriş', 'entry_price': 'Giriş Fiyatı', 'exit_date': 'Çıkış', 'exit_price': 'Çıkış Fiyatı', 'holding_days': 'Gün', 'return_pct': 'Getiri %', 'capital_after': 'Sermaye'}
                                    st.dataframe(log_df[display_cols].rename(columns=col_names), use_container_width=True, hide_index=True)

                    elif bt_mode == "Sepet (Multi-Asset) Test":
                        st.markdown("Seçilen stratejinin farklı varlıklarda eşzamanlı olarak nasıl çalıştığını gösterir.")
                        selected_strategy = st.selectbox("Strateji Seç", list(strategy_names.keys()), format_func=lambda x: strategy_names[x], key="bt_strat_multi")
                        
                        if st.button("🌐 Sinerji Testi Çalıştır", type="primary"):
                            default_basket = [final_ticker, "BTC-USD", "SPY", "GC=F"]
                            with st.spinner("Tüm varlıklar için test ediliyor..."):
                                mres = multi_asset_backtest(default_basket, selected_strategy, period=bt_period, commission_pct=user_comm, slippage_pct=user_slip)
                                
                            if "error" in mres:
                                st.error(mres["error"])
                            else:
                                st.success(f"Varlıkların **%{mres['successful_assets_pct']}** oranında kârlılık sağlandı. Ortalama Getiri: %{mres['avg_return_pct']}")
                                st.table(pd.DataFrame(mres['results']).rename(columns={
                                    "symbol": "Sembol", "return_pct": "Getiri %", "win_rate": "Win Rate %", 
                                    "sharpe": "Sharpe", "trades": "İşlem", "confidence": "Sistem Güveni"
                                }))

                
                except ImportError:
                    st.error("tradingview-mcp modülü yüklenemedi. 'tradingview-mcp-main/src' dizini kontrol edin.")
                except Exception as e:
                    st.error(f"Backtest Hatası: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                
                st.divider()
                
                # Risk Yönetimi (mevcut ATR bazlı)
                st.subheader("🛡️ Risk Yönetimi (ATR Bazlı)")
                st.markdown("Piyasanın gerçek oynaklığına (ATR) dayalı stop-loss hedefleri:")
                
                current_atr = df['ATR'].iloc[-1]
                sl, tp = get_risk_management(current_price, current_atr)
                
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.error("📉 ZARAR KES (STOP-LOSS)")
                    st.markdown(f"<h2>{sl:.2f}</h2>", unsafe_allow_html=True)
                    st.caption(f"Fiyat {sl:.2f} altına düşerse pozisyonu kapat.")
                    
                with cc2:
                    st.success("🎯 KAR AL (TAKE-PROFIT)")
                    st.markdown(f"<h2>{tp:.2f}</h2>", unsafe_allow_html=True)
                    st.caption(f"Fiyat {tp:.2f} seviyesine ulaşırsa karı realize et.")

            with tab5:
                st.subheader("🌐 Sosyal Medya Radarı (Reddit / X Sentimental)")
                st.markdown("API sınırları sebebiyle **Simüle Edilmiş (Mock)** anlık duygu durum analizini ve **Troll/Bot Tespit** filtresini görüntülüyorsunuz. (Trend yönü ve anomali tespiti hisse formasyonundan güç alır).")

                # ── Lazy-load guard: Sosyal analiz sadece butonla çalışır ──
                _social_key = f'social_{final_ticker}'
                if _social_key not in st.session_state:
                    if st.button("🌐 Sosyal Medya Taramasını Başlat", type="primary", key="btn_social_load"):
                        with st.spinner("Sosyal platformlar taranıyor..."):
                            from stock_analyzer import get_social_sentiment_mock
                            social_data = get_social_sentiment_mock(final_ticker, current_price, rsi_val)
                        st.session_state[_social_key] = social_data
                    else:
                        st.info("💡 Sosyal medya verilerini yüklemek için butona tıklayın.")

                if _social_key in st.session_state:
                    social_data = st.session_state[_social_key]

                    st.info(social_data["warning"])
                    
                    s_col1, s_col2, s_col3 = st.columns(3)
                    s_col1.metric("🔥 Hype (İlgi) Skoru", f"{social_data['hype_score']} / 100")
                    s_col2.metric("🤖 Veri Güvenilirliği (Troll Filtresi)", f"%{social_data['trust_score']}")
                    s_col3.metric("📉 Sosyal Medya Trendi", social_data["trend"])
                    
                    st.markdown("#### 💬 Son Yorumlar (Canlı Akış Simülasyonu)")
                    for post in social_data["posts"]:
                        with st.container():
                            p1, p2 = st.columns([1, 6])
                            with p1:
                                if post["color"] == "green":
                                    st.success("BULLISH 🟢")
                                elif post["color"] == "red":
                                    st.error("BEARISH 🔴")
                                else:
                                    st.warning("SCAM/TROLL ⚠️")
                            with p2:
                                st.markdown(f"**@{post['user']}**: {post['body']}")
                                st.caption("---")

            with tab6:
                st.subheader("📐 Formasyon AI & Fibonacci Destekleri")
                st.markdown("Bilgisayarlı Görü (Computer Vision) mantığıyla tepe/dip Pivot noktaları bulunur ve formasyon riskleri tespit edilir.")

                # ── Lazy-load guard: Formasyon/Fibonacci ───────────────────
                _fib_key = f'fib_{final_ticker}'
                if _fib_key not in st.session_state:
                    if st.button("📐 Formasyon Analizi Başlat", type="primary", key="btn_fib_load"):
                        with st.spinner("Geometrik formasyonlar çözümleniyor..."):
                            from stock_analyzer import calculate_fibonacci_and_patterns
                            fib_levels, patterns = calculate_fibonacci_and_patterns(df)
                        st.session_state[_fib_key] = {'levels': fib_levels, 'patterns': patterns}
                    else:
                        st.info("💡 Formasyon ve Fibonacci analizi için butona tıklayın.")

                if _fib_key in st.session_state:
                    fib_levels = st.session_state[_fib_key]['levels']
                    patterns = st.session_state[_fib_key]['patterns']

                    col_p1, col_p2 = st.columns([2, 1])
                    with col_p1:
                        st.markdown("#### 🚨 Formasyon Sinyalleri (SciPy ArgRelExtrema)")
                        for pat in patterns:
                            if "⚠️" in pat or "🚨" in pat:
                                st.error(pat)
                            elif "📈" in pat:
                                st.success(pat)
                            else:
                                st.info(pat)
                                
                    with col_p2:
                        st.markdown("#### 📏 Fibonacci Seviyeleri")
                        for label, price in fib_levels.items():
                            if "Dip" in label or "Tepe" in label:
                                st.markdown(f"**{label}:** {price:.2f}")
                            else:
                                st.markdown(f"- {label}: {price:.2f}")
                            
            with tab7:
                st.subheader("📚 RAG (Belge Okuma) Yapay Zekası")
                st.markdown("Piyasa raporlarını, bilanço PDF'lerini veya KAP bildirimlerini sizin için okur ve sorularınızı cevaplar.")
                
                st.info(f"💡 Şuan aktif hisse: **{final_ticker}**. RAG veritabanında bu hisseye ait güncel verilerin bulunabilmesi için aşağıdaki butonu kullanarak verileri güncelleyebilirsiniz.")
                
                if st.button(f"🔄 İnternetten Güncel Verileri Çek ve RAG'a Ekle ({final_ticker})", type="secondary", key="btn_rag_fetch"):
                    with st.spinner(f"{final_ticker} için finansal tablolar ve haberler (Yahoo, KAP, Midas, TradingView, Investing) çekiliyor..."):
                        from ai_integrations import fetch_and_prepare_rag_docs
                        res_msg = fetch_and_prepare_rag_docs(final_ticker)
                        st.success(res_msg)
                
                st.divider()
                
                rag_query = st.text_input("Belgelerle ilgili sorunuzu sorun (Örn: Şirketin brüt kar marjı nedir?):")
                if st.button("Belgelerde Ara", type="primary", key="btn_rag"):
                    if rag_query:
                        with st.spinner("Dokümanlar taranıyor ve yapay zeka cevaplıyor..."):
                            from ai_integrations import run_rag_query
                            rag_cevap = run_rag_query(rag_query, ticker=final_ticker)
                            st.success("🤖 Yapay Zeka Yanıtı:")
                            st.write(rag_cevap)
                    else:
                        st.warning("Lütfen bir soru girin.")
                        
        except Exception as e:
            st.error(f"Sistem Hatası: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

elif mode == "💼 Portföy Optimizasyonu (Markowitz)":
    st.header("💼 Modern Portföy Optimizasyonu (MPT)")
    st.markdown("Bu modül, girdiğiniz hisse sepetindeki varlıklar için binlerce simülasyon koşturarak, **kârınızı maksimize edip riskinizi (varyans) minimize eden** en optimum % dağılımlarını hesaplar.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sepet Seçimi")
    t1 = st.sidebar.text_input("Varlık 1", "AAPL")
    t2 = st.sidebar.text_input("Varlık 2", "TSLA")
    t3 = st.sidebar.text_input("Varlık 3", "MSFT")
    t4 = st.sidebar.text_input("Varlık 4 (Opsiyonel)", "BTC-USD")
    t5 = st.sidebar.text_input("Varlık 5 (Opsiyonel)", "")
    
    calc_btn = st.sidebar.button("Sepeti Optimize Et", type="primary")
    
    if calc_btn:
        tickers = [t.strip().upper() for t in [t1, t2, t3, t4, t5] if t.strip()]
        if len(tickers) < 2:
            st.warning("Lütfen portföy analizi için en az 2 geçerli sembol girin.")
        else:
            with st.spinner("Yıllık getiriler çekiliyor, kovaryans matrisleri hesaplanıyor ve 5,000 Monte Carlo senaryosu çalıştırılıyor..."):
                from stock_analyzer import calculate_portfolio_optimization
                opt_res = calculate_portfolio_optimization(tickers, period="1y")
                
            if "error" in opt_res:
                st.error(opt_res["error"])
            else:
                st.success("5,000 Portföy Varyasyonu Başarıyla Test Edildi!")
                
                # Plotly Chart
                import plotly.graph_objects as go
                results = opt_res["results"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results[1,:], # Volatility
                    y=results[0,:], # Return
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=results[2,:], # Sharpe ratio for color scale
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Sharpe Ratio")
                    ),
                    name='Rastgele Dağılımlar'
                ))
                
                # Sharpe Max and Min Volatility Highlights
                max_sharpe = opt_res["max_sharpe"]
                min_vol = opt_res["min_vol"]
                
                fig.add_trace(go.Scatter(
                    x=[max_sharpe["volatility"]],
                    y=[max_sharpe["return"]],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='star'),
                    name='Max Sharpe Oranı'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[min_vol["volatility"]],
                    y=[min_vol["return"]],
                    mode='markers',
                    marker=dict(size=12, color='blue', symbol='star'),
                    name='Minimum Risk (Varyans)'
                ))
                
                fig.update_layout(
                    title="Etkin Sınır (Efficient Frontier)",
                    xaxis_title="Yıllıklandırılmış Risk (Volatilite)",
                    yaxis_title="Yıllıklandırılmış Beklenen Getiri",
                    template='plotly_dark',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                w_col1, w_col2 = st.columns(2)
                
                with w_col1:
                    st.error("🏆 Maksimum Sharpe Portföyü (En İyi Risk/Getiri Oranı)")
                    st.metric("Beklenen Yıllık Getiri", f"%{max_sharpe['return']*100:.2f}")
                    st.metric("Beklenen Risk", f"%{max_sharpe['volatility']*100:.2f}")
                    
                    st.markdown("#### ⚖️ İdeal Dağılım (%):")
                    for sym, weight in max_sharpe["weights"].items():
                        st.write(f"- **{sym}**: %{weight*100:.1f}")
                        
                with w_col2:
                    st.info("🛡️ Minimum Varyans Portföyü (En Düşük Oynaklık/Risk)")
                    st.metric("Beklenen Yıllık Getiri", f"%{min_vol['return']*100:.2f}")
                    st.metric("Beklenen Risk", f"%{min_vol['volatility']*100:.2f}")
                    
                    st.markdown("#### ⚖️ Korumacı Dağılım (%):")
                    for sym, weight in min_vol["weights"].items():
                        st.write(f"- **{sym}**: %{weight*100:.1f}")

elif mode == "🤖 Gelişmiş AI Hisse Tarama":
    st.header("🤖 Yapay Zeka Destekli Filtreleme (Screener)")
    st.markdown("Piyasadaki hisseleri anlık olarak filtreler ve içlerinden yatırıma en uygun olanları (RSI şişmemiş, ML Yükseliş ihtimali > %60 olanlar) çıkarır.")
    
    t_tab1, t_tab2, t_tab3 = st.tabs(["🇺🇸 ABD Piyasası", "🇹🇷 Borsa İstanbul (BİST)", "🏦 TEFAS Fonları"])
    
    with t_tab1:
        st.subheader("ABD Hisse Senedi Piyasası")
        us_price_filter = st.selectbox("Fiyat Filtresi (Penny Stocks / Düşük Ücretli vs.)", 
            ["Any", "Under $1", "Under $5", "Under $10", "Under $20", "Under $50", "Over $10", "Over $50"])
            
        us_sector_filter = st.selectbox("Şirketin Faaliyet Sektörü", [
            "Any", "Basic Materials (Hammadde)", "Communication Services (İletişim)", 
            "Consumer Cyclical (Döngüsel Tüketim)", "Consumer Defensive (Gıda/Sabit Tüketim)", 
            "Energy (Enerji)", "Financial (Finans)", "Healthcare (Sağlık)", 
            "Industrials (Sanayi)", "Real Estate (Emlak)", "Technology (Teknoloji)", "Utilities (Altyapı)"
        ])
        
        us_min_prob = st.slider("Yapay Zeka Minimum Yükseliş İhtimali (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        
        scan_us_btn = st.button("🇺🇸 ABD Piyasasını Tara", type="primary")
        if scan_us_btn:
            # Finviz arka planı için Türkçe detayı kes (: "Technology (Teknoloji)" -> "Technology")
            raw_sector = us_sector_filter.split(" (")[0]
            with st.spinner("Finviz üzerinden piyasa filtreleri uygulanıyor ve model koşturuluyor... (1-2 dakika sürebilir)"):
                champions, scan_errors = screen_us_stocks(price_filter=us_price_filter, sector_filter=raw_sector, min_prob=us_min_prob)
            
            if scan_errors:
                st.caption("Tarama sirasinda atlanan veya veri yetersizligi nedeniyle elenen bazi semboller oldu. Ilk birkaci: " + " | ".join(scan_errors[:3]))

            if champions:
                st.success(f"Kriterleri ve AI Elemesini Geçen **{len(champions)}** Hisse Bulundu!")
                for champ in champions:
                    with st.expander(f"⭐ {champ['Ticker']} — Fiyat: ${champ['Price']:.2f} | ML Yükseliş: %{champ['Prob']:.1f} | RSI: {champ['RSI']:.1f}", expanded=False):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Yükseliş İhtimali", f"%{champ['Prob']:.1f}")
                        col_b.metric("RSI", f"{champ['RSI']:.1f}")
                        col_c.metric("Fiyat", f"${champ['Price']:.2f}")
                        
                        if champ['Prob'] >= 60:
                            st.success(f"📈 Model bu hisse için **güçlü yükseliş** sinyali veriyor.")
                        elif champ['Prob'] >= 40:
                            st.info(f"⚖️ Model kararsız, dikkatli olun.")
                        else:
                            st.warning(f"📉 Düşük olasılık, model yükseliş beklemiyor.")
            else:
                st.warning("Bu kriterlerde uygun ABD hissesi bulunamadı. Fiyat filtresini veya minimum yükseliş ihtimalini düşürmeyi deneyin.")

    with t_tab2:
        st.subheader("Borsa İstanbul (BİST) Piyasası")
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            bist_index = st.selectbox("Tarama Endeksi", ["XU100", "XU050", "XU030", "XKTUM", "XBANK", "XUSIN"], index=0)
            bist_max_price = st.number_input("Maksimum Hisse Fiyatı (TL)", min_value=1.0, value=500.0, step=10.0)
            bist_min_prob = st.slider("Minimum Yükseliş İhtimali (%)", min_value=0.0, max_value=100.0, value=52.0, step=1.0)
        
        with col_b2:
            pe_max = st.number_input("Maksimum F/K Oranı (Opsiyonel)", min_value=0.0, value=0.0, step=1.0, help="0 = Filtre kapalı")
            pe_val = pe_max if pe_max > 0 else None
            
            mcap_min = st.number_input("Minimum Piyasa Değeri (Milyon TL) (Opsiyonel)", min_value=0.0, value=0.0, step=100.0, help="0 = Filtre kapalı")
            mcap_val = mcap_min if mcap_min > 0 else None
            
        scan_bist_btn = st.button("🇹🇷 BİST Piyasasını Tara", type="primary")
        if scan_bist_btn:
            with st.spinner(f"BORSAPY: {bist_index} bileşenleri taranıyor, temel ve teknik analiz yapılıyor..."):
                champions, scan_errors = screen_bist_stocks(
                    index_name=bist_index, 
                    max_price=bist_max_price, 
                    min_prob=bist_min_prob,
                    pe_max=pe_val,
                    market_cap_min=mcap_val
                )
            
            if scan_errors:
                st.caption("Tarama sirasinda atlanan veya veri yetersizligi nedeniyle elenen bazi semboller oldu. Ilk birkaci: " + " | ".join(scan_errors[:3]))

            if champions:
                st.success(f"Kriterleri ve AI Elemesini Geçen **{len(champions)}** Hisse Bulundu!")
                for champ in champions:
                    with st.expander(f"⭐ {champ['Ticker']} — Fiyat: {champ['Price']:.2f} TL | ML Yükseliş: %{champ['Prob']:.1f} | RSI: {champ['RSI']:.1f}", expanded=False):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Yükseliş İhtimali", f"%{champ['Prob']:.1f}")
                        col_b.metric("RSI", f"{champ['RSI']:.1f}")
                        col_c.metric("Fiyat", f"{champ['Price']:.2f} TL")
                        
                        if "Borsapy" in champ:
                            b_data = champ["Borsapy"]
                            st.markdown("###### Borsapy Temel Analiz")
                            b_c1, b_c2 = st.columns(2)
                            b_c1.metric("Son Temettü", b_data.get("Son Temettü", "-"))
                            b_c2.metric("Yabancı Oranı", b_data.get("Yabancı Oranı", "-"))
                        
                        if champ['Prob'] >= 60:
                            st.success(f"📈 Model bu hisse için **güçlü yükseliş** sinyali veriyor.")
                        elif champ['Prob'] >= 40:
                            st.info(f"⚖️ Model kararsız, dikkatli olun.")
                        else:
                            st.warning(f"📉 Düşük olasılık, model yükseliş beklemiyor.")
            else:
                st.warning("Bu fiyat aralığında modeli geçebilen BİST hissesi bulunamadı. Maksimum fiyatı artırın veya minimum ihtimali düşürün.")

    with t_tab3:
        st.subheader("🏦 TEFAS Fon Tarama")
        st.markdown("Borsapy üzerinden TEFAS verilerini çekerek en iyi performans gösteren fonları filtreleyin.")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fund_type = st.radio("Fon Tipi", ["YAT (Yatırım Fonu)", "EMK (Emeklilik Fonu)"], horizontal=True)
            f_type = "YAT" if "YAT" in fund_type else "EMK"
        with col_f2:
            min_ret_1y = st.number_input("Minimum Yıllık Getiri (%)", min_value=0.0, value=50.0, step=5.0)
            
        if st.button("🔍 Fonları Filtrele", type="primary"):
            with st.spinner("TEFAS verileri analiz ediliyor..."):
                fund_df, f_err = screen_tefas_funds(fund_type=f_type, min_return_1y=min_ret_1y)
            
            if f_err:
                st.error(f"Hata: {f_err}")
            elif not fund_df.empty:
                st.success(f"Kriterlere uygun {len(fund_df)} fon bulundu.")
                st.dataframe(fund_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Uygun fon bulunamadı.")

elif mode == "🌍 Makro Ekonomi ve Global":
    # Dashboard123'ün macro dashboard bileşenini entegre ettik
    try:
        # Zorla çevresel değişkene API yazalım ki başka hiçbir yere bakmasın
        fred_api_key = os.getenv("FRED_API_KEY", "").strip()
        if not fred_api_key:
            st.error("FRED_API_KEY .env icinde tanimli degil. Makro paneli acmak icin once anahtari ekleyin.")
            st.stop()

        os.environ["FRED_API_KEY"] = fred_api_key

        from components.macro_dashboard import render_macro_dashboard
        
        # Standart Dark Tema renk kodları (Dashboard123 beklentisine uygun)
        colors = {
            "bg_main": "#0E1117",
            "bg_card": "#1E212A",
            "border": "#2D3139",
            "text": "#FAFAFA",
            "text_header": "#FFFFFF",
            "text_muted": "#A0A0A0",
            "green": "#28A745",
            "red": "#DC3545",
            "blue": "#007BFF"
        } 
        
        st.info("💡 Not: Resesyon ve S&P 500 P/E geçmiş verileri için FRED API anahtarı sisteme otomatik (gömülü) olarak sağlandı.")
        render_macro_dashboard(colors, theme="dark")
        
    except ImportError as e:
        st.error("Dashboard123 'macro_dashboard' modülleri yüklenemedi. 'trade/Dashboard123-main' dizin yapısı kontrol edilmeli.")
        st.code(str(e))
    except Exception as e:
        st.error(f"FRED Veri çekme hatası: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

elif mode == "🤖 Otomatik Al-Sat & Paper Trading (Bot)":
    st.header("🤖 Yapay Zeka Destekli Otomatik Al-Sat (Paper Trading)")
    st.markdown("4 farklı kanıtlanmış strateji arasından seç veya YZ'ye bırak — bot geçmiş veriler üzerinde simüle eder.")

    bt_tab1, bt_tab2 = st.tabs(["🇹🇷 Borsa (Paper Trading) Simülasyonu", "🌐 Kripto (CCXT) Canlı Bağlantı"])

    with bt_tab1:
        # ── Strateji Tanımları (araştırmaya dayalı) ──────────────────────────
        STRATEGY_INFO = {
            "⚡ Agresif Day Trading": {
                "desc": "RSI(5) < 30 + MACD kesişimi + Hacim patlaması (Vol Ratio > 1.5). Kısa vadeli, sık işlem.",
                "giriş": "RSI(5) 30'un altına düşüp tekrar çıkarken + MACD bullish kesişim + hacim ortalamanın 1.5 katı üstünde",
                "çıkış": "RSI(5) > 80 veya MACD bearish kesişim",
                "risk": "🔴 Yüksek",
                "uygun": "Kripto & volatil hisseler",
                "kaynak": "QuantifiedStrategies: RSI(5) stratejisi %80+ başarı oranı raporlandı",
            },
            "🛡️ Güvenli Swing Trading": {
                "desc": "Williams %R(10) < -90 girişi, -20 çıkışı. Fiyat SMA200 üstünde. Backtest: sadece 2 negatif yıl (1993'ten beri).",
                "giriş": "Fiyat > SMA200 + Williams %R < -90 (aşırı satım)",
                "çıkış": "Williams %R > -20 (aşırı alım) veya fiyat SMA200 altına düşerse",
                "risk": "🟢 Düşük",
                "uygun": "BIST & ABD büyük şirketler",
                "kaynak": "QuantifiedStrategies: Williams %R swing trading için en iyi indikatör seçildi",
            },
            "📈 Trend Takip (MACD+ADX)": {
                "desc": "MACD sıfır üstünde + ADX > 25 + +DI > -DI. Trend güçlü olduğu sürece pozisyonda kal.",
                "giriş": "MACD > sinyal + MACD > 0 + ADX > 25 + Plus_DI > Minus_DI",
                "çıkış": "MACD sinyal altına düşerse veya ADX < 20 (trend zayıflıyor)",
                "risk": "🟡 Orta",
                "uygun": "Güçlü trend dönemleri, her piyasa",
                "kaynak": "ForexTester backtest: MACD+ADX kombinasyonu güçlü trend teyidi sağlar",
            },
            "🔄 Ortalamaya Dönüş (Bollinger+RSI)": {
                "desc": "Bollinger alt bandı dokunuşu + RSI < 35 + VWAP sapması negatif. Mean-reversion.",
                "giriş": "Fiyat BB alt bandına değerse + RSI < 35 + Fiyat VWAP altında",
                "çıkış": "Fiyat BB orta bandına (SMA20) ulaştığında veya RSI > 65",
                "risk": "🟡 Düşük-Orta",
                "uygun": "Yatay piyasalar, konsolide hisseler",
                "kaynak": "Cloudzy/LiteFinance: BB + RSI kombinasyonu yatay piyasada etkili",
            },
            "🚀 Beat the Market (Zarattini/Aziz)": {
                "desc": "Kurumsal Day Trading: Noise Area kırılımı + VWAP Trailing Stop + Volatilite Hedefli Büyüklük.",
                "giriş": "Fiyat > Noise Area Üst Sınırı (veya < Alt Sınırı) + Sadece HH:00/HH:30 dilimleri",
                "çıkış": "VWAP Trailing Stop veya 16:00 Gün Kapanışı",
                "risk": "🟡 Orta-Yüksek",
                "uygun": "SPY, QQQ ve Yüksek Likiditeli Hisseler",
                "kaynak": "Concretum Research: Intraday Momentum Strategy for S&P500 ETF (2024)",
            },
        }

        # ── UI ────────────────────────────────────────────────────────────────
        col_left, col_right = st.columns([1, 1])

        with col_left:
            sim_ticker = st.text_input(
                "📊 Hisse / Kripto Sembolü",
                "THYAO.IS",
                key="sim_ticker",
                help="BİST: THYAO.IS | ABD: AAPL | Kripto: BTC-USD"
            ).upper()
            sim_period = st.selectbox("📅 Dönem", ["6mo", "1y", "2y"], index=1, key="sim_period")
            initial_balance = st.number_input("💰 Başlangıç Bakiyesi ($)", value=10000, step=1000, key="sim_balance")

        with col_right:
            st.markdown("#### 🤖 Strateji Seçimi")
            strategy_mode = st.radio(
                "",
                ["🧠 YZ Otomatik Seçsin", "👤 Ben Seçeyim"],
                key="strategy_mode",
                horizontal=True
            )
            if strategy_mode == "👤 Ben Seçeyim":
                selected_strategy = st.selectbox(
                    "Strateji:",
                    list(STRATEGY_INFO.keys()),
                    key="selected_strat"
                )
            else:
                selected_strategy = None
                st.info("YZ, piyasa koşullarını analiz ederek en uygun stratejiyi otomatik belirleyecek.")

        # ── Strateji kartları ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("##### 📚 Strateji Detayları")
        show_strats = [selected_strategy] if selected_strategy else list(STRATEGY_INFO.keys())
        cols = st.columns(len(show_strats))
        for col, sname in zip(cols, show_strats):
            info = STRATEGY_INFO[sname]
            with col:
                st.markdown(f"""
                <div style='background:#1a1d2e;border-radius:10px;padding:12px;border:1px solid #2d3153;height:100%'>
                <b style='color:#00d4ff'>{sname}</b><br>
                <small style='color:#8892b0'>{info['desc']}</small><br><br>
                <b style='color:#64ffda'>Giriş:</b> <small>{info['giriş']}</small><br>
                <b style='color:#ff6b6b'>Çıkış:</b> <small>{info['çıkış']}</small><br>
                <b>Risk:</b> {info['risk']} | <b>Uygun:</b> <small>{info['uygun']}</small><br>
                <small style='color:#4a5568'>📖 {info['kaynak']}</small>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        if st.button("🚀 Botu Başlat", type="primary", key="start_bot"):
            with st.spinner(f"{sim_ticker} için strateji analizi yapılıyor..."):
                try:
                    if selected_strategy == "🚀 Beat the Market (Zarattini/Aziz)":
                        from intraday_strategy_engine import IntradayMomentumEngine
                        engine = IntradayMomentumEngine()
                        # Makale stratejisi için 1dk veri çekiyoruz (Maksimum 30 gün yfinance sınırı)
                        with st.spinner("1 Dakikalık Intraday veriler çekiliyor (Borsapy/yFinance)..."):
                            df_sim = engine.fetch_intraday_data(sim_ticker, period="1mo", interval="1m")
                            final_ticker_sim = sim_ticker
                    else:
                        df_sim, final_ticker_sim = fetch_data(sim_ticker, period=sim_period)
                        df_sim = calculate_technical_indicators(df_sim)
                        df_sim = df_sim.dropna()
                    
                    if df_sim.empty:
                        st.error("⚠️ Yeterli veri bulunamadı! Lütfen 'Dönem' ayarını 1y veya 2y olarak değiştirin (SMA200 gibi uzun vadeli göstergelerin hesaplanabilmesi için en az 1 yıllık geçmiş veriye ihtiyaç vardır).")
                        st.stop()
                        
                    market_type = detect_market_type(final_ticker_sim)

                    # ── YZ Otomatik Strateji Seçimi ──────────────────────────
                    if strategy_mode == "🧠 YZ Otomatik Seçsin":
                        last = df_sim.iloc[-1]
                        adx_val  = last.get('ADX', 0)
                        rsi_val  = last.get('RSI', 50)
                        vol_r    = last.get('Volume_Ratio', 1)
                        bb_pos   = last.get('BB_Position', 0.5)
                        sma200   = last.get('SMA_200', last['Close'])
                        price    = last['Close']
                        wr_val   = last.get('WilliamsR', -50)

                        # Piyasa rejimi tespiti
                        is_trending   = adx_val > 25
                        is_oversold   = rsi_val < 35
                        is_highvol    = vol_r > 1.5
                        above_sma200  = price > sma200
                        is_bb_low     = bb_pos < 0.2

                        # YZ karar mantığı
                        if is_trending and adx_val > 30:
                            auto_strategy = "📈 Trend Takip (MACD+ADX)"
                            reason = f"ADX={adx_val:.1f} güçlü trend tespit edildi (>25)"
                        elif above_sma200 and wr_val < -80:
                            auto_strategy = "🛡️ Güvenli Swing Trading"
                            reason = f"Fiyat SMA200 üstünde ve Williams %R={wr_val:.1f} aşırı satım"
                        elif is_oversold and is_bb_low:
                            auto_strategy = "🔄 Ortalamaya Dönüş (Bollinger+RSI)"
                            reason = f"RSI={rsi_val:.1f} aşırı satım + Bollinger alt bandı yakını"
                        elif is_highvol and market_type == 'crypto':
                            auto_strategy = "⚡ Agresif Day Trading"
                            reason = f"Kripto + yüksek hacim (Vol Ratio={vol_r:.1f}x)"
                        else:
                            auto_strategy = "🛡️ Güvenli Swing Trading"
                            reason = "Belirsiz piyasa — en güvenli strateji seçildi"

                        selected_strategy = auto_strategy
                        st.success(f"🧠 YZ Kararı: **{auto_strategy}**")
                        st.caption(f"Sebep: {reason}")

                    st.markdown(f"#### ▶ Strateji: {selected_strategy}")

                    # ── Simülasyon Döngüsü (vektörleştirilmiş erişim) ──────────
                    balance  = float(initial_balance)
                    position = 0.0
                    trade_log   = []
                    equity_list = [balance]
                    commission  = 0.001  # %0.1

                    # ── Numpy dizilerine ön-çıkarma (DataFrame.iloc elimine) ──
                    import numpy as _np
                    _n_sim = len(df_sim)
                    _closes    = df_sim['Close'].values
                    _rsi_arr   = df_sim['RSI'].values if 'RSI' in df_sim.columns else _np.full(_n_sim, 50.0)
                    _macd_arr  = df_sim['MACD'].values if 'MACD' in df_sim.columns else _np.zeros(_n_sim)
                    _macd_s_arr= df_sim['MACD_Signal'].values if 'MACD_Signal' in df_sim.columns else _np.zeros(_n_sim)
                    _adx_arr   = df_sim['ADX'].values if 'ADX' in df_sim.columns else _np.zeros(_n_sim)
                    _pdi_arr   = df_sim['Plus_DI'].values if 'Plus_DI' in df_sim.columns else _np.zeros(_n_sim)
                    _mdi_arr   = df_sim['Minus_DI'].values if 'Minus_DI' in df_sim.columns else _np.zeros(_n_sim)
                    _sma200_arr= df_sim['SMA_200'].values if 'SMA_200' in df_sim.columns else _closes.copy()
                    _bbl_arr   = df_sim['BB_Lower'].values if 'BB_Lower' in df_sim.columns else _closes * 0.97
                    _bbm_arr   = df_sim['BB_Middle'].values if 'BB_Middle' in df_sim.columns else _closes.copy()
                    _wr_arr    = df_sim['Williams_R'].values if 'Williams_R' in df_sim.columns else _np.full(_n_sim, -50.0)
                    _vr_arr    = df_sim['Volume_Ratio'].values if 'Volume_Ratio' in df_sim.columns else _np.ones(_n_sim)
                    _vwap_arr  = df_sim['VWAP'].values if 'VWAP' in df_sim.columns else _closes.copy()
                    _sim_dates = df_sim.index

                    for i in range(1, _n_sim - 1):
                        cp = _closes[i]

                        rsi     = _rsi_arr[i]
                        macd    = _macd_arr[i]
                        macd_s  = _macd_s_arr[i]
                        adx_v   = _adx_arr[i]
                        plus_di = _pdi_arr[i]
                        minus_di= _mdi_arr[i]
                        sma200  = _sma200_arr[i]
                        bb_low  = _bbl_arr[i]
                        bb_mid  = _bbm_arr[i]
                        wr      = _wr_arr[i]
                        vol_r   = _vr_arr[i]
                        vwap    = _vwap_arr[i]

                        prev_macd   = _macd_arr[i - 1]
                        prev_macd_s = _macd_s_arr[i - 1]

                        buy_cond  = False
                        sell_cond = False

                        if selected_strategy == "⚡ Agresif Day Trading":
                            # RSI(5) yerine RSI(14) kullanıyoruz (mevcut indikatörden)
                            # Giriş: RSI aşırı satımdan çıkıyor + MACD bullish + yüksek hacim
                            buy_cond  = (rsi < 32 and
                                        macd > macd_s and
                                        prev_macd <= prev_macd_s and
                                        vol_r > 1.5)
                            sell_cond = (rsi > 75 or
                                        (macd < macd_s and prev_macd >= prev_macd_s))

                        elif selected_strategy == "🛡️ Güvenli Swing Trading":
                            # Williams %R: < -80 giriş, > -25 çıkış + SMA200 filtresi
                            buy_cond  = (cp > sma200 and wr < -80)
                            sell_cond = (wr > -25 or cp < sma200)

                        elif selected_strategy == "📈 Trend Takip (MACD+ADX)":
                            # MACD sıfır üstünde + ADX > 25 + +DI > -DI
                            buy_cond  = (macd > macd_s and
                                        macd > 0 and
                                        adx_v > 25 and
                                        plus_di > minus_di)
                            sell_cond = ((macd < macd_s and prev_macd >= prev_macd_s) or
                                        adx_v < 20)

                        elif selected_strategy == "🚀 Beat the Market (Zarattini/Aziz)":
                            # Bu strateji kendi motoruyla (intraday_strategy_engine) çalışır
                            # Bu yüzden döngü yerine doğrudan motoru çağırıp sonuçları alıyoruz
                            res, err = engine.run_backtest(df_sim, initial_balance=balance, commission=commission)
                            if err:
                                st.error(err)
                                st.stop()
                            
                            # Log ve Equity'yi senkronize et
                            trade_log = res["trade_log"]
                            equity_list = res["equity_curve"].tolist()
                            balance = res["final_balance"]
                            
                            # Döngüden çık (motor tüm süreci yönetti)
                            break

                        # İşlem uygula
                        if buy_cond and position == 0 and balance > 0:
                            shares   = balance / (cp * (1 + commission))
                            position = shares
                            balance  = 0.0
                            
                            tarih_str = _sim_dates[i].strftime("%d.%m.%Y")
                            trade_log.append({
                                "Tarih": tarih_str,
                                "İşlem": "🟢 AL",
                                "Fiyat": f"{cp:.4f}",
                                "Hisse": f"{shares:.4f}",
                                "Bakiye": "Pozisyonda"
                            })
                            
                            # Telegram Bildirimi
                            if st.session_state.get("tg_notify_buy") and st.session_state.get("tg_bot_token") and st.session_state.get("tg_chat_id"):
                                import requests
                                msg = f"🟢 *Hisse Avcısı SİNYAL AL*\nSembol: {final_ticker_sim}\nFiyat: {cp:.4f}\nStrateji: {selected_strategy}\nTarih: {tarih_str}"
                                try:
                                    requests.post(
                                        f"https://api.telegram.org/bot{st.session_state['tg_bot_token']}/sendMessage",
                                        json={"chat_id": st.session_state["tg_chat_id"], "text": msg, "parse_mode": "Markdown"},
                                        timeout=3
                                    )
                                except: pass

                        elif sell_cond and position > 0:
                            proceeds = position * cp * (1 - commission)
                            balance  = proceeds
                            position = 0.0
                            
                            tarih_str = _sim_dates[i].strftime("%d.%m.%Y")
                            trade_log.append({
                                "Tarih": tarih_str,
                                "İşlem": "🔴 SAT",
                                "Fiyat": f"{cp:.4f}",
                                "Hisse": "-",
                                "Bakiye": f"${proceeds:,.2f}"
                            })
                            
                            # Telegram Bildirimi
                            if st.session_state.get("tg_notify_sell") and st.session_state.get("tg_bot_token") and st.session_state.get("tg_chat_id"):
                                import requests
                                msg = f"🔴 *Hisse Avcısı SİNYAL SAT*\nSembol: {final_ticker_sim}\nFiyat: {cp:.4f}\nStrateji: {selected_strategy}\nTarih: {tarih_str}"
                                try:
                                    requests.post(
                                        f"https://api.telegram.org/bot{st.session_state['tg_bot_token']}/sendMessage",
                                        json={"chat_id": st.session_state["tg_chat_id"], "text": msg, "parse_mode": "Markdown"},
                                        timeout=3
                                    )
                                except: pass

                        equity_list.append(balance + position * cp)

                    # Dönem sonu açık pozisyonu kapat
                    final_price = df_sim['Close'].iloc[-1]
                    if position > 0:
                        proceeds = position * final_price * (1 - commission)
                        balance  = proceeds
                        trade_log.append({
                            "Tarih": df_sim.index[-1].strftime("%d.%m.%Y"),
                            "İşlem": "⚪ OTOMATİK KAPANIŞ",
                            "Fiyat": f"{final_price:.4f}",
                            "Hisse": "-",
                            "Bakiye": f"${proceeds:,.2f}"
                        })

                    profit_pct  = (balance - initial_balance) / initial_balance * 100
                    bah_return  = (df_sim['Close'].iloc[-1] / df_sim['Close'].iloc[0] - 1) * 100
                    alfa        = profit_pct - bah_return
                    n_trades    = len([t for t in trade_log if "AL" in t["İşlem"]])

                    # Equity eğrisi için index
                    eq_index = df_sim.index[:len(equity_list)]
                    eq_series = pd.Series(equity_list[:len(eq_index)], index=eq_index)

                    # Max drawdown
                    roll_max  = eq_series.cummax()
                    drawdown  = (eq_series - roll_max) / roll_max * 100
                    max_dd    = drawdown.min()

                    # ── Sonuç Gösterimi ───────────────────────────────────────
                    st.success("✅ Simülasyon tamamlandı!")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Son Bakiye", f"${balance:,.0f}", f"%{profit_pct:+.1f}")
                    m2.metric("Buy & Hold Getiri", f"%{bah_return:+.1f}")
                    m3.metric("Alfa (Strateji − B&H)", f"%{alfa:+.1f}",
                              delta_color="normal" if alfa > 0 else "inverse")
                    m4.metric("Max Drawdown", f"%{max_dd:.1f}")

                    c2_1, c2_2 = st.columns(2)
                    c2_1.metric("Toplam İşlem Sayısı", n_trades)
                    c2_2.metric("Başlangıç Sermaye", f"${initial_balance:,.0f}")

                    # Equity eğrisi grafiği
                    import plotly.graph_objects as go
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=eq_series.index, y=eq_series.values,
                        mode='lines', name='Strateji',
                        line=dict(color='#00d4ff', width=2)
                    ))
                    # Buy & Hold referans çizgisi
                    bah_vals = df_sim['Close'] / df_sim['Close'].iloc[0] * initial_balance
                    fig_eq.add_trace(go.Scatter(
                        x=bah_vals.index, y=bah_vals.values,
                        mode='lines', name='Buy & Hold',
                        line=dict(color='#888', width=1, dash='dash')
                    ))
                    fig_eq.update_layout(
                        title=f"{final_ticker_sim} — {selected_strategy}",
                        xaxis_title="Tarih", yaxis_title="Portföy Değeri ($)",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font=dict(color="#e0e0e0"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        height=350
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)

                    # İşlem defteri
                    with st.expander("📋 Detaylı İşlem Defteri", expanded=False):
                        if trade_log:
                            st.dataframe(pd.DataFrame(trade_log), use_container_width=True)
                        else:
                            st.warning("Bu dönemde hiç işlem sinyali üretilmedi.")

                    # ── Quantstats Tarzı Detaylı Rapor ───────────────────────
                    with st.expander("📊 Detaylı Performans Raporu", expanded=False):
                        try:
                            eq_s  = eq_series.dropna()
                            rets  = eq_s.pct_change().dropna()
                            if len(rets) > 5:
                                import plotly.graph_objects as go

                                # Aylık getiri tablosu
                                monthly    = rets.resample("ME").apply(lambda x: (1+x).prod()-1) * 100
                                monthly_df = monthly.reset_index()
                                monthly_df.columns = ["Tarih", "Getiri_%"]
                                monthly_df["Ay"]   = monthly_df["Tarih"].dt.strftime("%b")
                                monthly_df["Yıl"]  = monthly_df["Tarih"].dt.year
                                pivot = monthly_df.pivot_table(index="Yıl", columns="Ay", values="Getiri_%", aggfunc="first")
                                ay_sirasi = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                                pivot = pivot.reindex(columns=[a for a in ay_sirasi if a in pivot.columns])

                                st.markdown("##### 📅 Aylık Getiri Tablosu (%)")
                                st.dataframe(
                                    pivot.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.1f}%", na_rep="-"),
                                    use_container_width=True
                                )

                                # Gelişmiş metrikler
                                TD       = 252
                                ann_ret  = (1 + rets.mean()) ** TD - 1
                                ann_vol  = rets.std() * (TD ** 0.5)
                                sharpe   = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
                                downside = rets[rets < 0].std() * (TD ** 0.5)
                                sortino  = (ann_ret - 0.02) / downside if downside > 0 else 0
                                roll_max = eq_s.cummax()
                                dd_s     = (eq_s - roll_max) / roll_max * 100
                                max_dd   = dd_s.min()
                                calmar   = ann_ret / abs(max_dd / 100) if max_dd != 0 else 0
                                win_rate = (rets > 0).sum() / len(rets) * 100
                                best_day = rets.max() * 100
                                worst_day= rets.min() * 100

                                st.markdown("##### 📈 Gelişmiş Risk Metrikleri")
                                rm1, rm2, rm3, rm4 = st.columns(4)
                                rm1.metric("Yıllık Getiri",  f"%{ann_ret*100:.1f}")
                                rm2.metric("Sharpe Oranı",   f"{sharpe:.2f}",  help="2+ mükemmel, 1+ iyi, 0+ kabul edilebilir")
                                rm3.metric("Sortino Oranı",  f"{sortino:.2f}", help="Sadece negatif volatilite bazlı risk")
                                rm4.metric("Calmar Oranı",   f"{calmar:.2f}",  help="Yıllık getiri / Max Drawdown")

                                rm5, rm6, rm7, rm8 = st.columns(4)
                                rm5.metric("Max Drawdown",   f"%{max_dd:.1f}")
                                rm6.metric("Kazanma Günü",   f"%{win_rate:.1f}")
                                rm7.metric("En İyi Gün",     f"%{best_day:.2f}")
                                rm8.metric("En Kötü Gün",    f"%{worst_day:.2f}")

                                # Drawdown grafiği
                                fig_dd = go.Figure()
                                fig_dd.add_trace(go.Scatter(
                                    x=dd_s.index, y=dd_s.values,
                                    fill="tozeroy", mode="lines",
                                    line=dict(color="#e74c3c", width=1),
                                    fillcolor="rgba(231,76,60,0.2)",
                                    name="Drawdown"
                                ))
                                fig_dd.update_layout(
                                    title="Drawdown Eğrisi (%)", xaxis_title="Tarih", yaxis_title="DD %",
                                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                    font=dict(color="#e0e0e0"), height=220
                                )
                                st.plotly_chart(fig_dd, use_container_width=True)
                        except Exception as qe:
                            st.warning(f"Rapor üretilemedi: {qe}")

                    # ── Telegram Bildirim ─────────────────────────────────────
                    tg_token = st.session_state.get("tg_bot_token", "")
                    tg_chat  = st.session_state.get("tg_chat_id", "")
                    if tg_token and tg_chat and trade_log:
                        last_trade = trade_log[-1]
                        if "AL" in last_trade["İşlem"] or "SAT" in last_trade["İşlem"]:
                            try:
                                import requests as _req
                                msg = (
                                    f"🤖 *Hisse Avcısı Bot Bildirimi*\n"
                                    f"Sembol: `{final_ticker_sim}`\n"
                                    f"İşlem: {last_trade['İşlem']}\n"
                                    f"Fiyat: {last_trade['Fiyat']}\n"
                                    f"Tarih: {last_trade['Tarih']}\n"
                                    f"Strateji: {selected_strategy}"
                                )
                                _req.post(
                                    f"https://api.telegram.org/bot{tg_token}/sendMessage",
                                    json={"chat_id": tg_chat, "text": msg, "parse_mode": "Markdown"},
                                    timeout=5
                                )
                                st.toast("📱 Telegram bildirimi gönderildi!", icon="✅")
                            except Exception as te:
                                st.caption(f"Telegram bildirimi gönderilemedi: {te}")

                except Exception as e:
                    st.error(f"Bot Hatası: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with bt_tab2:
        st.subheader("CCXT ile Gerçek Zamanlı Kripto Piyasa Monitörü")
        st.markdown("Dünyadaki kripto piyasasını **CCXT kütüphanesi** ile Binance motorundan saniyelik sıfır gecikmeyle (Live) çeker.")
        try:
            import ccxt
            exchange = ccxt.binance({'enableRateLimit': True})

            crypto_ticker = st.text_input("Kripto Çifti (Örn: BTC/USDT, ETH/USDT, XRP/USDT)", "BTC/USDT", key="crypto_ticker").upper()

            if st.button("🚀 Miktarı ve Defteri Canlı Çek", type="primary"):
                with st.spinner(f"Binance üzerinden {crypto_ticker} defteri ve anlık tick'i alınıyor..."):
                    ticker   = exchange.fetch_ticker(crypto_ticker)
                    order_book = exchange.fetch_order_book(crypto_ticker, limit=7)

                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Anlık Fiyat (CCXT Live)", f"${ticker['last']}")
                    cc2.metric("24s Fiyat Değişimi", f"%{ticker['percentage']:.2f}")
                    cc3.metric("24s Hacim", f"{ticker['baseVolume']:,.2f} {crypto_ticker.split('/')[0]}")

                    st.divider()
                    st.subheader("Binance Canlı Sipariş Defteri (Order Book)")
                    o1, o2 = st.columns(2)
                    with o1:
                        st.markdown("##### 🟢 Bekleyen Alış Emirleri (Bids)")
                        for bid in order_book['bids']:
                            st.success(f"Fiyat: **${bid[0]:.4f}** | Miktar: {bid[1]:.4f}")
                    with o2:
                        st.markdown("##### 🔴 Bekleyen Satış Emirleri (Asks)")
                        for ask in order_book['asks']:
                            st.error(f"Fiyat: **${ask[0]:.4f}** | Miktar: {ask[1]:.4f}")

                    st.info("💡 **Gelişmiş Fikir:** CCXT kütüphanesi sisteme entegre edildiği için ilerleyen aşamada kendi API Anahtarlarınızı girerek, yandaki simülasyon botunun ürettiği Al/Sat kararlarını burası üzerinden doğrudan Binance hesabınıza gönderen otomatik bir tüccar (trader) oluşturabilirsiniz!")

        except ImportError:
            st.error("ccxt kütüphanesi bulunamadı! Lütfen sistemi 'BASLAT.bat' üzerinden tekrar başlattığınızdan emin olun.")
        except Exception as e:
            st.error(f"CCXT Bağlantı Hatası: Borsa geçici olarak yanıt vermiyor olabilir veya sembol yanlış. Detay: {str(e)}")

elif mode == "🧠 Quant-AI Sinyal Motoru":

    # ── CSS Overrides for premium signal cards ─────────────────────────────
    st.markdown("""
    <style>
    .quant-badge {
        display: flex; align-items: center; justify-content: center;
        border-radius: 16px; padding: 18px 32px; font-size: 2.4rem;
        font-weight: 900; letter-spacing: 4px; margin-bottom: 8px;
        box-shadow: 0 0 40px rgba(0,0,0,0.45);
    }
    .quant-buy  { background: linear-gradient(135deg,#0d6e3f,#1a9e5c); color:#e0ffe8; }
    .quant-sell { background: linear-gradient(135deg,#6e0d0d,#c0392b); color:#ffe0e0; }
    .quant-hold { background: linear-gradient(135deg,#3a3d4a,#5a5e70); color:#e8e8f0; }
    .quant-card {
        background: #1a1d2e; border-radius: 12px; padding: 14px 18px;
        border: 1px solid #2d3153; margin-bottom: 8px;
    }
    .quant-label { font-size:0.75rem; color:#8892b0; text-transform:uppercase; letter-spacing:1px; }
    .quant-value { font-size:1.15rem; font-weight:700; color:#ccd6f6; }
    .vote-bull { color:#2ecc71; font-weight:700; }
    .vote-bear { color:#e74c3c; font-weight:700; }
    .vote-neut { color:#95a5a6; }
    </style>
    """, unsafe_allow_html=True)

    st.header("🧠 Quant-AI Sinyal Motoru")
    st.markdown(
        """Akademik yöntemler **(Kakushadze & Serur – 151 Trading Strategies)** üzerine inşa edilmiş 
        beş kantitatif strateji modülü paralel olarak hesaplanır ve tek bir **konsensüs sinyal**e dönüştürülür.
        Sezgiye değil **matematiksel modellere** dayanır."""
    )

    # ── Sidebar controls ───────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Analiz Parametreleri")
    qai_ticker = st.sidebar.text_input(
        "Hisse / Kripto Sembolü", "AAPL",
        help="Örn: AAPL, BTC-USD, THYAO, SPY",
        key="qai_ticker"
    ).upper()
    qai_sentiment_toggle = st.sidebar.toggle(
        "📰 FinBERT Haber Duygusu (yavaş)", value=False,
        help="Açık iken gerçek haber analizi yapılır. Kapalı iken nötr kabul edilir."
    )
    qai_btn = st.sidebar.button("🚀 Sinyal Üret", type="primary", key="qai_analyze_btn")

    if qai_btn:
        st.session_state["qai_run"] = True
        st.session_state["qai_active_ticker"] = qai_ticker

    if st.session_state.get("qai_run", False):
        _ticker = st.session_state.get("qai_active_ticker", qai_ticker)

        try:
            from quant_ai_engine import run_quant_ai_analysis
            import json

            # ── Data loading ────────────────────────────────────────────
            with st.spinner(f"📡 {_ticker} için piyasa verisi yükleniyor..."):
                _df, _final_ticker = fetch_data(_ticker, period="2y")
                _df = calculate_technical_indicators(_df)

            # ── Sentiment (optional) ────────────────────────────────────
            _sentiment_score = 0.0
            if qai_sentiment_toggle:
                with st.spinner("📰 FinBERT haber analizi yapılıyor..."):
                    _sentiment_score, _ = get_finbert_sentiment(_final_ticker)

            # ── Run engine ──────────────────────────────────────────────
            with st.spinner("🧮 Beş strateji modülü hesaplanıyor..."):
                result = run_quant_ai_analysis(_df, _sentiment_score)

            _signal     = result["consensus_signal"]
            _conf       = result["confidence_score"]
            _regime     = result["market_regime"]
            _primary    = result["primary_strategy_triggered"]
            _summary    = result["analysis_summary"]
            _breakdown  = result["strategy_breakdown"]
            _risk       = result["risk_management"]
            _detail     = result.get("_detail", {})
            _votes      = _detail.get("votes", {})
            _price      = _detail.get("current_price", float(_df["Close"].iloc[-1]))

            # ═══════════════════════════════════════════════════════════
            # 1. CONSENSUS SIGNAL BADGE
            # ═══════════════════════════════════════════════════════════
            badge_class = {"BUY": "quant-buy", "SELL": "quant-sell", "HOLD": "quant-hold"}.get(_signal, "quant-hold")
            badge_emoji = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️"}.get(_signal, "⏸️")
            badge_tr    = {"BUY": "AL", "SELL": "SAT", "HOLD": "BEKLE"}.get(_signal, "BEKLE")

            st.markdown(f"""
            <div class='quant-badge {badge_class}'>
                {badge_emoji}&nbsp;&nbsp;{badge_tr}&nbsp;({_signal})
            </div>
            <p style='text-align:center;color:#8892b0;font-size:0.85rem;margin-top:4px;'>
                Analiz edilen sembol: <strong style='color:#ccd6f6'>{_final_ticker}</strong> &nbsp;|&nbsp;
                Güncel fiyat: <strong style='color:#ccd6f6'>{_price:,.4f}</strong>
            </p>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # ═══════════════════════════════════════════════════════════
            # 2. TOP METRICS ROW
            # ═══════════════════════════════════════════════════════════
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🎯 Güven Skoru",     f"%{_conf}")
            m2.metric("🌍 Piyasa Rejimi",   _regime)
            m3.metric("🏆 Birincil Strateji", _primary[:32] + ("..." if len(_primary) > 32 else ""))
            m4.metric("📰 Haber Duygusu",   _breakdown["sentiment"])

            # Confidence bar
            bar_color = "#2ecc71" if _signal == "BUY" else "#e74c3c" if _signal == "SELL" else "#7f8c8d"
            st.markdown(f"""
            <div style='background:#1a1d2e;border-radius:8px;padding:4px 8px;margin-bottom:6px;'>
                <div style='font-size:0.72rem;color:#8892b0;'>GÜVEN / CONFLUENCE SKORU</div>
                <div style='background:#2d3153;border-radius:6px;height:18px;overflow:hidden;'>
                    <div style='background:{bar_color};width:{_conf}%;height:100%;border-radius:6px;
                                transition:width 1s ease;display:flex;align-items:center;
                                padding-left:8px;font-size:0.72rem;color:#fff;font-weight:700;'>
                        {_conf}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.info(f"💬 **Analiz Özeti:** {_summary}")
            st.divider()

            # ═══════════════════════════════════════════════════════════
            # 3. STRATEGY BREAKDOWN  (5 modules)
            # ═══════════════════════════════════════════════════════════
            st.subheader("📊 Strateji Modülü Döküm Paneli")
            st.caption("Her modül bağımsız akademik formülle hesaplanır, ardından birleştirilir.")

            # Helper to render vote badge
            def _vote_badge(v):
                if v > 0:  return "<span class='vote-bull'>▲ YUKARI</span>"
                if v < 0:  return "<span class='vote-bear'>▼ AŞAĞI</span>"
                return "<span class='vote-neut'>━ NÖTR</span>"

            _mom_d   = _detail.get("momentum", {})
            _ma_d    = _detail.get("moving_avg", {})
            _pivot_d = _detail.get("pivot", {})
            _don_d   = _detail.get("donchian", {})
            _sent_d  = _detail.get("sentiment", {})

            # --- Row 1: Momentum + MA ---
            col_s1, col_s2 = st.columns(2)

            with col_s1:
                st.markdown(f"""
                <div class='quant-card'>
                    <div class='quant-label'>📖 §3.1 — Fiyat Momentumu</div>
                    <div class='quant-value' style='margin:6px 0;'>{_votes.get('momentum','?'):+d} soy &nbsp;&nbsp;{_vote_badge(_votes.get('momentum',0))}</div>
                    <table style='width:100%;font-size:0.82rem;color:#a8b2d8;'>
                        <tr><td>Kümülatif Getiri</td><td align='right'><b>%{_mom_d.get('cumulative_return_pct', 0):+.2f}</b></td></tr>
                        <tr><td>Risk-Adj. Getiri</td><td align='right'><b>{_mom_d.get('risk_adjusted_return', 0):+.3f}</b></td></tr>
                        <tr><td>Aylık Volatilite</td><td align='right'><b>%{_mom_d.get('monthly_volatility_pct', 0):.2f}</b></td></tr>
                        <tr><td>Formasyon Periyodu</td><td align='right'>{_mom_d.get('formation_months', 12)} ay</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            with col_s2:
                st.markdown(f"""
                <div class='quant-card'>
                    <div class='quant-label'>📖 §3.12/3.13 — İkili & Üçlü Hareketli Ortalama</div>
                    <div class='quant-value' style='margin:6px 0;'>{_votes.get('moving_avg','?'):+d} soy &nbsp;&nbsp;{_vote_badge(_votes.get('moving_avg',0))}</div>
                    <table style='width:100%;font-size:0.82rem;color:#a8b2d8;'>
                        <tr><td>SMA-5</td>  <td align='right'><b>{_ma_d.get('sma5', 0):.4f}</b></td></tr>
                        <tr><td>SMA-20</td> <td align='right'><b>{_ma_d.get('sma20', 0):.4f}</b></td></tr>
                        <tr><td>SMA-50</td> <td align='right'><b>{_ma_d.get('sma50', 0):.4f}</b></td></tr>
                        <tr><td>İkili MA Cross</td><td align='right'>{_ma_d.get('two_ma_crossover','—')}</td></tr>
                        <tr><td>Üçlü MA Hizalama</td><td align='right'>{_ma_d.get('three_ma_alignment','—')}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            # --- Row 2: Pivot + Donchian ---
            col_s3, col_s4 = st.columns(2)

            with col_s3:
                st.markdown(f"""
                <div class='quant-card'>
                    <div class='quant-label'>📖 §3.14 — Pivot Destek &amp; Direnç</div>
                    <div class='quant-value' style='margin:6px 0;'>{_votes.get('pivot','?'):+d} soy &nbsp;&nbsp;{_vote_badge(_votes.get('pivot',0))}</div>
                    <table style='width:100%;font-size:0.82rem;color:#a8b2d8;'>
                        <tr><td>Pivot Merkez (C)</td><td align='right'><b>{_pivot_d.get('pivot_centre', 0):.4f}</b></td></tr>
                        <tr><td>Direnç (R)</td>      <td align='right'><b style='color:#e74c3c'>{_pivot_d.get('resistance', 0):.4f}</b></td></tr>
                        <tr><td>Destek (S)</td>      <td align='right'><b style='color:#2ecc71'>{_pivot_d.get('support', 0):.4f}</b></td></tr>
                        <tr><td>Bar Genişliği</td>   <td align='right'>%{_pivot_d.get('band_pct', 0):.2f}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            with col_s4:
                st.markdown(f"""
                <div class='quant-card'>
                    <div class='quant-label'>📖 §3.15 — Donchian Kanal Kırılımı</div>
                    <div class='quant-value' style='margin:6px 0;'>{_votes.get('donchian','?'):+d} soy &nbsp;&nbsp;{_vote_badge(_votes.get('donchian',0))}</div>
                    <table style='width:100%;font-size:0.82rem;color:#a8b2d8;'>
                        <tr><td>Üst Bant (B_up)</td>    <td align='right'><b style='color:#e74c3c'>{_don_d.get('donchian_upper', 0):.4f}</b></td></tr>
                        <tr><td>Alt Bant (B_down)</td>  <td align='right'><b style='color:#2ecc71'>{_don_d.get('donchian_lower', 0):.4f}</b></td></tr>
                        <tr><td>Kanal Genişliği</td>    <td align='right'>%{_don_d.get('channel_width_pct', 0):.2f}</td></tr>
                        <tr><td>Fiyat Konumu</td>       <td align='right'>%{_don_d.get('price_position_pct', 50):.1f} (kanalda)</td></tr>
                        <tr><td>Periyot</td>            <td align='right'>{_don_d.get('channel_period', 20)} gün</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            # --- Row 3: Sentiment full width ---
            st.markdown(f"""
            <div class='quant-card'>
                <div class='quant-label'>📖 §18.3 — Naïve Bayes Haber Duygusu (FinBERT NLP)</div>
                <div class='quant-value' style='margin:6px 0;'>{_votes.get('sentiment','?'):+d} soy &nbsp;&nbsp;{_vote_badge(_votes.get('sentiment',0))}
                    &nbsp;&nbsp;<span style='color:#a8b2d8;font-size:0.9rem;font-weight:400;'>Ham Skor: {_sent_d.get('raw_score', 0):+.4f}</span>
                </div>
                <div style='font-size:0.88rem;color:#8892b0;'>
                    FinBERT (ProsusAI/finbert) ile analiz edilen güncel haber başlıklarının duygu ağırlıklı ortalaması.
                    {'Duygu analizi bu çalışmada devre dışı bırakıldı (sidebar geçişi ile açın).' if not qai_sentiment_toggle else f'Tespit edilen duygu: <strong style="color:#ccd6f6">{_sent_d.get("label")}</strong>'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # ═══════════════════════════════════════════════════════════
            # 4. VOTE SCOREBOARD
            # ═══════════════════════════════════════════════════════════
            st.subheader("🗳️ Oy Sayım Tablosu")
            _tb = _votes.get("total_bull", 0)
            _ts = _votes.get("total_bear", 0)
            _tn = _votes.get("total_neutral", 0)
            _total_strategies = _tb + _ts + _tn  # dinamik toplam (şu an 12)

            vc1, vc2, vc3 = st.columns(3)
            vc1.metric("🟢 Yükseliş Oyu", f"{_tb} / {_total_strategies}")
            vc2.metric("🔴 Düşüş Oyu",   f"{_ts} / {_total_strategies}")
            vc3.metric("⚪ Nötr Oy",      f"{_tn} / {_total_strategies}")

            # Visual scoreboard table
            _strategy_rows = [
                ("Price Momentum",        "§3.1",      _votes.get("momentum", 0)),
                ("İkili Hareketli Ort.",  "§3.12",     _votes.get("moving_avg", 0)),
                ("Pivot Destek/Direnç",   "§3.14",     _votes.get("pivot", 0)),
                ("Donchian Kanal",        "§3.15",     _votes.get("donchian", 0)),
                ("Haber Duygusu NLP",     "§18.3",     _votes.get("sentiment", 0)),
                ("ASO Duygu Osc.",        "ASO",       _votes.get("aso", 0)),
                ("Bollinger + RSI",       "BB+RSI",    _votes.get("bb_rsi", 0)),
                ("MACD + EMA",            "Freqtrade", _votes.get("macd_ema", 0)),
                ("Adaptive MACD v2.4",   "MACD-v2.4", _votes.get("macd_v24", 0)),
                ("StochRSI v2.4",         "StRSI-v2.4",_votes.get("stochrsi_v24", 0)),
                ("WaveTrend v5.0 PRO",    "WT-v5.0",   _votes.get("wavetrend", 0)),
                ("Bollinger Hunter v5.7", "BBH-v5.7",  _votes.get("bbhunter", 0)),
            ]
            _arrows = {1: "▲ Yukarı", -1: "▼ Aşağı", 0: "━ Nötr"}
            _colors = {1: "#2ecc71",  -1: "#e74c3c",  0: "#7f8c8d"}

            table_html = "<table style='width:100%;border-collapse:collapse;'>"
            table_html += "<tr style='border-bottom:1px solid #2d3153;'><th style='text-align:left;color:#8892b0;padding:6px;'>Strateji</th><th style='color:#8892b0;'>Referans</th><th style='color:#8892b0;'>Sinyal</th></tr>"
            for name, ref, vote in _strategy_rows:
                color = _colors[vote]
                label = _arrows[vote]
                table_html += (f"<tr style='border-bottom:1px solid #1a1d2e;'>"
                               f"<td style='padding:7px 6px;color:#ccd6f6;'>{name}</td>"
                               f"<td style='text-align:center;color:#8892b0;font-size:0.8rem;'>{ref}</td>"
                               f"<td style='text-align:center;color:{color};font-weight:700;'>{label}</td></tr>")
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)

            st.divider()

            # ═══════════════════════════════════════════════════════════
            # 5. RISK MANAGEMENT PANEL
            # ═══════════════════════════════════════════════════════════
            st.subheader("🛡️ Risk Yönetimi Paneli")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("📉 Stop-Loss",     f"{_risk['suggested_stop_loss']:,.4f}")
            r2.metric("🎯 Take-Profit",   f"{_risk['suggested_take_profit']:,.4f}")
            r3.metric("⚖️ Risk/Ödül",     _risk["risk_reward_ratio"])
            r4.metric("🚦 Risk Seviyesi", _risk["risk_level"])

            # Contextual ATR note
            _atr_pct = _detail.get("atr_pct", 0)
            if _atr_pct > 3.0:
                st.error(f"⚠️ Piyasa **çok oynak** (ATR = %{_atr_pct:.2f}). Stop-Loss seviyeleri slipaj riskine karşı geniş tutulmuştur.")
            elif _atr_pct > 1.5:
                st.warning(f"💛 Orta oynaklık (ATR = %{_atr_pct:.2f}). Pozisyon büyüklüğünüzü buna göre sınırlayın.")
            else:
                st.success(f"✅ Düşük oynaklık (ATR = %{_atr_pct:.2f}). Stop-Loss seviyelerine güvenle yaklaşılabilir.")

            st.divider()

            # ═══════════════════════════════════════════════════════════
            # 6. RAW JSON OUTPUT (developer panel)
            # ═══════════════════════════════════════════════════════════
            import json as _json
            with st.expander("🔧 Ham JSON Çıktısı (Geliştiriciler İçin)"):
                _export = {k: v for k, v in result.items() if k != "_detail"}
                st.code(_json.dumps(_export, indent=2, ensure_ascii=False), language="json")

        except Exception as _e:
            st.error(f"Quant-AI Motor Hatası: {str(_e)}")
            import traceback
            st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# 📈  BACKTEST SİMÜLATÖRÜ
# quant_ai_engine sinyallerini geçmiş veri üzerinde simüle eder
# Look-ahead bias yok | Komisyon dahil | Equity Curve görselleştirme
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📈 Backtest Simülatörü":

    # ── CSS: Premium dark kart stili ─────────────────────────────────────────
    st.markdown("""
    <style>
    .bt-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #16192a 100%);
        border: 1px solid #2d3153; border-radius: 14px;
        padding: 18px 22px; margin-bottom: 10px;
    }
    .bt-label { font-size:0.72rem; color:#8892b0; letter-spacing:1px;
                text-transform:uppercase; margin-bottom:4px; }
    .bt-val-pos { font-size:1.6rem; font-weight:800; color:#2ecc71; }
    .bt-val-neg { font-size:1.6rem; font-weight:800; color:#e74c3c; }
    .bt-val-neu { font-size:1.6rem; font-weight:800; color:#ccd6f6; }
    .bt-sub     { font-size:0.78rem; color:#8892b0; margin-top:2px; }
    .bt-badge-buy  { display:inline-block; background:#0d3d23; color:#2ecc71;
                     border:1px solid #2ecc71; border-radius:6px;
                     padding:1px 8px; font-size:0.72rem; font-weight:700; }
    .bt-badge-sell { display:inline-block; background:#3d0d0d; color:#e74c3c;
                     border:1px solid #e74c3c; border-radius:6px;
                     padding:1px 8px; font-size:0.72rem; font-weight:700; }
    .bt-badge-hold { display:inline-block; background:#2a2d3a; color:#8892b0;
                     border:1px solid #4a4d5a; border-radius:6px;
                     padding:1px 8px; font-size:0.72rem; font-weight:700; }
    </style>
    """, unsafe_allow_html=True)

    st.header("📈 Backtest Simülatörü")
    st.markdown(
        """**Quant-AI Sinyal Motoru**'nun 5 akademik strateji sinyalini gerçek geçmiş veri 
        üzerinde simüle eder. **Look-ahead bias yok** — sinyal t günü kapanışından üretilir, 
        işlem t+1 gününün **açılış fiyatından** gerçekleşir. Komisyon her alım ve satımda ayrı uygulanır."""
    )

    # ── Sidebar: Backtest Parametreleri ──────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Backtest Parametreleri")

    bt_ticker = st.sidebar.text_input(
        "Hisse / Kripto Sembolü", "AAPL",
        help="Örn: AAPL, BTC-USD, THYAO.IS, SPY",
        key="bt_ticker_input"
    ).upper()

    bt_period = st.sidebar.selectbox(
        "Test Periyodu",
        options=["6mo", "1y", "2y"],
        index=1,
        format_func=lambda x: {"6mo": "Son 6 Ay", "1y": "Son 1 Yıl", "2y": "Son 2 Yıl"}[x],
        key="bt_period_select"
    )

    bt_capital = st.sidebar.number_input(
        "Başlangıç Sermayesi ($)",
        min_value=100.0, max_value=10_000_000.0,
        value=10_000.0, step=1000.0,
        format="%.2f",
        key="bt_capital_input"
    )

    bt_commission = st.sidebar.slider(
        "İşlem Komisyonu (%)",
        min_value=0.0, max_value=0.5,
        value=0.1, step=0.05,
        help="Her alım VE satım işlemi için ayrı ayrı uygulanır.",
        key="bt_comm_slider"
    )

    bt_slippage_factor = st.sidebar.slider(
        "Dinamik Slippage Çarpanı (ATR)",
        min_value=0.0, max_value=1.0,
        value=0.10, step=0.05,
        help="ATR_Pct * Çarpan oranında dolum fiyatı kaydırılır. 0 = Kayma yok.",
        key="bt_slip_factor_slider"
    )

    bt_start_btn = st.sidebar.button(
        "🚀 Backtest'i Başlat", type="primary", key="bt_start_btn"
    )

    if bt_start_btn:
        st.session_state["bt_run"]    = True
        st.session_state["bt_params"] = {
            "ticker"    : bt_ticker,
            "period"    : bt_period,
            "capital"   : bt_capital,
            "commission": bt_commission / 100.0,   # % → ondalık
            "slippage_factor": bt_slippage_factor,
        }

    # ── Çalıştırma Bloğu ─────────────────────────────────────────────────────
    if st.session_state.get("bt_run", False):
        params  = st.session_state.get("bt_params", {})
        _ticker = params.get("ticker", bt_ticker)
        _period = params.get("period", bt_period)
        _cap    = params.get("capital", bt_capital)
        _comm   = params.get("commission", bt_commission / 100.0)
        _slip_factor = params.get("slippage_factor", 0.10)

        try:
            from backtest_engine import BacktestEngine

            # Aşamalı progress bar
            prog = st.progress(0, text="📡 Piyasa verisi indiriliyor...")

            with st.spinner(f"📡 {_ticker} için {_period} veri çekiliyor..."):
                _bt_df, _bt_ticker = fetch_data(_ticker, period=_period)
                _bt_df = calculate_technical_indicators(_bt_df)
            prog.progress(35, text="🧮 Teknik göstergeler hesaplandı...")

            with st.spinner("⚙️ Vektörel sinyaller üretiliyor (look-ahead bias yok)..."):
                engine = BacktestEngine(
                    df=_bt_df,
                    initial_capital=_cap,
                    commission_pct=_comm,
                    slippage_factor=_slip_factor
                )
            prog.progress(65, text="📊 Serma eğrisi ve metrikler hesaplanıyor...")

            with st.spinner("📊 Simülasyon tamamlanıyor..."):
                bt_result = engine.run()
            prog.progress(100, text="✅ Backtest tamamlandı!")
            prog.empty()

            # ── Sonuçları aç ──────────────────────────────────────────────────
            _eq       = bt_result['equity_curve']
            _ret      = bt_result['total_return_pct']
            _hit      = bt_result['hit_rate_pct']
            _dd       = bt_result['max_drawdown_pct']
            _sharpe   = bt_result['sharpe_ratio']
            _sortino  = bt_result['sortino_ratio']
            _trades   = bt_result['total_trades']
            _final    = bt_result['final_capital']
            _bnh      = bt_result['buy_and_hold_pct']
            _log      = bt_result['trade_log']
            _sig_df   = bt_result['signals_df']
            _period_label = {"6mo": "6 Ay", "1y": "1 Yıl", "2y": "2 Yıl"}.get(_period, _period)

            # ════════════════════════════════════════════════════════════════
            # BÖLÜM 1: BAŞLIK & ÖZET BADGE
            # ════════════════════════════════════════════════════════════════
            badge_color = "#2ecc71" if _ret >= 0 else "#e74c3c"
            badge_icon  = "📈" if _ret >= 0 else "📉"
            vs_bnh_diff = round(_ret - _bnh, 2)
            vs_label    = f"+{vs_bnh_diff}%" if vs_bnh_diff >= 0 else f"{vs_bnh_diff}%"
            vs_color    = "#2ecc71" if vs_bnh_diff >= 0 else "#e74c3c"

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1a1d2e,#0e1117);
                        border:1px solid {badge_color}33; border-radius:16px;
                        padding:16px 24px; margin-bottom:12px;
                        box-shadow: 0 0 30px {badge_color}18;'>
                <div style='font-size:0.75rem;color:#8892b0;letter-spacing:2px;
                            text-transform:uppercase;'>Backtest Sonucu — {_bt_ticker} / {_period_label}</div>
                <div style='font-size:2.2rem;font-weight:900;
                            color:{badge_color};margin:6px 0;'>
                    {badge_icon} Net Getiri: {_ret:+.2f}%
                </div>
                <div style='font-size:0.85rem;color:#a8b2d8;'>
                    Başlangıç: <b>${_cap:,.2f}</b>
                    &nbsp;→&nbsp; Son Değer: <b style='color:{badge_color}'>${_final:,.2f}</b>
                    &nbsp;|&nbsp; Al-Tut (Buy &amp; Hold): <b>{_bnh:+.2f}%</b>
                    &nbsp;|&nbsp; Quant-AI farkı: <b style='color:{vs_color}'>{vs_label}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ════════════════════════════════════════════════════════════════
            # BÖLÜM 2: 4'LÜ METRİK KARTI SATIRI
            # ════════════════════════════════════════════════════════════════
            mc1, mc2, mc3, mc4 = st.columns(4)

            def _delta_color(val, inverse=False):
                """Pozitif → yeşil delta oku, negatif → kırmızı delta oku."""
                positive = val >= 0 if not inverse else val <= 0
                return "normal" if positive else "inverse"

            mc1.metric(
                "💰 Son Sermaye",
                f"${_final:,.2f}",
                f"{_ret:+.2f}%",
                delta_color=_delta_color(_ret)
            )
            mc2.metric(
                "🎯 Kazanma Oranı (Hit Rate)",
                f"%{_hit:.1f}",
                f"{_trades} işlem",
                delta_color="off"
            )
            mc3.metric(
                "📉 Max Drawdown",
                f"%{_dd:.2f}",
                delta_color="off"
            )
            mc4.metric(
                "📐 Sharpe / Sortino",
                f"{_sharpe:.2f} / {_sortino:.2f}",
                delta_color="off"
            )

            # İkinci satır: Al-tut karşılaştırması
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("🔄 Al-Tut Getirisi",   f"%{_bnh:+.2f}")
            rc2.metric("⚡ Quant-AI vs Al-Tut", f"%{vs_bnh_diff:+.2f}",
                       delta_color=_delta_color(vs_bnh_diff))
            rc3.metric("💸 Toplam Komisyon & Slippage", f"${(bt_result.get('total_slippage_cost', 0.0)):,.2f}", f"Slippage Faktör: {_slip_factor}x ATR")
            rc4.metric("📅 Test Periyodu",     _period_label)

            st.divider()

            # ════════════════════════════════════════════════════════════════
            # BÖLÜM 3: EQUITY CURVE (Sermaye Eğrisi) — Plotly İnteraktif
            # ════════════════════════════════════════════════════════════════
            st.subheader("📊 Sermaye Eğrisi (Equity Curve)")
            st.caption(
                "Günlük portföy değerinin zaman içindeki seyri. "
                "Açık pozisyonlar anlık piyasa fiyatından (mark-to-market) değerlenir."
            )

            # Buy & Hold karşılaştırma serisi
            bnh_series = _bt_df.loc[_eq.index, 'Close'] / float(_bt_df['Close'].iloc[0]) * _cap

            fig_eq = go.Figure()

            # Quant-AI eğrisi
            fig_eq.add_trace(go.Scatter(
                x=_eq.index, y=_eq.values,
                name="🧠 Quant-AI Stratejisi",
                line=dict(color="#2ecc71" if _ret >= 0 else "#e74c3c", width=2.5),
                fill='tozeroy',
                fillcolor="rgba(46,204,113,0.06)" if _ret >= 0 else "rgba(231,76,60,0.06)",
                hovertemplate="%{x|%d %b %Y}<br>Sermaye: $%{y:,.2f}<extra></extra>"
            ))

            # Al-Tut karşılaştırma eğrisi
            try:
                fig_eq.add_trace(go.Scatter(
                    x=bnh_series.index, y=bnh_series.values,
                    name="📌 Al & Tut (Buy & Hold)",
                    line=dict(color="#f39c12", width=1.5, dash="dot"),
                    hovertemplate="%{x|%d %b %Y}<br>Al-Tut: $%{y:,.2f}<extra></extra>"
                ))
            except Exception:
                pass

            # Başlangıç referans çizgisi
            fig_eq.add_hline(
                y=_cap, line_dash="dash",
                line_color="#4a4d5a", line_width=1,
                annotation_text=f"Başlangıç: ${_cap:,.0f}",
                annotation_position="bottom left"
            )

            fig_eq.update_layout(
                template="plotly_dark",
                height=420,
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                legend=dict(orientation="h", yanchor="bottom", y=1.01,
                            xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(showgrid=True, gridcolor="#1e2130",
                           zeroline=False, tickformat="%b %Y"),
                yaxis=dict(showgrid=True, gridcolor="#1e2130",
                           zeroline=False, tickprefix="$", tickformat=",.0f"),
                hovermode="x unified"
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # Drawdown grafiği
            roll_max_eq  = _eq.cummax()
            drawdown_ser = (_eq - roll_max_eq) / roll_max_eq * 100

            st.markdown(
                "<p style='font-size:0.9rem;color:#8892b0;margin:4px 0 2px 0;'>"
                "📉 <b>Geri Çekilme (Drawdown) Grafiği</b></p>",
                unsafe_allow_html=True
            )
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=drawdown_ser.index, y=drawdown_ser.values,
                name="Max Drawdown",
                fill='tozeroy', fillcolor="rgba(231,76,60,0.15)",
                line=dict(color="#e74c3c", width=1.5),
                hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>"
            ))
            fig_dd.update_layout(
                template="plotly_dark", height=200,
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(l=0, r=0, t=4, b=0),
                xaxis=dict(showgrid=True, gridcolor="#1e2130", tickformat="%b %Y"),
                yaxis=dict(showgrid=True, gridcolor="#1e2130", ticksuffix="%"),
                showlegend=False
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            st.divider()

            # ════════════════════════════════════════════════════════════════
            # BÖLÜM 4: DETAY PANELLERI (tab yapısı)
            # ════════════════════════════════════════════════════════════════
            det_tab1, det_tab2 = st.tabs(["📋 İşlem Geçmişi (Trade Log)",
                                          "📡 Günlük Sinyal Tablosu"])

            with det_tab1:
                st.caption(
                    "Her satır bir tamamlanmış alım veya satım işlemini gösterir. "
                    "Komisyon her işlemde net fiyata yansıtılmıştır."
                )
                if _log:
                    log_df = pd.DataFrame(_log)
                    # Sinyale göre renkli rozet
                    def _badge_html(tip):
                        if "ALIŞ" in str(tip):    return "<span class='bt-badge-buy'>ALIŞ</span>"
                        if "SATIŞ" in str(tip):   return "<span class='bt-badge-sell'>SATIŞ</span>"
                        return "<span class='bt-badge-hold'>KAPANIŞ</span>"

                    log_df['_badge'] = log_df['tip'].apply(_badge_html)

                    # Gösterilecek sütunlar
                    show_cols = [c for c in
                                 ['işlem_no', 'tip', 'tarih', 'fiyat', 'ham_fiyat', 'slippage',
                                  'getiri_pct', 'komisyon_pct', 'sl', 'tp', 'not']
                                 if c in log_df.columns]
                    rename_map = {
                        'işlem_no':    '#',
                        'tip':         'İşlem Tipi',
                        'tarih':       'Tarih',
                        'fiyat':       'Gerçekleşen Fiyat',
                        'ham_fiyat':   'Ham Fiyat',
                        'slippage':    'Kayma (Slip)',
                        'getiri_pct':  'Getiri %',
                        'komisyon_pct':'Komisyon %',
                        'sl':          'Stop-Loss',
                        'tp':          'Take-Profit',
                        'not':         'Not',
                    }
                    st.dataframe(
                        log_df[show_cols].rename(columns=rename_map),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("Bu periyotta sinyal eşiğini (≥3/5 oy) aşan işlem oluşmadı. "
                            "Daha uzun bir test periyodu deneyin.")

            with det_tab2:
                st.caption(
                    "Her gün için Quant-AI'ın ürettiği oy ve sinyal dağılımı. "
                    "+1 = Yukarı oy, -1 = Aşağı oy, 0 = Nötr."
                )
                # En son 100 günü göster (performans için)
                show_sig = _sig_df.tail(100).copy()

                # Sinyal sütununu emoji ile zenginleştir
                sig_map = {'BUY': '🟢 AL', 'SELL': '🔴 SAT', 'HOLD': '⚪ BEKLE'}
                show_sig['Sinyal'] = show_sig['Sinyal'].map(sig_map).fillna('⚪ BEKLE')

                st.dataframe(
                    show_sig,
                    use_container_width=True,
                    height=380
                )
                st.caption("ℹ️ Tablo son 100 işlem gününü göstermektedir.")

        except Exception as _bt_err:
            st.error(f"Backtest Motoru Hatası: {str(_bt_err)}")
            import traceback
            st.code(traceback.format_exc())

elif mode == "📊 Performans Raporu (Quantstats)":
    st.header("📊 Detaylı Performans & Risk Analizi")
    st.markdown("Herhangi bir hisse veya portföy için Quantstats tarzı kapsamlı rapor üretin.")

    qs_col1, qs_col2 = st.columns([2, 1])
    with qs_col1:
        qs_ticker  = st.text_input("Hisse / Kripto Sembolü", "THYAO.IS", key="qs_ticker").upper()
    with qs_col2:
        qs_period  = st.selectbox("Dönem", ["1y", "2y", "3y", "5y"], index=1, key="qs_period")

    if st.button("📊 Rapor Oluştur", type="primary"):
        with st.spinner(f"{qs_ticker} için rapor hazırlanıyor..."):
            try:
                import plotly.graph_objects as go
                import plotly.express as px

                df_qs, final_qs = fetch_data(qs_ticker, period=qs_period)
                close = df_qs["Close"].squeeze()
                rets  = close.pct_change().dropna()
                TD    = 252

                # Temel metrikler
                ann_ret  = (1 + rets.mean()) ** TD - 1
                ann_vol  = rets.std() * (TD ** 0.5)
                sharpe   = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
                downside = rets[rets < 0].std() * (TD ** 0.5)
                sortino  = (ann_ret - 0.02) / downside if downside > 0 else 0
                roll_max = close.cummax()
                dd_s     = (close - roll_max) / roll_max * 100
                max_dd   = dd_s.min()
                calmar   = ann_ret / abs(max_dd / 100) if max_dd != 0 else 0
                win_rate = (rets > 0).sum() / len(rets) * 100
                skew     = float(rets.skew())
                kurt     = float(rets.kurt())
                best_day = rets.max() * 100
                worst_day= rets.min() * 100
                total_ret= (close.iloc[-1] / close.iloc[0] - 1) * 100

                # Başlık badge
                bc = "#2ecc71" if total_ret >= 0 else "#e74c3c"
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#1a1d2e,#0e1117);
                            border:1px solid {bc}44;border-radius:16px;padding:16px 24px;margin-bottom:16px'>
                    <div style='font-size:0.75rem;color:#8892b0;letter-spacing:2px;text-transform:uppercase'>
                        Performans Raporu — {final_qs} / {qs_period}
                    </div>
                    <div style='font-size:2rem;font-weight:900;color:{bc};margin:6px 0'>
                        {"📈" if total_ret >= 0 else "📉"} {total_ret:+.2f}% Toplam Getiri
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Metrik kartları
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Yıllık Getiri",  f"%{ann_ret*100:.1f}")
                c2.metric("Sharpe",          f"{sharpe:.2f}")
                c3.metric("Sortino",         f"{sortino:.2f}")
                c4.metric("Calmar",          f"{calmar:.2f}")

                c5,c6,c7,c8 = st.columns(4)
                c5.metric("Max Drawdown",    f"%{max_dd:.1f}")
                c6.metric("Yıllık Volatilite",f"%{ann_vol*100:.1f}")
                c7.metric("Kazanma Günü",    f"%{win_rate:.1f}")
                c8.metric("Skewness",        f"{skew:.2f}")

                # Aylık getiri ısı haritası
                st.markdown("---")
                st.markdown("##### 📅 Aylık Getiri Isı Haritası (%)")
                monthly    = rets.resample("ME").apply(lambda x: (1+x).prod()-1) * 100
                monthly_df = monthly.reset_index()
                monthly_df.columns = ["Tarih","Getiri"]
                monthly_df["Ay"]  = monthly_df["Tarih"].dt.strftime("%b")
                monthly_df["Yıl"] = monthly_df["Tarih"].dt.year
                ay_s = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                pivot = monthly_df.pivot_table(index="Yıl", columns="Ay", values="Getiri", aggfunc="first")
                pivot = pivot.reindex(columns=[a for a in ay_s if a in pivot.columns])
                st.dataframe(
                    pivot.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.1f}%", na_rep="-"),
                    use_container_width=True
                )

                # Fiyat + Drawdown grafikleri yan yana
                st.markdown("---")
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines",
                        line=dict(color="#00d4ff", width=2), name="Fiyat"))
                    fig_p.update_layout(title="Fiyat Grafiği", height=280,
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font=dict(color="#e0e0e0"), margin=dict(l=10,r=10,t=40,b=10))
                    st.plotly_chart(fig_p, use_container_width=True)

                with gcol2:
                    fig_d = go.Figure()
                    fig_d.add_trace(go.Scatter(x=dd_s.index, y=dd_s.values, mode="lines",
                        fill="tozeroy", line=dict(color="#e74c3c", width=1),
                        fillcolor="rgba(231,76,60,0.2)", name="Drawdown"))
                    fig_d.update_layout(title="Drawdown (%)", height=280,
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font=dict(color="#e0e0e0"), margin=dict(l=10,r=10,t=40,b=10))
                    st.plotly_chart(fig_d, use_container_width=True)

                # Getiri dağılımı
                fig_hist = px.histogram(rets * 100, nbins=60, title="Günlük Getiri Dağılımı (%)",
                    color_discrete_sequence=["#00d4ff"])
                fig_hist.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font=dict(color="#e0e0e0"), height=280, showlegend=False,
                    xaxis_title="Günlük Getiri (%)", yaxis_title="Frekans"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            except Exception as e:
                st.error(f"Rapor hatası: {e}")
                import traceback; st.code(traceback.format_exc())

elif mode == "📱 Telegram Bot Ayarları":
    st.header("📱 Telegram Bot Entegrasyonu")
    st.markdown("Botun ürettiği alım/satım sinyallerini Telegram'a anlık bildirim olarak al.")

    with st.expander("📖 Nasıl Kurulur?", expanded=False):
        st.markdown("""
        **Adım 1:** Telegram'da [@BotFather](https://t.me/BotFather)'a git ve `/newbot` komutunu gönder.

        **Adım 2:** Bot adı ve kullanıcı adı belirle. BotFather sana bir **API Token** verecek.

        **Adım 3:** Chat ID almak için botuna herhangi bir mesaj gönder, 
        sonra tarayıcıdan `https://api.telegram.org/bot<TOKEN>/getUpdates` adresini aç.
        Gelen JSON'daki `"chat":{"id":...}` değeri senin **Chat ID**'ndir.

        **Adım 4:** Token ve Chat ID'yi aşağıya gir, test mesajı gönder.
        """)

    tg_col1, tg_col2 = st.columns(2)
    with tg_col1:
        tg_token_input = st.text_input(
            "🔑 Bot API Token",
            value=st.session_state.get("tg_bot_token", ""),
            type="password",
            key="tg_token_input",
            placeholder="1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
        )
    with tg_col2:
        tg_chat_input = st.text_input(
            "💬 Chat ID",
            value=st.session_state.get("tg_chat_id", ""),
            key="tg_chat_input",
            placeholder="-100123456789 veya 123456789"
        )

    st.markdown("##### 🔔 Bildirim Tercihleri")
    n1, n2, n3 = st.columns(3)
    tg_notify_buy  = n1.checkbox("AL Sinyali", value=True)
    tg_notify_sell = n2.checkbox("SAT Sinyali", value=True)
    tg_notify_risk = n3.checkbox("Risk Limiti Aşıldı", value=True)

    tg_col_save, tg_col_test = st.columns(2)

    with tg_col_save:
        if st.button("💾 Kaydet", type="primary"):
            st.session_state["tg_bot_token"]    = tg_token_input
            st.session_state["tg_chat_id"]      = tg_chat_input
            st.session_state["tg_notify_buy"]   = tg_notify_buy
            st.session_state["tg_notify_sell"]  = tg_notify_sell
            st.session_state["tg_notify_risk"]  = tg_notify_risk
            st.success("✅ Telegram ayarları kaydedildi! Paper Trading botunu çalıştırdığında bildirim alacaksın.")

    with tg_col_test:
        if st.button("📤 Test Mesajı Gönder"):
            _tok = tg_token_input or st.session_state.get("tg_bot_token", "")
            _cid = tg_chat_input  or st.session_state.get("tg_chat_id",  "")
            if not _tok or not _cid:
                st.error("Önce Token ve Chat ID gir!")
            else:
                try:
                    import requests as _req
                    resp = _req.post(
                        f"https://api.telegram.org/bot{_tok}/sendMessage",
                        json={
                            "chat_id": _cid,
                            "text": (
                                "✅ *Hisse Avcısı — Bağlantı Testi*\n"
                                "Telegram entegrasyonu başarıyla kuruldu!\n"
                                "Artık alım/satım sinyallerini buradan alacaksın. 🚀"
                            ),
                            "parse_mode": "Markdown"
                        },
                        timeout=8
                    )
                    if resp.status_code == 200:
                        st.success("✅ Test mesajı gönderildi! Telegram'ı kontrol et.")
                    else:
                        st.error(f"Gönderim başarısız: {resp.text}")
                except Exception as te:
                    st.error(f"Bağlantı hatası: {te}")

    # Mevcut durum
    if st.session_state.get("tg_bot_token"):
        st.markdown("---")
        st.success("✅ Telegram bağlantısı aktif — Paper Trading botundaki işlemler bildirim gönderecek.")
    else:
        st.info("ℹ️ Telegram ayarlanmadı. Yukarıdan token ve chat ID girerek aktif et.")

