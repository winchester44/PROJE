import streamlit.components.v1 as components
import json
import pandas as pd
import numpy as np

def map_ticker_to_tradingview(ticker: str) -> str:
    if not ticker:
        return "BIST:THYAO"
    ticker = ticker.upper().strip()
    if ticker.endswith('.IS'):
        base = ticker.replace('.IS', '')
        return f"BIST:{base}"
    elif '-USD' in ticker:
        base = ticker.replace('-USD', '')
        return f"BINANCE:{base}USD"
    elif '-USDT' in ticker:
        base = ticker.replace('-USDT', '')
        return f"BINANCE:{base}USDT"
    else:
        return ticker

def render_lightweight_chart(df, ticker_symbol=None, height=550):
    """
    Renders TradingView advanced chart widget as an iframe if ticker_symbol is provided.
    Falls back to the local Lightweight Charts implementation using df if ticker_symbol is None
    or if the stock is a BIST stock (ends with .IS) due to TradingView embedding restrictions.
    """
    if ticker_symbol and not ticker_symbol.upper().strip().endswith('.IS'):
        tv_symbol = map_ticker_to_tradingview(ticker_symbol)
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    height: 100%;
                    background-color: #0E1117;
                    overflow: hidden;
                }}
                .tradingview-widget-container {{
                    width: 100%;
                    height: 100%;
                }}
            </style>
        </head>
        <body>
            <div class="tradingview-widget-container">
                <div id="tradingview_chart" style="width: 100%; height: 100%;"></div>
                <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                <script type="text/javascript">
                new TradingView.widget({{
                    "width": "100%",
                    "height": "100%",
                    "symbol": "{tv_symbol}",
                    "interval": "D",
                    "timezone": "Europe/Istanbul",
                    "theme": "dark",
                    "style": "1",
                    "locale": "tr",
                    "enable_publishing": false,
                    "hide_side_toolbar": false,
                    "allow_symbol_change": true,
                    "container_id": "tradingview_chart"
                }});
                </script>
            </div>
        </body>
        </html>
        """
        return components.html(html_code, height=height)

    # Fallback to local Lightweight Charts (original code logic)
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index(ascending=True)
        
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df['Volume'] = df['Volume'].fillna(0)
    
    if df.empty:
        return components.html("<div style='color: white; font-family: sans-serif; text-align: center; padding-top: 50px;'>Geçerli fiyat verisi bulunamadı.</div>", height=height)

    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.strftime('%Y-%m-%d').tolist()
    else:
        dates = df['Date'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d')).tolist()
        
    opens = df['Open'].tolist()
    highs = df['High'].tolist()
    lows = df['Low'].tolist()
    closes = df['Close'].tolist()
    volumes = df['Volume'].tolist()
    
    candle_data = []
    volume_data = []
    
    for i in range(len(dates)):
        color = 'rgba(38, 166, 154, 0.5)' if closes[i] >= opens[i] else 'rgba(239, 83, 80, 0.5)'
        
        candle_data.append({
            "time": dates[i],
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i]
        })
        
        volume_data.append({
            "time": dates[i],
            "value": volumes[i],
            "color": color
        })
        
    candle_json = json.dumps(candle_data)
    volume_json = json.dumps(volume_data)
    
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {
                margin: 0;
                padding: 0;
                background-color: #0E1117;
                color: #d1d4dc;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
            }
            #chart-container {
                width: 100vw;
                height: 100vh;
                position: relative;
            }
            #watermark {
                position: absolute;
                bottom: 20px;
                left: 20px;
                color: rgba(255, 255, 255, 0.1);
                font-size: 24px;
                font-weight: bold;
                pointer-events: none;
                z-index: 10;
            }
        </style>
    </head>
    <body>
        <div id="chart-container">
            <div id="watermark">AI Trading Platform</div>
        </div>
        <script>
            const container = document.getElementById('chart-container');
            
            const chartOptions = {
                width: container.clientWidth || window.innerWidth,
                height: container.clientHeight || window.innerHeight,
                autoSize: true,
                layout: {
                    textColor: '#d1d4dc',
                    background: { type: 'solid', color: '#0E1117' },
                },
                grid: {
                    vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
                    horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
                },
                crosshair: {
                    mode: LightweightCharts.CrosshairMode.Normal,
                },
                rightPriceScale: {
                    borderColor: 'rgba(197, 203, 206, 0.8)',
                    autoScale: true,
                },
                timeScale: {
                    borderColor: 'rgba(197, 203, 206, 0.8)',
                    timeVisible: true,
                    secondsVisible: false,
                },
                handleScroll: {
                    vertTouchDrag: false,
                },
            };
            
            const chart = LightweightCharts.createChart(container, chartOptions);
            
            const candlestickSeries = chart.addCandlestickSeries({
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderVisible: false,
                wickUpColor: '#26a69a',
                wickDownColor: '#ef5350',
            });
            
            candlestickSeries.setData(__CANDLE_JSON__);
            
            const volumeSeries = chart.addHistogramSeries({
                color: '#26a69a',
                priceFormat: {
                    type: 'volume',
                },
                priceScaleId: '',
                scaleMargins: {
                    top: 0.8,
                    bottom: 0,
                },
            });
            
            volumeSeries.setData(__VOLUME_JSON__);
            
            chart.timeScale().fitContent();
            
            window.addEventListener('resize', () => {
                chart.applyOptions({ 
                    width: container.clientWidth || window.innerWidth,
                    height: container.clientHeight || window.innerHeight
                });
            });
        </script>
    </body>
    </html>
    """
    
    html_code = html_code.replace('__CANDLE_JSON__', candle_json)
    html_code = html_code.replace('__VOLUME_JSON__', volume_json)
    
    return components.html(html_code, height=height)
