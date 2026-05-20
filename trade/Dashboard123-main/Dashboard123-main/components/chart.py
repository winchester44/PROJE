import json
import streamlit as st
import streamlit.components.v1 as components

# yfinance suffix -> (TradingView exchange prefix, share-class separator)
# Yahoo always uses dashes for share classes (BERNER-B.ST, AKT-A.TO).
# TradingView's separator varies by exchange:
#   Nordic = underscore  (OMXSTO:BERNER_B)
#   Canada = dot         (TSX:AKT.A)
#   Most others = dot
_YF_TO_TV = {
    ".ST": ("OMXSTO", "_"),
    ".OL": ("OSL", "_"),
    ".HE": ("OMXHEX", "_"),
    ".CO": ("OMXCOP", "_"),
    ".DE": ("XETR", "."),
    ".L":  ("LSE", "."),
    ".PA": ("EPA", "."),
    ".AX": ("ASX", "."),
    ".T":  ("TSE", "."),
    ".HK": ("HKEX", "."),
    ".SI": ("SGX", "."),
    ".AS": ("EURONEXT", "."),
    ".BR": ("EBR", "."),
    ".MC": ("BME", "."),
    ".MI": ("MIL", "."),
    ".SW": ("SIX", "."),
    ".VI": ("VIE", "."),
    ".IR": ("ISE", "."),
    ".LS": ("ELI", "."),
    ".AT": ("ATHEX", "."),
    ".LU": ("LUXSE", "."),
    ".NZ": ("NZX", "."),
    ".TA": ("TASE", "."),
    ".TO": ("TSX", "."),
    ".V":  ("TSXV", "."),
    ".IS": ("BIST", ""),
}


def _to_tradingview_symbol(ticker: str) -> str:
    """Convert yfinance ticker to TradingView symbol format.

    Yahoo uses dashes for share-class separators (BERNER-B.ST, AKT-A.TO).
    TradingView uses underscores for Nordic exchanges (OMXSTO:BERNER_B)
    and dots for most others including Canada (TSX:AKT.A).
    """
    # Handle special tickers
    if ticker.startswith("^"):
        return f"TVC:{ticker[1:]}"

    # Check for yfinance exchange suffixes
    for suffix, (tv_exchange, sep) in _YF_TO_TV.items():
        if ticker.endswith(suffix):
            symbol = ticker[: -len(suffix)]
            symbol = symbol.replace("-", sep)
            return f"{tv_exchange}:{symbol}"

    # No suffix = US ticker
    # Try to add common exchange prefixes if known, or let TradingView resolve
    # For now, we'll keep it as is but ensure it's clean
    return ticker.strip().upper()


def render_tradingview_chart(symbol: str, theme: str = "dark", height: int = 620):
    """Render TradingView Advanced Chart widget (embed-widget format)."""
    tv_theme = "dark" if theme == "dark" else "light"
    tv_symbol = _to_tradingview_symbol(symbol)

    widget_config = json.dumps({
        "autosize": True,
        "symbol": tv_symbol,
        "interval": "W",
        "range": "12M",
        "timezone": "America/New_York",
        "theme": tv_theme,
        "style": "1",
        "locale": "en",
        "hide_side_toolbar": False,
        "allow_symbol_change": True,
        "studies": [
            "STD;MA%Ribbon",
            "Volume@tv-basicstudies",
        ],
        "support_host": "https://www.tradingview.com",
    })

    chart_html = f"""
    <div style="height:{height}px;width:100%;">
    <div class="tradingview-widget-container" style="height:100%;width:100%;">
      <div class="tradingview-widget-container__widget" style="height:100%;width:100%;"></div>
      <script type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
        async>{widget_config}</script>
    </div>
    </div>
    """
    # Use components.html (iframe) for better script execution in Streamlit
    components.html(chart_html, height=height + 20)
