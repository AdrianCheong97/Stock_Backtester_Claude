"""
╔══════════════════════════════════════════════════════════╗
║  STOCK SCREENER  —  Swing Trade Backtester Suite        ║
║  TradingView-style filters + Sharpe Ratio screening     ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Screener · Swing Backtester",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  STYLES  (same dark terminal aesthetic as main app)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"]       { font-family:'IBM Plex Sans',sans-serif; background:#0d0f14; color:#c9d1d9; }
.stApp                           { background:#0d0f14; }
section[data-testid="stSidebar"] { background:#111318; border-right:1px solid #21262d; }

h1 { font-family:'IBM Plex Mono',monospace !important; color:#58a6ff !important; letter-spacing:-1px; }
h2,h3 { font-family:'IBM Plex Mono',monospace !important; color:#79c0ff !important; }

[data-testid="metric-container"] {
  background:#161b22; border:1px solid #21262d; border-radius:8px; padding:12px 16px;
}
[data-testid="metric-container"] label {
  color:#8b949e !important; font-size:.72rem !important; letter-spacing:.08em; text-transform:uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family:'IBM Plex Mono',monospace; font-size:1.3rem !important; color:#e6edf3 !important;
}

.stButton > button {
  background:#238636; color:#fff; border:none; border-radius:6px;
  font-family:'IBM Plex Mono',monospace; font-weight:600;
  padding:10px 24px; width:100%; transition:background .2s;
}
.stButton > button:hover { background:#2ea043; }

.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stMultiSelect > div > div {
  background:#161b22 !important; border:1px solid #30363d !important;
  color:#e6edf3 !important; font-family:'IBM Plex Mono',monospace;
}

.stSlider > div > div > div > div { background:#238636 !important; }

.stTabs [data-baseweb="tab-list"] { background:#161b22; border-bottom:1px solid #21262d; gap:0; }
.stTabs [data-baseweb="tab"] {
  font-family:'IBM Plex Mono',monospace; color:#8b949e;
  border-bottom:2px solid transparent; padding:10px 20px; font-size:.82rem;
}
.stTabs [aria-selected="true"] {
  color:#58a6ff !important; border-bottom:2px solid #58a6ff !important; background:transparent !important;
}

.stDataFrame { background:#161b22; border:1px solid #21262d; border-radius:8px; }
.stInfo  { background:#161b22; border-left:3px solid #58a6ff; }
hr       { border-color:#21262d; }

.filter-section {
  background:#161b22; border:1px solid #21262d; border-radius:8px;
  padding:14px 16px; margin-bottom:10px;
}
.filter-section h4 {
  font-family:'IBM Plex Mono',monospace; font-size:.8rem;
  color:#58a6ff; margin:0 0 10px 0; text-transform:uppercase; letter-spacing:.08em;
}
.tag { display:inline-block; background:#1f2937; border:1px solid #30363d; border-radius:12px; padding:2px 10px; font-size:.72rem; font-family:'IBM Plex Mono',monospace; color:#8b949e; margin:2px; }
.tag-green { border-color:#238636; color:#3fb950; background:#0d1f0e; }
.tag-blue  { border-color:#1f6feb; color:#58a6ff; background:#0d1929; }
.result-count { font-family:'IBM Plex Mono',monospace; font-size:.85rem; color:#8b949e; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

BG    = "#0d0f14"
GRID  = "#1a1f28"
MUTED = "#8b949e"
PLOTLY_CFG = dict(scrollZoom=True, displaylogo=False,
                  modeBarButtonsToRemove=["select2d", "lasso2d"])

# ─────────────────────────────────────────────────────────────────────────────
#  UNIVERSE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Curated universe organised by sector — extend as needed
UNIVERSE = {
    "Technology": {
        "AAPL":  ("Apple Inc.", "NASDAQ", "Mega Cap"),
        "MSFT":  ("Microsoft Corp.", "NASDAQ", "Mega Cap"),
        "NVDA":  ("NVIDIA Corp.", "NASDAQ", "Large Cap"),
        "META":  ("Meta Platforms", "NASDAQ", "Mega Cap"),
        "GOOGL": ("Alphabet Inc.", "NASDAQ", "Mega Cap"),
        "AVGO":  ("Broadcom Inc.", "NASDAQ", "Large Cap"),
        "ORCL":  ("Oracle Corp.", "NYSE", "Large Cap"),
        "CRM":   ("Salesforce Inc.", "NYSE", "Large Cap"),
        "AMD":   ("Advanced Micro Devices", "NASDAQ", "Large Cap"),
        "INTC":  ("Intel Corp.", "NASDAQ", "Large Cap"),
        "QCOM":  ("Qualcomm Inc.", "NASDAQ", "Large Cap"),
        "ADBE":  ("Adobe Inc.", "NASDAQ", "Large Cap"),
        "NOW":   ("ServiceNow Inc.", "NYSE", "Large Cap"),
        "SNOW":  ("Snowflake Inc.", "NYSE", "Mid Cap"),
        "PLTR":  ("Palantir Technologies", "NYSE", "Mid Cap"),
    },
    "Financials": {
        "JPM":   ("JPMorgan Chase", "NYSE", "Mega Cap"),
        "BAC":   ("Bank of America", "NYSE", "Large Cap"),
        "GS":    ("Goldman Sachs", "NYSE", "Large Cap"),
        "MS":    ("Morgan Stanley", "NYSE", "Large Cap"),
        "V":     ("Visa Inc.", "NYSE", "Mega Cap"),
        "MA":    ("Mastercard Inc.", "NYSE", "Mega Cap"),
        "AXP":   ("American Express", "NYSE", "Large Cap"),
        "BLK":   ("BlackRock Inc.", "NYSE", "Large Cap"),
        "WFC":   ("Wells Fargo", "NYSE", "Large Cap"),
        "C":     ("Citigroup Inc.", "NYSE", "Large Cap"),
    },
    "Healthcare": {
        "JNJ":   ("Johnson & Johnson", "NYSE", "Mega Cap"),
        "UNH":   ("UnitedHealth Group", "NYSE", "Mega Cap"),
        "LLY":   ("Eli Lilly & Co.", "NYSE", "Mega Cap"),
        "PFE":   ("Pfizer Inc.", "NYSE", "Large Cap"),
        "ABBV":  ("AbbVie Inc.", "NYSE", "Large Cap"),
        "MRK":   ("Merck & Co.", "NYSE", "Large Cap"),
        "TMO":   ("Thermo Fisher Scientific", "NYSE", "Large Cap"),
        "DHR":   ("Danaher Corp.", "NYSE", "Large Cap"),
        "AMGN":  ("Amgen Inc.", "NASDAQ", "Large Cap"),
        "ISRG":  ("Intuitive Surgical", "NASDAQ", "Large Cap"),
    },
    "Consumer Discretionary": {
        "AMZN":  ("Amazon.com Inc.", "NASDAQ", "Mega Cap"),
        "TSLA":  ("Tesla Inc.", "NASDAQ", "Mega Cap"),
        "HD":    ("Home Depot Inc.", "NYSE", "Large Cap"),
        "NKE":   ("Nike Inc.", "NYSE", "Large Cap"),
        "MCD":   ("McDonald's Corp.", "NYSE", "Large Cap"),
        "SBUX":  ("Starbucks Corp.", "NASDAQ", "Large Cap"),
        "TGT":   ("Target Corp.", "NYSE", "Large Cap"),
        "LOW":   ("Lowe's Companies", "NYSE", "Large Cap"),
        "BKNG":  ("Booking Holdings", "NASDAQ", "Large Cap"),
    },
    "Energy": {
        "XOM":   ("Exxon Mobil Corp.", "NYSE", "Mega Cap"),
        "CVX":   ("Chevron Corp.", "NYSE", "Large Cap"),
        "COP":   ("ConocoPhillips", "NYSE", "Large Cap"),
        "SLB":   ("SLB (Schlumberger)", "NYSE", "Large Cap"),
        "EOG":   ("EOG Resources", "NYSE", "Large Cap"),
        "PSX":   ("Phillips 66", "NYSE", "Large Cap"),
    },
    "Industrials": {
        "CAT":   ("Caterpillar Inc.", "NYSE", "Large Cap"),
        "BA":    ("Boeing Co.", "NYSE", "Large Cap"),
        "GE":    ("GE Aerospace", "NYSE", "Large Cap"),
        "HON":   ("Honeywell Intl.", "NASDAQ", "Large Cap"),
        "UPS":   ("United Parcel Service", "NYSE", "Large Cap"),
        "RTX":   ("RTX Corp.", "NYSE", "Large Cap"),
        "LMT":   ("Lockheed Martin", "NYSE", "Large Cap"),
        "DE":    ("Deere & Company", "NYSE", "Large Cap"),
    },
    "ETFs & Indices": {
        "SPY":   ("SPDR S&P 500 ETF", "NYSE", "ETF"),
        "QQQ":   ("Invesco QQQ Trust", "NASDAQ", "ETF"),
        "IWM":   ("iShares Russell 2000", "NYSE", "ETF"),
        "DIA":   ("SPDR Dow Jones ETF", "NYSE", "ETF"),
        "GLD":   ("SPDR Gold Shares", "NYSE", "ETF"),
        "TLT":   ("iShares 20+ Yr Treasury", "NASDAQ", "ETF"),
        "VXX":   ("iPath VIX Short-Term", "CBOE", "ETF"),
        "ARKK":  ("ARK Innovation ETF", "NYSE", "ETF"),
        "SOXX":  ("iShares Semiconductor", "NASDAQ", "ETF"),
        "XLE":   ("Energy Select SPDR", "NYSE", "ETF"),
    },
}

ALL_SECTORS   = sorted(UNIVERSE.keys())
ALL_EXCHANGES = ["NASDAQ", "NYSE", "CBOE"]
ALL_CAP_SIZES = ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "ETF"]

# ─────────────────────────────────────────────────────────────────────────────
#  DATA & INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price(ticker: str, period_days: int = 365) -> pd.DataFrame:
    try:
        import yfinance as yf
        end   = datetime.today()
        start = end - timedelta(days=period_days + 60)  # extra buffer for indicators
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index   = pd.to_datetime(df.index)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


def _ema(s, n):   return s.ewm(span=n, adjust=False).mean()
def _sma(s, n):   return s.rolling(n).mean()
def _atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()
def _rsi(s, n=14):
    d = s.diff(); g = d.clip(lower=0).rolling(n).mean(); lo = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / lo.replace(0, np.nan))

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute all screening metrics from OHLCV dataframe."""
    if df.empty or len(df) < 30:
        return {}

    c = df["Close"]
    v = df["Volume"]
    last = c.iloc[-1]

    # Returns
    ret_1d  = (last / c.iloc[-2]  - 1) * 100 if len(c) > 1  else np.nan
    ret_5d  = (last / c.iloc[-6]  - 1) * 100 if len(c) > 5  else np.nan
    ret_1m  = (last / c.iloc[-22] - 1) * 100 if len(c) > 21 else np.nan
    ret_3m  = (last / c.iloc[-66] - 1) * 100 if len(c) > 65 else np.nan
    ret_6m  = (last / c.iloc[-126]- 1) * 100 if len(c) > 125 else np.nan
    ret_ytd = (last / c.iloc[-252]- 1) * 100 if len(c) > 251 else np.nan

    # 52-week hi/lo
    w52_hi = c.iloc[-252:].max() if len(c) > 251 else c.max()
    w52_lo = c.iloc[-252:].min() if len(c) > 251 else c.min()
    pct_from_hi = (last / w52_hi - 1) * 100
    pct_from_lo = (last / w52_lo - 1) * 100

    # Moving averages
    sma50  = _sma(c, 50).iloc[-1]  if len(c) >= 50  else np.nan
    sma200 = _sma(c, 200).iloc[-1] if len(c) >= 200 else np.nan
    ema20  = _ema(c, 20).iloc[-1]  if len(c) >= 20  else np.nan
    ema50  = _ema(c, 50).iloc[-1]  if len(c) >= 50  else np.nan

    above_sma50  = last > sma50  if not np.isnan(sma50)  else False
    above_sma200 = last > sma200 if not np.isnan(sma200) else False
    above_ema20  = last > ema20  if not np.isnan(ema20)  else False

    # Momentum oscillators
    rsi14  = _rsi(c, 14).iloc[-1] if len(c) >= 15 else np.nan

    # MACD
    macd_line   = _ema(c, 12) - _ema(c, 26)
    macd_signal = _ema(macd_line, 9)
    macd_val    = macd_line.iloc[-1]
    macd_sig    = macd_signal.iloc[-1]
    macd_bull   = macd_val > macd_sig

    # Volatility
    daily_ret  = c.pct_change().dropna()
    vol_30d    = daily_ret.iloc[-30:].std() * np.sqrt(252) * 100 if len(daily_ret) >= 30 else np.nan
    atr14      = _atr(df, 14).iloc[-1] if len(df) >= 15 else np.nan
    atr_pct    = (atr14 / last * 100) if (not np.isnan(atr14) and last > 0) else np.nan

    # Volume
    vol_avg20 = v.iloc[-20:].mean() if len(v) >= 20 else np.nan
    vol_ratio = (v.iloc[-1] / vol_avg20) if (vol_avg20 and vol_avg20 > 0) else np.nan

    # Sharpe (annualised, past 1 year, risk-free=0 approximation)
    yr_ret = daily_ret.iloc[-252:] if len(daily_ret) >= 252 else daily_ret
    sharpe = (yr_ret.mean() / max(yr_ret.std(), 1e-9)) * np.sqrt(252) if len(yr_ret) > 5 else np.nan

    # Trend score: simple composite 0-5
    trend_pts = sum([
        above_sma50,
        above_sma200,
        above_ema20,
        macd_bull,
        ret_1m > 0 if not np.isnan(ret_1m) else False,
    ])

    return {
        "Price":         round(last, 2),
        "1D %":          round(ret_1d,  2) if not np.isnan(ret_1d)  else None,
        "5D %":          round(ret_5d,  2) if not np.isnan(ret_5d)  else None,
        "1M %":          round(ret_1m,  2) if not np.isnan(ret_1m)  else None,
        "3M %":          round(ret_3m,  2) if not np.isnan(ret_3m)  else None,
        "6M %":          round(ret_6m,  2) if not np.isnan(ret_6m)  else None,
        "YTD %":         round(ret_ytd, 2) if not np.isnan(ret_ytd) else None,
        "52W High":      round(w52_hi, 2),
        "52W Low":       round(w52_lo, 2),
        "% from 52W Hi": round(pct_from_hi, 2),
        "% from 52W Lo": round(pct_from_lo, 2),
        "RSI(14)":       round(rsi14, 1)   if not np.isnan(rsi14)   else None,
        "MACD Bull":     macd_bull,
        "Volatility(30D)%": round(vol_30d, 1) if not np.isnan(vol_30d) else None,
        "ATR%":          round(atr_pct, 2) if not np.isnan(atr_pct) else None,
        "Vol/Avg20":     round(vol_ratio, 2) if not np.isnan(vol_ratio) else None,
        "Sharpe(1Y)":    round(sharpe, 2)  if not np.isnan(sharpe)  else None,
        "Above SMA50":   above_sma50,
        "Above SMA200":  above_sma200,
        "Above EMA20":   above_ema20,
        "Trend Score":   trend_pts,
        "SMA50":         round(sma50,  2)  if not np.isnan(sma50)  else None,
        "SMA200":        round(sma200, 2)  if not np.isnan(sma200) else None,
    }


def run_screen(tickers_info: dict, period_days: int, progress_bar) -> pd.DataFrame:
    """Fetch data and compute metrics for a list of tickers."""
    rows = []
    total = len(tickers_info)
    for i, (ticker, (name, exchange, cap)) in enumerate(tickers_info.items()):
        df = fetch_price(ticker, period_days)
        progress_bar.progress((i + 1) / total, text=f"Analysing {ticker}…")
        if df.empty:
            continue
        m = compute_metrics(df)
        if not m:
            continue
        rows.append({
            "Ticker":   ticker,
            "Name":     name,
            "Exchange": exchange,
            "Cap":      cap,
            **m,
        })
        time.sleep(0.05)   # gentle rate limit for yfinance
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  MINI SPARKLINE
# ─────────────────────────────────────────────────────────────────────────────
def sparkline_fig(ticker: str, days: int = 90) -> go.Figure:
    df = fetch_price(ticker, days)
    if df.empty:
        return go.Figure()
    c = df["Close"].iloc[-days:]
    colour = "#3fb950" if c.iloc[-1] >= c.iloc[0] else "#f85149"
    fig = go.Figure(go.Scatter(y=c.values, mode="lines",
                               line=dict(color=colour, width=1.5),
                               fill="tozeroy", fillcolor=colour.replace(")", ",0.08)").replace("rgb", "rgba")))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=60,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR — FILTER CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Screener Filters")
    st.markdown("---")

    # Universe
    st.markdown("**Universe**")
    selected_sectors = st.multiselect(
        "Sectors", ALL_SECTORS, default=ALL_SECTORS,
        help="Filter by GICS sector"
    )
    selected_exchanges = st.multiselect(
        "Exchanges", ALL_EXCHANGES, default=ALL_EXCHANGES
    )
    selected_caps = st.multiselect(
        "Market Cap", ALL_CAP_SIZES, default=ALL_CAP_SIZES
    )
    custom_tickers_raw = st.text_input(
        "Add Custom Tickers (comma-separated)",
        placeholder="e.g. COIN, RBLX, HOOD",
        help="These are added to the scan regardless of sector/cap filters"
    )

    st.markdown("---")

    # Period
    st.markdown("**Lookback Period**")
    period_map = {"3 Months": 90, "6 Months": 180, "1 Year": 365, "2 Years": 730}
    period_label = st.selectbox("Data Window", list(period_map.keys()), index=2)
    period_days  = period_map[period_label]

    st.markdown("---")

    # ── Price Filters ─────────────────────────────────────────────────────
    st.markdown("**💲 Price**")
    col_pa, col_pb = st.columns(2)
    with col_pa: price_min = st.number_input("Min ($)", value=1.0,  step=1.0, min_value=0.0)
    with col_pb: price_max = st.number_input("Max ($)", value=9999.0, step=10.0)

    st.markdown("---")

    # ── Performance Filters ───────────────────────────────────────────────
    st.markdown("**📈 Performance**")
    ret1d_min,  ret1d_max  = st.slider("1-Day Return (%)",  -20.0, 20.0, (-20.0, 20.0), 0.5)
    ret1m_min,  ret1m_max  = st.slider("1-Month Return (%)",-40.0, 40.0, (-40.0, 40.0), 1.0)
    ret3m_min,  ret3m_max  = st.slider("3-Month Return (%)",-60.0, 60.0, (-60.0, 60.0), 2.0)

    st.markdown("---")

    # ── Momentum Filters ──────────────────────────────────────────────────
    st.markdown("**⚡ Momentum**")
    rsi_min, rsi_max = st.slider("RSI(14)", 0, 100, (0, 100))
    macd_filter = st.radio("MACD Signal", ["Any", "Bullish only", "Bearish only"],
                           horizontal=True)
    trend_min = st.slider("Min Trend Score (0–5)", 0, 5, 0,
                          help="Composite: Above SMA50 + SMA200 + EMA20 + MACD Bull + 1M positive")

    st.markdown("---")

    # ── Moving Average Filters ────────────────────────────────────────────
    st.markdown("**〰 Moving Averages**")
    ma_filters = st.multiselect(
        "Price must be above…",
        ["SMA 50", "SMA 200", "EMA 20"],
        default=[],
    )

    st.markdown("---")

    # ── 52-Week Range Filters ─────────────────────────────────────────────
    st.markdown("**📏 52-Week Range**")
    near_hi = st.checkbox("Near 52W High (within 10%)", value=False)
    near_lo = st.checkbox("Near 52W Low (within 10%)",  value=False)

    st.markdown("---")

    # ── Volatility Filters ────────────────────────────────────────────────
    st.markdown("**〜 Volatility**")
    vol_max   = st.slider("Max 30D Annualised Vol (%)", 0, 200, 200)
    atr_max   = st.slider("Max ATR% (daily range)",     0.0, 15.0, 15.0, 0.5)

    st.markdown("---")

    # ── Volume Filters ────────────────────────────────────────────────────
    st.markdown("**📊 Volume**")
    vol_ratio_min = st.slider("Min Volume / 20D Avg", 0.0, 5.0, 0.0, 0.1,
                              help=">1.0 = above average volume today")

    st.markdown("---")

    # ── Sharpe Filter (key feature) ───────────────────────────────────────
    st.markdown("**⭐ Risk-Adjusted (Sharpe)**")
    sharpe_min = st.slider(
        "Min Sharpe Ratio (1Y)", -3.0, 5.0, 0.0, 0.1,
        help="Annualised Sharpe using 1-year daily returns. >1 = good, >2 = excellent."
    )
    sharpe_max = st.slider("Max Sharpe Ratio (1Y)", -3.0, 10.0, 10.0, 0.1)

    st.markdown("---")

    # ── Sort & Display ────────────────────────────────────────────────────
    st.markdown("**⬆ Sort Results**")
    sort_col = st.selectbox("Sort by", [
        "Sharpe(1Y)", "Trend Score", "RSI(14)", "1D %", "1M %", "3M %",
        "% from 52W Hi", "% from 52W Lo", "Vol/Avg20", "Volatility(30D)%",
    ])
    sort_asc = st.radio("Order", ["Descending ↓", "Ascending ↑"],
                        horizontal=True) == "Ascending ↑"

    st.markdown("---")
    run_screen_btn = st.button("🔍  RUN SCREENER", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 📡 Stock Screener")
st.markdown(
    '<span class="tag tag-blue">TradingView-style filters</span>'
    '<span class="tag tag-green">Sharpe screening</span>'
    '<span class="tag">100+ instruments</span>',
    unsafe_allow_html=True,
)
st.markdown("")

st.markdown("""
Use the filters in the sidebar to build your watchlist.  
Screener computes **price action, momentum, volatility, volume, and Sharpe ratio** from live data.  
Results link back to the backtester — click any ticker to analyse it in detail.
""")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  RUN SCREEN
# ─────────────────────────────────────────────────────────────────────────────
if run_screen_btn:

    # Build ticker universe from sidebar selections
    tickers_to_scan: dict = {}
    for sector in selected_sectors:
        for ticker, (name, exchange, cap) in UNIVERSE[sector].items():
            if exchange in selected_exchanges and cap in selected_caps:
                tickers_to_scan[ticker] = (name, exchange, cap)

    # Add custom tickers
    if custom_tickers_raw.strip():
        for t in [x.strip().upper() for x in custom_tickers_raw.split(",") if x.strip()]:
            tickers_to_scan[t] = (t, "Custom", "Custom")

    if not tickers_to_scan:
        st.warning("No tickers match the universe filters. Broaden your sector/exchange/cap selections.")
        st.stop()

    st.info(f"Scanning **{len(tickers_to_scan)}** instruments…")
    progress = st.progress(0, text="Starting scan…")

    with st.spinner(""):
        df_results = run_screen(tickers_to_scan, period_days, progress)

    progress.empty()

    if df_results.empty:
        st.error("No data returned. Check your internet connection or try a smaller universe.")
        st.stop()

    # ── Apply Filters ────────────────────────────────────────────────────
    def safe_filter(mask, col, lo=None, hi=None):
        if col not in df_results.columns:
            return mask
        s = pd.to_numeric(df_results[col], errors="coerce")
        if lo is not None: mask = mask & (s.fillna(-np.inf) >= lo)
        if hi is not None: mask = mask & (s.fillna(np.inf)  <= hi)
        return mask

    mask = pd.Series([True] * len(df_results), index=df_results.index)
    mask = safe_filter(mask, "Price",            lo=price_min,     hi=price_max)
    mask = safe_filter(mask, "1D %",             lo=ret1d_min,     hi=ret1d_max)
    mask = safe_filter(mask, "1M %",             lo=ret1m_min,     hi=ret1m_max)
    mask = safe_filter(mask, "3M %",             lo=ret3m_min,     hi=ret3m_max)
    mask = safe_filter(mask, "RSI(14)",          lo=rsi_min,       hi=rsi_max)
    mask = safe_filter(mask, "Volatility(30D)%", hi=vol_max)
    mask = safe_filter(mask, "ATR%",             hi=atr_max)
    mask = safe_filter(mask, "Vol/Avg20",        lo=vol_ratio_min)
    mask = safe_filter(mask, "Sharpe(1Y)",       lo=sharpe_min,    hi=sharpe_max)
    mask = safe_filter(mask, "Trend Score",      lo=trend_min)

    if macd_filter == "Bullish only":
        mask &= df_results["MACD Bull"] == True
    elif macd_filter == "Bearish only":
        mask &= df_results["MACD Bull"] == False

    if "SMA 50"  in ma_filters: mask &= df_results["Above SMA50"]  == True
    if "SMA 200" in ma_filters: mask &= df_results["Above SMA200"] == True
    if "EMA 20"  in ma_filters: mask &= df_results["Above EMA20"]  == True

    if near_hi:
        mask &= pd.to_numeric(df_results["% from 52W Hi"], errors="coerce").fillna(-np.inf) >= -10
    if near_lo:
        mask &= pd.to_numeric(df_results["% from 52W Lo"], errors="coerce").fillna(np.inf) <= 10

    df_filtered = df_results[mask].copy()

    # Sort
    if sort_col in df_filtered.columns:
        df_filtered = df_filtered.sort_values(
            sort_col, ascending=sort_asc, na_position="last"
        )

    # ── Summary KPIs ─────────────────────────────────────────────────────
    st.markdown("### Scan Results")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Scanned",       len(df_results))
    m2.metric("Passed Filters",len(df_filtered))
    m3.metric("Avg Sharpe",
              f"{df_filtered['Sharpe(1Y)'].mean():.2f}"
              if not df_filtered.empty and df_filtered['Sharpe(1Y)'].notna().any() else "—")
    m4.metric("Avg RSI",
              f"{df_filtered['RSI(14)'].mean():.1f}"
              if not df_filtered.empty and df_filtered['RSI(14)'].notna().any() else "—")
    m5.metric("Avg 1M Return",
              f"{df_filtered['1M %'].mean():+.1f}%"
              if not df_filtered.empty and df_filtered['1M %'].notna().any() else "—")

    st.markdown("---")

    if df_filtered.empty:
        st.warning("No stocks passed all filters. Try relaxing one or more criteria.")
        st.stop()

    # ── Tabs: Table / Charts / Heatmap ───────────────────────────────────
    tab_table, tab_charts, tab_heatmap, tab_scatter = st.tabs(
        ["📋 Results Table", "📊 Top Movers", "🌡 Heatmap", "🔵 Sharpe vs Return"]
    )

    # Display columns for the table
    display_cols = [
        "Ticker", "Name", "Sector", "Cap", "Exchange",
        "Price", "1D %", "1M %", "3M %",
        "RSI(14)", "Sharpe(1Y)", "Trend Score",
        "Volatility(30D)%", "Vol/Avg20",
        "% from 52W Hi", "% from 52W Lo",
        "Above SMA50", "Above SMA200", "MACD Bull",
    ]

    # Attach sector label
    ticker_to_sector = {}
    for sec, tmap in UNIVERSE.items():
        for tk in tmap:
            ticker_to_sector[tk] = sec
    df_filtered["Sector"] = df_filtered["Ticker"].map(
        lambda t: ticker_to_sector.get(t, "Custom")
    )

    show_cols = [c for c in display_cols if c in df_filtered.columns]
    df_display = df_filtered[show_cols].reset_index(drop=True)

    # ── TABLE TAB ────────────────────────────────────────────────────────
    with tab_table:
        st.markdown(
            f'<div class="result-count">{len(df_filtered)} results · sorted by <b>{sort_col}</b></div>',
            unsafe_allow_html=True,
        )

        def colour_val(val, col_name=""):
            if isinstance(val, bool):
                return f"color: {'#3fb950' if val else '#f85149'}"
            if isinstance(val, (int, float)) and not pd.isna(val):
                if col_name in ["1D %", "1M %", "3M %", "6M %", "YTD %",
                                 "% from 52W Lo", "Sharpe(1Y)", "Trend Score"]:
                    return f"color: {'#3fb950' if val >= 0 else '#f85149'}"
                if col_name == "% from 52W Hi":
                    return f"color: {'#f85149' if val < -20 else '#d29922' if val < -10 else '#3fb950'}"
                if col_name == "RSI(14)":
                    return f"color: {'#f85149' if val > 70 else '#3fb950' if val < 30 else '#c9d1d9'}"
                if col_name == "Vol/Avg20":
                    return f"color: {'#3fb950' if val > 1.5 else '#c9d1d9'}"
            return "color: #c9d1d9"

        def style_row(s):
            return [colour_val(v, s.name) for v in s]

        styled = df_display.style.apply(
            lambda col: [colour_val(v, col.name) for v in col], axis=0
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=520)

        # Download
        csv = df_filtered.to_csv(index=False).encode()
        st.download_button(
            "⬇  Export Results (CSV)", csv,
            f"screener_results_{datetime.today().strftime('%Y%m%d')}.csv", "text/csv"
        )

    # ── TOP MOVERS TAB ───────────────────────────────────────────────────
    with tab_charts:
        n_show = min(20, len(df_filtered))
        top_n  = df_filtered.head(n_show)

        ca, cb = st.columns(2)

        with ca:
            st.markdown("**Top by 1-Month Return**")
            top_1m = df_filtered.dropna(subset=["1M %"]).nlargest(15, "1M %")
            colors = ["#3fb950" if v >= 0 else "#f85149" for v in top_1m["1M %"]]
            fig1m  = go.Figure(go.Bar(
                x=top_1m["1M %"], y=top_1m["Ticker"],
                orientation="h", marker_color=colors, opacity=.85,
                text=[f"{v:+.1f}%" for v in top_1m["1M %"]],
                textposition="outside", textfont=dict(color=MUTED, size=9),
            ))
            fig1m.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                xaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED), title="1M Return (%)"),
                yaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED, size=9), autorange="reversed"),
                height=420, margin=dict(l=10, r=60, t=10, b=30),
                dragmode="pan",
            )
            st.plotly_chart(fig1m, use_container_width=True, config=PLOTLY_CFG)

        with cb:
            st.markdown("**Top by Sharpe Ratio**")
            top_sh = df_filtered.dropna(subset=["Sharpe(1Y)"]).nlargest(15, "Sharpe(1Y)")
            sh_colors = ["#58a6ff" if v >= 1 else "#d29922" if v >= 0 else "#f85149"
                         for v in top_sh["Sharpe(1Y)"]]
            fig_sh = go.Figure(go.Bar(
                x=top_sh["Sharpe(1Y)"], y=top_sh["Ticker"],
                orientation="h", marker_color=sh_colors, opacity=.85,
                text=[f"{v:.2f}" for v in top_sh["Sharpe(1Y)"]],
                textposition="outside", textfont=dict(color=MUTED, size=9),
            ))
            fig_sh.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                xaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED), title="Sharpe Ratio (1Y)"),
                yaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED, size=9), autorange="reversed"),
                height=420, margin=dict(l=10, r=60, t=10, b=30),
                dragmode="pan",
            )
            st.plotly_chart(fig_sh, use_container_width=True, config=PLOTLY_CFG)

        # RSI distribution
        st.markdown("**RSI Distribution**")
        df_rsi = df_filtered.dropna(subset=["RSI(14)"])
        if not df_rsi.empty:
            fig_rsi = go.Figure(go.Histogram(
                x=df_rsi["RSI(14)"], nbinsx=20,
                marker_color="#58a6ff", opacity=.75,
            ))
            fig_rsi.add_vline(x=30, line_dash="dot", line_color="#f85149", opacity=.6)
            fig_rsi.add_vline(x=70, line_dash="dot", line_color="#3fb950", opacity=.6)
            fig_rsi.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                xaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED), title="RSI(14)"),
                yaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED), title="Count"),
                height=250, margin=dict(l=10, r=10, t=10, b=30),
            )
            st.plotly_chart(fig_rsi, use_container_width=True, config=PLOTLY_CFG)

    # ── HEATMAP TAB ──────────────────────────────────────────────────────
    with tab_heatmap:
        st.markdown("**1-Month Return Heatmap** (grouped by sector)")

        heat_df = df_filtered.dropna(subset=["1M %", "Sector"]).copy()
        if not heat_df.empty:
            heat_df = heat_df.sort_values(["Sector", "1M %"], ascending=[True, False])
            sectors_present = heat_df["Sector"].unique()

            sector_col_map = {
                "Technology": "#1f6feb", "Financials": "#238636",
                "Healthcare": "#9e6a03", "Consumer Discretionary": "#d29922",
                "Energy": "#b08800", "Industrials": "#8250df",
                "ETFs & Indices": "#0d419d", "Custom": "#484f58",
            }

            COLS = 8
            all_tickers = heat_df["Ticker"].tolist()
            n_rows = -(-len(all_tickers) // COLS)

            cells_text  = []
            cells_color = []
            hover_text  = []

            for i in range(n_rows):
                row_t, row_c, row_h = [], [], []
                for j in range(COLS):
                    idx = i * COLS + j
                    if idx < len(heat_df):
                        row_data = heat_df.iloc[idx]
                        t   = row_data["Ticker"]
                        ret = row_data["1M %"]
                        sec = row_data.get("Sector", "")
                        sh  = row_data.get("Sharpe(1Y)", np.nan)
                        rsi_v = row_data.get("RSI(14)", np.nan)
                        row_t.append(f"<b>{t}</b><br>{ret:+.1f}%")
                        # colour by performance intensity
                        intensity = min(abs(ret) / 20, 1.0)
                        if ret >= 0:
                            g = int(50 + 150 * intensity)
                            row_c.append(f"rgb(15,{g},30)")
                        else:
                            r = int(60 + 150 * intensity)
                            row_c.append(f"rgb({r},20,20)")
                        row_h.append(
                            f"<b>{t}</b> — {row_data['Name']}<br>"
                            f"Price: ${row_data['Price']}<br>"
                            f"1M: {ret:+.1f}%  |  RSI: {rsi_v if pd.notna(rsi_v) else '—'}<br>"
                            f"Sharpe: {sh:.2f if pd.notna(sh) else '—'}<br>"
                            f"Sector: {sec}"
                        )
                    else:
                        row_t.append(""); row_c.append("rgb(20,20,28)"); row_h.append("")
                cells_text.append(row_t)
                cells_color.append(row_c)
                hover_text.append(row_h)

            fig_heat = go.Figure(go.Heatmap(
                z=[[0]*COLS]*n_rows,   # dummy z — we use custom colours
                text=cells_text,
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
                texttemplate="%{text}",
                textfont=dict(size=10, color="#e6edf3"),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                xgap=3, ygap=3,
            ))
            # Overlay coloured rectangles via annotations is complex;
            # instead use a simpler px treemap for the heatmap visual
            fig_heat = px.treemap(
                heat_df,
                path=["Sector", "Ticker"],
                values=[1] * len(heat_df),
                color="1M %",
                color_continuous_scale=[
                    [0.0,  "#7f1d1d"],
                    [0.35, "#3a1a1a"],
                    [0.5,  "#161b22"],
                    [0.65, "#1a3a1a"],
                    [1.0,  "#14532d"],
                ],
                color_continuous_midpoint=0,
                custom_data=["Name", "1M %", "Sharpe(1Y)", "RSI(14)", "Price"],
            )
            fig_heat.update_traces(
                texttemplate="<b>%{label}</b><br>%{customdata[1]:+.1f}%",
                hovertemplate=(
                    "<b>%{label}</b><br>%{customdata[0]}<br>"
                    "1M: %{customdata[1]:+.1f}%<br>"
                    "Sharpe: %{customdata[2]:.2f}<br>"
                    "RSI: %{customdata[3]:.0f}<br>"
                    "Price: $%{customdata[4]:.2f}"
                    "<extra></extra>"
                ),
                textfont=dict(size=11),
            )
            fig_heat.update_layout(
                paper_bgcolor=BG, height=500,
                coloraxis_colorbar=dict(
                    title="1M%", tickfont=dict(color=MUTED),
                    titlefont=dict(color=MUTED),
                ),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_heat, use_container_width=True, config=PLOTLY_CFG)
        else:
            st.info("Not enough data to build the heatmap.")

    # ── SHARPE vs RETURN SCATTER ─────────────────────────────────────────
    with tab_scatter:
        st.markdown("**Sharpe Ratio vs 3-Month Return** — bubble size = volatility")

        scatter_df = df_filtered.dropna(subset=["Sharpe(1Y)", "3M %", "Volatility(30D)%"]).copy()
        if not scatter_df.empty:
            scatter_df["Sector"] = scatter_df["Ticker"].map(
                lambda t: ticker_to_sector.get(t, "Custom")
            )
            scatter_df["_size"] = scatter_df["Volatility(30D)%"].clip(5, 100)

            fig_sc = px.scatter(
                scatter_df,
                x="3M %", y="Sharpe(1Y)",
                size="_size", size_max=30,
                color="Sector",
                text="Ticker",
                hover_data={"Ticker": True, "Name": True, "3M %": ":.1f",
                            "Sharpe(1Y)": ":.2f", "RSI(14)": ":.0f",
                            "_size": False},
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_sc.add_hline(y=0, line_dash="dot", line_color="#30363d", opacity=.6)
            fig_sc.add_vline(x=0, line_dash="dot", line_color="#30363d", opacity=.6)
            # Quadrant labels
            for txt, x, y in [
                ("Strong + Safe", scatter_df["3M %"].max() * .7, scatter_df["Sharpe(1Y)"].max() * .7),
                ("Weak + Risky",  scatter_df["3M %"].min() * .7, scatter_df["Sharpe(1Y)"].min() * .7),
            ]:
                fig_sc.add_annotation(x=x, y=y, text=txt, showarrow=False,
                                      font=dict(color="#484f58", size=10))

            fig_sc.update_traces(textposition="top center",
                                 textfont=dict(size=8, color="#8b949e"))
            fig_sc.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                legend=dict(bgcolor=BG, font=dict(color=MUTED, size=10)),
                xaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED),
                           title="3-Month Return (%)", zeroline=False),
                yaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED),
                           title="Sharpe Ratio (1Y)", zeroline=False),
                height=520, margin=dict(l=10, r=10, t=20, b=30),
                dragmode="pan",
            )
            st.plotly_chart(fig_sc, use_container_width=True, config=PLOTLY_CFG)
        else:
            st.info("Not enough data for scatter plot. Run with a larger universe.")

else:
    # Landing
    st.markdown("""
<div style="text-align:center; padding:60px 20px; color:#8b949e;">
  <div style="font-size:3rem; margin-bottom:16px;">🔍</div>
  <div style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem; color:#58a6ff; margin-bottom:8px;">
    Configure filters in the sidebar, then click RUN SCREENER
  </div>
  <div style="font-size:.85rem; max-width:600px; margin:0 auto;">
    Screens 100+ instruments across 7 sectors.
    Filters include price, performance, RSI, MACD, moving averages,
    52-week range, volume, volatility, and <b>Sharpe ratio</b>.
    Results export to CSV.
  </div>
</div>
""", unsafe_allow_html=True)

    # Filter preview cards
    st.markdown("### Available Filter Categories")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
<div class="filter-section">
<h4>💲 Price & Performance</h4>
Price range · 1D / 1M / 3M / 6M / YTD returns
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="filter-section">
<h4>⚡ Momentum</h4>
RSI(14) range · MACD bullish/bearish · Trend Score composite
</div>
""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div class="filter-section">
<h4>〰 Technicals</h4>
SMA 50/200 · EMA 20 · 52-week hi/lo proximity · ATR%
</div>
""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
<div class="filter-section">
<h4>⭐ Risk-Adjusted</h4>
Sharpe ratio (1Y) · 30D annualised volatility · Volume vs avg
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#484f58; font-size:.7rem; font-family:IBM Plex Mono,monospace;">'
    'Screener data sourced from Yahoo Finance via yfinance. '
    'For educational use only — not financial advice.'
    '</div>',
    unsafe_allow_html=True,
)
