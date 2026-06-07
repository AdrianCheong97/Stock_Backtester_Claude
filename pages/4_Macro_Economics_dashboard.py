"""
╔══════════════════════════════════════════════════════════╗
║         MACRO ECONOMICS MONITORING DASHBOARD             ║
║  Data Sources: US Treasury API | FRED | Yahoo Finance    ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE LAYOUT CONFIG & THEME MATCHING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macro Economics Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0d0f14;
        color: #c9d1d9;
    }
    .stCodeBlock, code, pre, .stMarkdown pre {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    .macro-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .macro-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .macro-value {
        font-size: 1.9rem;
        font-weight: 700;
        margin-top: 4px;
        color: #f0f6fc;
        font-family: 'IBM Plex Mono', monospace;
    }
    .macro-delta {
        font-size: 0.85rem;
        margin-top: 2px;
        font-weight: 500;
    }
    .status-badge {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
    }
    .stExpander {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        margin-bottom: 20px;
    }
    .help-header {
        font-family: 'IBM Plex Mono', monospace;
        color: #58a6ff;
        font-size: 0.95rem;
        margin-top: 14px;
        margin-bottom: 4px;
        border-bottom: 1px dashed #21262d;
        padding-bottom: 2px;
    }
    .source-badge {
        display: inline-block;
        padding: 1px 5px;
        border-radius: 3px;
        font-size: 0.65rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        background-color: #21262d;
        color: #8b949e;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TREASURY_API_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
    "/v2/accounting/od/daily_treasury_yield_curve_rates"
)
TREASURY_FIELDS = (
    "record_date,bc_3month,bc_6month,bc_1year,bc_2year,"
    "bc_5year,bc_7year,bc_10year,bc_20year,bc_30year"
)
TREASURY_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Maturity label mapping (API field → display label, years for sorting)
MATURITY_MAP = {
    "bc_3month":  ("3M",  0.25),
    "bc_6month":  ("6M",  0.5),
    "bc_1year":   ("1Y",  1.0),
    "bc_2year":   ("2Y",  2.0),
    "bc_5year":   ("5Y",  5.0),
    "bc_7year":   ("7Y",  7.0),
    "bc_10year":  ("10Y", 10.0),
    "bc_20year":  ("20Y", 20.0),
    "bc_30year":  ("30Y", 30.0),
}


# ─────────────────────────────────────────────────────────────────────────────
#  TREASURY YIELD DATA — LAYERED FETCH STRATEGY
# ─────────────────────────────────────────────────────────────────────────────
def _treasury_api_fetch(start_date_str: str) -> pd.DataFrame:
    """
    Pull yield curve history from the US Treasury FiscalData API.

    FIX vs original:
      • Param name is 'filters' (plural), NOT 'filter' — the old code sent
        an unrecognised key so the API returned 0 records silently.
      • Explicit 'fields' param: only fetch the maturity columns we need,
        which reduces response size and prevents breakage when Treasury
        adds new columns.
      • Sort newest-first ('-record_date') so partial pages still give us
        the latest data.
      • page[size] capped at 3000 to stay well within the API's hard 10 000
        row limit per page while covering ~12 years of trading days.
    """
    params = {
        "filters": f"record_date:gte:{start_date_str}",
        "fields":  TREASURY_FIELDS,
        "page[size]": 3000,
        "sort": "-record_date",
    }
    resp = requests.get(
        TREASURY_API_URL, params=params,
        headers=TREASURY_HEADERS, timeout=20
    )
    resp.raise_for_status()
    raw = resp.json().get("data", [])
    if not raw:
        return pd.DataFrame()

    records = []
    for row in raw:
        date_str = row.get("record_date")
        if not date_str:
            continue
        rec = {"Date": pd.to_datetime(date_str).tz_localize(None)}
        for field, (label, _) in MATURITY_MAP.items():
            val = row.get(field)
            rec[label] = pd.to_numeric(val, errors="coerce") if val is not None else np.nan
        records.append(rec)

    df = (
        pd.DataFrame(records)
        .dropna(subset=["2Y", "10Y"])   # require at minimum these two
        .sort_values("Date")
        .reset_index(drop=True)
    )
    df["Spread_10Y2Y"] = df["10Y"] - df["2Y"]
    return df


def _yfinance_yield_fallback(start_date_str: str) -> pd.DataFrame:
    """
    Secondary fallback using yfinance treasury ETF proxies.

    Tickers used:
      ^TNX  → 10-Year Treasury yield (CBOE)
      ^IRX  → 13-Week Treasury yield  (closest public proxy to 2Y)
      ^TYX  → 30-Year Treasury yield
      ^FVX  → 5-Year Treasury yield

    The 2Y doesn't have a direct yfinance ticker, so we use ^IRX (3-month)
    as a directionally similar short-end proxy and label it accordingly.
    This is a degraded signal — always prefer the Treasury API path.
    """
    tickers = {
        "^TNX": "10Y",
        "^IRX": "3M",   # 13-week, closest available short-end proxy
        "^TYX": "30Y",
        "^FVX": "5Y",
    }
    frames = []
    for ticker, label in tickers.items():
        try:
            raw = yf.download(ticker, start=start_date_str, progress=False, auto_adjust=True)
            if raw.empty:
                continue
            close_col = "Close" if "Close" in raw.columns else raw.columns[0]
            tmp = pd.DataFrame({
                "Date": pd.to_datetime(raw.index).tz_localize(None),
                label: pd.to_numeric(raw[close_col].values, errors="coerce") / 10.0  # ^TNX is in tenths
            }).dropna()
            frames.append(tmp)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = frames[0]
    for f in frames[1:]:
        df = pd.merge(df, f, on="Date", how="outer")
    df = df.sort_values("Date").reset_index(drop=True)

    # Build a spread using 10Y minus 3M (labelled clearly as proxy)
    if "10Y" in df.columns and "3M" in df.columns:
        df["Spread_10Y2Y"] = df["10Y"] - df["3M"]
        df.rename(columns={"3M": "2Y_proxy(3M)"}, inplace=True)
    elif "10Y" in df.columns:
        df["Spread_10Y2Y"] = np.nan

    return df


@st.cache_data(ttl=3600)
def fetch_treasury_yields(lookback_years: int = 10) -> tuple[pd.DataFrame, str]:
    """
    Layered fetch: Treasury API → yfinance fallback → static stub.
    Returns (DataFrame, source_label).
    """
    start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")

    # ── Layer 1: US Treasury FiscalData API (authoritative, keyless) ──
    try:
        df = _treasury_api_fetch(start_date)
        if not df.empty and len(df) > 30:
            return df, "US Treasury FiscalData API"
    except Exception as e:
        st.sidebar.warning(f"Treasury API: {e}")

    # ── Layer 2: yfinance proxy tickers ──
    try:
        df = _yfinance_yield_fallback(start_date)
        if not df.empty and len(df) > 30:
            return df, "Yahoo Finance (proxy tickers — degraded)"
    except Exception as e:
        st.sidebar.warning(f"yfinance fallback: {e}")

    # ── Layer 3: Static stub so the dashboard doesn't crash ──
    dates = pd.date_range(start=start_date, end=datetime.now(), freq="B").tz_localize(None)
    df = pd.DataFrame({
        "Date": dates,
        "2Y": 4.10,
        "10Y": 4.25,
        "Spread_10Y2Y": 0.15,
    })
    return df, "Static Stub (all data sources unavailable)"


# ─────────────────────────────────────────────────────────────────────────────
#  FRED CREDIT SPREAD — fixed URL + yfinance HYG fallback
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_credit_spread() -> tuple[pd.DataFrame, str]:
    """
    Fetch ICE BofA HY credit spread.

    FIX vs original: the old scraping URL
      /series/BAMLH0A0HYM2/downloaddata?form=csv
    was a UI page endpoint that FRED started blocking. The correct
    machine-readable endpoint is /graph/fredgraph.csv?id=SERIES_ID.
    """
    HEADERS = {"User-Agent": TREASURY_HEADERS["User-Agent"]}

    # ── Layer 1: FRED fredgraph CSV (correct machine-readable endpoint) ──
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"])
        df.columns = ["Date", "HY_Spread"]
        df["HY_Spread"] = pd.to_numeric(df["HY_Spread"].replace(".", np.nan), errors="coerce")
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.dropna().reset_index(drop=True)
        if not df.empty:
            return df, "FRED (BAMLH0A0HYM2)"
    except Exception:
        pass

    # ── Layer 2: HYG–LQD spread via yfinance as proxy ──
    # HYG = iShares HY Bond ETF yield, LQD = IG Bond ETF yield
    # Their price ratio isn't a spread, but HYG's 30-day SEC yield data
    # isn't available via yfinance — so we use the HYG/IEF ratio as a
    # directional risk-on/risk-off proxy instead.
    try:
        raw = yf.download(["HYG", "LQD"], start="2003-01-01", progress=False, auto_adjust=True)
        hyg = raw["Close"]["HYG"].rename("HYG")
        lqd = raw["Close"]["LQD"].rename("LQD")
        df = pd.concat([hyg, lqd], axis=1).dropna()
        # Express as a relative underperformance ratio — not true OAS but directional
        df["HY_Spread"] = ((lqd / hyg) - 1) * 100
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.reset_index().rename(columns={"index": "Date", "Date": "Date"})
        df = df[["Date", "HY_Spread"]].dropna()
        if not df.empty:
            return df, "Yahoo Finance HYG/LQD ratio (proxy — directional only)"
    except Exception:
        pass

    # ── Layer 3: Static stub ──
    dates = pd.date_range(start="2020-01-01", end=datetime.now(), freq="D").tz_localize(None)
    return pd.DataFrame({"Date": dates, "HY_Spread": 4.15}), "Static Stub"


# ─────────────────────────────────────────────────────────────────────────────
#  YFINANCE MARKET DATA
# ─────────────────────────────────────────────────────────────────────────────
def _extract_yf_close(raw_df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Normalise yfinance output to a clean timezone-naive Date + value frame."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["Date", col_name])
    try:
        if isinstance(raw_df.columns, pd.MultiIndex):
            close_col = [c for c in raw_df.columns if c[0] in ("Close", "Adj Close")]
            series = raw_df[close_col[0]] if close_col else None
        else:
            close_col = [c for c in raw_df.columns if c in ("Close", "Adj Close")]
            series = raw_df[close_col[0]] if close_col else None

        if series is None:
            return pd.DataFrame(columns=["Date", col_name])

        df = pd.DataFrame({
            "Date": pd.to_datetime(raw_df.index).tz_localize(None),
            col_name: pd.to_numeric(series.values, errors="coerce"),
        }).dropna().sort_values("Date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["Date", col_name])


@st.cache_data(ttl=3600)
def load_market_data() -> dict[str, pd.DataFrame]:
    """Fetch VIX, S&P 500, DXY, and Gold from yfinance."""
    tickers = {
        "VIX":   "^VIX",
        "SP500": "^GSPC",
        "DXY":   "DX-Y.NYB",
        "Gold":  "GC=F",
    }
    out = {}
    for name, ticker in tickers.items():
        try:
            raw = yf.download(ticker, start="2000-01-01", progress=False, auto_adjust=True)
            out[name] = _extract_yf_close(raw, name)
        except Exception:
            out[name] = pd.DataFrame(columns=["Date", name])
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  KPI HELPER
# ─────────────────────────────────────────────────────────────────────────────
def kpi(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """Return (latest_value, 30-day absolute delta)."""
    if df.empty or col not in df.columns:
        return 0.0, 0.0
    s = df.dropna(subset=[col]).sort_values("Date")
    if len(s) < 2:
        return 0.0, 0.0
    latest = float(s[col].iloc[-1])
    cutoff = s["Date"].iloc[-1] - timedelta(days=30)
    past = s[s["Date"] <= cutoff]
    prior = float(past[col].iloc[-1]) if not past.empty else float(s[col].iloc[0])
    return latest, latest - prior


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD ALL DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Fetching macro data…"):
    df_treasury, treasury_source = fetch_treasury_yields(lookback_years=10)
    df_credit, credit_source = fetch_credit_spread()
    market = load_market_data()

df_vix   = market["VIX"]
df_sp500 = market["SP500"]
df_dxy   = market["DXY"]
df_gold  = market["Gold"]


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🌐 Macro Parameters")
st.sidebar.markdown("---")
lookback_selection = st.sidebar.selectbox(
    "Select Historical Window",
    options=["1 Year", "3 Years", "5 Years", "10 Years", "Max Records"],
    index=2,
)

now = datetime.now()
_window_map = {
    "1 Year": 365, "3 Years": 365*3,
    "5 Years": 365*5, "10 Years": 365*10,
}
start_filter = now - timedelta(days=_window_map[lookback_selection]) if lookback_selection != "Max Records" else datetime(2000, 1, 1)
p_start = pd.Timestamp(start_filter).tz_localize(None)

# Apply date filter
def filt(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Date" not in df.columns:
        return df
    return df[df["Date"] >= p_start].copy()

f_tsy    = filt(df_treasury)
f_credit = filt(df_credit)
f_vix    = filt(df_vix)
f_sp     = filt(df_sp500)
f_dxy    = filt(df_dxy)
f_gold   = filt(df_gold)

# Data source status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sources**")
src_color = "#3fb950" if "Stub" not in treasury_source else "#f85149"
st.sidebar.markdown(
    f'<span style="color:{src_color}; font-size:0.75rem; font-family:IBM Plex Mono,monospace;">'
    f'● Yields: {treasury_source}</span>', unsafe_allow_html=True
)
src_color2 = "#3fb950" if "Stub" not in credit_source else "#f85149"
st.sidebar.markdown(
    f'<span style="color:{src_color2}; font-size:0.75rem; font-family:IBM Plex Mono,monospace;">'
    f'● Credit: {credit_source}</span>', unsafe_allow_html=True
)


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🌐 Macro Economics Dashboard")
st.markdown(
    "Institutional monitoring workspace tracking systemic credit stress, "
    "risk premiums, yield curve dynamics, and global liquidity indicators."
)

with st.expander("📖 Indicator Reference Manual & Playbook", expanded=False):
    st.markdown('<div class="help-header">🏛️ 10Y - 2Y Treasury Yield Spread</div>', unsafe_allow_html=True)
    st.markdown("""
    * **Core Concept:** Mathematical delta between 10-Year and 2-Year US Constant Maturity yields. Under stable conditions, longer durations require higher yields to compensate for duration risk.
    * **Recession Signaling:** When short-term yields eclipse long-term yields, the spread drops **below 0.0% (Inversion)**. This has accurately preceded every US recession since 1970 with a **12–18 month lead time**.
    """)
    st.markdown('<div class="help-header">📊 Full Yield Curve Snapshot</div>', unsafe_allow_html=True)
    st.markdown("""
    * Plots the entire term structure from 3-Month to 30-Year in a single chart. A humped or inverted curve signals monetary tightening stress; a steep upward slope signals expansion expectations.
    """)
    st.markdown('<div class="help-header">📊 VIX Volatility Index (The Fear Gauge)</div>', unsafe_allow_html=True)
    st.markdown("""
    * **Core Concept:** Compiled by the CBOE, the VIX tracks the 30-day implied volatility derived from S&P 500 options order flow. Readings above 20 indicate elevated uncertainty; above 30 indicate market stress.
    """)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  TOP-ROW KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    v, d = kpi(df_treasury, "Spread_10Y2Y")
    status = "INVERTED ⚠️" if v < 0 else "NORMAL 🟢"
    bg, fg = ("#f8514915", "#f85149") if v < 0 else ("#3fb95015", "#3fb950")
    st.markdown(f"""
    <div class="macro-card">
        <div class="macro-title">10Y – 2Y Yield Spread</div>
        <div class="macro-value">{v:.2f}%</div>
        <div class="macro-delta" style="color:{'#3fb950' if d >= 0 else '#f85149'}">
            {'+' if d >= 0 else ''}{d:.2f}% <span style="color:#8b949e; font-size:0.7rem; font-weight:normal;">MoM</span>
        </div>
        <div style="margin-top:10px;"><span class="status-badge" style="background-color:{bg}; color:{fg};">{status}</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    v, d = kpi(df_vix, "VIX")
    if v > 25:   status, bg, fg = "PANIC 🔥", "#f8514915", "#f85149"
    elif v > 17: status, bg, fg = "ELEVATED 🟡", "#d2992215", "#d29922"
    else:        status, bg, fg = "CALM 🟢", "#58a6ff15", "#58a6ff"
    st.markdown(f"""
    <div class="macro-card">
        <div class="macro-title">VIX Volatility Index</div>
        <div class="macro-value">{v:.2f}</div>
        <div class="macro-delta" style="color:{'#f85149' if d >= 0 else '#3fb950'}">
            {'+' if d >= 0 else ''}{d:.2f} <span style="color:#8b949e; font-size:0.7rem; font-weight:normal;">MoM</span>
        </div>
        <div style="margin-top:10px;"><span class="status-badge" style="background-color:{bg}; color:{fg};">{status}</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    v, d = kpi(df_credit, "HY_Spread")
    status = "CREDIT DISTRESS 🚨" if v > 4.5 else "STABLE 🟢"
    bg, fg = ("#f8514915", "#f85149") if v > 4.5 else ("#3fb95015", "#3fb950")
    st.markdown(f"""
    <div class="macro-card">
        <div class="macro-title">High Yield Credit Spread</div>
        <div class="macro-value">{v:.2f}%</div>
        <div class="macro-delta" style="color:{'#f85149' if d >= 0 else '#3fb950'}">
            {'+' if d >= 0 else ''}{d:.2f}% <span style="color:#8b949e; font-size:0.7rem; font-weight:normal;">MoM</span>
        </div>
        <div style="margin-top:10px;"><span class="status-badge" style="background-color:{bg}; color:{fg};">{status}</span></div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    v, d = kpi(df_sp500, "SP500")
    pct = (d / (v - d)) * 100 if (v - d) != 0 else 0.0
    status = "BULLISH 📈" if d >= 0 else "BEARISH 📉"
    bg, fg = ("#3fb95015", "#3fb950") if d >= 0 else ("#f8514915", "#f85149")
    st.markdown(f"""
    <div class="macro-card">
        <div class="macro-title">S&P 500 Index Benchmark</div>
        <div class="macro-value">{v:,.1f}</div>
        <div class="macro-delta" style="color:{'#3fb950' if d >= 0 else '#f85149'}">
            {'+' if d >= 0 else ''}{pct:.2f}% <span style="color:#8b949e; font-size:0.7rem; font-weight:normal;">MoM</span>
        </div>
        <div style="margin-top:10px;"><span class="status-badge" style="background-color:{bg}; color:{fg};">{status}</span></div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Yield Curve & Recession Tracker",
    "🏛️ Full Curve Snapshot",
    "📊 Market Sentiment & Risk Gauges",
    "🌍 Systemic Liquidity & Flight to Safety",
])

# ── TAB 1: 10Y/2Y history + spread ──────────────────────────────────────────
with tab1:
    st.markdown("### 🏛️ US Treasury Curve Inversion Modelling")

    have_10y = "10Y" in f_tsy.columns and not f_tsy["10Y"].dropna().empty
    have_2y  = "2Y"  in f_tsy.columns and not f_tsy["2Y"].dropna().empty
    have_spr = "Spread_10Y2Y" in f_tsy.columns

    fig1 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08, row_heights=[0.5, 0.5]
    )

    if have_10y:
        fig1.add_trace(go.Scatter(
            x=f_tsy["Date"], y=f_tsy["10Y"],
            name="US 10-Year Yield",
            line=dict(color="#58a6ff", width=2)
        ), row=1, col=1)

    if have_2y:
        fig1.add_trace(go.Scatter(
            x=f_tsy["Date"], y=f_tsy["2Y"],
            name="US 2-Year Yield",
            line=dict(color="#ff7b72", width=1.5)
        ), row=1, col=1)
    elif "2Y_proxy(3M)" in f_tsy.columns:
        fig1.add_trace(go.Scatter(
            x=f_tsy["Date"], y=f_tsy["2Y_proxy(3M)"],
            name="US 3-Month Yield (2Y proxy)",
            line=dict(color="#ff7b72", width=1.5, dash="dot")
        ), row=1, col=1)

    if have_spr and not f_tsy["Spread_10Y2Y"].dropna().empty:
        spr_series = f_tsy["Spread_10Y2Y"]
        fig1.add_trace(go.Scatter(
            x=f_tsy["Date"], y=spr_series,
            name="10Y – 2Y Spread",
            line=dict(color="#d29922", width=2),
            fill="tozeroy", fillcolor="rgba(210,153,34,0.04)"
        ), row=2, col=1)
        # Shade inversion periods
        inversion_mask = spr_series < 0
        if inversion_mask.any():
            fig1.add_trace(go.Scatter(
                x=f_tsy["Date"], y=spr_series.where(inversion_mask),
                name="Inversion",
                fill="tozeroy", fillcolor="rgba(248,81,73,0.12)",
                line=dict(color="rgba(0,0,0,0)", width=0),
                showlegend=True,
            ), row=2, col=1)
        fig1.add_hline(y=0, line_dash="dash", line_color="#f85149", line_width=1.5, row=2, col=1)

    fig1.update_layout(
        template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14",
        height=560, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig1.update_yaxes(title_text="Yield (%)", row=1, col=1, gridcolor="#21262d")
    fig1.update_yaxes(title_text="Spread (%)", row=2, col=1, gridcolor="#21262d")
    fig1.update_xaxes(gridcolor="#21262d")
    st.plotly_chart(fig1, use_container_width=True)

    if "Stub" in treasury_source:
        st.warning("⚠️ Yield data is unavailable — showing static placeholder values. Check your network connection.")


# ── TAB 2: Full Yield Curve Snapshot (NEW) ───────────────────────────────────
with tab2:
    st.markdown("### 🏛️ Full Yield Curve — Term Structure Snapshot")

    maturity_cols = [label for field, (label, _) in MATURITY_MAP.items() if label in f_tsy.columns]

    if maturity_cols and not f_tsy.empty:
        # Current curve (latest date)
        latest_row = f_tsy[["Date"] + maturity_cols].dropna(subset=maturity_cols, how="all").iloc[-1]
        latest_date = latest_row["Date"].strftime("%d %b %Y")

        # 1-year ago curve for comparison
        one_yr_ago = f_tsy["Date"].iloc[-1] - timedelta(days=365)
        past_rows = f_tsy[f_tsy["Date"] <= one_yr_ago]
        past_row = past_rows[["Date"] + maturity_cols].dropna(subset=maturity_cols, how="all")
        has_past = not past_row.empty

        tenors = [years for _, (label, years) in MATURITY_MAP.items() if label in maturity_cols]
        labels = [label for _, (label, _) in MATURITY_MAP.items() if label in maturity_cols]
        current_yields = [float(latest_row.get(lbl, np.nan)) for lbl in labels]

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=labels, y=current_yields,
            mode="lines+markers",
            name=f"Current ({latest_date})",
            line=dict(color="#58a6ff", width=2.5),
            marker=dict(size=8, color="#58a6ff"),
        ))

        if has_past:
            past_r = past_row.iloc[-1]
            past_date = past_r["Date"].strftime("%d %b %Y")
            past_yields = [float(past_r.get(lbl, np.nan)) for lbl in labels]
            fig_curve.add_trace(go.Scatter(
                x=labels, y=past_yields,
                mode="lines+markers",
                name=f"1 Year Ago ({past_date})",
                line=dict(color="#ff7b72", width=1.5, dash="dot"),
                marker=dict(size=6, color="#ff7b72"),
            ))

        fig_curve.add_hline(y=0, line_dash="dash", line_color="#30363d", line_width=1)
        fig_curve.update_layout(
            template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14",
            height=420, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Maturity", yaxis_title="Yield (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        fig_curve.update_xaxes(gridcolor="#21262d")
        fig_curve.update_yaxes(gridcolor="#21262d")
        st.plotly_chart(fig_curve, use_container_width=True)

        # Curve stats table
        if maturity_cols:
            st.markdown("**Current Yields by Maturity**")
            tbl = pd.DataFrame({
                "Maturity": labels,
                "Yield (%)": [f"{y:.2f}" if not np.isnan(y) else "—" for y in current_yields],
            })
            st.dataframe(
                tbl.set_index("Maturity").T,
                use_container_width=True,
                hide_index=False,
            )
    else:
        st.info("Full yield curve data is not available with the current data source. The Treasury API provides all maturities; ensure it is reachable.")


# ── TAB 3: VIX + Credit Spread ───────────────────────────────────────────────
with tab3:
    st.markdown("### 📊 Market Volatility Framework & Systemic Credit Stress")
    col_vix, col_cred = st.columns(2)

    with col_vix:
        st.markdown("**CBOE Volatility Index (VIX)**")
        fig_vix = go.Figure()
        if not f_vix.empty:
            fig_vix.add_trace(go.Scatter(
                x=f_vix["Date"], y=f_vix["VIX"],
                name="VIX", line=dict(color="#58a6ff", width=1.5)
            ))
            fig_vix.add_hline(y=20, line_dash="dot", line_color="#d29922", line_width=1, annotation_text="Elevated (20)", annotation_position="top left")
            fig_vix.add_hline(y=30, line_dash="dot", line_color="#f85149", line_width=1, annotation_text="Panic (30)", annotation_position="top left")
        fig_vix.update_layout(
            template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14",
            height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified"
        )
        fig_vix.update_yaxes(title_text="Index Value", gridcolor="#21262d")
        fig_vix.update_xaxes(gridcolor="#21262d")
        st.plotly_chart(fig_vix, use_container_width=True)

    with col_cred:
        st.markdown(f"**ICE BofA High Yield Credit Spread** <span class='source-badge'>{credit_source.split('(')[0].strip()}</span>", unsafe_allow_html=True)
        fig_hy = go.Figure()
        if not f_credit.empty:
            fig_hy.add_trace(go.Scatter(
                x=f_credit["Date"], y=f_credit["HY_Spread"],
                name="HY Spread", line=dict(color="#ff7b72", width=1.5)
            ))
            fig_hy.add_hline(y=4.5, line_dash="dot", line_color="#f85149", line_width=1, annotation_text="Distress (4.5%)", annotation_position="top left")
        fig_hy.update_layout(
            template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14",
            height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified"
        )
        fig_hy.update_yaxes(title_text="Spread (%)", gridcolor="#21262d")
        fig_hy.update_xaxes(gridcolor="#21262d")
        st.plotly_chart(fig_hy, use_container_width=True)


# ── TAB 4: DXY + Gold ────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🌍 Global Currency Valuation & Alternative Safe Assets")
    col_dxy, col_gold = st.columns(2)

    with col_dxy:
        st.markdown("**US Dollar Index (DXY)**")
        fig_dxy = go.Figure()
        if not f_dxy.empty:
            fig_dxy.add_trace(go.Scatter(
                x=f_dxy["Date"], y=f_dxy["DXY"],
                name="DXY Index", line=dict(color="#79c0ff", width=1.5)
            ))
        fig_dxy.update_layout(
            template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14",
            height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified"
        )
        fig_dxy.update_yaxes(title_text="Index Value", gridcolor="#21262d")
        fig_dxy.update_xaxes(gridcolor="#21262d")
        st.plotly_chart(fig_dxy, use_container_width=True)

    with col_gold:
        st.markdown("**Gold Spot Commodity**")
        fig_gold = go.Figure()
        if not f_gold.empty:
            fig_gold.add_trace(go.Scatter(
                x=f_gold["Date"], y=f_gold["Gold"],
                name="Gold", line=dict(color="#d29922", width=1.5)
            ))
        fig_gold.update_layout(
            template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14",
            height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified"
        )
        fig_gold.update_yaxes(title_text="Price (USD/oz)", gridcolor="#21262d")
        fig_gold.update_xaxes(gridcolor="#21262d")
        st.plotly_chart(fig_gold, use_container_width=True)

st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#484f58; font-size:.7rem; '
    'font-family:IBM Plex Mono,monospace;">'
    'Macro indicators are lagged and intended for architectural market assessment. '
    'Sourced via US Treasury FiscalData API, FRED &amp; Yahoo Finance.'
    '</div>',
    unsafe_allow_html=True
)
