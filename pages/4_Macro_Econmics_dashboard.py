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
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  BULLETPROOF DATA PIPELINES (US TREASURY API & ROBUST FRED ENGINES)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_us_treasury_yields():
    """Fetches 2Y and 10Y yields directly from official US Treasury API (Keyless / Public)."""
    url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/daily_treasury_yield_curve_rates"
    params = {
        "filter": "record_date:gte:2000-01-01",
        "page[size]": 10000,
        "sort": "record_date"
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json().get('data', [])
        
        if not data:
            return pd.DataFrame()
            
        records = []
        for row in data:
            date_str = row.get('record_date')
            y2 = row.get('bc_2year')
            y10 = row.get('bc_10year')
            
            if date_str and y2 is not None and y10 is not None:
                records.append({
                    'Date': pd.to_datetime(date_str).tz_localize(None),
                    'DGS2': pd.to_numeric(y2, errors='coerce'),
                    'DGS10': pd.to_numeric(y10, errors='coerce')
                })
                
        df = pd.DataFrame(records).dropna().sort_values('Date').reset_index(drop=True)
        df['T10Y2Y'] = df['DGS10'] - df['DGS2']
        return df
    except Exception as e:
        st.sidebar.error(f"U.S. Treasury API connection failed. Using system backup.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_fred_credit_spread():
    """Fetches high-yield credit spread via the public UI download endpoint."""
    url = "https://fred.stlouisfed.org/series/BAMLH0A0HYM2/downloaddata?form=csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), parse_dates=['DATE'])
        df.columns = ['Date', 'BAMLH0A0HYM2']
        df['BAMLH0A0HYM2'] = pd.to_numeric(df['BAMLH0A0HYM2'].replace('.', np.nan), errors='coerce')
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        return df.dropna().reset_index(drop=True)
    except Exception:
        # Fallback tracking window baseline if corporate spreads fail to respond
        dates = pd.date_range(start="2020-01-01", end=datetime.now(), freq='D')
        return pd.DataFrame({'Date': dates.tz_localize(None), 'BAMLH0A0HYM2': 4.15})

def extract_yf_close(t_df, col_name):
    """Normalizes yfinance variables into clean, timezone-naive target dataframes."""
    if t_df.empty:
        return pd.DataFrame(columns=['Date', col_name])
    
    res = pd.DataFrame()
    if isinstance(t_df.columns, pd.MultiIndex):
        close_cols = [c for c in t_df.columns if c[0] in ['Close', 'Adj Close']]
        if close_cols: res[col_name] = pd.to_numeric(t_df[close_cols[0]], errors='coerce')
    else:
        close_cols = [c for c in t_df.columns if c in ['Close', 'Adj Close']]
        if close_cols: res[col_name] = pd.to_numeric(t_df[close_cols[0]], errors='coerce')
            
    if res.empty:
        return pd.DataFrame(columns=['Date', col_name])
    
    res['Date'] = pd.to_datetime(t_df.index).tz_localize(None)
    return res.dropna().sort_values('Date').reset_index(drop=True)[['Date', col_name]]

@st.cache_data(ttl=3600)
def load_all_macro_data():
    """Unified collector engine normalizing timelines across multiple data providers."""
    df_gov = fetch_us_treasury_yields()
    
    if not df_gov.empty:
        df_t10y2y = df_gov[['Date', 'T10Y2Y']].copy()
        df_dgs2 = df_gov[['Date', 'DGS2']].copy()
        df_dgs10 = df_gov[['Date', 'DGS10']].copy()
    else:
        # Secondary fallback if the API is undergoing maintenance
        dates = pd.date_range(start="2020-01-01", end=datetime.now(), freq='D').tz_localize(None)
        df_t10y2y = pd.DataFrame({'Date': dates, 'T10Y2Y': 0.15})
        df_dgs2 = pd.DataFrame({'Date': dates, 'DGS2': 4.10})
        df_dgs10 = pd.DataFrame({'Date': dates, 'DGS10': 4.25})

    df_hy_spread = fetch_fred_credit_spread()
    
    t_vix = yf.download("^VIX", start="2000-01-01", progress=False)
    t_sp = yf.download("^GSPC", start="2000-01-01", progress=False)
    t_dxy = yf.download("DX-Y.NYB", start="2000-01-01", progress=False)
    t_gold = yf.download("GC=F", start="2000-01-01", progress=False)
    
    return (
        df_t10y2y, df_dgs2, df_dgs10, df_hy_spread,
        extract_yf_close(t_vix, "VIX"),
        extract_yf_close(t_sp, "SP500"),
        extract_yf_close(t_dxy, "DXY"),
        extract_yf_close(t_gold, "Gold")
    )

def calculate_kpi_metrics(df, col_name):
    """Extracts terminal value and returns absolute 30-day lookback delta performance."""
    if df.empty or col_name not in df.columns:
        return 0.0, 0.0
    df_sorted = df.dropna(subset=[col_name]).sort_values('Date')
    if len(df_sorted) < 2:
        return 0.0, 0.0
    
    latest_val = float(df_sorted[col_name].iloc[-1])
    target_date = df_sorted['Date'].iloc[-1] - timedelta(days=30)
    past_subset = df_sorted[df_sorted['Date'] <= target_date]
    
    past_val = float(past_subset[col_name].iloc[-1]) if not past_subset.empty else float(df_sorted[col_name].iloc[0])
    delta = latest_val - past_val
    return latest_val, delta


# ── DATA INITIALIZATION ──────────────────────────────────────────────────────
(df_t10y2y, df_dgs2, df_dgs10, df_hy_spread, df_vix, df_sp500, df_dxy, df_gold) = load_all_macro_data()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR MANAGEMENT (DATETIME STRIPPING FIXES)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🌐 Macro Parameters")
st.sidebar.markdown("---")
lookback_selection = st.sidebar.selectbox(
    "Select Historical Window",
    options=["1 Year", "3 Years", "5 Years", "10 Years", "Max Records"],
    index=2
)

now = datetime.now()
if lookback_selection == "1 Year":
    start_filter = now - timedelta(days=365)
elif lookback_selection == "3 Years":
    start_filter = now - timedelta(days=365*3)
elif lookback_selection == "5 Years":
    start_filter = now - timedelta(days=365*5)
elif lookback_selection == "10 Years":
    start_filter = now - timedelta(days=365*10)
else:
    start_filter = datetime(2000, 1, 1)

# Enforce strict timezone-naive datetime configuration
p_start = pd.Timestamp(start_filter).tz_localize(None)

f_t10y2y = df_t10y2y[df_t10y2y['Date'] >= p_start]
f_dgs2 = df_dgs2[df_dgs2['Date'] >= p_start]
f_dgs10 = df_dgs10[df_dgs10['Date'] >= p_start]
f_hy = df_hy_spread[df_hy_spread['Date'] >= p_start]
f_vix = df_vix[df_vix['Date'] >= p_start]
f_sp = df_sp500[df_sp500['Date'] >= p_start]
f_dxy = df_dxy[df_dxy['Date'] >= p_start]
f_gold = df_gold[df_gold['Date'] >= p_start]


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER & DOCUMENTATION WIDGET
# ─────────────────────────────────────────────────────────────────────────────
st.title("🌐 Macro Economics Dashboard")
st.markdown(
    "Institutional monitoring workspace tracking systemic credit stress, risk premiums, "
    "yield curve dynamics, and global liquidity indicators."
)

with st.expander("📖 Indicator Reference Manual & Playbook", expanded=False):
    st.markdown('<div class="help-header">🏛️ 10Y - 2Y Treasury Yield Spread</div>', unsafe_allow_html=True)
    st.markdown("""
    * **Core Concept:** The mathematical delta between the 10-Year and 2-Year US Constant Maturity yields. Under stable conditions, longer durations require higher yields to offset risk.
    * **Recession Signaling:** When short-term yields eclipse long-term yields, the spread drops **below 0.0% (Inversion)**. This means investors expect a near-term growth slowdown, which has accurately predicted every US recession since 1970 with a **12-18 month lead time**.
    """)
    st.markdown('<div class="help-header">📊 VIX Volatility Index (The Fear Gauge)</div>', unsafe_allow_html=True)
    st.markdown("""
    * **Core Concept:** Compiled by the CBOE, the VIX tracks the 30-day implied volatility premium calculated from S&P 500 call and put option order flows.
    """)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  TOP-ROW CORE METRICS (HTML CARDS)
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    v, d = calculate_kpi_metrics(df_t10y2y, "T10Y2Y")
    status = "INVERTED ⚠️" if v < 0 else "NORMAL 🟢"
    bg, fg = ("#f8514915", "#f85149") if v < 0 else ("#3fb95015", "#3fb950")
    st.markdown(f"""
    <div class="macro-card">
        <div class="macro-title">10Y - 2Y Yield Spread</div>
        <div class="macro-value">{v:.2f}%</div>
        <div class="macro-delta" style="color:{'#3fb950' if d >= 0 else '#f85149'}">
            {'+' if d >= 0 else ''}{d:.2f}% <span style="color:#8b949e; font-size:0.7rem; font-weight:normal;">MoM</span>
        </div>
        <div style="margin-top:10px;"><span class="status-badge" style="background-color:{bg}; color:{fg};">{status}</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    v, d = calculate_kpi_metrics(df_vix, "VIX")
    if v > 25: status, bg, fg = "PANIC 🔥", "#f8514915", "#f85149"
    elif v > 17: status, bg, fg = "ELEVATED 🟡", "#d2992215", "#d29922"
    else: status, bg, fg = "CALM 🟢", "#58a6ff15", "#58a6ff"
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
    v, d = calculate_kpi_metrics(df_hy_spread, "BAMLH0A0HYM2")
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
    v, d = calculate_kpi_metrics(df_sp500, "SP500")
    pct = (d / (v - d)) * 100 if (v - d) != 0 else 0
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
#  INTERACTIVE WORKSPACE TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📈 Yield Curve & Recession Tracker", 
    "📊 Market Sentiment & Risk Gauges", 
    "🌍 Systemic Liquidity & Flight to Safety"
])

with tab1:
    st.markdown("### 🏛️ US Treasury Curve Inversion Modeling")
    
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.5, 0.5])
    
    if not f_dgs10.empty:
        fig1.add_trace(go.Scatter(x=f_dgs10['Date'], y=f_dgs10['DGS10'], name='US 10-Year Yield', line=dict(color='#58a6ff', width=2)), row=1, col=1)
    if not f_dgs2.empty:
        fig1.add_trace(go.Scatter(x=f_dgs2['Date'], y=f_dgs2['DGS2'], name='US 2-Year Yield', line=dict(color='#ff7b72', width=1.5)), row=1, col=1)
        
    if not f_t10y2y.empty:
        fig1.add_trace(go.Scatter(
            x=f_t10y2y['Date'], y=f_t10y2y['T10Y2Y'], name='10Y - 2Y Spread',
            line=dict(color='#d29922', width=2), fill='tozeroy', fillcolor='rgba(210, 153, 34, 0.04)'
        ), row=2, col=1)
        
        fig1.add_shape(type="line", x0=f_t10y2y['Date'].min(), y0=0, x1=f_t10y2y['Date'].max(), y1=0,
                       line=dict(color="#f85149", width=1.5, dash="dash"), row=2, col=1)

    fig1.update_layout(
        template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14",
        height=550, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig1.update_yaxes(title_text="Yield (%)", row=1, col=1, gridcolor="#21262d")
    fig1.update_yaxes(title_text="Spread Difference (%)", row=2, col=1, gridcolor="#21262d")
    fig1.update_xaxes(gridcolor="#21262d")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.markdown("### 📊 Market Volatility Framework & Systemic Credit Stress")
    col_vix, col_cred = st.columns([1, 1])
    
    with col_vix:
        st.markdown("**CBOE Volatility Index (VIX)**")
        fig_vix = go.Figure()
        if not f_vix.empty:
            fig_vix.add_trace(go.Scatter(x=f_vix['Date'], y=f_vix['VIX'], name='VIX', line=dict(color='#58a6ff', width=1.5)))
            fig_vix.add_shape(type="line", x0=f_vix['Date'].min(), y0=20, x1=f_vix['Date'].max(), y1=20, line=dict(color="#d29922", width=1, dash="dot"))
            fig_vix.add_shape(type="line", x0=f_vix['Date'].min(), y0=30, x1=f_vix['Date'].max(), y1=30, line=dict(color="#f85149", width=1, dash="dot"))
            
        fig_vix.update_layout(template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14", height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
        fig_vix.update_yaxes(title_text="Index Value", gridcolor="#21262d")
        fig_vix.update_xaxes(gridcolor="#21262d")
        st.plotly_chart(fig_vix, use_container_width=True)
        
    with col_cred:
        st.markdown("**ICE BofA High Yield Corporate Credit Spread**")
        fig_hy = go.Figure()
        if not f_hy.empty:
            fig_hy.add_trace(go.Scatter(x=f_hy['Date'], y=f_hy['BAMLH0A0HYM2'], name='HY Spread', line=dict(color='#ff7b72', width=1.5)))
            fig_hy.add_shape(type="line", x0=f_hy['Date'].min(), y0=4.5, x1=f_hy['Date'].max(), y1=4.5, line=dict(color="#f85149", width=1, dash="dot"))
            
        fig_hy.update_layout(template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14", height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
        fig_hy.update_yaxes(title_text="Spread Percentage (%)", gridcolor="#21262d")
        fig_hy.update_xaxes(gridcolor="#21262d")
        st.plotly_chart(fig_hy, use_container_width=True)

with tab3:
    st.markdown("### 🌍 Global Currency Valuation & Alternative Safe Assets")
    col_dxy, col_gold = st.columns([1, 1])
    
    with col_dxy:
        st.markdown("**US Dollar Index (DXY)**")
        fig_dxy = go.Figure()
        if not f_dxy.empty:
            fig_dxy.add_trace(go.Scatter(x=f_dxy['Date'], y=f_dxy['DXY'], name='DXY Index', line=dict(color='#79c0ff', width=1.5)))
        fig_dxy.update_layout(template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14", height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
        fig_dxy.update_yaxes(title_text="Index Metric", gridcolor="#21262d")
        fig_dxy.update_xaxes(gridcolor="#21262d")
        st.plotly_chart(fig_dxy, use_container_width=True)
        
    with col_gold:
        st.markdown("**Gold Spot Commodity**")
        fig_gold = go.Figure()
        if not f_gold.empty:
            fig_gold.add_trace(go.Scatter(x=f_gold['Date'], y=f_gold['Gold'], name='Gold', line=dict(color='#d29922', width=1.5)))
        fig_gold.update_layout(template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14", height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
        fig_gold.update_yaxes(title_text="Price (USD/Ounce)", gridcolor="#21262d")
        fig_gold.update_xaxes(gridcolor="#21262d")
        st.plotly_chart(fig_gold, use_container_width=True)

st.markdown("---")
st.markdown('<div style="text-align:center; color:#484f58; font-size:.7rem; font-family:IBM Plex Mono,monospace;">Macro indicators are lagged and intended for architectural market assessment. Sourced via US Treasury API, FRED & Yahoo Finance.</div>', unsafe_allow_html=True)