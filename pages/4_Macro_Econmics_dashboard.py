"""
╔══════════════════════════════════════════════════════════╗
║         MACRO ECONOMICS MONITORING DASHBOARD             ║
║  Data Sources: FRED (Direct) | Yahoo Finance (yfinance)  ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
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

# Dark theme CSS matching original application styling rules
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
    
    /* Document Widget Custom Style Adjustments */
    .stExpander {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        margin-bottom: 20px;
    }
    .stExpander p {
        font-size: 0.9rem;
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
#  ROBUST DATA PIPELINES (FRED & YFINANCE HELPERS)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_fred_csv(series_id):
    """Fetches clean data arrays directly from St. Louis Fed endpoint without keys."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url, parse_dates=['DATE'])
        df.columns = ['Date', series_id]
        df[series_id] = pd.to_numeric(df[series_id].replace('.', np.nan), errors='coerce')
        df = df.dropna()
        return df
    except Exception:
        return pd.DataFrame(columns=['Date', series_id])

def extract_yf_close(t_df, col_name):
    """Safely handles single or multi-index structural columns from yfinance."""
    if t_df.empty:
        return pd.DataFrame(columns=['Date', col_name])
    t_df = t_df.copy()
    close_col = [col for col in t_df.columns if 'Close' in col or (isinstance(col, tuple) and 'Close' in col[0])]
    if not close_col:
        return pd.DataFrame(columns=['Date', col_name])
    
    res = pd.DataFrame(index=t_df.index)
    res[col_name] = pd.to_numeric(t_df[close_col[0]].values.flatten(), errors='coerce')
    res = res.dropna().reset_index()
    res.rename(columns={res.columns[0]: 'Date'}, inplace=True)
    return res

@st.cache_data(ttl=3600)
def load_all_macro_data():
    """Unified collection engine for system efficiency."""
    df_t10y2y = fetch_fred_csv("T10Y2Y") # 10Y-2Y Spread
    df_dgs2 = fetch_fred_csv("DGS2")     # 2-Year Constant Yield
    df_dgs10 = fetch_fred_csv("DGS10")   # 10-Year Constant Yield
    df_hy_spread = fetch_fred_csv("BAMLH0A0HYM2") # High Yield Credit Spread
    
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

def calculate_kpi_metrics(df, col_name, is_percentage=False):
    """Extracts final data value and calculates a trailing 30-day lookback delta."""
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


# ── EXECUTION & PIPELINE ─────────────────────────────────────────────────────
(df_t10y2y, df_dgs2, df_dgs10, df_hy_spread, df_vix, df_sp500, df_dxy, df_gold) = load_all_macro_data()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR MANAGEMENT
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

p_start = pd.Timestamp(start_filter)
f_t10y2y = df_t10y2y[df_t10y2y['Date'] >= p_start]
f_dgs2 = df_dgs2[df_dgs2['Date'] >= p_start]
f_dgs10 = df_dgs10[df_dgs10['Date'] >= p_start]
f_hy = df_hy_spread[df_hy_spread['Date'] >= p_start]
f_vix = df_vix[df_vix['Date'] >= p_start]
f_sp = df_sp500[df_sp500['Date'] >= p_start]
f_dxy = df_dxy[df_dxy['Date'] >= p_start]
f_gold = df_gold[df_gold['Date'] >= p_start]


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER HERO SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.title("🌐 Macro Economics Dashboard")
st.markdown(
    "Institutional monitoring workspace tracking systemic credit stress, risk premiums, "
    "yield curve dynamics, and global liquidity indicators."
)

# ── ON-DEMAND HELP DOCUMENTATION WIDGET ──────────────────────────────────────
with st.expander("📖 Indicator Reference Manual & Playbook", expanded=False):
    st.markdown("""
    This panel outlines structural mechanics, baseline calibrations, and execution frameworks for monitored indexes.
    """)
    
    st.markdown('<div class="help-header">🏛️ 10Y - 2Y Treasury Yield Spread</div>', unsafe_allow_html=True)
    st.markdown("""
    * **Core Concept:** The mathematical delta between the 10-Year and 2-Year US Constant Maturity yields. Under stable conditions, longer durations require higher yields to offset risk.
    * **Recession Signaling:** When short-term yields eclipse long-term yields, the spread drops **below 0.0% (Inversion)**. This means investors expect a near-term growth slowdown, which has accurately predicted every US recession since 1970 with a **12-18 month lead time**.
    * **The De-Inversion Trap:** Risk peaks not during the inversion itself, but when the curve rapidly un-inverts and jumps back above zero. This usually means the Federal Reserve is rushing to slash interest rates to stabilize a weakening credit landscape.
    """)
    
    st.markdown('<div class="help-header">📊 VIX Volatility Index (The Fear Gauge)</div>', unsafe_allow_html=True)
    st.markdown("""
    * **Core Concept:** Compiled by the CBOE, the VIX tracks the 30-day implied volatility premium calculated from S&P 500 call and put option order flows.
    * **Operational Thresholds:**
        * **< 15 (Calm):** Suggests structural market stability or potential investor complacency.
        * **17 - 25 (Elevated):** Signals tactical correction zones, macro uncertainty, or upcoming systemic events.
        * **> 30 (Panic/Capitulation):** Indicates forced liquidations and extreme institutional hedging, which often align with long-term equity buying opportunities.
    """)
    
    st.markdown('<div class="help-header">🚨 ICE BofA High Yield Corporate Credit Spread</div>', unsafe_allow_html=True)
    st.markdown("""
    * **Core Concept:** The extra yield required by the market to hold non-investment grade (junk) corporate debt over risk-free US Treasury notes.
    * **Why It Matters:** Equities are heavily reliant on corporate credit lines. When this spread expands above **4.5%**, it signals that low-tier corporate borrowers are facing higher funding strains, which serves as a leading indicator for defaults and financial stress.
    """)
    
    st.markdown('<div class="help-header">🌍 US Dollar Currency Index (DXY)</div>', unsafe_allow_html=True)
    st.markdown("""
    * **Core Concept:** Geometrical currency weight checking the purchasing power of the US Dollar against a basket of international foreign reserve currencies (primarily the Euro and Yen).
    * **Market Mechanics:** A rising DXY exerts a deflationary weight on risk assets. Because global commodity trades and multinational earnings rely on international funding liquidity, a surging dollar effectively acts as a monetary break on asset prices.
    """)
    
    st.markdown('<div class="help-header">🪙 Spot Gold Commodity (GC=F)</div>', unsafe_allow_html=True)
    st.markdown("""
    * **Core Concept:** Global macro safe-haven benchmark tracking tangible asset value outside the fiat monetary framework.
    * **Strategic Purpose:** Monitors structural debasement trends, unexpected real inflation shocks, and geopolitical risk. It helps identify capital flight away from traditional stock and bond allocations.
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

# ── TAB 1: YIELD CURVE DETAILED TRACKER ──────────────────────────────────────
with tab1:
    st.markdown("### 🏛️ US Treasury Curve Inversion Modeling")
    st.markdown(
        "A yield curve inversion occurs when yields on short-term horizons exceed long-term yields. "
        "Historically, a sustained drop below **0.0%** in the 10Y-2Y Spread is a premier leading recession indicator, "
        "preceding US economic contractions by roughly 12 to 18 months."
    )
    
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
    
    st.markdown("""
    <div style="background-color:#161b22; border:1px solid #30363d; padding:15px; border-radius:6px; font-size:0.85rem; line-height:1.5; color:#8b949e;">
        <strong style="color:#c9d1d9;">💡 Critical Analysis Framework:</strong><br/>
        • <strong>Inversion Phase:</strong> Reflects heavy policy tightening by the central bank. Short term rates rise to curb inflation while long-term rates stay suppressed due to lower growth forecasts.<br/>
        • <strong>The 'De-Inversion' Trap:</strong> Note that historical market selloffs and recessions often don't occur *during* deep inversion. Risk peaks when the curve rapidly un-inverts and moves back above zero, signaling aggressive policy rate cuts to salvage a cracking economy.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 2: SENTIMENT & VOLATILITY RISK GAUGES ────────────────────────────────
with tab2:
    st.markdown("### 📊 Market Volatility Framework & Systemic Credit Stress")
    
    col_vix, col_cred = st.columns([1, 1])
    
    with col_vix:
        st.markdown("**CBOE Volatility Index (VIX)**")
        st.markdown("Derived from option pricing architectures on the S&P 500. Represents a measure of short-term implied volatility.")
        
        fig_vix = go.Figure()
        if not f_vix.empty:
            fig_vix.add_trace(go.Scatter(x=f_vix['Date'], y=f_vix['VIX'], name='VIX', line=dict(color='#58a6ff', width=1.5)))
            fig_vix.add_shape(type="line", x0=f_vix['Date'].min(), y0=20, x1=f_vix['Date'].max(), y1=20, line=dict(color="#d29922", width=1, dash="dot"))
            fig_vix.add_shape(type="line", x0=f_vix['Date'].min(), y0=30, x1=f_vix['Date'].max(), y1=30, line=dict(color="#f85149", width=1, dash="dot"))
            
        fig_vix.update_layout(
            template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14",
            height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified"
        )
        fig_vix.update_yaxes(title_text="Index Value", gridcolor="#21262d")
        fig_vix.update_xaxes(gridcolor="#21262d")
        st.plotly_chart(fig_vix, use_container_width=True)
        
    with col_cred:
        st.markdown("**ICE BofA High Yield Corporate Credit Spread**")
        st.markdown("Measures the premium yield high-risk/junk corporate borrowers must offer over safe-haven government bonds.")
        
        fig_hy = go.Figure()
        if not f_hy.empty:
            fig_hy.add_trace(go.Scatter(x=f_hy['Date'], y=f_hy['BAMLH0A0HYM2'], name='HY Spread', line=dict(color='#ff7b72', width=1.5)))
            fig_hy.add_shape(type="line", x0=f_hy['Date'].min(), y0=4.5, x1=f_hy['Date'].max(), y1=4.5, line=dict(color="#f85149", width=1, dash="dot"))
            
        fig_hy.update_layout(
            template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14",
            height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified"
        )
        fig_hy.update_yaxes(title_text="Spread Percentage (%)", gridcolor="#21262d")
        fig_hy.update_xaxes(gridcolor="#21262d")
        st.plotly_chart(fig_hy, use_container_width=True)

# ── TAB 3: SYSTEMIC LIQUIDITY MATRIX ─────────────────────────────────────────
with tab3:
    st.markdown("### 🌍 Global Currency Valuation & Alternative Safe Assets")
    
    col_dxy, col_gold = st.columns([1, 1])
    
    with col_dxy:
        st.markdown("**US Dollar Index (DXY)**")
        st.markdown("Tracks USD strength relative to a basket of key foreign currencies. A rising DXY strains international dollar liquidity.")
        
        fig_dxy = go.Figure()
        if not f_dxy.empty:
            fig_dxy.add_trace(go.Scatter(x=f_dxy['Date'], y=f_dxy['DXY'], name='DXY Index', line=dict(color='#79c0ff', width=1.5)))
        fig_dxy.update_layout(
            template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14",
            height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified"
        )
        fig_dxy.update_yaxes(title_text="Index Metric", gridcolor="#21262d")
        fig_dxy.update_xaxes(gridcolor="#21262d")
        st.plotly_chart(fig_dxy, use_container_width=True)
        
    with col_gold:
        st.markdown("**Gold Spot Commodity**")
        st.markdown("The classic hedge against structural inflation and real purchasing power decay within fiat currencies.")
        
        fig_gold = go.Figure()
        if not f_gold.empty:
            fig_gold.add_trace(go.Scatter(x=f_gold['Date'], y=f_gold['Gold'], name='Gold', line=dict(color='#d29922', width=1.5)))
        fig_gold.update_layout(
            template="plotly_dark", plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14",
            height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified"
        )
        fig_gold.update_yaxes(title_text="Price (USD/Ounce)", gridcolor="#21262d")
        fig_gold.update_xaxes(gridcolor="#21262d")
        st.plotly_chart(fig_gold, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER BLOCK
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#484f58; font-size:.7rem; font-family:IBM Plex Mono,monospace;">'
    'Macro indicators are lagged and intended for architectural market assessment. Data sourced via FRED & Yahoo Finance APIs.</div>',
    unsafe_allow_html=True
)