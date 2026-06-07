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
    # 1. Yield parameters from FRED
    df_t10y2y = fetch_fred_csv("T10Y2Y") # 10Y-2Y Spread
    df_dgs2 = fetch_fred_csv("DGS2")     # 2-Year Constant Yield
    df_dgs10 = fetch_fred_csv("DGS10")   # 10-Year Constant Yield
    df_hy_spread = fetch_fred_csv("BAMLH0A0HYM2") # High Yield Credit Spread
    
    # 2. Broad market parameters from Yahoo Finance
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
    options=