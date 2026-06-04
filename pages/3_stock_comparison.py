import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL STYLES — Inheriting your dark trading terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght=300;400;600&family=IBM+Plex+Sans:wght=300;400;600&display=swap');

html, body, [class*="css"]      { font-family:'IBM Plex Sans',sans-serif; background:#0d0f14; color:#c9d1d9; }
.stApp                          { background:#0d0f14; }
section[data-testid="stSidebar"]{ background:#111318; border-right:1px solid #21262d; }

h1 { font-family:'IBM Plex Mono',monospace !important; color:#58a6ff !important; letter-spacing:-1px; }
h2,h3 { font-family:'IBM Plex Mono',monospace !important; color:#79c0ff !important; }

[data-testid="metric-container"] {
  background:#161b22; border:1px solid #21262d; border-radius:8px; padding:12px 16px;
}
[data-testid="metric-container"] label {
  color:#8b949e !important; font-size:0.72rem !important;
  letter-spacing:.08em; text-transform:uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family:'IBM Plex Mono',monospace; font-size:1.4rem !important; color:#e6edf3 !important;
}

.stButton > button {
  background:#238636; color:#fff; border:none; border-radius:6px;
  font-family:'IBM Plex Mono',monospace; font-weight:600;
  padding:10px 24px; width:100%; transition:background .2s;
}
.stButton > button:hover { background:#2ea043; }

.stSelectbox > div > div,
.stTextInput > div > div > input {
  background:#161b22 !important; border:1px solid #30363d !important;
  color:#e6edf3 !important; font-family:'IBM Plex Mono',monospace;
}

.stTabs [data-baseweb="tab-list"] { background:#161b22; border-bottom:1px solid #21262d; gap:0; }
.stTabs [data-baseweb="tab"] {
  font-family:'IBM Plex Mono',monospace; color:#8b949e;
  border-bottom:2px solid transparent; padding:10px 20px; font-size:.82rem;
}
.stTabs [aria-selected="true"] {
  color:#58a6ff !important; border-bottom:2px solid #58a6ff !important; background:transparent !important;
}

.stDataFrame { background:#161b22; border:1px solid #21262d; border-radius:8px; }
.stInfo      { background:#161b22; border-left:3px solid #58a6ff; color:#c9d1d9; }
.stWarning   { background:#161b22; border-left:3px solid #d29922; color:#c9d1d9; }

.custom-banner {
  background:linear-gradient(135deg,#0d1f0e 0%,#0d1929 100%);
  border:1px solid #238636; border-radius:8px; padding:14px 18px; margin:8px 0;
  font-family:'IBM Plex Mono',monospace; font-size:.78rem; color:#8b949e;
}
.custom-banner b { color:#3fb950; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DATA FETCHING W/ CACHE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_close(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch historical close prices safely for comparison plotting."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float)
        # Flatten columns if MultiIndex returned by yfinance
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if "Close" in df.columns:
            series = df["Close"]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series.index = pd.to_datetime(series.index)
            return series.dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Configuration")

tickers_input = st.sidebar.text_input(
    "Enter Tickers / ETFs (comma-separated)", 
    value="AAPL, MSFT, NVDA, SPY"
)

timeframe = st.sidebar.selectbox(
    "Select Duration",
    ["1 Month", "3 Months", "6 Months", "Year-to-Date (YTD)", "1 Year", "3 Years", "5 Years", "Custom Range"]
)

# Parse timeframes
end_date = datetime.today()
if timeframe == "1 Month":
    start_date = end_date - timedelta(days=30)
elif timeframe == "3 Months":
    start_date = end_date - timedelta(days=90)
elif timeframe == "6 Months":
    start_date = end_date - timedelta(days=180)
elif timeframe == "Year-to-Date (YTD)":
    start_date = datetime(end_date.year, 1, 1)
elif timeframe == "1 Year":
    start_date = end_date - timedelta(days=365)
elif timeframe == "3 Years":
    start_date = end_date - timedelta(days=365 * 3)
elif timeframe == "5 Years":
    start_date = end_date - timedelta(days=365 * 5)
else:
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start", end_date - timedelta(days=365))
    end_date = col2.date_input("End", end_date)

norm_mode = st.sidebar.selectbox(
    "Chart Return Type",
    ["Cumulative Return (%)", "Normalized to $100 Base", "Raw Stock Price ($)"]
)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DASHBOARD PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Multi-Asset Performance Comparator")
st.markdown("Overlay historical pricing metrics, scale performance parameters, and review asset correlation trees simultaneously.")

# Process ticker string input safely
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if not tickers:
    st.info("Please enter at least one valid ticker symbol in the sidebar.")
else:
    start_str = start_date.strftime("%Y-%m-%d") if isinstance(start_date, datetime) else str(start_date)
    end_str = end_date.strftime("%Y-%m-%d") if isinstance(end_date, datetime) else str(end_date)
    
    data_dict = {}
    invalid_tickers = []
    
    with st.spinner("Fetching market data..."):
        for t in tickers:
            series = fetch_ticker_close(t, start_str, end_str)
            if not series.empty and len(series) >= 2:
                data_dict[t] = series
            else:
                invalid_tickers.append(t)

    if invalid_tickers:
        st.sidebar.warning(f"Skipped invalid/empty data for: {', '.join(invalid_tickers)}")

    if not data_dict:
        st.error("No valid data could be found for any provided asset tokens during this period.")
    else:
        # Calculate analytics & data vectors
        metrics_list = []
        returns_dict = {}
        
        for ticker, series in data_dict.items():
            daily_rets = series.pct_change().dropna()
            returns_dict[ticker] = daily_rets
            
            # Mathematical calculations
            total_ret = (series.iloc[-1] / series.iloc[0] - 1) * 100
            days = (series.index[-1] - series.index[0]).days
            ann_ret = ((series.iloc[-1] / series.iloc[0]) ** (365.25 / days) - 1) * 100 if days > 0 else 0.0
            ann_vol = daily_rets.std() * np.sqrt(252) * 100
            sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0.0
            
            cum_max = series.cummax()
            max_dd = ((series - cum_max) / cum_max).min() * 100
            
            metrics_list.append({
                "Asset Ticker": ticker,
                "Total Return": total_ret,
                "Annualized Return": f"{ann_ret:.2f}%",
                "Annualized Volatility": f"{ann_vol:.2f}%",
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Max Drawdown": f"{max_dd:.2f}%"
            })
            
        metrics_df = pd.DataFrame(metrics_list)
        
        # Display Top Performer Winner Banner using your CSS tokens
        top_asset = max(metrics_list, key=lambda x: x["Total Return"])
        st.markdown(f"""
        <div class="custom-banner">
            🏆 <b>Top Performer Matrix:</b> {top_asset['Asset Ticker']} dominates the group with a total return profile of <b>{top_asset['Total Return']:.2f}%</b> inside this window.
        </div>
        """, unsafe_allow_html=True)
        
        # ── TAB PANELS ──
        tab1, tab2, tab3 = st.tabs(["📊 Performance Line Graph", "📋 Detailed Statistics", "🔀 Returns Correlation Matrix"])
        
        with tab1:
            fig = go.Figure()
            yaxis_title = ""
            
            for ticker, series in data_dict.items():
                if norm_mode == "Cumulative Return (%)":
                    plot_series = (series / series.iloc[0] - 1) * 100
                    yaxis_title = "Cumulative Performance Change (%)"
                elif norm_mode == "Normalized to $100 Base":
                    plot_series = (series / series.iloc[0]) * 100
                    yaxis_title = "Growth Scaled Value (Base $100)"
                else:
                    plot_series = series
                    yaxis_title = "Raw Closing Valuation ($)"
                    
                fig.add_trace(go.Scatter(
                    x=plot_series.index,
                    y=plot_series.values,
                    mode='lines',
                    name=ticker,
                    line=dict(width=2),
                    hovertemplate=f"<b>{ticker}</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.2f}}<extra></extra>"
                ))
                
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d0f14",
                plot_bgcolor="#0d0f14",
                xaxis=dict(showgrid=True, gridcolor="#21262d", title="Timeline", titlefont=dict(family="IBM Plex Mono", color="#8b949e")),
                yaxis=dict(showgrid=True, gridcolor="#21262d", title=yaxis_title, titlefont=dict(family="IBM Plex Mono", color="#8b949e")),
                legend=dict(font=dict(family="IBM Plex Mono", size=11), bgcolor="rgba(0,0,0,0)", bordercolor="#21262d"),
                margin=dict(l=40, r=40, t=20, b=40),
                height=520
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            # Reformat total return column for display cleanly
            display_df = metrics_df.copy()
            display_df["Total Return"] = display_df["Total Return"].apply(lambda x: f"{x:.2f}%")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
        with tab3:
            if len(returns_dict) > 1:
                corr_matrix = pd.DataFrame(returns_dict).corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmin=-1, zmax=1,
                    hovertemplate="Asset X: %{y}<br>Asset Y: %{x}<br>Correlation: %{z:.2f}<extra></extra>"
                ))
                fig_corr.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0d0f14",
                    plot_bgcolor="#0d0f14",
                    margin=dict(l=40, r=40, t=30, b=40),
                    height=450,
                    xaxis=dict(gridcolor="#21262d", fontfamily="IBM Plex Mono"),
                    yaxis=dict(gridcolor="#21262d", fontfamily="IBM Plex Mono")
                )
                st.markdown("### 🧮 Daily Percent Change Returns Correlation")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Input at least 2 valid tickers to compute a dynamic variance-covariance correlation map.")