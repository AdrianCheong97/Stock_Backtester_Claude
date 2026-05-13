"""
╔══════════════════════════════════════════════════════════╗
║         SWING TRADE BACKTESTER  •  v1.0                 ║
║  Data: yfinance (default) | FMP (optional API key)      ║
╚══════════════════════════════════════════════════════════╝

Strategies:
  1. Mean Reversion  — RSI + Bollinger Bands
  2. Momentum        — MACD Crossover + ATR Stop
  3. Trend Following — Golden/Death Cross (50/200 EMA)
  4. ★ Custom        — Donchian + Multi-EMA Stack + Candle Breakout
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

# ── ML model imports (optional — graceful fallback if files missing) ──
try:
    import xgboost as xgb
    import joblib
    from scipy import stats as scipy_stats
    from sklearn.preprocessing import RobustScaler
    _ML_LIBS_OK = True
except ImportError:
    _ML_LIBS_OK = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Swing Trade Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL STYLES  — dark trading terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

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
[data-testid="stMetricDelta"] { font-family:'IBM Plex Mono',monospace; font-size:.78rem !important; }

.stButton > button {
  background:#238636; color:#fff; border:none; border-radius:6px;
  font-family:'IBM Plex Mono',monospace; font-weight:600;
  padding:10px 24px; width:100%; transition:background .2s;
}
.stButton > button:hover { background:#2ea043; }

.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
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
.stInfo      { background:#161b22; border-left:3px solid #58a6ff; color:#c9d1d9; }
.stWarning   { background:#161b22; border-left:3px solid #d29922; color:#c9d1d9; }
.stSuccess   { background:#161b22; border-left:3px solid #3fb950; color:#c9d1d9; }
hr           { border-color:#21262d; }

.tag        { display:inline-block; background:#1f2937; border:1px solid #30363d; border-radius:12px; padding:2px 10px; font-size:.72rem; font-family:'IBM Plex Mono',monospace; color:#8b949e; margin:2px; }
.tag-green  { border-color:#238636; color:#3fb950; background:#0d1f0e; }
.tag-blue   { border-color:#1f6feb; color:#58a6ff; background:#0d1929; }
.tag-amber  { border-color:#9e6a03; color:#d29922; background:#1a1300; }

.custom-banner {
  background:linear-gradient(135deg,#0d1f0e 0%,#0d1929 100%);
  border:1px solid #238636; border-radius:8px; padding:14px 18px; margin:8px 0;
  font-family:'IBM Plex Mono',monospace; font-size:.78rem; color:#8b949e;
}
.custom-banner b { color:#3fb950; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        # Flatten MultiIndex columns (yfinance sometimes returns these)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index   = pd.to_datetime(df.index)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        st.error(f"yfinance error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fmp(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Fetch historical daily OHLCV from Financial Modelling Prep."""
    url = (
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
        f"?from={start}&to={end}&apikey={api_key}"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        historical = r.json().get("historical", [])
        if not historical:
            return pd.DataFrame()
        df = pd.DataFrame(historical)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        st.error(f"FMP error: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(s, n)
    std = s.rolling(n).std()
    return mid - k * std, mid, mid + k * std

def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    m = ema(s, fast) - ema(s, slow)
    sg = ema(m, signal)
    return m, sg

def donchian(high: pd.Series, low: pd.Series, n: int = 20):
    upper = high.rolling(n).max()
    lower = low.rolling(n).min()
    mid   = (upper + lower) / 2
    return upper, lower, mid


# ─────────────────────────────────────────────────────────────────────────────
#  XGBoost MODEL — loader + feature builder
#  Mirrors the exact feature engineering in stock_direction_predictor.py
# ─────────────────────────────────────────────────────────────────────────────

XGB_MODEL_PATH      = "xgb_stock_model.json"
XGB_ARTIFACTS_PATH  = "model_artifacts.joblib"
LABEL_NAMES         = ["BEARISH", "SIDEWAYS", "BULLISH"]
# colour tokens for each class
PRED_COLOURS = {
    "BEARISH":  "#f85149",
    "SIDEWAYS": "#d29922",
    "BULLISH":  "#3fb950",
}


@st.cache_resource(show_spinner=False)
def load_xgb_model():
    """Load saved XGBoost model + artifacts. Returns (model, artifacts) or (None, None)."""
    if not _ML_LIBS_OK:
        return None, None
    if not (Path(XGB_MODEL_PATH).exists() and Path(XGB_ARTIFACTS_PATH).exists()):
        return None, None
    try:
        model = xgb.XGBClassifier()
        model.load_model(XGB_MODEL_PATH)
        artifacts = joblib.load(XGB_ARTIFACTS_PATH)
        return model, artifacts
    except Exception as e:
        st.warning(f"Could not load XGB model: {e}")
        return None, None


def _add_emas(df):
    for p in [10, 20, 50, 200]:
        df[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
    return df

def _add_atr_ml(df, period=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=period, adjust=False).mean()
    return df

def _add_donchian_ml(df, period=20):
    df["DC_upper"] = df["High"].rolling(period).max()
    df["DC_lower"] = df["Low"].rolling(period).min()
    df["DC_mid"]   = (df["DC_upper"] + df["DC_lower"]) / 2
    df["DC_width"] = df["DC_upper"] - df["DC_lower"]
    return df

def _add_sharpe_ml(df, period=20):
    ret = df["Close"].pct_change()
    mu  = ret.rolling(period).mean()
    sd  = ret.rolling(period).std()
    df["Sharpe"] = (mu / sd.replace(0, np.nan)) * np.sqrt(252)
    return df

def _add_alpha_beta_ml(df, spy_close: pd.Series, period=60):
    """Vectorised rolling OLS beta & alpha vs SPY — no row-by-row loop."""
    df["Beta"]  = np.nan
    df["Alpha"] = np.nan
    stk_ret = df["Close"].pct_change()
    spy_ret = spy_close.reindex(df.index).pct_change()
    aligned = pd.concat([stk_ret.rename("stk"), spy_ret.rename("spy")], axis=1).dropna()
    betas, alphas = [], []
    for i in range(len(aligned)):
        if i < period:
            betas.append(np.nan); alphas.append(np.nan)
        else:
            w = aligned.iloc[i - period : i]
            slope, intercept, *_ = scipy_stats.linregress(w["spy"], w["stk"])
            betas.append(slope)
            alphas.append(intercept * 252)
    beta_s  = pd.Series(betas,  index=aligned.index)
    alpha_s = pd.Series(alphas, index=aligned.index)
    df["Beta"]  = beta_s.reindex(df.index)
    df["Alpha"] = alpha_s.reindex(df.index)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_spy(years_back: int) -> pd.Series:
    """Fetch SPY close prices for Beta/Alpha calculation."""
    try:
        import yfinance as yf
        end   = datetime.today()
        start = end - timedelta(days=years_back * 365 + 90)   # extra buffer
        spy = yf.download("SPY", start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        return spy["Close"]
    except Exception:
        return pd.Series(dtype=float)


def build_ml_features(df: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    """
    Reproduce every feature from stock_direction_predictor.engineer_features().
    Input df must have OHLCV columns. Returns df with all ML feature columns appended.
    """
    d = df.copy()
    d = _add_emas(d)
    d = _add_atr_ml(d)
    d = _add_donchian_ml(d)
    d = _add_sharpe_ml(d)
    if not spy_close.empty:
        d = _add_alpha_beta_ml(d, spy_close)
    else:
        d["Beta"] = np.nan; d["Alpha"] = np.nan

    # OHLC deltas
    d["HL_range"]     = d["High"]  - d["Low"]
    d["OC_delta"]     = d["Close"] - d["Open"]
    d["OC_pct"]       = d["OC_delta"] / d["Open"]
    d["HO_gap"]       = d["High"]  - d["Open"]
    d["OL_gap"]       = d["Open"]  - d["Low"]
    d["HLC_avg"]      = (d["High"] + d["Low"] + d["Close"]) / 3
    d["gap_open"]     = d["Open"]  - d["Close"].shift(1)
    d["gap_open_pct"] = d["gap_open"] / d["Close"].shift(1)

    # Log returns
    d["log_ret"]      = np.log(d["Close"] / d["Close"].shift(1))
    d["log_ret_2"]    = np.log(d["Close"] / d["Close"].shift(2))
    d["log_ret_5"]    = np.log(d["Close"] / d["Close"].shift(5))
    d["log_ret_10"]   = np.log(d["Close"] / d["Close"].shift(10))
    d["realised_vol"] = d["log_ret"].rolling(20).std() * np.sqrt(252)

    # Volume features
    d["vol_ma20"]  = d["Volume"].rolling(20).mean()
    d["vol_ratio"] = d["Volume"] / d["vol_ma20"].replace(0, np.nan)
    d["vol_log"]   = np.log1p(d["Volume"])

    # Price vs EMA flags
    for p in [10, 20, 50, 200]:
        d[f"above_EMA{p}"] = (d["Close"] > d[f"EMA_{p}"]).astype(int)
        d[f"dist_EMA{p}"]  = (d["Close"] - d[f"EMA_{p}"]) / d["ATR"].replace(0, np.nan)

    d["EMA10_slope"]   = d["EMA_10"].diff(3) / d["EMA_10"].shift(3)
    d["EMA10_x_EMA20"] = (d["EMA_10"] > d["EMA_20"]).astype(int)
    d["EMA20_x_EMA50"] = (d["EMA_20"] > d["EMA_50"]).astype(int)

    # Donchian position
    d["DC_pos"] = ((d["Close"] - d["DC_lower"]) /
                   d["DC_width"].replace(0, np.nan))
    return d


# Columns excluded from feature matrix (same EXCLUDE set as training script)
_EXCLUDE_COLS = {"Open", "High", "Low", "Close", "Volume", "ticker",
                 "target", "target_enc", "signal"}

def _get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in _EXCLUDE_COLS]


def run_ml_predictions(df_raw: pd.DataFrame,
                       model,
                       artifacts: dict,
                       spy_close: pd.Series) -> pd.DataFrame:
    """
    Build features → scale → predict probabilities for every row.
    Returns a DataFrame indexed like df_raw with columns:
        pred_label, pred_BEARISH, pred_SIDEWAYS, pred_BULLISH, pred_correct
    pred_correct compares the predicted direction to the ACTUAL next-day
    close direction so we know accuracy at backtest time.
    """
    feat_df   = build_ml_features(df_raw, spy_close)
    feat_cols = artifacts.get("feature_cols", _get_feature_cols(feat_df))

    # Only keep columns that actually exist after feature building
    feat_cols = [c for c in feat_cols if c in feat_df.columns]

    X = feat_df[feat_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    scaler = artifacts.get("scaler")
    if scaler is not None:
        X_vals = scaler.transform(X.fillna(0))
    else:
        X_vals = X.fillna(0).values

    proba  = model.predict_proba(X_vals)          # (n, 3) — BEARISH/SIDEWAYS/BULLISH
    labels = [LABEL_NAMES[i] for i in proba.argmax(axis=1)]

    out = pd.DataFrame({
        "pred_label":   labels,
        "pred_BEARISH":  proba[:, 0],
        "pred_SIDEWAYS": proba[:, 1],
        "pred_BULLISH":  proba[:, 2],
    }, index=feat_df.index)

    # Actual next-day direction (ground truth for accuracy calculation)
    actual_ret  = np.log(df_raw["Close"] / df_raw["Close"].shift(1)).shift(-1)
    THRESH      = 0.003
    actual_dir  = pd.Series("SIDEWAYS", index=df_raw.index)
    actual_dir[actual_ret >  THRESH] = "BULLISH"
    actual_dir[actual_ret < -THRESH] = "BEARISH"
    out["actual_dir"]   = actual_dir
    out["pred_correct"] = (out["pred_label"] == out["actual_dir"]).astype(int)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 1 — RSI Mean Reversion + Bollinger Bands
# ─────────────────────────────────────────────────────────────────────────────
def strategy_rsi_bollinger(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    LONG when RSI < oversold AND price <= lower Bollinger Band.
    EXIT when RSI > midline OR price >= upper Bollinger Band.
    """
    d = df.copy()
    d["RSI"]                    = rsi(d["Close"], params["rsi_period"])
    d["BB_L"], d["BB_M"], d["BB_U"] = bollinger(d["Close"], params["bb_period"], params["bb_std"])
    d["signal"] = 0
    in_trade = False

    for i in range(1, len(d)):
        r_val = d["RSI"].iloc[i]
        c     = d["Close"].iloc[i]
        if not in_trade:
            if r_val < params["rsi_oversold"] and c <= d["BB_L"].iloc[i]:
                d.iloc[i, d.columns.get_loc("signal")] = 1
                in_trade = True
        else:
            if r_val > params["rsi_midline"] or c >= d["BB_U"].iloc[i]:
                d.iloc[i, d.columns.get_loc("signal")] = -1
                in_trade = False
    return d


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 2 — MACD Momentum Crossover
# ─────────────────────────────────────────────────────────────────────────────
def strategy_macd(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    LONG on MACD histogram flipping positive (line crosses above signal).
    EXIT on histogram flipping negative OR ATR-based hard stop.
    """
    d = df.copy()
    d["MACD"], d["MACD_S"] = macd(d["Close"], params["macd_fast"],
                                   params["macd_slow"], params["macd_signal"])
    d["MACD_H"] = d["MACD"] - d["MACD_S"]
    d["ATR"]    = atr(d, 14)
    d["signal"] = 0
    in_trade    = False
    entry_price = 0.0

    for i in range(1, len(d)):
        prev_h = d["MACD_H"].iloc[i - 1]
        curr_h = d["MACD_H"].iloc[i]
        c      = d["Close"].iloc[i]
        if not in_trade:
            if prev_h < 0 and curr_h > 0:
                d.iloc[i, d.columns.get_loc("signal")] = 1
                in_trade    = True
                entry_price = c
        else:
            stop = entry_price - params["atr_filter"] * d["ATR"].iloc[i]
            if (prev_h > 0 and curr_h < 0) or c < stop:
                d.iloc[i, d.columns.get_loc("signal")] = -1
                in_trade = False
    return d


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 3 — Golden / Death Cross  (50 / 200 EMA)
# ─────────────────────────────────────────────────────────────────────────────
def strategy_ema_cross(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    LONG on 50 EMA crossing above 200 EMA (Golden Cross).
    Optional RSI > 50 confirmation filter.
    EXIT on 50 EMA crossing below 200 EMA (Death Cross).
    """
    d = df.copy()
    d["EMA_F"]  = ema(d["Close"], params["ema_fast"])
    d["EMA_S"]  = ema(d["Close"], params["ema_slow"])
    d["RSI14"]  = rsi(d["Close"], 14)
    d["signal"] = 0
    in_trade    = False

    for i in range(1, len(d)):
        prev_diff = d["EMA_F"].iloc[i - 1] - d["EMA_S"].iloc[i - 1]
        curr_diff = d["EMA_F"].iloc[i]     - d["EMA_S"].iloc[i]
        rsi_ok    = (d["RSI14"].iloc[i] > 50) if params.get("rsi_confirm") else True
        curr_EMA_F = d["EMA_F"].iloc[i] 

        if not in_trade:
            if (prev_diff < 0 and curr_diff > 0 and rsi_ok):
                d.iloc[i, d.columns.get_loc("signal")] = 1
                in_trade = True
        else:
            if (prev_diff > 0 and curr_diff) < 0 or (d["Close"].iloc[i] < curr_EMA_F):
                d.iloc[i, d.columns.get_loc("signal")] = -1
                in_trade = False
    return d



# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 4 — CUSTOM  (Donchian + Multi-EMA Stack + Candle Breakout)
# ─────────────────────────────────────────────────────────────────────────────
def strategy_custom(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    ┌────────────────────────────────────────────────────────────┐
    │  CUSTOM STRATEGY — intended for you to modify and extend   │
    │                                                            │
    │  DEFAULT LOGIC:                                            │
    │  Entry conditions (ALL must be true):                      │
    │    1. Donchian breakout — close > prior N-bar upper band   │
    │    2. EMA stack aligned — ema_fast > ema_mid > ema_slow             │
    │    3. Strong candle     — body size > ATR × body_factor    │
    │                                                            │
    │  Exit conditions (ANY triggers exit):                      │
    │    1. Close below Donchian midline                         │
    │    2. Close below ema_mid                                    │
    │    3. ATR trailing stop from swing high                    │
    └────────────────────────────────────────────────────────────┘
    """
    d = df.copy()
    n  = params["donchian_period"]

    d["DON_U"], d["DON_L"], d["DON_M"] = donchian(d["High"], d["Low"], n)
    d["ema_fast"]   = ema(d["Close"], params["ema_fast"])
    d["ema_mid"]  = ema(d["Close"], params["ema_mid"])
    d["ema_slow"]  = ema(d["Close"], params["ema_slow"])
    d["ATR14"]  = atr(d, 14)
    d["body"]   = (d["Close"] - d["Open"]).abs()
    d["signal"] = 0

    in_trade          = False
    high_since_entry  = 0.0

    for i in range(1, len(d)):
        prev_close = d["Close"].iloc[i - 1]
        prev_open  = d["Open"].iloc[i - 1]
        curr_close = d["Close"].iloc[i]
        curr_open = d["Open"].iloc[i]
        don_u_prev = d["DON_U"].iloc[i - 1]   # shifted to avoid lookahead
        ema_fast       = d["ema_fast"].iloc[i]
        ema_mid      = d["ema_mid"].iloc[i]
        ema_slow      = d["ema_slow"].iloc[i]
        curr_atr   = d["ATR14"].iloc[i]
        curr_body  = d["body"].iloc[i]
        is_bull    = curr_close > d["Open"].iloc[i]
        curr_low = d["Low"].iloc[i]
        
        ema_aligned   = ema_fast > ema_mid > ema_slow  # overwritten
        ema_aligned   = ema_fast > ema_mid 
        strong_candle = is_bull and (curr_body > params["body_factor"] * curr_atr)  # overwritten
        strong_candle = is_bull and d["Open"].iloc[i] < d["ema_fast"].iloc[i] and d["Close"].iloc[i] > d["ema_fast"].iloc[i] 
        don_breakout  = (prev_close < don_u_prev) and (curr_close > don_u_prev)
        gap_up        = (curr_close > prev_open) and (curr_close > prev_close) and (curr_open > prev_open) and (curr_open > prev_close) and (curr_close > curr_open) 
        
        if not in_trade:
            if ema_aligned and (strong_candle or gap_up) and curr_low > ema_slow and curr_open> ema_fast: # and don_breakout: 
                d.iloc[i, d.columns.get_loc("signal")] = 1
                in_trade         = True
                high_since_entry = curr_close
        else:
            high_since_entry = max(high_since_entry, curr_close)
            trail_stop       = high_since_entry - params["atr_trail"] * curr_atr
            don_mid          = d["DON_M"].iloc[i]

            if curr_close < ema_fast or curr_close < trail_stop or curr_close < ema_mid or curr_close < don_mid:
                d.iloc[i, d.columns.get_loc("signal")] = -1
                in_trade = False
    return d


# ─────────────────────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame,
                 initial_capital: float = 10_000.0,
                 position_size_pct: float = 0.95,
                 commission_pct: float = 0.001) -> dict:
    """
    Event-driven backtest.  signal=+1 → buy, signal=-1 → sell/close.
    Returns equity curve, trade log, and summary stats.
    """
    if "signal" not in df.columns or df.empty:
        return {}

    capital     = initial_capital
    shares      = 0.0
    entry_price = 0.0
    entry_date  = None
    trades      = []
    equity      = []

    for i, (idx, row) in enumerate(df.iterrows()):
        sig   = row["signal"]
        price = row["Close"]
        exec_price = df["Open"].iloc[i + 1] if i + 1 < len(df) else df["Close"].iloc[i]
        if sig == 1 and shares == 0:
            invest     = capital * position_size_pct
            commission = invest * commission_pct
            shares     = (invest - commission) / exec_price
            entry_price = exec_price
            entry_date  = idx
            capital    -= invest

        elif sig == -1 and shares > 0:
            gross      = shares * price
            commission = gross * commission_pct
            net        = gross - commission
            pnl        = net - (shares * entry_price)
            pnl_pct    = pnl / (shares * entry_price) * 100
            capital   += net
            trades.append({
                "Entry Date":      entry_date,
                "Exit Date":       idx,
                "Entry Price":     round(entry_price, 4),
                "Exit Price":      round(price, 4),
                "Shares":          round(shares, 4),
                "P&L ($)":         round(pnl, 2),
                "P&L (%)":         round(pnl_pct, 2),
                "Duration (days)": (idx - entry_date).days,
            })
            shares = 0.0

        equity.append({"Date": idx,
                        "Equity": capital + (shares * price if shares > 0 else 0)})

    # Close any open position at last price
    if shares > 0:
        price      = df["Close"].iloc[-1]
        gross      = shares * price
        commission = gross * commission_pct
        net        = gross - commission
        pnl        = net - (shares * entry_price)
        pnl_pct    = pnl / (shares * entry_price) * 100
        capital   += net
        trades.append({
            "Entry Date":      entry_date,
            "Exit Date":       df.index[-1],
            "Entry Price":     round(entry_price, 4),
            "Exit Price":      round(price, 4),
            "Shares":          round(shares, 4),
            "P&L ($)":         round(pnl, 2),
            "P&L (%)":         round(pnl_pct, 2),
            "Duration (days)": (df.index[-1] - entry_date).days,
        })

    equity_df = pd.DataFrame(equity).set_index("Date")
    trade_df  = pd.DataFrame(trades) if trades else pd.DataFrame()

    # ── Performance Stats ──────────────────────────────────────────────────
    final_equity = equity_df["Equity"].iloc[-1] if not equity_df.empty else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital * 100
    bh_return    = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

    wins   = trade_df[trade_df["P&L ($)"] > 0]  if not trade_df.empty else pd.DataFrame()
    losses = trade_df[trade_df["P&L ($)"] <= 0] if not trade_df.empty else pd.DataFrame()
    win_rate      = len(wins) / max(len(trade_df), 1) * 100
    avg_win       = wins["P&L (%)"].mean()  if not wins.empty  else 0.0
    avg_loss      = losses["P&L (%)"].mean() if not losses.empty else 0.0
    profit_factor = (wins["P&L ($)"].sum() / max(abs(losses["P&L ($)"].sum()), 1e-9)
                     if not trade_df.empty else 0.0)

    eq_ret   = equity_df["Equity"].pct_change().dropna()
    sharpe   = (eq_ret.mean() / max(eq_ret.std(), 1e-9)) * np.sqrt(252) if len(eq_ret) > 1 else 0.0
    roll_max = equity_df["Equity"].cummax()
    max_dd   = ((equity_df["Equity"] - roll_max) / roll_max * 100).min()

    return {
        "equity_curve": equity_df,
        "trade_log":    trade_df,
        "stats": {
            "Total Return (%)":  round(total_return, 2),
            "Buy & Hold (%)":    round(bh_return, 2),
            "Final Equity ($)":  round(final_equity, 2),
            "Total Trades":      len(trade_df),
            "Win Rate (%)":      round(win_rate, 2),
            "Avg Win (%)":       round(avg_win, 2),
            "Avg Loss (%)":      round(avg_loss, 2),
            "Profit Factor":     round(profit_factor, 2),
            "Sharpe Ratio":      round(sharpe, 2),
            "Max Drawdown (%)":  round(max_dd, 2),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────────────────────
BG    = "#0d0f14"
GRID  = "#1a1f28"
MUTED = "#8b949e"

def chart_price(df: pd.DataFrame, result: dict, title: str,
                ml_preds: pd.DataFrame = None) -> go.Figure:
    trade_df = result.get("trade_log", pd.DataFrame())

    # must be defined before make_subplots which uses it for row count
    has_ml = ml_preds is not None and not ml_preds.empty

    fig = make_subplots(
        rows=4 if has_ml else 3, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.50, 0.17, 0.18, 0.15] if has_ml else [0.55, 0.2, 0.25],
    )

    # ── Build per-candle hover text incorporating ML predictions ─────
    has_ml = ml_preds is not None and not ml_preds.empty
    custom_hover = []
    for idx, row in df.iterrows():
        lines = [
            f"<b>{idx.strftime('%Y-%m-%d')}</b>",
            f"O: {row['Open']:.2f}  H: {row['High']:.2f}  "
            f"L: {row['Low']:.2f}  C: {row['Close']:.2f}",
        ]
        if has_ml and idx in ml_preds.index:
            mp = ml_preds.loc[idx]
            label   = mp["pred_label"]
            colour  = PRED_COLOURS.get(label, "#8b949e")
            correct = "✓" if mp["pred_correct"] == 1 else "✗"
            lines += [
                "─────────────────",
                f"<b>ML Prediction: "
                f"<span style='color:{colour}'>{label}</span></b>  {correct}",
                f"  🔴 BEARISH  {mp['pred_BEARISH']*100:5.1f}%",
                f"  🟡 SIDEWAYS {mp['pred_SIDEWAYS']*100:5.1f}%",
                f"  🟢 BULLISH  {mp['pred_BULLISH']*100:5.1f}%",
                f"  Actual next-day: {mp['actual_dir']}",
            ]
        custom_hover.append("<br>".join(lines))

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        increasing_fillcolor="#1a3a1a",  decreasing_fillcolor="#3a1a1a",
        name="Price", line_width=1,
        text=custom_hover,
        hoverinfo="text",
    ), row=1, col=1)

    # ── ML confidence band overlay (faint background tint per candle) ─
    if has_ml:
        # Scatter invisible markers that carry the prediction colour as a
        # faint rectangle — achieved via shape-less markers at candle midpoint
        mid_price = (df["High"] + df["Low"]) / 2
        ml_colours_rgba = []
        for idx in df.index:
            if idx in ml_preds.index:
                lbl = ml_preds.loc[idx, "pred_label"]
                if lbl == "BULLISH":
                    ml_colours_rgba.append("rgba(63,185,80,0.13)")
                elif lbl == "BEARISH":
                    ml_colours_rgba.append("rgba(248,81,73,0.13)")
                else:
                    ml_colours_rgba.append("rgba(210,153,34,0.10)")
            else:
                ml_colours_rgba.append("rgba(0,0,0,0)")

        fig.add_trace(go.Bar(
            x=df.index,
            y=df["High"] - df["Low"],
            base=df["Low"],
            marker_color=ml_colours_rgba,
            marker_line_width=0,
            name="ML Prediction",
            showlegend=True,
            hoverinfo="skip",
        ), row=1, col=1)

    # Indicator overlays
    overlay_cfg = {
        "ema_fast":   ("#58a6ff", 1),  
        "ema_mid":  ("#d2a8ff", 1),  
        "ema_slow":  ("#ffa657", 1),
        "EMA_F":  ("#58a6ff", 1.5),
        "EMA_S":  ("#ffa657", 1.5),
        "BB_M":   ("#388bfd", 1),  
        # FIXED: Changed 8-digit hex to standard hex or rgba
        "DON_U":  ("#3fb950", 1), 
        "DON_L":  ("#f85149", 1),
        "DON_M":  ("#d29922", 1),
    }
    for col, (colour, w) in overlay_cfg.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                     line=dict(color=colour, width=w), opacity=.85), row=1, col=1)

    if "BB_L" in df.columns and "BB_U" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_U"], fill=None,
                                line=dict(color="#388bfd", width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_L"], fill="tonexty",
                                fillcolor="rgba(56, 139, 253, 0.07)", line=dict(color="#388bfd", width=0),
                                showlegend=False), row=1, col=1)

    # Trade markers
    if not trade_df.empty:
        entries = df.loc[df.index.isin(trade_df["Entry Date"]), "Close"]
        exits   = df.loc[df.index.isin(trade_df["Exit Date"]),  "Close"]
        fig.add_trace(go.Scatter(x=entries.index, y=entries.values * .988,
                                 mode="markers", name="Entry",
                                 marker=dict(symbol="triangle-up", size=11, color="#3fb950",
                                             line=dict(color="#fff", width=.8))), row=1, col=1)
        fig.add_trace(go.Scatter(x=exits.index,   y=exits.values   * 1.012,
                                 mode="markers", name="Exit",
                                 marker=dict(symbol="triangle-down", size=11, color="#f85149",
                                             line=dict(color="#fff", width=.8))), row=1, col=1)

    # Volume
    vol_colors = ["#3fb950" if c >= o else "#f85149"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                         marker_color=vol_colors, opacity=.55, name="Volume"), row=2, col=1)

    # Sub-oscillator
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
                                 line=dict(color="#d2a8ff", width=1.2), name="RSI"), row=3, col=1)
        for level, col in [(30, "#f85149"), (70, "#3fb950"), (50, "#8b949e")]:
            fig.add_hline(y=level, line_dash="dot", line_color=col, opacity=.4, row=3, col=1)
    elif "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
                                 line=dict(color="#58a6ff", width=1.2), name="MACD"),   row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_S"],
                                 line=dict(color="#ffa657", width=1.2), name="Signal"), row=3, col=1)
        h_colors = ["#3fb950" if v >= 0 else "#f85149" for v in df["MACD_H"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_H"],
                             marker_color=h_colors, opacity=.7, name="Hist"), row=3, col=1)
    elif "RSI14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"],
                                 line=dict(color="#d2a8ff", width=1.2), name="RSI"), row=3, col=1)
    elif "ATR14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ATR14"],
                                 line=dict(color="#d2a8ff", width=1.2), name="ATR"), row=3, col=1)

    # ── Row 4: ML confidence stacked bar (Bearish / Sideways / Bullish) ──
    if has_ml:
        aligned_preds = ml_preds.reindex(df.index)
        fig.add_trace(go.Bar(
            x=df.index,
            y=aligned_preds["pred_BEARISH"],
            name="P(Bearish)",
            marker_color="#f85149",
            opacity=0.85,
        ), row=4, col=1)
        fig.add_trace(go.Bar(
            x=df.index,
            y=aligned_preds["pred_SIDEWAYS"],
            name="P(Sideways)",
            marker_color="#d29922",
            opacity=0.85,
        ), row=4, col=1)
        fig.add_trace(go.Bar(
            x=df.index,
            y=aligned_preds["pred_BULLISH"],
            name="P(Bullish)",
            marker_color="#3fb950",
            opacity=0.85,
        ), row=4, col=1)
        fig.update_layout(barmode="stack")
        fig.add_hline(y=0.5, line_dash="dot", line_color="#484f58",
                      opacity=0.6, row=4, col=1)
        fig.update_yaxes(title_text="P(class)", title_font=dict(size=9, color=MUTED),
                         range=[0, 1], row=4, col=1)

    n_rows = 4 if has_ml else 3
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(family="IBM Plex Mono", size=13, color="#58a6ff")),
        paper_bgcolor=BG, plot_bgcolor=BG,
        legend=dict(bgcolor=BG, font=dict(color=MUTED, size=10),
                    orientation="h", y=1.02),
        xaxis_rangeslider_visible=False,
        dragmode="pan",
        height=820 if has_ml else 700,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    for r in range(1, n_rows + 1):
        fig.update_xaxes(row=r, col=1, gridcolor=GRID, zerolinecolor=GRID,
                         tickfont=dict(color=MUTED, size=9))
        fig.update_yaxes(row=r, col=1, gridcolor=GRID, zerolinecolor=GRID,
                         tickfont=dict(color=MUTED, size=9))
    return fig


def chart_equity(equity_df: pd.DataFrame, capital: float, price_df: pd.DataFrame) -> go.Figure:
    bh  = capital * (price_df["Close"] / price_df["Close"].iloc[0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_df.index, y=bh, name="Buy & Hold",
                             line=dict(color="#8b949e", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df["Equity"], name="Strategy",
                             line=dict(color="#58a6ff", width=2),
                             fill="tozeroy", fillcolor="rgba(88,166,255,0.07)"))
    fig.add_hline(y=capital, line_dash="dot", line_color="#30363d", opacity=.5)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                      dragmode="pan",
                      legend=dict(bgcolor=BG, font=dict(color=MUTED)),
                      xaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED)),
                      yaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED),
                                 title="Portfolio ($)", fixedrange=False),
                      height=320, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def chart_monthly(equity_df: pd.DataFrame) -> go.Figure:
    monthly  = equity_df["Equity"].resample("ME").last().pct_change().dropna() * 100
    m_colors = ["#3fb950" if v >= 0 else "#f85149" for v in monthly]
    fig = go.Figure(go.Bar(x=monthly.index.strftime("%b %Y"),
                           y=monthly.values, marker_color=m_colors, opacity=.85))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                      xaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED, size=9), tickangle=-45),
                      yaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED), title="Monthly Return (%)"),
                      height=300, margin=dict(l=10, r=10, t=10, b=60))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
STRATEGIES = {
    "1 — RSI Mean Reversion + Bollinger Bands":          "rsi_bb",
    "2 — MACD Momentum Crossover":                       "macd",
    "3 — Golden / Death Cross (50/200 EMA)":             "ema_cross",
    "4 — ★ Custom: Donchian + Multi-EMA + Breakout":     "custom",
}

with st.sidebar:
    st.markdown("## ⚙ Configuration")
    st.markdown("---")

    st.markdown("**Strategy**")
    strat_label = st.selectbox("", list(STRATEGIES.keys()), label_visibility="collapsed")
    strat_key   = STRATEGIES[strat_label]

    st.markdown("---")
    st.markdown("**Instrument**")
    ticker    = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
    ca, cb    = st.columns(2)
    with ca: years_back = st.selectbox("History (yrs)", [1, 2, 3, 5, 10], index=2)
    with cb: interval   = st.selectbox("Interval",      ["1d", "1wk"], index=0)

    start_date = (datetime.today() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
    end_date   = datetime.today().strftime("%Y-%m-%d")

    st.markdown("---")
    st.markdown("**Data Source**")
    data_source = st.radio("", ["yfinance (recommended)", "FMP (API key required)"],
                           label_visibility="collapsed")
    fmp_key = ""
    if "FMP" in data_source:
        fmp_key = st.text_input("FMP API Key", type="password",
                                help="Get a free key at financialmodelingprep.com")

    st.markdown("---")
    st.markdown("**Backtest Settings**")
    initial_capital = st.number_input("Initial Capital ($)", value=10_000, step=1_000, min_value=1_000)
    position_size   = st.slider("Position Size (%)", 10, 100, 95, step=5) / 100
    commission      = st.slider("Commission (bps)",  0,  50, 10, step=5)  / 10_000

    st.markdown("---")
    st.markdown("**Strategy Parameters**")
    params = {}

    if strat_key == "rsi_bb":
        params["rsi_period"]   = st.slider("RSI Period",       7, 30, 14)
        params["bb_period"]    = st.slider("Bollinger Period", 10, 50, 20)
        params["bb_std"]       = st.slider("BB Std Dev",       1.0, 3.0, 2.0, 0.1)
        params["rsi_oversold"] = st.slider("RSI Oversold",     15, 40, 30)
        params["rsi_midline"]  = st.slider("RSI Exit Level",   45, 70, 55)

    elif strat_key == "macd":
        params["macd_fast"]   = st.slider("MACD Fast",   5, 20, 12)
        params["macd_slow"]   = st.slider("MACD Slow",  15, 50, 26)
        params["macd_signal"] = st.slider("Signal Line",  5, 20,  9)
        params["atr_filter"]  = st.slider("ATR Stop Multiplier", 0.5, 3.0, 0.5, 0.25)

    elif strat_key == "ema_cross":
        params["ema_fast"]    = st.slider("Fast EMA",  10, 100,  50, 5)
        params["ema_slow"]    = st.slider("Slow EMA",  50, 300, 200, 10)
        params["rsi_confirm"] = st.checkbox("RSI > 50 Confirmation", value=True)

    elif strat_key == "custom":
        st.markdown(
            '<div class="custom-banner"><b>★ CUSTOM STRATEGY</b><br/>'
            'Donchian + EMA Stack + Candle Breakout<br/>'
            'Edit <code>strategy_custom()</code> in <code>app.py</code></div>',
            unsafe_allow_html=True,
        )
        params["donchian_period"] = st.slider("Donchian Period",   10,  55, 20)
        params["ema_fast"]        = st.slider("EMA Fast (stack)",   5,  20,  8)
        params["ema_mid"]         = st.slider("EMA Mid (stack)",   15,  50, 21)
        params["ema_slow"]        = st.slider("EMA Slow (stack)",  34, 200, 55)
        params["body_factor"]     = st.slider("Body/ATR Filter",  0.2, 1.5, 0.6, 0.05)
        params["atr_trail"]       = st.slider("ATR Trailing Stop",0.5, 5.0, 2.0, 0.25)

    st.markdown("---")
    run_btn = st.button("▶  RUN BACKTEST", use_container_width=True)

    st.markdown("---")
    st.markdown("**🤖 ML Model**")
    use_ml = st.toggle("Enable XGBoost Predictions", value=True,
                       help="Loads xgb_stock_model.json + model_artifacts.joblib "
                            "from the working directory")
    if use_ml:
        st.caption(f"Model: `{XGB_MODEL_PATH}`")
        st.caption(f"Artifacts: `{XGB_ARTIFACTS_PATH}`")


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 📈 Swing Trade Backtester")
st.markdown(
    '<span class="tag tag-blue">Python · Streamlit</span>'
    '<span class="tag tag-green">yfinance</span>'
    '<span class="tag tag-amber">FMP optional</span>'
    '<span class="tag">4 Strategies</span>',
    unsafe_allow_html=True,
)
st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY DESCRIPTION TABS
# ─────────────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["RSI + Bollinger", "MACD Crossover", "EMA Cross", "★ Custom"])

with t1:
    st.markdown("""
**Mean Reversion — RSI + Bollinger Bands**
Profits from short-term overextension snapping back to the mean.
- 🟢 **Entry**: RSI below oversold threshold *AND* price at or below the lower Bollinger Band
- 🔴 **Exit**: RSI crosses above midline *OR* price reaches the upper Bollinger Band
- **Best on**: Range-bound markets, large-cap equities. Struggles in strong trending conditions.
""")

with t2:
    st.markdown("""
**Momentum — MACD Crossover with ATR Stop**
Captures directional momentum; protects capital with volatility-scaled stops.
- 🟢 **Entry**: MACD line crosses above signal line (histogram flips positive)
- 🔴 **Exit**: MACD crosses back below signal *OR* ATR hard stop triggered
- **Best on**: Trending equities / sector ETFs. Frequent signals; watch for whipsaws in choppy regimes.
""")

with t3:
    st.markdown("""
**Trend Following — Golden / Death Cross (50/200 EMA)**
Institutional-grade long-term trend confirmation.
- 🟢 **Entry**: 50 EMA crosses above 200 EMA (Golden Cross) + optional RSI > 50 filter
- 🔴 **Exit**: 50 EMA crosses below 200 EMA (Death Cross)
- **Best on**: Indices, sector ETFs, blue-chip equities with 3+ years of data. Low trade frequency, high conviction.
""")

with t4:
    st.markdown("""
**★ Custom — Donchian Channel + Multi-EMA Stack + Candle Breakout**
A breakout-momentum hybrid. - NOTED DONCHIAN AND ATR DISABLED FOR TESTING
- 🟢 **Entry**: PRICE > EMA_FAST AND IF GAP UP ABOVE EMA FAST
- 🔴 **Exit**: PRICE BREAKDOWN THROUGH EMA_FAST
- **Customise**: Edit `strategy_custom()` in `app.py`. All parameters are tunable in the sidebar.
""")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  RUN BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not ticker:
        st.warning("Please enter a ticker symbol.")
        st.stop()

    with st.spinner(f"Fetching {ticker} ({years_back}y, {interval})…"):
        if "FMP" in data_source:
            if not fmp_key:
                st.error("Enter your FMP API key in the sidebar.")
                st.stop()
            df_raw = fetch_fmp(ticker, start_date, end_date, fmp_key)
        else:
            df_raw = fetch_yfinance(ticker, start_date, end_date, interval)

    if df_raw.empty:
        st.error(f"No data returned for **{ticker}**. Verify the ticker and try again.")
        st.stop()

    bar_count = len(df_raw)
    st.success(
        f"✓ {bar_count:,} bars loaded for **{ticker}** "
        f"({df_raw.index[0].date()} → {df_raw.index[-1].date()})"
    )

    if strat_key == "ema_cross" and bar_count < 300:
        st.warning("⚠ The 50/200 EMA strategy needs ~300+ bars. Consider extending the history window.")

    with st.spinner("Running backtest…"):
        fn_map = {
            "rsi_bb":    strategy_rsi_bollinger,
            "macd":      strategy_macd,
            "ema_cross": strategy_ema_cross,
            "custom":    strategy_custom,
        }
        df_signals = fn_map[strat_key](df_raw.copy(), params)
        result     = run_backtest(df_signals, initial_capital, position_size, commission)

    if not result:
        st.error("Backtest produced no results. Check data length and parameters.")
        st.stop()

    stats = result["stats"]

    # ── ML Predictions ───────────────────────────────────────────────────────
    ml_preds    = pd.DataFrame()
    ml_loaded   = False
    ml_acc_all  = None    # overall accuracy across all candles
    ml_acc_bull = None    # accuracy on BULLISH predictions only
    ml_acc_bear = None    # accuracy on BEARISH predictions only

    if use_ml:
        xgb_model, artifacts = load_xgb_model()
        if xgb_model is not None:
            with st.spinner("Running ML predictions…"):
                spy_close = _fetch_spy(years_back + 1)
                ml_preds  = run_ml_predictions(df_raw, xgb_model, artifacts, spy_close)
            ml_loaded = True

            # ── Accuracy statistics ──────────────────────────────────────
            valid = ml_preds.dropna(subset=["actual_dir", "pred_label"])
            # Exclude last row — actual_dir requires a future candle
            valid = valid.iloc[:-1]
            total_candles = len(valid)
            if total_candles > 0:
                ml_acc_all  = valid["pred_correct"].mean() * 100
                bull_mask   = valid["pred_label"] == "BULLISH"
                bear_mask   = valid["pred_label"] == "BEARISH"
                side_mask   = valid["pred_label"] == "SIDEWAYS"
                ml_acc_bull = valid.loc[bull_mask, "pred_correct"].mean() * 100 if bull_mask.any() else 0.0
                ml_acc_bear = valid.loc[bear_mask, "pred_correct"].mean() * 100 if bear_mask.any() else 0.0
                ml_acc_side = valid.loc[side_mask, "pred_correct"].mean() * 100 if side_mask.any() else 0.0
                n_bull = bull_mask.sum(); n_bear = bear_mask.sum(); n_side = side_mask.sum()
        else:
            if use_ml:
                st.warning(
                    f"⚠ XGBoost model files not found in working directory.  "
                    f"Expected: `{XGB_MODEL_PATH}` and `{XGB_ARTIFACTS_PATH}`"
                )

    # ── KPI Row 1 ───────────────────────────────────────────────────────────
    st.markdown("### Performance Summary")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Return",   f"{stats['Total Return (%)']:+.2f}%",
              delta=f"vs B&H {stats['Buy & Hold (%)']:+.2f}%")
    k2.metric("Final Equity",   f"${stats['Final Equity ($)']:,.0f}")
    k3.metric("Total Trades",   str(stats["Total Trades"]))
    k4.metric("Win Rate",       f"{stats['Win Rate (%)']:.1f}%")
    k5.metric("Sharpe Ratio",   f"{stats['Sharpe Ratio']:.2f}")
    k6.metric("Max Drawdown",   f"{stats['Max Drawdown (%)']:.2f}%")

    st.markdown("")

    # ── KPI Row 2 ───────────────────────────────────────────────────────────
    ka, kb, kc, kd = st.columns(4)
    ka.metric("Profit Factor",  f"{stats['Profit Factor']:.2f}")
    kb.metric("Avg Win",        f"{stats['Avg Win (%)']:.2f}%")
    kc.metric("Avg Loss",       f"{stats['Avg Loss (%)']:.2f}%")
    kd.metric("Buy & Hold",     f"{stats['Buy & Hold (%)']:+.2f}%")

    # ── KPI Row 3: ML Accuracy ───────────────────────────────────────────────
    if ml_loaded and ml_acc_all is not None:
        st.markdown("")
        st.markdown("#### 🤖 ML Prediction Accuracy  *(XGBoost — next-session direction)*")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Overall Accuracy",
            f"{ml_acc_all:.1f}%",
            delta=f"{total_candles:,} candles assessed",
            help="% of candles where predicted direction matched actual next-day direction"
        )
        m2.metric(
            f"Bullish Calls  ({n_bull})",
            f"{ml_acc_bull:.1f}%",
            help="Accuracy when model predicted BULLISH"
        )
        m3.metric(
            f"Bearish Calls  ({n_bear})",
            f"{ml_acc_bear:.1f}%",
            help="Accuracy when model predicted BEARISH"
        )
        m4.metric(
            f"Sideways Calls ({n_side})",
            f"{ml_acc_side:.1f}%",
            help="Accuracy when model predicted SIDEWAYS"
        )

        # Per-class breakdown bar
        class_acc_df = pd.DataFrame({
            "Class": ["BULLISH", "SIDEWAYS", "BEARISH"],
            "Accuracy (%)": [ml_acc_bull, ml_acc_side, ml_acc_bear],
            "# Predictions": [n_bull, n_side, n_bear],
        })
        with st.expander("📊 Per-class breakdown", expanded=False):
            st.dataframe(class_acc_df.style.format({
                "Accuracy (%)": "{:.1f}",
                "# Predictions": "{:,}",
            }), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Chart Tabs ──────────────────────────────────────────────────────────
    ct1, ct2, ct3, ct4 = st.tabs(
        ["📊 Price & Signals", "📈 Equity Curve", "🗓 Monthly Returns", "📋 Trade Log"]
    )

    PLOTLY_CFG = dict(scrollZoom=True, displaylogo=False,
                      modeBarButtonsToRemove=["select2d", "lasso2d"])

    with ct1:
        st.plotly_chart(
            chart_price(df_signals, result, strat_label,
                        ml_preds=ml_preds if ml_loaded else None),
            use_container_width=True, config=PLOTLY_CFG,
        )
        if ml_loaded:
            st.caption(
                "🟢 Green tint = ML predicted BULLISH · "
                "🔴 Red tint = BEARISH · 🟡 Amber tint = SIDEWAYS  |  "
                "Hover a candle for full probability breakdown."
            )

    with ct2:
        st.plotly_chart(chart_equity(result["equity_curve"], initial_capital, df_raw),
                        use_container_width=True, config=PLOTLY_CFG)

    with ct3:
        if len(result["equity_curve"]) > 30:
            st.plotly_chart(chart_monthly(result["equity_curve"]),
                            use_container_width=True, config=PLOTLY_CFG)
        else:
            st.info("Not enough data for a monthly breakdown. Use a longer time window.")

    with ct4:
        tdf = result["trade_log"]
        if not tdf.empty:
            def _style_pnl(val):
                return f"color: {'#3fb950' if val > 0 else '#f85149'}"
            styled = tdf.style.map(_style_pnl, subset=["P&L ($)", "P&L (%)"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇  Download Trade Log (CSV)",
                tdf.to_csv(index=False).encode(),
                f"{ticker}_{strat_key}_trades.csv",
                "text/csv",
            )
        else:
            st.info("No completed trades. Extend the backtest window or adjust parameters.")

else:
    # ── Landing placeholder ─────────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center; padding:60px 20px; color:#8b949e;">
  <div style="font-size:3rem; margin-bottom:16px;">📊</div>
  <div style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem; color:#58a6ff; margin-bottom:8px;">
    Configure your backtest in the sidebar
  </div>
  <div style="font-size:.85rem;">
    Pick a strategy → enter a ticker → click <strong>RUN BACKTEST</strong>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#484f58; font-size:.7rem; font-family:IBM Plex Mono,monospace;">'
    'Past performance is not indicative of future results. For educational use only.<br/>'
    'Data: Yahoo Finance (yfinance) · Financial Modelling Prep (optional)'
    '</div>',
    unsafe_allow_html=True,
)