# 📈 Swing Trade Backtester

A Streamlit app for backtesting swing trading strategies with clean charting,
performance metrics, and a fully customisable strategy slot.

---

## Quick Start

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Strategies

| # | Name | Style | Key Indicators |
|---|------|-------|----------------|
| 1 | RSI Mean Reversion + Bollinger | Mean reversion | RSI, Bollinger Bands |
| 2 | MACD Momentum Crossover | Momentum | MACD, ATR stop |
| 3 | Golden / Death Cross | Trend following | 50/200 EMA |
| 4 ★ | Custom: Donchian + Multi-EMA + Breakout | Breakout-momentum | Donchian, EMA8/21/55, ATR |

---

## Data Sources

| Source | Setup | Best For |
|--------|-------|----------|
| **yfinance** (default) | Zero setup, no key needed | Development, backtesting — unlimited pulls |
| **FMP** (optional) | Requires free/paid API key at financialmodelingprep.com | Production, fundamental data |

> **Why yfinance over FMP for backtesting?**  
> FMP's free tier allows only 250 API calls/day, and each historical fetch counts.
> A single backtest scan over multiple tickers will hit that limit quickly.
> yfinance is uncapped, has 20+ years of history, and requires no key.
> Use FMP if you need fundamental data alongside price data or need a paid SLA.

---

## Customising Strategy 4

Open `app.py` and find the `strategy_custom()` function (~line 215).
The entry/exit logic is clearly commented — replace or extend it:

```python
def strategy_custom(df, params):
    # --- YOUR LOGIC HERE ---
    # Use params dict for sidebar-tunable values
    # Set df["signal"] = 1 for entries, -1 for exits
    ...
```

Sidebar parameters for the custom strategy are in the `elif strat_key == "custom":` 
block in the sidebar section. Add `st.slider()` / `st.checkbox()` calls there and
reference them via `params["your_key"]` in the strategy function.

---

## Performance Metrics Explained

| Metric | Description |
|--------|-------------|
| Total Return | Strategy PnL vs initial capital |
| Buy & Hold | Passive benchmark (buy first bar, hold to last) |
| Win Rate | % of trades that closed profitable |
| Profit Factor | Gross wins ÷ gross losses (>1.5 = good) |
| Sharpe Ratio | Risk-adjusted return, annualised (>1.0 = acceptable) |
| Max Drawdown | Largest peak-to-trough equity decline |
| Avg Win / Avg Loss | Average % gain on winners vs losers |

---

## Notes

- **Commissions**: Set in basis points (bps). 10 bps = 0.10% per trade side.
- **Position sizing**: % of capital deployed per trade. 95% leaves a cash buffer.
- **No shorting**: All strategies are long-only in this version.
- **No lookahead bias**: Entry signals use prior bar's indicator values where relevant (e.g. Donchian breakout uses `DON_U.shift(1)`).

---

*Past performance is not indicative of future results. For educational use only.*
"# Stock_Backtester_Claude" 
