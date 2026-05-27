import streamlit as st
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Live Portfolio", page_icon="💼", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"]      { font-family:'IBM Plex Sans',sans-serif; background:#0d0f14; color:#c9d1d9; }
.stApp                          { background:#0d0f14; }
h1, h2, h3 { font-family:'IBM Plex Mono',monospace !important; color:#58a6ff !important; }

[data-testid="metric-container"] {
  background:#161b22; border:1px solid #21262d; border-radius:8px; padding:12px 16px;
}
[data-testid="metric-container"] label {
  color:#8b949e !important; font-size:0.72rem !important; letter-spacing:.08em; text-transform:uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family:'IBM Plex Mono',monospace; font-size:1.4rem !important; color:#e6edf3 !important;
}
.stButton > button {
  background:#238636; color:#fff; border:none; border-radius:6px;
  font-family:'IBM Plex Mono',monospace; font-weight:600; width:100%;
}
.stButton > button:hover { background:#2ea043; }
.stDataFrame { background:#161b22; border:1px solid #21262d; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("💼 Live Portfolio Dashboard")
st.markdown("### Read-Only Account Aggregator")
st.info("🔒 API connections execute locally. Ensure your API keys and parameters are configured for read-only access.")

# Create Tabs for the two brokers
tab_ibkr, tab_webull = st.tabs(["Interactive Brokers (IBKR)", "Webull"])

# ═════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE BROKERS (IBKR)
# ═════════════════════════════════════════════════════════════════════════════
with tab_ibkr:
    st.subheader("IBKR Account Overview")
    
    # Connection Config
    with st.expander("🔌 IBKR Connection Configuration", expanded=False):
        st.markdown("""
        **Note:** Requires an active **IB Gateway** or **TWS** instance running locally. 
        Ensure your API settings in TWS have *'Read-Only API'* checked.
        """)
        ib_host = st.text_input("IB Gateway Host", value="127.0.0.1")
        ib_port = st.number_input("IB Gateway Port (Default: 7497)", value=7497, step=1)
        ib_client_id = st.number_input("Client ID", value=1, step=1)
    
    if st.button("🔄 Fetch IBKR Data", key="btn_ibkr"):
        with st.spinner("Connecting to IB Gateway..."):
            try:
                from ib_insync import IB
                
                ib = IB()
                # Short timeout to prevent the Streamlit thread from hanging indefinitely
                ib.connect(ib_host, ib_port, clientId=ib_client_id, timeout=5)
                
                if ib.isConnected():
                    summary = ib.accountSummary()
                    portfolio = ib.portfolio()
                    ib.disconnect()
                    
                    summary_data = [{"Account": s.account, "Tag": s.tag, "Value": s.value, "Currency": s.currency} for s in summary]
                    df_summary = pd.DataFrame(summary_data)
                    
                    core_tags = ["NetLiquidation", "TotalCashValue", "UnrealizedPnL"]
                    df_filtered = df_summary[df_summary["Tag"].isin(core_tags)]
                    
                    # Core Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        net_liq = df_summary[df_summary["Tag"] == "NetLiquidation"]["Value"].values[0]
                        st.metric("Net Liquidation", f"${float(net_liq):,.2f}")
                    with col2:
                        cash = df_summary[df_summary["Tag"] == "TotalCashValue"]["Value"].values[0]
                        st.metric("Total Cash", f"${float(cash):,.2f}")
                    with col3:
                        pnl = df_summary[df_summary["Tag"] == "UnrealizedPnL"]["Value"].values[0]
                        st.metric("Unrealized P&L", f"${float(pnl):,.2f}")
                        
                    # Positions Table
                    st.markdown("#### Open Positions")
                    pos_data = []
                    for p in portfolio:
                        pos_data.append({
                            "Symbol": p.contract.localSymbol,
                            "SecType": p.contract.secType,
                            "Position": p.position,
                            "Avg Cost": f"${p.marketPrice:,.2f}",
                            "Market Value": f"${p.marketValue:,.2f}",
                            "Unrealized P&L": f"${p.unrealizedPNL:,.2f}"
                        })
                    
                    if pos_data:
                        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
                    else:
                        st.info("No open positions found.")
                        
            except ImportError:
                st.error("Missing dependency: Run `pip install ib-insync`")
            except Exception as e:
                st.error(f"Could not connect to IB Gateway: {e}")
                st.warning("Ensure IB Gateway or TWS is running on your machine with 'Enable ActiveX and Socket Clients' enabled in API settings.")

# ═════════════════════════════════════════════════════════════════════════════
#  WEBULL
# ═════════════════════════════════════════════════════════════════════════════
with tab_webull:
    st.subheader("Webull Account Overview")
    
    with st.expander("🔌 Webull Authentication Settings", expanded=False):
        wb_email = st.text_input("Account Email / Phone")
        wb_password = st.text_input("Password", type="password")
        wb_mfa = st.text_input("MFA Code (If prompted via SMS/App)", help="Leave blank on first run; fill if prompted.")
        wb_trade_pin = st.text_input("Trading PIN", type="password", help="Required by Webull to unlock portfolio details.")

    if st.button("🔄 Fetch Webull Data", key="btn_webull"):
        if not wb_email or not wb_password:
            st.error("Please provide your Webull login credentials inside the configuration panel.")
        else:
            with st.spinner("Authenticating with Webull..."):
                try:
                    from webull import webull
                    
                    wb = webull()
                    login_res = wb.login(wb_email, wb_password)
                    
                    if 'code' in login_res and login_res['code'] == 'mfa_required':
                        st.warning("MFA Required. Retrieve the code sent to your device, enter it in the settings expander, and fetch again.")
                    else:
                        # Accessing portfolio requires unlocking with the 6-digit trade PIN
                        if wb_trade_pin:
                            wb.get_trade_token(wb_trade_pin)
                        
                        account_details = wb.get_account_details()
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            net_asset = account_details.get('netAsset', 0)
                            st.metric("Net Asset Value", f"${float(net_asset):,.2f}")
                        with col2:
                            cash_bal = account_details.get('cashBalance', 0)
                            st.metric("Cash Balance", f"${float(cash_bal):,.2f}")
                        with col3:
                            day_pnl = account_details.get('totalProfitLoss', 0)
                            st.metric("Total P&L", f"${float(day_pnl):,.2f}")
                            
                        # Positions
                        st.markdown("#### Open Positions")
                        positions = account_details.get('positions', [])
                        wb_pos_list = []
                        for p in positions:
                            wb_pos_list.append({
                                "Symbol": p.get('ticker', {}).get('symbol', 'N/A'),
                                "Name": p.get('ticker', {}).get('name', 'N/A'),
                                "Qty": p.get('quantity', 0),
                                "Cost Basis": f"${float(p.get('cost', 0)):,.2f}",
                                "Market Value": f"${float(p.get('marketValue', 0)):,.2f}",
                                "Total P&L": f"${float(p.get('unrealizedProfitLoss', 0)):,.2f}"
                            })
                        
                        if wb_pos_list:
                            st.dataframe(pd.DataFrame(wb_pos_list), use_container_width=True)
                        else:
                            st.info("No open Webull positions found.")
                            
                except ImportError:
                    st.error("Missing dependency: Run `pip install webull`")
                except Exception as e:
                    st.error(f"Failed to pull Webull data: {e}")