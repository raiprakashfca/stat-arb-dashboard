import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from kiteconnect import KiteConnect
import time

# Read API credentials securely
api_key = st.secrets["api_key"]
access_token = st.secrets["access_token"]

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Define selected stock pairs for stat arb (correcting HPCL to HINDPETRO)
pairs = [
    ("NTPC", "POWERGRID"),
    ("BPCL", "HINDPETRO"),
    ("JSWSTEEL", "HINDALCO"),
    ("SUNPHARMA", "AUROPHARMA"),
    ("SBIN", "BANKBARODA")
]

st.set_page_config(page_title="📈 StatArb Dashboard", layout="wide")
st.title("📊 OLS-Based Statistical Arbitrage Dashboard")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }
    .metric-label, .metric-value {
        font-size: 16px !important;
    }
    .stMetric { margin-bottom: 10px !important; }
</style>
""", unsafe_allow_html=True)

if "price_history" not in st.session_state:
    st.session_state.price_history = {pair: {"x": [], "y": []} for pair in pairs}

st.info("🔄 Auto-refreshing every 30 seconds")
st.query_params["ts"] = str(time.time())

# MAIN LOOP TO DISPLAY PAIRS
for stock1, stock2 in pairs:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"🟢 {stock1} vs {stock2}")

    try:
        ltp_data = kite.ltp([f"NSE:{stock1}", f"NSE:{stock2}"])
        price1 = ltp_data[f"NSE:{stock1}"]["last_price"]
        price2 = ltp_data[f"NSE:{stock2}"]["last_price"]

        st.session_state.price_history[(stock1, stock2)]["x"].append(price1)
        st.session_state.price_history[(stock1, stock2)]["y"].append(price2)

        df = pd.DataFrame({
            stock1: st.session_state.price_history[(stock1, stock2)]["x"],
            stock2: st.session_state.price_history[(stock1, stock2)]["y"]
        })

        if len(df) >= 30:
            X = sm.add_constant(df[stock2])
            model = sm.OLS(df[stock1], X).fit()
            df['residuals'] = model.resid
            zscore = (df['residuals'] - df['residuals'].rolling(30).mean()) / df['residuals'].rolling(30).std()

            latest_zscore = zscore.iloc[-1]
            signal = "No Signal"
            if latest_zscore > 2:
                signal = f"🔻 SELL {stock1}, BUY {stock2}"
            elif latest_zscore < -2:
                signal = f"🔺 BUY {stock1}, SELL {stock2}"

            with col1:
                st.metric(label=f"{stock1} Price", value=price1)
                st.metric(label=f"{stock2} Price", value=price2)
                st.metric(label="Z-score", value=f"{latest_zscore:.2f}")
                st.success(f"📣 Signal: **{signal}**")

            with col2:
                st.line_chart(zscore.tail(50))
        else:
            with col1:
                st.warning("Waiting for at least 30 data points...")

    except Exception as e:
        st.error(f"Error fetching data for {stock1} vs {stock2}: {e}")

# Backtesting module using regression residuals
with st.expander("🧪 Run OLS-Based Backtest"):
    selected_pair = st.selectbox("Choose a pair to backtest", [f"{s1} & {s2}" for s1, s2 in pairs])
    entry_z = st.slider("Z-score Entry Threshold", 1.0, 3.0, 2.0, 0.1)
    exit_z = st.slider("Z-score Exit Threshold", -1.0, 1.0, 0.0, 0.1)

    if st.button("Run Backtest"):
        s1, s2 = selected_pair.split(" & ")
        df_bt = pd.DataFrame({
            s1: st.session_state.price_history[(s1, s2)]["x"],
            s2: st.session_state.price_history[(s1, s2)]["y"]
        })

        if len(df_bt) < 30:
            st.warning("Not enough data for backtest (need at least 30 samples).")
        else:
            X_bt = sm.add_constant(df_bt[s2])
            model_bt = sm.OLS(df_bt[s1], X_bt).fit()
            df_bt['residuals'] = model_bt.resid
            z_bt = (df_bt['residuals'] - df_bt['residuals'].rolling(30).mean()) / df_bt['residuals'].rolling(30).std()

            position = 0
            pnl = []
            entry_price = 0

            for i in range(30, len(z_bt)):
                if position == 0:
                    if z_bt.iloc[i] > entry_z:
                        position = -1
                        entry_price = df_bt['residuals'].iloc[i]
                    elif z_bt.iloc[i] < -entry_z:
                        position = 1
                        entry_price = df_bt['residuals'].iloc[i]
                elif position == -1 and z_bt.iloc[i] < exit_z:
                    pnl.append(entry_price - df_bt['residuals'].iloc[i])
                    position = 0
                elif position == 1 and z_bt.iloc[i] > exit_z:
                    pnl.append(df_bt['residuals'].iloc[i] - entry_price)
                    position = 0

            if pnl:
                st.success(f"Backtest Completed. Trades: {len(pnl)}, Total PnL: ₹{sum(pnl):.2f}, Avg: ₹{np.mean(pnl):.2f}")
                st.line_chart(pd.Series(pnl).cumsum())
            else:
                st.info("No trades triggered in backtest.")

# Auto-refresh every 30 seconds
st.markdown("<meta http-equiv='refresh' content='30'>", unsafe_allow_html=True)
