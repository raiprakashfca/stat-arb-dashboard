import streamlit as st
import pandas as pd
import numpy as np
from kiteconnect import KiteConnect
import datetime
import time

# ---- USER INPUTS ----
api_key = st.secrets["api_key"]
access_token = st.secrets["access_token"]

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Define multiple commonly traded stock pairs
pairs = [
    ("TCS", "INFY"),
    ("HDFCBANK", "ICICIBANK"),
    ("SBIN", "PNB")
]

st.set_page_config(page_title="Live StatArb Dashboard", layout="wide")
st.title("📈 Live Statistical Arbitrage Dashboard")

# Initialize session state
if "price_history" not in st.session_state:
    st.session_state.price_history = {pair: {"x": [], "y": []} for pair in pairs}

# Info message and auto-refresh
st.info("🔄 Auto-refreshing every 30 seconds...")
st.experimental_set_query_params(ts=time.time())  # trigger rerun

# Main display
for stock1, stock2 in pairs:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{stock1} vs {stock2}")

    try:
        # Fetch live prices
        ltp_data = kite.ltp([f"NSE:{stock1}", f"NSE:{stock2}"])
        price1 = ltp_data[f"NSE:{stock1}"]["last_price"]
        price2 = ltp_data[f"NSE:{stock2}"]["last_price"]

        # Update price history
        st.session_state.price_history[(stock1, stock2)]["x"].append(price1)
        st.session_state.price_history[(stock1, stock2)]["y"].append(price2)

        # Create DataFrame
        df = pd.DataFrame({
            stock1: st.session_state.price_history[(stock1, stock2)]["x"],
            stock2: st.session_state.price_history[(stock1, stock2)]["y"]
        })

        # Calculate spread and z-score
        hedge_ratio = 1.0  # Simplified static hedge ratio
        spread = df[stock1] - hedge_ratio * df[stock2]
        zscore = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()

        latest_spread = spread.iloc[-1] if not spread.empty else 0
        latest_zscore = zscore.iloc[-1] if not zscore.empty else 0

        # Signal logic
        signal = "No Signal"
        if latest_zscore > 2:
            signal = f"🔻 SELL {stock1}, BUY {stock2}"
        elif latest_zscore < -2:
            signal = f"🔺 BUY {stock1}, SELL {stock2}"

        with col1:
            st.metric(label=f"{stock1} Price", value=price1)
            st.metric(label=f"{stock2} Price", value=price2)
            st.metric(label="Spread", value=f"{latest_spread:.2f}")
            st.metric(label="Z-score", value=f"{latest_zscore:.2f}")
            st.markdown(f"### Signal: **{signal}**")

        with col2:
            st.line_chart(zscore.tail(50))

    except Exception as e:
        st.error(f"Error fetching data for {stock1} vs {stock2}: {e}")

# Auto-refresh after 30 seconds
st.markdown("<meta http-equiv='refresh' content='30'>", unsafe_allow_html=True)
