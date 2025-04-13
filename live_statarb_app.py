import streamlit as st
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import pandas as pd

st.set_page_config(page_title="📈 Kalman + Scalping Dashboard", layout="wide")
tabs = st.tabs(["📊 StatArb", "⚡ Scalping"])

# Load API credentials
api_key = st.secrets["api_key"]
access_token = st.secrets["access_token"]

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Fetch historical data from Zerodha
def fetch_historical_data(symbol, interval="minute", days=1):
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)
    instrument_token = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"]
    data = kite.historical_data(instrument_token, from_date, to_date, interval)
    return pd.DataFrame(data)

with tabs[1]:
    st.title("⚡ Microstructure-Based EMA Bounce Scalping")
    st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)
    st.subheader("🔁 Auto-Refreshing Every 60 Seconds")

    symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "SBIN", "RELIANCE", "TCS"])

    try:
        live_df = fetch_historical_data(symbol, interval="minute", days=1)
        live_df["EMA9"] = live_df["close"].ewm(span=9).mean()
        live_df["EMA21"] = live_df["close"].ewm(span=21).mean()
        live_df.dropna(inplace=True)

        latest = live_df.iloc[-1]
        prev = live_df.iloc[-2]

        signal = "No Signal"
        if prev.EMA9 < prev.EMA21 and latest.EMA9 > latest.EMA21:
            signal = "🔼 LONG SIGNAL"
        elif prev.EMA9 > prev.EMA21 and latest.EMA9 < latest.EMA21:
            signal = "🔽 SHORT SIGNAL"

        st.metric("Last Price", f"₹{latest.close:.2f}")
        st.metric("Signal", signal)
        st.line_chart(live_df.set_index("date")[["EMA9", "EMA21", "close"]].tail(100))

    except Exception as e:
        st.error(f"Error fetching live data: {e}")
