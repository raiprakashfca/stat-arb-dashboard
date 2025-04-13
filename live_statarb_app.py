import streamlit as st
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pykalman import KalmanFilter

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

# Kalman beta function
def kalman_beta(y, x):
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([x, np.ones(len(x))]).T, axis=1)
    kf = KalmanFilter(transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      initial_state_mean=np.zeros(2),
                      initial_state_covariance=np.ones((2, 2)),
                      observation_covariance=1.0,
                      transition_covariance=trans_cov)
    state_means, _ = kf.filter(y)
    return state_means[:, 0], y - (state_means[:, 0] * x + state_means[:, 1])

with tabs[0]:
    st.title("📊 Kalman Filter-Based Statistical Arbitrage")
    st.subheader("🔁 Auto-Refreshing Every 30 Seconds")
    st.markdown("<meta http-equiv='refresh' content='30'>", unsafe_allow_html=True)

    pair = st.selectbox("Select Pair", [("SBIN", "BANKBARODA"), ("HINDALCO", "JSWSTEEL"), ("NTPC", "POWERGRID")])

    try:
        df1 = fetch_historical_data(pair[0], interval="minute", days=1)
        if df1.empty:
            st.warning(f"No data returned for {pair[0]} — market may be closed or holiday.")
            st.stop()
        st.write(f"**Preview {pair[0]} data:**")
        st.dataframe(df1.head())
        df2 = fetch_historical_data(pair[1], interval="minute", days=1)
        if df2.empty:
            st.warning(f"No data returned for {pair[1]} — market may be closed or holiday.")
            st.stop()
        st.write(f"**Preview {pair[1]} data:**")
        st.dataframe(df2.head())
        df = pd.merge(df1, df2, on="date", suffixes=(f"_{pair[0]}", f"_{pair[1]}"))

        beta_series, residuals = kalman_beta(df[f"close_{pair[0]}"][:], df[f"close_{pair[1]}"][:])
        zscore = (pd.Series(residuals) - pd.Series(residuals).rolling(30).mean()) / pd.Series(residuals).rolling(30).std()
        latest_z = zscore.iloc[-1]
        latest_beta = beta_series[-1]

        signal = "No Signal"
        if latest_z > 2:
            signal = f"🔻 SELL {pair[0]}, BUY {pair[1]}"
        elif latest_z < -2:
            signal = f"🔺 BUY {pair[0]}, SELL {pair[1]}"

        st.metric(label="Z-score", value=f"{latest_z:.2f}")
        st.metric(label="Kalman Beta", value=f"{latest_beta:.2f}")
        st.success(f"📣 Signal: **{signal}**")
        st.line_chart(pd.DataFrame({"Z-Score": zscore, "Beta": beta_series}))

    except Exception as e:
        st.error(f"Error in Kalman StatArb: {e}")

with tabs[1]:
    st.title("⚡ Microstructure-Based EMA Bounce Scalping")
    st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)
    st.subheader("🔁 Auto-Refreshing Every 60 Seconds")

    symbol = st.selectbox("Select Symbol", ["SBIN", "RELIANCE", "TCS", "HDFCBANK", "INFY"])

    try:
        live_df = fetch_historical_data(symbol, interval="minute", days=1)
        if live_df.empty:
            st.warning(f"No data returned for {symbol} — market may be closed or holiday.")
            st.stop()
        st.write(f"**Preview {symbol} data:**")
        st.dataframe(live_df.head())
        if live_df.empty:
            st.warning(f"No data returned for {symbol} — market may be closed or holiday.")
            st.stop()
        st.write(f"**Preview {symbol} data:**")
        st.dataframe(live_df.head())
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
