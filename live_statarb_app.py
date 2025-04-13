import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pykalman import KalmanFilter
from kiteconnect import KiteConnect
import time
import os

# Read API credentials securely
api_key = st.secrets["api_key"]
access_token = st.secrets["access_token"]

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Define selected stock pairs for stat arb
pairs = [
    ("NTPC", "POWERGRID"),
    ("BPCL", "HINDPETRO"),
    ("JSWSTEEL", "HINDALCO"),
    ("SUNPHARMA", "AUROPHARMA"),
    ("SBIN", "BANKBARODA")
]

st.set_page_config(page_title="📈 Kalman Filter StatArb", layout="wide")
st.title("📊 Kalman Filter-Based Statistical Arbitrage Dashboard")

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

# Kalman Filter-based beta estimator
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

        if len(df) >= 60:
            beta_series, residuals = kalman_beta(df[stock1].values, df[stock2].values)
            zscore = (pd.Series(residuals) - pd.Series(residuals).rolling(30).mean()) / pd.Series(residuals).rolling(30).std()
            latest_z = zscore.iloc[-1]
            latest_beta = beta_series[-1]

            signal = "No Signal"
            if latest_z > 2:
                signal = f"🔻 SELL {stock1}, BUY {stock2}"
            elif latest_z < -2:
                signal = f"🔺 BUY {stock1}, SELL {stock2}"

            with col1:
                st.metric(label=f"{stock1} Price", value=price1)
                st.metric(label=f"{stock2} Price", value=price2)
                st.metric(label="Z-score", value=f"{latest_z:.2f}")
                st.metric(label="Kalman Beta", value=f"{latest_beta:.2f}")
                st.success(f"📣 Signal: **{signal}**")

            with col2:
                st.line_chart(pd.DataFrame({
                    "Beta": beta_series,
                    "Z-Score": zscore
                }))

            # Logging to CSV
            log_df = pd.DataFrame({
                "Timestamp": [pd.Timestamp.now()],
                "Pair": [f"{stock1}-{stock2}"],
                "Beta": [latest_beta],
                "Z-Score": [latest_z],
                "Signal": [signal]
            })
            log_file = f"log_{stock1}_{stock2}.csv"
            if os.path.exists(log_file):
                log_df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                log_df.to_csv(log_file, index=False)
        else:
            with col1:
                st.warning("Waiting for at least 60 data points for Kalman Filter...")

    except Exception as e:
        st.error(f"Error fetching data for {stock1} vs {stock2}: {e}")

# Auto-refresh every 30 seconds
st.markdown("<meta http-equiv='refresh' content='30'>", unsafe_allow_html=True)
