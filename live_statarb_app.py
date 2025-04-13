...

    st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)
    st.subheader("🔁 Auto-Refreshing Every 60 Seconds")

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
