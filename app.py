import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import ta

# ------------------------------------
# Page Config
# ------------------------------------
st.set_page_config(page_title="Stock Price Prediction using LSTM", layout="wide")

st.title(" Stock Price Trend Prediction using LSTM")
st.write("Predict future stock prices using historical data and LSTM deep learning model")

# ------------------------------------
# Sidebar Inputs
# ------------------------------------
st.sidebar.header("User Input")

ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

run_button = st.sidebar.button("Run Prediction")

# ------------------------------------
# Load Model
# ------------------------------------
@st.cache_resource
def load_lstm():
    return load_model("lstm_stock_model.h5")

model = load_lstm()

# ------------------------------------
# Main Logic
# ------------------------------------
if run_button:
    st.subheader(f" Stock Data: {ticker}")

    df = yf.download(ticker, start=start_date, end=end_date)

    # FIX for MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        st.error("Invalid ticker or no data available.")
        st.stop()

    # ------------------------------------
    # Feature Engineering
    # ------------------------------------
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
    df.dropna(inplace=True)

    st.dataframe(df.tail())

    # ------------------------------------
    # Price + MA Plot
    # ------------------------------------
    st.subheader(" Closing Price with Moving Averages")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df["Close"], label="Close")
    ax1.plot(df["MA_20"], label="MA 20")
    ax1.plot(df["MA_50"], label="MA 50")
    ax1.legend()
    st.pyplot(fig1)

    # ------------------------------------
    # RSI Plot
    # ------------------------------------
    st.subheader(" Relative Strength Index (RSI)")

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(df["RSI"])
    ax2.axhline(70, linestyle="--")
    ax2.axhline(30, linestyle="--")
    st.pyplot(fig2)

    # ------------------------------------
    # Scaling
    # ------------------------------------
    features = ["Close", "MA_20", "MA_50", "RSI"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    # ------------------------------------
    # Create Sequences
    # ------------------------------------
    TIME_STEPS = 60
    X = []

    for i in range(TIME_STEPS, len(scaled)):
        X.append(scaled[i - TIME_STEPS:i])

    X = np.array(X)

    # ------------------------------------
    # Prediction
    # ------------------------------------
    preds = model.predict(X)

    close_scaler = MinMaxScaler()
    close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]

    actual = close_scaler.inverse_transform(
        scaled[TIME_STEPS:, 0].reshape(-1, 1)
    )
    predicted = close_scaler.inverse_transform(preds)

    # ------------------------------------
    # Actual vs Predicted Plot
    # ------------------------------------
    st.subheader(" Actual vs Predicted Prices")

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(actual, label="Actual")
    ax3.plot(predicted, label="Predicted")
    ax3.legend()
    st.pyplot(fig3)

    # ------------------------------------
    # Next Day Prediction
    # ------------------------------------
    last_seq = scaled[-TIME_STEPS:].reshape(1, TIME_STEPS, len(features))
    next_day = model.predict(last_seq)
    next_price = close_scaler.inverse_transform(next_day)


    st.success(f" Predicted Next Day Closing Price: ${next_price[0][0]:.2f}")
