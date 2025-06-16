import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

model = load_model("models/stock_model.keras")


st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title('📈 Stock Market Predictor')

# Input field for any stock symbol
stock = st.text_input('Enter Stock Symbol (e.g. GOOG, AAPL, TSLA)', 'GOOG')

start = '2012-01-01'
end = '2025-5-21'

# Validate symbol and load data
if stock:
    try:
        data = yf.download(stock, start=start, end=end)
        if data.empty:
            st.error("❌ Invalid stock symbol or no data found.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        st.stop()

    st.subheader('Raw Stock Data')
    st.write(data)

    # Split into training/testing
    data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Prepare test data for prediction
    past_100_days = data_train.tail(100)
    data_test_full = pd.concat([past_100_days, data_test], ignore_index=True)
    data_test_scaled = scaler.fit_transform(data_test_full)

    # Moving Average Plots
    st.subheader('Price vs MA50')
    ma_50 = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(8,6))
    plt.plot(ma_50, 'r', label='MA50')
    plt.plot(data.Close, 'g', label='Close')
    plt.legend()
    st.pyplot(fig1)

    st.subheader('Price vs MA50 vs MA100')
    ma_100 = data.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(8,6))
    plt.plot(ma_50, 'r', label='MA50')
    plt.plot(ma_100, 'b', label='MA100')
    plt.plot(data.Close, 'g', label='Close')
    plt.legend()
    st.pyplot(fig2)

    st.subheader('Price vs MA100 vs MA200')
    ma_200 = data.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(8,6))
    plt.plot(ma_100, 'r', label='MA100')
    plt.plot(ma_200, 'b', label='MA200')
    plt.plot(data.Close, 'g', label='Close')
    plt.legend()
    st.pyplot(fig3)

    # Prepare data for model
    x, y = [], []
    for i in range(100, data_test_scaled.shape[0]):
        x.append(data_test_scaled[i-100:i])
        y.append(data_test_scaled[i, 0])

    x, y = np.array(x), np.array(y)

    # Predict and rescale
    predict = model.predict(x)
    scale = 1 / scaler.scale_[0]
    predict = predict * scale
    y = y * scale

    # Plot results
    st.subheader('📊 Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8,6))
    plt.plot(y, 'r', label='Original Price')
    plt.plot(predict, 'g', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig4)
