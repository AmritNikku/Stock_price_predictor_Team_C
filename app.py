
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Ensure UTF-8 encoding

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Add updated CSS for aesthetic changes
custom_css = """
<style>
    /* Body styling */
    body {
        background-color: rgb(240, 242, 246);
        color: rgb(240, 242, 246);
        font-family: 'Arial', sans-serif;
    }

    /* Gradient background for header */
    .main > .block-container {
        background-color: rgb(144, 213, 255);
        border-radius: 15px;
        padding: 1rem;
    }

    /* Sidebar styling */
    .css-18e3th9 {
        background: linear-gradient(180deg, #b3e5fc, #d6eefc);
        color: #333333;
    }
    .css-18e3th9 a {
        color: #01579b;
        text-decoration: none;
    }
    .css-18e3th9 a:hover {
        color: #0288d1;
    }

    /* Headings styling */
    h1, h2, h3 {
        color: #0277bd;
        font-weight: 600;
    }

    /* Subheaders and text */
    .stMarkdown p {
        font-size: 16px;
        line-height: 1.6;
        color: #333333;
    }

    /* Tables */
    .stDataFrame, .stTable {
        background-color: #e8f5e9;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Title and Introduction
st.title("📈 Stock Price Predicting Stocks with ML")
st.markdown("""
Our Project is an **ML-powered stock price prediction app** that analyzes stock market trends using machine learning.  
With this app, you can explore historical stock data, moving averages, and future price predictions.

---

Get started by entering a stock ticker symbol below!
""")

# Sidebar
st.sidebar.title("Menu")
menu = st.sidebar.radio("Navigate", ["Stock Info", "Moving Averages", "Future Prediction"])

# Input for stock symbol
stock = st.text_input("Enter the Stock Ticker (e.g., GOOG, AAPL)", "GOOG")

# Time range for stock data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Fetch stock data using yfinance
st.markdown("### Fetching Stock Data...")
try:
    stock_data = yf.download(stock, start, end)
    if stock_data.empty:
        st.error("No data found. Please check the stock ticker.")
    else:
        st.success("Data successfully fetched!")
except Exception as e:
    st.error(f"Error fetching data: {str(e)}")

# Display stock data based on selected menu
if not stock_data.empty:
    # Flatten MultiIndex if it exists
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = [col[0] for col in stock_data.columns]

    # Ensure 'Close' column exists
    if 'Close' not in stock_data.columns:
        st.error("The 'Close' column is not found in the data. Please check the stock ticker.")
    else:
        if menu == "Stock Info":
            st.subheader("📊 Stock Data Overview")
            st.write(stock_data.describe())
            st.subheader("📈 Closing Price Over Time")
            st.line_chart(stock_data['Close'])

        elif menu == "Moving Averages":
            st.subheader("📉 Moving Averages (MA)")
            st.markdown("""
            Moving averages smooth out fluctuations to show trends over time.  
            - **250 Days MA**: Long-term trend  
            - **200 Days MA**: Medium-term trend  
            - **100 Days MA**: Short-term trend  
            """)

            for ma in [250, 200, 100]:
                column_name = f"MA_{ma}"
                stock_data[column_name] = stock_data['Close'].rolling(ma).mean()
                st.line_chart(stock_data[['Close', column_name]].dropna())

        elif menu == "Future Prediction":
            try:
                model = load_model("Latest_stock_price_model.keras")
                st.success("Model loaded successfully!")

                # Data preprocessing for prediction
                scaler = MinMaxScaler(feature_range=(0, 1))
                closing_prices = stock_data[['Close']]
                scaled_data = scaler.fit_transform(closing_prices)

                # Create sequences for the model
                sequence_length = 100
                x_data, y_data = [], []
                for i in range(sequence_length, len(scaled_data)):
                    x_data.append(scaled_data[i-sequence_length:i])
                    y_data.append(scaled_data[i])

                x_data, y_data = np.array(x_data), np.array(y_data)

                # Predict stock prices
                st.markdown("### 🚀 Predicting Stock Prices...")
                predictions = model.predict(x_data)
                predictions = scaler.inverse_transform(predictions)
                y_data = scaler.inverse_transform(y_data)

                # Display predictions
                st.subheader("📌 Predictions vs Actual Values")
                results = pd.DataFrame({
                    'Actual': y_data.flatten(),
                    'Predicted': predictions.flatten()
                }, index=stock_data.index[sequence_length:])

                st.line_chart(results)

                # Forecast future prices
                st.subheader("📅 Future Stock Price Forecast")
                last_sequence = scaled_data[-sequence_length:]
                future_predictions = []
                for _ in range(10):  # Predict next 10 days
                    pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
                    future_predictions.append(scaler.inverse_transform(pred)[0, 0])
                    last_sequence = np.append(last_sequence[1:], pred, axis=0)

                future_dates = pd.date_range(stock_data.index[-1], periods=11, freq='D')[1:]
                future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
                st.write(future_df)

                st.line_chart(future_df.set_index('Date'))
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
