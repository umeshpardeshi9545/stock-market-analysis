import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import urllib.request
from datetime import timedelta

# App Configuration
st.set_page_config(page_title="📈 Apple Stock Predictor", layout="centered")
st.title("🍏 Apple Stock Price Forecast App")
st.markdown("Forecast Apple's stock closing price using a trained **SRIMA** model.")

# Load ARIMA Model from GitHub
model_url = "https://raw.githubusercontent.com/umeshpardeshi9545/stock-market-analysis/main/sarima_model"

try:
    with urllib.request.urlopen(model_url) as response:
        model_data = response.read()
        model = pickle.loads(model_data)
except Exception as e:
    model = None
    st.error(f"❌ Error loading the model: {e}")

# Load Apple Stock Data
df = pd.read_csv("AAPL(4).csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')
last_date = df['Date'].max()

# Forecast Settings
st.sidebar.header("🔢 Forecast Settings")
days = st.sidebar.slider("Select number of days to forecast", min_value=1, max_value=30, value=30)

# Forecast Generation
if model:
    forecast = model.forecast(steps=days)
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(days)]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_Close': forecast})
else:
    forecast_df = pd.DataFrame(columns=['Date', 'Predicted_Close'])

# 📉 Historical Plot
st.subheader("📉 Historical Close Price")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df['Date'], df['Close'], label='Historical Close', color='blue')
ax1.set_title('Historical Apple Stock Prices')
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price ($)')
ax1.legend()
st.pyplot(fig1)

# 📈 Forecast Plot
if not forecast_df.empty:
    st.subheader(f"📈 Forecast for Next {days} Days")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(forecast_df['Date'], forecast_df['Predicted_Close'], label='Forecasted Close', color='orange')
    ax2.set_title('Forecasted Apple Stock Prices')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Predicted Price ($)')
    ax2.legend()
    st.pyplot(fig2)

    # 📊 Combined Historical + Forecast Plot
    st.subheader("📊 Combined View: Historical + Forecast")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(df['Date'], df['Close'], label='Historical', color='blue')
    ax3.plot(forecast_df['Date'], forecast_df['Predicted_Close'], label='Forecast', color='orange')
    ax3.axvline(last_date, linestyle='--', color='red', label='Forecast Start')
    ax3.set_title('Apple Stock Prices: Historical + Forecast')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Close Price ($)')
    ax3.legend()
    st.pyplot(fig3)

    # 📋 Forecast Data Table
    st.subheader("📋 Forecast Data Table")
    st.dataframe(forecast_df)

    # 📥 Download CSV
    st.download_button(
        "📥 Download Forecast as CSV",
        forecast_df.to_csv(index=False).encode('utf-8'),
        "forecast.csv",
        "text/csv"
    )

    st.success(f"✅ Forecast complete for {days} days from {last_date.date()}!")
else:
    st.warning("⚠️ Forecasting unavailable. Please check model load or try again later.")
