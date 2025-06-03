import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import timedelta

# Config
st.set_page_config(page_title="📈 Apple Stock Predictor", layout="centered")
st.title("🍏 Apple Stock Price Forecast App")
st.markdown("Forecast Apple's stock closing price using a trained **ARIMA** model.")

# Load model
@st.cache_resource
def load_model():
    with open('arima_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Load data
df = pd.read_csv('AAPL (4).csv')  # Make sure file name is clean
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')
last_date = df['Date'].max()

# 📆 Date range input (optional - forecasting always happens from last available date)
st.sidebar.header("🔢 Forecast Settings")
days = st.sidebar.slider("Select number of days to forecast", min_value=1, max_value=30, value=30)

# Forecast
forecast = model.forecast(steps=days)
forecast_dates = [last_date + timedelta(days=i+1) for i in range(days)]
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_Close': forecast})

# 📊 Visualization
st.subheader("📉 Historical Close Price")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df['Date'], df['Close'], label='Historical Close', color='blue')
ax1.set_title('Historical Apple Stock Prices')
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price ($)')
ax1.legend()
st.pyplot(fig1)

st.subheader(f"📈 Forecast for Next {days} Days")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(forecast_df['Date'], forecast_df['Predicted_Close'], label='Forecasted Close', color='orange')
ax2.set_title('Forecasted Apple Stock Prices')
ax2.set_xlabel('Date')
ax2.set_ylabel('Predicted Price ($)')
ax2.legend()
st.pyplot(fig2)

# 🧾 Display forecast data
st.subheader("📋 Forecast Data Table")
st.dataframe(forecast_df)

st.success(f"✅ Forecast complete for {days} days from {last_date.date()}!")

