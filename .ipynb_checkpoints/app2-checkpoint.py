import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="AQI Dashboard", layout="wide")

st.title("🌍 Construction Pollution Intelligence System")

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    xgb = joblib.load("xgb_model.pkl")

    # ✅ FIX: disable compilation while loading
    lstm = load_model("lstm_model.h5", compile=False)

    # ✅ OPTIONAL: recompile manually (recommended)
    lstm.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    features = joblib.load("features.pkl")
    return xgb, lstm, features

xgb_model, lstm_model, feature_cols = load_models()

SEQ_LENGTH = 24

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.sidebar.header("📂 Upload Data")

uploaded_file = st.sidebar.file_uploader("Upload merged_df CSV")

if uploaded_file is None:
    st.warning("Please upload a dataset to proceed")
    st.stop()

df = pd.read_csv(uploaded_file)
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
df = df.sort_values("TIMESTAMP")

# -------------------------------
# SENSOR FILTER
# -------------------------------
if "sensor_id" in df.columns:
    sensor = st.sidebar.selectbox("Select Sensor", ["All"] + list(df["sensor_id"].unique()))
    if sensor != "All":
        df = df[df["sensor_id"] == sensor]

# -------------------------------
# DATE SELECTION
# -------------------------------
today = datetime.now().date()
max_date = today + timedelta(days=7)

start_date = st.sidebar.date_input("Start Date", today)
end_date = st.sidebar.date_input("End Date", max_date)

# -------------------------------
# FEATURE ALIGNMENT
# -------------------------------
X = df.copy()

# Ensure required features exist
missing_cols = [col for col in feature_cols if col not in X.columns]
for col in missing_cols:
    X[col] = 0

X = X[feature_cols]

# -------------------------------
# CURRENT PREDICTION (XGBoost)
# -------------------------------
df["Predicted_PM25"] = xgb_model.predict(X)

# -------------------------------
# LSTM FORECAST
# -------------------------------
def forecast_lstm(model, data, steps=168):
    preds = []
    current = data.copy()

    for _ in range(steps):
        pred = model.predict(current.reshape(1, SEQ_LENGTH, data.shape[1]))[0][0]
        preds.append(pred)

        current = np.roll(current, -1, axis=0)
        current[-1] = pred

    return preds

future_df = pd.DataFrame()

if end_date > today:
    last_seq = X.tail(SEQ_LENGTH).values
    steps = (end_date - today).days * 24

    preds = forecast_lstm(lstm_model, last_seq, steps)

    future_dates = pd.date_range(start=today, periods=steps, freq="H")

    future_df = pd.DataFrame({
        "TIMESTAMP": future_dates,
        "PM2.5": preds
    })

# -------------------------------
# FILTER HISTORICAL
# -------------------------------
hist_df = df[
    (df["TIMESTAMP"].dt.date >= start_date) &
    (df["TIMESTAMP"].dt.date <= min(end_date, today))
]

hist_df = hist_df[["TIMESTAMP", "Predicted_PM25"]]
hist_df.columns = ["TIMESTAMP", "PM2.5"]

# -------------------------------
# MERGE
# -------------------------------
display_df = pd.concat([hist_df, future_df])

# -------------------------------
# DECISION LOGIC
# -------------------------------
def decision(pm25):
    if pm25 < 100:
        return "Safe"
    elif pm25 < 200:
        return "Moderate"
    else:
        return "Unsafe"

display_df["Decision"] = display_df["PM2.5"].apply(decision)

# -------------------------------
# KPI CARDS
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg PM2.5", round(display_df["PM2.5"].mean(), 2))
col2.metric("Max PM2.5", round(display_df["PM2.5"].max(), 2))
col3.metric("Unsafe Hours", (display_df["Decision"] == "Unsafe").sum())
col4.metric("Safe Hours", (display_df["Decision"] == "Safe").sum())

# -------------------------------
# TREND CHART
# -------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=display_df["TIMESTAMP"],
    y=display_df["PM2.5"],
    mode="lines",
    name="PM2.5"
))

fig.add_hrect(y0=0, y1=100, fillcolor="green", opacity=0.1)
fig.add_hrect(y0=100, y1=200, fillcolor="yellow", opacity=0.1)
fig.add_hrect(y0=200, y1=500, fillcolor="red", opacity=0.1)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# DAILY SUMMARY
# -------------------------------
display_df["date"] = display_df["TIMESTAMP"].dt.date

daily = display_df.groupby("date")["PM2.5"].mean().reset_index()
daily["Decision"] = daily["PM2.5"].apply(decision)

fig2 = px.bar(daily, x="date", y="PM2.5", color="Decision")
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# SENSOR CONTRIBUTION
# -------------------------------
if "sensor_id" in df.columns:
    st.subheader("Sensor Contribution")

    contrib = df.groupby("sensor_id")["PM2.5"].mean()

    st.bar_chart(contrib)

# -------------------------------
# MAP
# -------------------------------
if {"lat", "lon"}.issubset(df.columns):
    st.subheader("Map")

    map_df = df.groupby("sensor_id").agg({
        "PM2.5": "mean",
        "lat": "first",
        "lon": "first"
    }).reset_index()

    st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}))