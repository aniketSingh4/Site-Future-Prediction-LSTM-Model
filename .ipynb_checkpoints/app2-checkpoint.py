import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import requests
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
# SHAP EXPLAINER (AFTER MODEL LOAD)
# -------------------------------
@st.cache_resource
def load_explainer(_model):
    return shap.Explainer(_model)

explainer = load_explainer(xgb_model)

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
# DATE RANGE SLIDER (Today → Next 7 Days)
# -------------------------------
today = datetime.now().date()
max_date = today + timedelta(days=7)

start_date, end_date = st.sidebar.slider(
    "📅 Select Date Range (Next 7 Days)",
    min_value=today,
    max_value=max_date,
    value=(today, max_date)
)

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
# FUTURE STEPS CALCULATION
# -------------------------------
if end_date > today:
    steps = int((end_date - today).days * 24)
else:
    steps = 0

    
# -------------------------------
# Future Weather CALCULATION
# -------------------------------
def get_weather_forecast(lat, lon, steps):
    API_KEY = "c5f66654ed6c90413a409155db5c61b7"

    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

    res = requests.get(url).json()

    weather_data = []

    for item in res["list"]:
        weather_data.append({
            "TIMESTAMP": item["dt_txt"],
            "AT": item["main"]["temp"],
            "RH": item["main"]["humidity"],
            "WS": item["wind"]["speed"],
            "WD": item["wind"]["deg"],
            "RF": item.get("rain", {}).get("3h", 0),
            "SR": 0  # Not available → keep 0 or estimate
        })

    weather_df = pd.DataFrame(weather_data)

    weather_df["TIMESTAMP"] = pd.to_datetime(weather_df["TIMESTAMP"])

    # Convert 3-hour data → hourly (important for your LSTM)
    weather_df = weather_df.set_index("TIMESTAMP").resample("H").interpolate().reset_index()

    return weather_df.head(steps)
# -------------------------------
# LSTM FORECAST
# -------------------------------
def forecast_lstm_with_weather(model, last_seq, future_weather, feature_cols, target_col="PM2.5"):
    preds = []
    current_seq = last_seq.copy()

    # ✅ FIX HERE
    target_idx = feature_cols.get_loc(target_col)

    for i in range(len(future_weather)):
        pred = model.predict(
            current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1]),
            verbose=0
        )[0][0]

        preds.append(pred)

        next_row = current_seq[-1].copy()

        # Update PM2.5
        next_row[target_idx] = pred

        # Update weather
        for col in future_weather.columns:
            if col in feature_cols:
                idx = feature_cols.get_loc(col)   # ✅ FIX HERE ALSO
                next_row[idx] = future_weather.iloc[i][col]

        current_seq = np.vstack([current_seq[1:], next_row])

    return preds



last_seq = X.tail(SEQ_LENGTH).values

# Calculate steps
steps = int((end_date - today).days * 24)

# Generate future weather
#future_weather = generate_future_weather(df.iloc[-1], steps)


# -------------------------------
# FUTURE WEATHER
# -------------------------------
future_df = pd.DataFrame()

if end_date > today:
    steps = int((end_date - today).days * 24)

    last_seq = X.tail(SEQ_LENGTH).values

    # ✅ Use API (Option 2B)
    future_weather = get_weather_forecast(19.0760, 72.8777, steps)

    # ✅ Save timestamp BEFORE dropping columns
    future_timestamps = future_weather["TIMESTAMP"].copy()

    # Ensure feature alignment
    for col in feature_cols:
        if col not in future_weather.columns:
            future_weather[col] = 0

    future_weather = future_weather[feature_cols]

    preds = forecast_lstm_with_weather(
        lstm_model,
        last_seq,
        future_weather,
        feature_cols
    )

    future_df = pd.DataFrame({
    "TIMESTAMP": future_timestamps.iloc[:len(preds)].values,
    "PM2.5": preds
    })

# -------------------------------
# FILTER HISTORICAL
# -------------------------------
hist_df = df[
    (df["TIMESTAMP"].dt.date >= start_date) &
    (df["TIMESTAMP"].dt.date <= today)
]

hist_df = hist_df[["TIMESTAMP", "Predicted_PM25"]]
hist_df.columns = ["TIMESTAMP", "PM2.5"]

# -------------------------------
# MERGE
# -------------------------------
display_df = pd.concat([hist_df, future_df])
# -------------------------------
# FIX TIMESTAMP BEFORE USING
# -------------------------------
display_df["TIMESTAMP"] = pd.to_datetime(display_df["TIMESTAMP"], errors="coerce")

# Remove invalid timestamps (like 1970 / NaT)
display_df = display_df.dropna(subset=["TIMESTAMP"])

# Remove any accidental old dates
display_df = display_df[
    display_df["TIMESTAMP"].dt.date >= today
]

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
# Extract date
display_df["date"] = display_df["TIMESTAMP"].dt.date

daily = display_df.groupby("date")["PM2.5"].mean().reset_index()
daily["Decision"] = daily["PM2.5"].apply(decision)

fig2 = px.bar(daily, x="date", y="PM2.5", color="Decision")
fig.update_xaxes(
    range=[
        pd.Timestamp(today),
        pd.Timestamp(today + timedelta(days=7))
    ]
)
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

# -------------------------------
# SHAP EXPLAINABILITY
# -------------------------------
st.subheader("🧠 Explain Why Pollution Changed (SHAP)")

if st.checkbox("Show SHAP Explainability"):

    # -------------------------------
    # GLOBAL IMPORTANCE
    # -------------------------------
    st.write("### 🌍 Global Feature Importance")

    sample_X = X.sample(min(300, len(X)))  # speed optimization

    shap_values = explainer(sample_X)

    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig1)

    # -------------------------------
    # LOCAL EXPLANATION
    # -------------------------------
    st.write("### 🔍 Explain Specific Prediction")

    selected_index = st.slider(
        "Select Row Index",
        0,
        len(X) - 1,
        len(X) - 1
    )

    single_row = X.iloc[[selected_index]]
    shap_val = explainer(single_row)

    fig2 = plt.figure()
    shap.plots.waterfall(shap_val[0], show=False)
    st.pyplot(fig2)

    # -------------------------------
    # TOP FEATURES TABLE
    # -------------------------------
    st.write("### 🔥 Top Pollution Drivers")

    vals = np.abs(shap_values.values).mean(axis=0)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Impact": vals
    }).sort_values(by="Impact", ascending=False)

    st.dataframe(importance.head(10))

    # -------------------------------
    # BEESWARM (DISTRIBUTION)
    # -------------------------------
    st.write("### 📊 Feature Impact Distribution")

    fig3 = plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig3)

    # -------------------------------
    # AI SUMMARY
    # -------------------------------
    st.write("### 🤖 AI Insight")

    top_features = importance.head(3)["Feature"].tolist()

    st.success(
        f"Pollution is mainly influenced by: {', '.join(top_features)}"
    )