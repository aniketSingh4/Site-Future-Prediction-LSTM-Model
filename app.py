import streamlit as st
import pandas as pd
import numpy as np
import joblib
import smtplib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="Construction Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================
# Login for Dashboard
# =============================
# =========================
# LOGIN SYSTEM
# =========================
import streamlit as st

# Dummy users (replace later with DB)
USERS = {
    "admin": "admin123",
    "manager": "manager123",
    "aniket": "aniket123"
}

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login function
def login():
    st.markdown("## 🔐 Login to Dashboard")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.success("✅ Login successful")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")


# ===============================================================
# Add Logout button
# ===============================================================
col1, col2 = st.columns([8, 1])

with col2:
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

# ===================================
# Adding Logging Check Before Dashboard
# =======================================
# If not logged in → show login page
if not st.session_state.logged_in:
    login()
    st.stop()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Construction Dashboard", layout="wide")

# =========================
# LOAD MODEL + SCALER
# =========================
model = load_model("lstm_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# =========================
# TITLE
# =========================
st.markdown("## 🏗️ Construction Safety Dashboard")
st.markdown("---")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("merged_df.csv")

df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
df = df.sort_values('TIMESTAMP')
df.set_index('TIMESTAMP', inplace=True)

# =========================
# FEATURES
# =========================
features = [
    'PM2.5', 'PM10', 'TEMPERATURE', 'HUMIDITY',
    'WS', 'WD', 'RF', 'NO2', 'Ozone'
]

data = df[features]

# =========================
# SCALE DATA
# =========================
scaled_data = scaler.transform(data)

# =========================
# LAST 72 HOURS
# =========================
if len(data) < 72:
    st.error("❌ Not enough data (minimum 72 rows required)")
    st.stop()

last_72 = scaled_data[-72:].reshape(1, 72, len(features))

# =========================
# PREDICT
# =========================
future = model.predict(last_72).reshape(24, 2)

# =========================
# INVERSE SCALING
# =========================
dummy = np.zeros((24, len(features)))
dummy[:, 0:2] = future

future_actual = scaler.inverse_transform(dummy)

pm25_values = future_actual[:, 0]
pm10_values = future_actual[:, 1]

# =========================
# AVERAGE VALUES
# =========================
pm25_next = pm25_values.mean()
pm10_next = pm10_values.mean()

# =========================
# DECISION FUNCTION
# =========================
def decision(pm25, pm10):
    return 0 if (pm25 > 100 or pm10 > 150) else 1

# =========================
# CONFIDENCE FUNCTIONS
# =========================
def confidence_score(pm25, pm10):
    PM25_LIMIT = 100
    PM10_LIMIT = 150
    
    pm25_diff = abs(pm25 - PM25_LIMIT) / PM25_LIMIT
    pm10_diff = abs(pm10 - PM10_LIMIT) / PM10_LIMIT
    
    return round(((pm25_diff + pm10_diff) / 2) * 100, 2)

def hourly_confidence(pm25_arr, pm10_arr):
    PM25_LIMIT = 100
    PM10_LIMIT = 150
    
    confidences = []
    
    for pm25, pm10 in zip(pm25_arr, pm10_arr):
        risk = (pm25/PM25_LIMIT + pm10/PM10_LIMIT) / 2
        distance = abs(risk - 1)
        conf = min(distance * 100, 100)
        confidences.append(conf)
    
    return confidences

conf = confidence_score(pm25_next, pm10_next)
conf_per_hour = hourly_confidence(pm25_values, pm10_values)

# =========================
# TIME LABELS (TOMORROW)
# =========================
start_time = (datetime.now() + timedelta(days=1)).replace(
    hour=0, minute=0, second=0, microsecond=0
)

formatted_date = start_time.strftime("%d %B %Y")

time_labels = [
    (start_time + timedelta(hours=i)).strftime("%d-%b %H:%M")
    for i in range(24)
]

# =========================
# DATAFRAME FOR GRAPH
# =========================
plot_df = pd.DataFrame({
    "Time": time_labels,
    "PM2.5": pm25_values,
    "PM10": pm10_values,
    "Confidence": conf_per_hour
})

plot_df["Risk"] = plot_df["Confidence"].apply(
    lambda x: "🔴 High Risk" if x < 30 else "🟢 Safe"
)

# =========================
# BEST WORKING HOURS
# =========================
safe_hours = plot_df[plot_df["Confidence"] > 50]

st.markdown("### 🕒 Best Working Hours")

if not safe_hours.empty:
    best_start = safe_hours.iloc[0]["Time"]
    best_end = safe_hours.iloc[-1]["Time"]
    
    st.success(f"✅ Recommended Work Window: {best_start} → {best_end}")
else:
    st.error("❌ No safe working hours tomorrow")

# Color Coded Graph
plot_df["Zone"] = plot_df["Confidence"].apply(
    lambda x: "🟢 Safe" if x > 60 else ("🟡 Moderate" if x > 30 else "🔴 Unsafe")
)

# =========================
# UI: METRICS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("PM2.5 (Tomorrow)", round(pm25_next, 2))

with col2:
    st.metric("PM10 (Tomorrow)", round(pm10_next, 2))

with col3:
    st.metric("Confidence", f"{conf}%")

# =========================
# CONFIDENCE BAR
# =========================
st.markdown("### 🔍 Confidence Level")
st.progress(conf / 100)

if conf > 70:
    st.success("🟢 High Confidence")
elif conf > 40:
    st.warning("🟡 Medium Confidence")
else:
    st.error("🔴 Low Confidence")

# =========================
# DECISION
# =========================
st.markdown("### 🚧 Construction Decision")

if decision(pm25_next, pm10_next):
    st.success("✅ SAFE TO CONTINUE WORK")
else:
    st.error("❌ STOP WORK – Unsafe Air Quality")

# =========================
# DATE
# =========================
st.markdown(f"### 📅 Forecast Date: **{formatted_date}**")

# =========================
# GRAPHS
# =========================
st.markdown("### 📈 24-Hour Air Quality Forecast")
st.line_chart(plot_df.set_index("Time")[["PM2.5", "PM10"]])

st.markdown("### 📊 Confidence Trend")
st.line_chart(plot_df.set_index("Time")[["Confidence"]])

# =========================
# RISK TABLE
# =========================
st.markdown("### ⚠️ Risk Analysis")
st.dataframe(plot_df, use_container_width=True)

risky_hours = plot_df[plot_df["Confidence"] < 30]

if not risky_hours.empty:
    st.warning(f"⚠️ {len(risky_hours)} risky hours detected tomorrow")
else:
    st.success("✅ No risky hours detected")

# =========================
# ALERT FUNCTION
# =========================
def send_alert():
    sender = "your_email@gmail.com"
    receiver = "receiver@gmail.com"
    password = "your_app_password"

    message = "Subject: Construction Alert\n\nStop work tomorrow due to high pollution."

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, message)

# =========================
# TRIGGER ALERT
# =========================
if not decision(pm25_next, pm10_next):
    send_alert()