import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timedelta
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="BuildSafe AI", layout="wide")

# =========================
# USER AUTH SYSTEM
# =========================
USER_DB = "users.json"

def load_users():
    if os.path.exists(USER_DB):
        with open(USER_DB, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)

def login(username, password):
    users = load_users()
    return username in users and users[username] == password

def signup(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = password
    save_users(users)
    return True

# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =========================
# LOGIN / SIGNUP UI
# =========================
if not st.session_state.logged_in:

    st.title("🔐 BuildSafe AI Login")

    menu = st.radio("Select Option", ["Login", "Signup"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if menu == "Login":
        if st.button("Login"):
            if login(username, password):
                st.session_state.logged_in = True
                st.success("Login Successful ✅")
                st.rerun()
            else:
                st.error("Invalid credentials ❌")

    else:
        if st.button("Signup"):
            if signup(username, password):
                st.success("Account created! Please login ✅")
            else:
                st.warning("User already exists ⚠️")

    st.stop()

# =========================
# LOGOUT BUTTON
# =========================
with st.sidebar:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# =========================
# LOAD MODEL (CORRECT WAY)
# =========================
@st.cache_resource
def load_model_bundle():
    return joblib.load("model_bundle.pkl")

bundle = load_model_bundle()

model_pm25 = bundle["model_pm25"]
model_pm10 = bundle["model_pm10"]
features = bundle["features"]

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("cleaned_ashapura_data.csv")

df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
df = df.sort_values('TIMESTAMP')
df.set_index('TIMESTAMP', inplace=True)

X = df[features]

# =========================
# FUTURE PREDICTION
# =========================
def predict_next_24(model_pm25, model_pm10, X, features):

    future_pm25 = []
    future_pm10 = []
    
    last_row = X.iloc[-1:].copy()

    for i in range(24):

        last_row = last_row[features]

        pred25 = model_pm25.predict(last_row)[0]
        pred10 = model_pm10.predict(last_row)[0]

        future_pm25.append(pred25)
        future_pm10.append(pred10)

        lag_list = [24, 12, 6, 3, 2, 1]

        for j in range(len(lag_list)-1, 0, -1):
            curr = lag_list[j-1]
            prev = lag_list[j]
            
            last_row[f'PM25_lag_{curr}'] = last_row[f'PM25_lag_{prev}']
            last_row[f'PM10_lag_{curr}'] = last_row[f'PM10_lag_{prev}']

        last_row['PM25_lag_1'] = pred25
        last_row['PM10_lag_1'] = pred10

    return future_pm25, future_pm10

future_pm25, future_pm10 = predict_next_24(model_pm25, model_pm10, X, features)

# =========================
# DECISION + CONFIDENCE
# =========================
PM25_LIMIT = 100
PM10_LIMIT = 150

pm25_avg = np.mean(future_pm25)
pm10_avg = np.mean(future_pm10)

def decision(pm25, pm10):
    return pm25 < PM25_LIMIT and pm10 < PM10_LIMIT

def confidence(pm25, pm10):
    risk = (pm25/PM25_LIMIT + pm10/PM10_LIMIT)/2
    return max(0, round((1 - risk) * 100, 2))

conf = confidence(pm25_avg, pm10_avg)

# =========================
# HEADER UI
# =========================
st.title("🏗️ BuildSafe AI Dashboard - XGBoost Model")

tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d %B %Y")

col1, col2, col3 = st.columns(3)
col1.metric("PM2.5", round(pm25_avg, 2))
col2.metric("PM10", round(pm10_avg, 2))
col3.metric("Confidence", f"{conf:.2f}%")

# =========================
# DECISION UI (COLOR BASED)
# =========================
st.subheader(f"📅 Decision for {tomorrow}")

if decision(pm25_avg, pm10_avg):
    st.success("✅ SAFE TO CONTINUE WORK")
else:
    st.error("❌ STOP WORK (High Pollution)")

# =========================
# INTERACTIVE GRAPH
# =========================
hours = list(range(24))

plot_df = pd.DataFrame({
    "Hour": hours,
    "PM2.5": future_pm25,
    "PM10": future_pm10
})

# COLOR ZONES
def get_zone(pm25):
    if pm25 < 60:
        return "Good"
    elif pm25 < 100:
        return "Moderate"
    else:
        return "Unsafe"

plot_df["Zone"] = plot_df["PM2.5"].apply(get_zone)

fig = px.line(
    plot_df,
    x="Hour",
    y=["PM2.5", "PM10"],
    title="📊 24-Hour Pollution Forecast",
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# BEST WORK HOURS
# =========================
safe_hours = plot_df[
    (plot_df["PM2.5"] < PM25_LIMIT) &
    (plot_df["PM10"] < PM10_LIMIT)
]

st.subheader("🟢 Best Working Hours")

if not safe_hours.empty:
    # Get tomorrow's date
    start_date = (datetime.now() + timedelta(days=1)).strftime("%d-%b")
    
    # First and last safe hour
    start_hour = int(safe_hours.iloc[0]["Hour"])
    end_hour = int(safe_hours.iloc[-1]["Hour"])
    
    start = f"{start_date} {start_hour:02d}:00"
    end = f"{start_date} {end_hour:02d}:00"
    
    st.success(f"✅ Recommended Work Window: {start} → {end}")
else:
    st.warning("❌ No safe hours tomorrow")


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Controls")
    

    auto_refresh = st.checkbox("🔄 Auto Refresh (1 hour)")

with st.sidebar:
    st.header("⚙️ Options")

    PM25_LIMIT = st.slider("PM2.5 Limit", 50, 200, 100)
    PM10_LIMIT = st.slider("PM10 Limit", 50, 300, 150)

    st.info("Adjust limits based on site rules")

# =========================
# AUTO REFRESH
# =========================
if auto_refresh:
    import time
    time.sleep(3600)
    st.rerun()