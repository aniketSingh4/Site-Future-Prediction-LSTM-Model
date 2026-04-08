import streamlit as st
import joblib
from tensorflow.keras.models import load_model

st.title("🌍 AQI 7-Day Forecast Dashboard")

# Load models
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

# Upload or use existing merged_df
if st.button("Predict Next 7 Days"):

    forecast_df = forecast_next_days(xgb_model, merged_df, feature_cols, days=7)

    forecast_df['Safety'] = forecast_df['prediction'].apply(classify_aqi)

    st.subheader("📅 Forecast Results")
    st.dataframe(forecast_df)

    # Chart
    st.line_chart(forecast_df.set_index('date')['prediction'])