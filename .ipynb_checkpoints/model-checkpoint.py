import streamlit as st
import pandas as pd
from predict import predict_future

st.title("Air Quality Prediction")

# Load your latest data
df = pd.read_csv("model_data.csv")

pred_df = predict_future(df)

st.subheader("7 Day Prediction")
st.dataframe(pred_df)

st.line_chart(pred_df[['PM2.5', 'AQI']])