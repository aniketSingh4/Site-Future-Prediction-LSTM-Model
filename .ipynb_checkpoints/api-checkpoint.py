from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = FastAPI()

# Load models
xgb_model = joblib.load("xgb_model.pkl")
lstm_model = load_model("lstm_model.h5")

SEQ_LENGTH = 24

# Decision logic
def construction_decision(pm25):
    if pm25 < 100:
        return "Allowed"
    elif pm25 < 200:
        return "Restricted"
    else:
        return "Not Allowed"


@app.get("/")
def home():
    return {"message": "Air Quality Prediction API Running"}

# 🔹 Predict current
@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    
    pred = xgb_model.predict(df)[0]
    decision = construction_decision(pred)

    return {
        "PM2.5_prediction": float(pred),
        "decision": decision
    }


# 🔹 7-day forecast (LSTM)
@app.post("/forecast")
def forecast(data: list):
    seq = np.array(data)

    preds = []
    current = seq.copy()

    for _ in range(168):
        pred = lstm_model.predict(current.reshape(1, SEQ_LENGTH, seq.shape[1]))[0][0]
        preds.append(float(pred))

        current = np.roll(current, -1, axis=0)
        current[-1] = pred

    decisions = [construction_decision(p) for p in preds]

    return {
        "forecast": preds,
        "decisions": decisions
    }

@app.post("/explain")
def explain(data: dict):
    df = pd.DataFrame([data])
    shap_values = explainer(df)

    return {"importance": shap_values.values.tolist()}