import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

model = load_model("lstm_model.h5", compile=False)

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)
feature_columns = joblib.load("features.pkl")
scaler = joblib.load("scaler.pkl")

n_past = 7
n_future = 7

def predict_future(input_df):
    scaled = scaler.transform(input_df)
    
    last_seq = scaled[-n_past:]
    last_seq = last_seq.reshape(1, n_past, scaled.shape[1])
    
    pred = model.predict(last_seq)
    pred = pred.reshape(n_future, scaled.shape[1])
    
    pred_actual = scaler.inverse_transform(pred)
    
    pred_df = pd.DataFrame(pred_actual, columns=input_df.columns)
    
    return pred_df