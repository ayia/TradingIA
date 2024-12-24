import os
import sys
import joblib
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import load_model

def fetch_data(pair_name):
    url = f"https://forex-bestshots-black-meadow-4133.fly.dev/api/FetchPairsData/Lats10IndicatorsBarHistoricaldata?currencyPairs={pair_name}&interval=OneHour"
    response = requests.get(url, headers={"accept": "text/plain"})
    response.raise_for_status()
    return response.json()  # Returns JSON data

def predict_next_bar(pair_name, models_root="Models", window_size=10):
    # Fetch the JSON data from the API
    json_data = fetch_data(pair_name)

    # Load the model and scalers
    model_path = os.path.join(models_root, pair_name, "my_lstm_model.h5")
    scaler_features_path = os.path.join(models_root, pair_name, "scaler_features.pkl")
    scaler_target_path = os.path.join(models_root, pair_name, "scaler_target.pkl")

    model = load_model(model_path)
    scaler_features = joblib.load(scaler_features_path)
    scaler_target = joblib.load(scaler_target_path)

    # Process the JSON data into a DataFrame
    df = pd.DataFrame(json_data)
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    df.sort_values(by='dateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_cols = [
        "smA_20", "smA_50", "emA_20", "emA_50", 
        "rsi", "macd", "macD_Signal", "macD_Diff",
        "bollinger_High", "bollinger_Low", "atr",
        "open", "close", "high", "low", "volume"
    ]
    target_cols = ["open", "close", "high", "low"]

    feature_data = df[feature_cols].values
    feature_data_scaled = scaler_features.transform(feature_data)

    # Take the last window (10 latest records) for prediction
    seq_x = feature_data_scaled[-window_size:]
    seq_x = np.array([seq_x])  # shape: (1, 10, nb_features)

    # Make predictions
    pred_scaled = model.predict(seq_x)  # shape (1, 4)
    pred = scaler_target.inverse_transform(pred_scaled)

    # pred = [ [open, close, high, low] ]
    open_pred, close_pred, high_pred, low_pred = pred[0]
    print(f"Prediction for {pair_name}:")
    print("  Open :", open_pred)
    print("  Close:", close_pred)
    print("  High :", high_pred)
    print("  Low  :", low_pred)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <currency_pair>")
        sys.exit(1)
    
    pair_name = sys.argv[1]
    predict_next_bar(pair_name, models_root="Models", window_size=10)
