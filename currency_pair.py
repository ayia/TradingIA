import os
import sys
import joblib
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import load_model
import tensorflow as tf
import logging

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO and WARNING logs
tf.get_logger().setLevel(logging.ERROR)   # Suppresses TensorFlow logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow root logger

# Suppress Abseil logs
logging.getLogger('absl').setLevel(logging.ERROR)

def fetch_data(pair_name):
    url = f"https://forex-bestshots-black-meadow-4133.fly.dev/api/FetchPairsData/Lats10IndicatorsBarHistoricaldata?currencyPairs={pair_name}&interval=OneHour"
    response = requests.get(url, headers={"accept": "text/plain"})
    response.raise_for_status()
    return response.json()  # Returns JSON data

def calculate_pips(price1, price2, pair_name):
    """
    Calculate the pip difference based on the currency pair conventions.
    """
    multiplier = 100 if "JPY" in pair_name.upper() else 10000
    return abs(price1 - price2) * multiplier

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

    # Determine the direction
    direction = "Buy" if close_pred > open_pred else "Sell"

    # Calculate TP pips
    tp_pips = calculate_pips(open_pred, close_pred, pair_name)

    # Calculate SL pips
    if direction == "Buy":
        sl_pips = calculate_pips(open_pred, low_pred, pair_name)
    else:  # Sell
        sl_pips = calculate_pips(open_pred, high_pred, pair_name)

    # Format the output based on the pair type
    decimal_format = ".5f" if "JPY" not in pair_name.upper() else ".3f"

    print(f"Prediction for {pair_name}:")
    print("  Direction:", direction)
    print(f"  TP Pip Difference: {tp_pips:.2f} pips")
    print(f"  SL Pip Difference: {sl_pips:.2f} pips")

    
    
    print(f"  Open : {open_pred:{decimal_format}}")
    print(f"  Close: {close_pred:{decimal_format}}")
    print(f"  High : {high_pred:{decimal_format}}")
    print(f"  Low  : {low_pred:{decimal_format}}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <currency_pair>")
        sys.exit(1)
    
    pair_name = sys.argv[1]
    predict_next_bar(pair_name, models_root="Models", window_size=10)
