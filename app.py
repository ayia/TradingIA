from flask import Flask, request, jsonify
import os
import joblib
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import load_model
import tensorflow as tf
import logging

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

app = Flask(__name__)

def fetch_data(pair_name):
    url = f"https://forex-bestshots-black-meadow-4133.fly.dev/api/FetchPairsData/Lats10IndicatorsBarHistoricaldata?currencyPairs={pair_name}&interval=OneHour"
    response = requests.get(url, headers={"accept": "text/plain"})
    response.raise_for_status()
    return response.json()

def calculate_pips(price1, price2, pair_name):
    multiplier = 100 if "JPY" in pair_name.upper() else 10000
    return abs(price1 - price2) * multiplier

def predict_next_bar(pair_name, models_root="Models", window_size=10):
    json_data = fetch_data(pair_name)

    model_path = os.path.join(models_root, pair_name, "my_lstm_model.h5")
    scaler_features_path = os.path.join(models_root, pair_name, "scaler_features.pkl")
    scaler_target_path = os.path.join(models_root, pair_name, "scaler_target.pkl")

    model = load_model(model_path)
    scaler_features = joblib.load(scaler_features_path)
    scaler_target = joblib.load(scaler_target_path)

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

    seq_x = np.array([feature_data_scaled[-window_size:]])
    pred_scaled = model.predict(seq_x)
    pred = scaler_target.inverse_transform(pred_scaled)

    open_pred, close_pred, high_pred, low_pred = map(float, pred[0])  # Convert to Python float
    direction = "Buy" if close_pred > open_pred else "Sell"

    # Validation based on Bollinger Bands
    latest_row = df.iloc[-1]
    bollinger_high = latest_row['bollinger_High']
    bollinger_low = latest_row['bollinger_Low']

    if not (bollinger_low <= close_pred <= bollinger_high):
        return {
            "error": f"Prediction for {pair_name} is outside Bollinger Bands",
            "close_pred": round(close_pred, 5),
            "bollinger_high": round(bollinger_high, 5),
            "bollinger_low": round(bollinger_low, 5),
        }

    # Validation based on MACD Cross
    macd = latest_row['macd']
    macd_signal = latest_row['macD_Signal']

    if direction == "Buy" and macd <= macd_signal:
        return {"error": f"MACD validation failed for Buy signal on {pair_name}"}
    if direction == "Sell" and macd >= macd_signal:
        return {"error": f"MACD validation failed for Sell signal on {pair_name}"}

    # Validation based on RSI
    rsi = latest_row['rsi']
    if direction == "Buy" and rsi > 70:
        return {"error": f"RSI indicates overbought conditions for Buy on {pair_name}"}
    if direction == "Sell" and rsi < 30:
        return {"error": f"RSI indicates oversold conditions for Sell on {pair_name}"}

    tp_pips = float(calculate_pips(open_pred, close_pred, pair_name))  # Convert to Python float
    if direction == "Buy":
        sl_pips = float(calculate_pips(open_pred, low_pred, pair_name))
    else:
        sl_pips = float(calculate_pips(open_pred, high_pred, pair_name))

    risk_reward_ratio = tp_pips / sl_pips if sl_pips > 0 else None

    return {
        "pair_name": pair_name,
        "direction": direction,
        "tp_pips": round(tp_pips, 2),
        "sl_pips": round(sl_pips, 2),
        "risk_reward_ratio": f"1:{round(risk_reward_ratio, 2)}" if risk_reward_ratio else "Undefined",
        "open_pred": round(open_pred, 5),
        "close_pred": round(close_pred, 5),
        "high_pred": round(high_pred, 5),
        "low_pred": round(low_pred, 5)
    }

def predict_next_bar_from_json(pair_name, json_data, models_root="Models", window_size=10):
    model_path = os.path.join(models_root, pair_name, "my_lstm_model.h5")
    scaler_features_path = os.path.join(models_root, pair_name, "scaler_features.pkl")
    scaler_target_path = os.path.join(models_root, pair_name, "scaler_target.pkl")

    # Load the model and scalers
    model = load_model(model_path)
    scaler_features = joblib.load(scaler_features_path)
    scaler_target = joblib.load(scaler_target_path)

    # Prepare the DataFrame
    df = pd.DataFrame(json_data)
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    df.sort_values(by='dateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Define feature and target columns
    feature_cols = [
        "smA_20", "smA_50", "emA_20", "emA_50",
        "rsi", "macd", "macD_Signal", "macD_Diff",
        "bollinger_High", "bollinger_Low", "atr",
        "open", "close", "high", "low", "volume"
    ]
    target_cols = ["open", "close", "high", "low"]

    # Scale features and make predictions
    feature_data = df[feature_cols].values
    feature_data_scaled = scaler_features.transform(feature_data)

    seq_x = np.array([feature_data_scaled[-window_size:]])
    pred_scaled = model.predict(seq_x)
    pred = scaler_target.inverse_transform(pred_scaled)

    open_pred, close_pred, high_pred, low_pred = map(float, pred[0])  # Convert to Python float
    direction = "Buy" if close_pred > open_pred else "Sell"

    # Validation based on Bollinger Bands
    latest_row = df.iloc[-1]
    bollinger_high = latest_row['bollinger_High']
    bollinger_low = latest_row['bollinger_Low']

    if not (bollinger_low <= close_pred <= bollinger_high):
        return {
            "error": f"Prediction for {pair_name} is outside Bollinger Bands",
            "close_pred": round(close_pred, 5),
            "bollinger_high": round(bollinger_high, 5),
            "bollinger_low": round(bollinger_low, 5),
        }

    # Validation based on MACD Cross
    macd = latest_row['macd']
    macd_signal = latest_row['macD_Signal']

    if direction == "Buy" and macd <= macd_signal:
        return {"error": f"MACD validation failed for Buy signal on {pair_name}"}
    if direction == "Sell" and macd >= macd_signal:
        return {"error": f"MACD validation failed for Sell signal on {pair_name}"}

    # Validation based on RSI
    rsi = latest_row['rsi']
    if direction == "Buy" and rsi > 70:
        return {"error": f"RSI indicates overbought conditions for Buy on {pair_name}"}
    if direction == "Sell" and rsi < 30:
        return {"error": f"RSI indicates oversold conditions for Sell on {pair_name}"}

    tp_pips = float(calculate_pips(open_pred, close_pred, pair_name))  # Convert to Python float
    if direction == "Buy":
        sl_pips = float(calculate_pips(open_pred, low_pred, pair_name))
    else:
        sl_pips = float(calculate_pips(open_pred, high_pred, pair_name))

    risk_reward_ratio = tp_pips / sl_pips if sl_pips > 0 else None

    return {
        "pair_name": pair_name,
        "direction": direction,
        "tp_pips": round(tp_pips, 2),
        "sl_pips": round(sl_pips, 2),
        "risk_reward_ratio": f"1:{round(risk_reward_ratio, 2)}" if risk_reward_ratio else "Undefined",
        "open_pred": round(open_pred, 5),
        "close_pred": round(close_pred, 5),
        "high_pred": round(high_pred, 5),
        "low_pred": round(low_pred, 5)
    }

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    pair_names = data.get('pair_names')  # Note: Expecting a list of pair names

    if not pair_names or not isinstance(pair_names, list):
        return jsonify({"error": "Missing or invalid 'pair_names' in the request body. Expected a list of pair names."}), 400

    results = []
    try:
        for pair_name in pair_names:
            prediction = predict_next_bar(pair_name)  # Call your existing function

            # Skip predictions with validation errors
            if "error" in prediction:
                continue

            # Extract the numeric risk/reward ratio
            risk_reward = float(prediction['risk_reward_ratio'].split(':')[1])
            
            # Filter based on the condition risk/reward > 2
            if risk_reward > 1:
                results.append(prediction)
                
        if not results:
            return jsonify({"message": "No valid predictions met the conditions."}), 200
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
       
@app.route('/predictfromjson', methods=['POST'])
def predict_from_json():
    data = request.json
    
    # Validate input
    pair_name = data.get('pair_name')
    json_data = data.get('json_data')

    if not pair_name or not json_data:
        return jsonify({"error": "Missing 'pair_name' or 'json_data' in the request body"}), 400

    try:
        # Call the prediction logic with the provided JSON data
        result = predict_next_bar_from_json(pair_name, json_data)

        # Exclude predictions that contain validation errors
        if "error" in result:
            return jsonify({"message": f"Prediction for {pair_name} did not meet validation criteria.", "error": result["error"]}), 200

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)