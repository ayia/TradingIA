from flask import Flask, request, jsonify
import os
import joblib
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import load_model
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Define the rename map for incorrect column casing
rename_map = {
    "smA_20": "SMA_20",
    "smA_50": "SMA_50",
    "emA_20": "EMA_20",
    "emA_50": "EMA_50",
    "macD_Signal": "MACD_Signal",
    "macD_Diff": "MACD_Diff"
}

# Function to calculate technical indicators
def calculate_indicators(df):
    if "bollinger_Middle" not in df.columns:
        df['bollinger_Middle'] = df['close'].rolling(window=20).mean()
    if "bollinger_Std" not in df.columns:
        df['bollinger_Std'] = df['close'].rolling(window=20).std()
    if "bollinger_High" not in df.columns:
        df['bollinger_High'] = df['bollinger_Middle'] + (df['bollinger_Std'] * 2)
    if "bollinger_Low" not in df.columns:
        df['bollinger_Low'] = df['bollinger_Middle'] - (df['bollinger_Std'] * 2)

    if "EMA_20" not in df.columns:
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    if "EMA_50" not in df.columns:
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    if "MACD" not in df.columns:
        df['MACD'] = df['EMA_20'] - df['EMA_50']
    if "MACD_Signal" not in df.columns:
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    if "MACD_Diff" not in df.columns:
        df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']

    if "RSI" not in df.columns:
        delta = df['close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

    if "ATR" not in df.columns:
        df['Prev_Close'] = df['close'].shift(1)
        df['TR'] = df.apply(lambda row: max(row['high'] - row['low'], abs(row['high'] - row['Prev_Close']), abs(row['low'] - row['Prev_Close'])), axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df.drop(columns=['Prev_Close', 'TR'], inplace=True, errors='ignore')

    df.dropna(inplace=True)
    return df

# Function to fetch data from the correct URL
def fetch_data(pair_name):
    url = f"https://forex-bestshots-black-meadow-4133.fly.dev/api/FetchPairsData/Lats10IndicatorsBarHistoricaldata?currencyPairs={pair_name}&interval=OneHour"
    try:
        response = requests.get(url, headers={"accept": "text/plain"})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Data fetching failed for {pair_name}: {e}")

# Function for predictions with validation
def predict_next_bar(pair_name, json_data, models_root="Models", window_size=10):
    model_path = os.path.join(models_root, pair_name, "my_lstm_model.h5")
    scaler_features_path = os.path.join(models_root, pair_name, "scaler_features.pkl")
    scaler_target_path = os.path.join(models_root, pair_name, "scaler_target.pkl")

    # Load model and scalers
    try:
        model = load_model(model_path)
        model.compile(optimizer='adam', loss='mean_squared_error')  # Compile model
    except Exception as e:
        raise Exception(f"Model loading failed for {pair_name}: {e}")

    try:
        scaler_features = joblib.load(scaler_features_path)
        scaler_target = joblib.load(scaler_target_path)
    except Exception as e:
        raise Exception(f"Scaler loading failed for {pair_name}: {e}")

    # Prepare the DataFrame
    df = pd.DataFrame(json_data)
    df.rename(columns=rename_map, inplace=True)
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    df.sort_values(by='dateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = calculate_indicators(df)

    feature_cols = [
        "SMA_20", "SMA_50", "EMA_20", "EMA_50",
        "RSI", "MACD", "MACD_Signal", "MACD_Diff",
        "bollinger_High", "bollinger_Low", "ATR",
        "open", "close", "high", "low", "volume"
    ]

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {', '.join(missing_cols)}")

    if df.empty:
        raise ValueError("DataFrame is empty after processing.")

    if len(df) < window_size:
        raise ValueError(f"Not enough data for prediction. Required: {window_size}, Found: {len(df)}")

    feature_data = df[feature_cols].values
    if df[feature_cols].isnull().any().any():
        raise ValueError(f"NaN values found in feature columns: {df[feature_cols].isnull().sum()}")

    feature_data_scaled = scaler_features.transform(feature_data)
    if feature_data_scaled.shape[0] < window_size:
        raise ValueError(f"Insufficient rows for LSTM input: {feature_data_scaled.shape[0]} < {window_size}")

    seq_x = np.array([feature_data_scaled[-window_size:]])

    try:
        pred_scaled = model.predict(seq_x)
        pred = scaler_target.inverse_transform(pred_scaled)
    except Exception as e:
        raise Exception(f"Prediction failed for {pair_name}: {e}")

    open_pred, close_pred, high_pred, low_pred = map(float, pred[0])
    direction = "Buy" if close_pred > open_pred else "Sell"

    latest_row = df.iloc[-1]

    if not (latest_row['bollinger_Low'] <= close_pred <= latest_row['bollinger_High']):
        raise Exception(f"Prediction outside Bollinger Bands for {pair_name}")

    if direction == "Buy" and latest_row['MACD'] <= latest_row['MACD_Signal']:
        raise Exception(f"MACD validation failed for Buy on {pair_name}")
    if direction == "Sell" and latest_row['MACD'] >= latest_row['MACD_Signal']:
        raise Exception(f"MACD validation failed for Sell on {pair_name}")

    if direction == "Buy" and latest_row['RSI'] > 70:
        raise Exception(f"RSI indicates overbought for Buy on {pair_name}")
    if direction == "Sell" and latest_row['RSI'] < 30:
        raise Exception(f"RSI indicates oversold for Sell on {pair_name}")

    return {
        "pair_name": pair_name,
        "direction": direction,
        "open_pred": round(open_pred, 5),
        "close_pred": round(close_pred, 5),
        "high_pred": round(high_pred, 5),
        "low_pred": round(low_pred, 5)
    }

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    pair_names = data.get('pair_names', [])

    if not pair_names or not isinstance(pair_names, list):
        return jsonify({"error": "Missing or invalid 'pair_names' in the request body. Expected a list of pair names."}), 400

    results = []
    for pair_name in pair_names:
        try:
            json_data = fetch_data(pair_name)
            # Make prediction
            prediction = predict_next_bar(pair_name, json_data)
            results.append(prediction)
        except Exception as e:
            logging.error(f"Error processing {pair_name}: {e}")
            continue

    if not results:
        return jsonify({"message": "No valid predictions met the conditions."}), 200

    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)