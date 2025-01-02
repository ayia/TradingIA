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
    logging.info(f"Starting indicator calculation. Initial rows: {len(df)}")
    if len(df) < 20:
        logging.error("Insufficient rows for indicator calculation. Skipping.")
        return pd.DataFrame()  # Return an empty DataFrame if not enough rows

    try:
        df['bollinger_Middle'] = df['close'].rolling(window=20).mean()
        df['bollinger_Std'] = df['close'].rolling(window=20).std()
        df['bollinger_High'] = df['bollinger_Middle'] + (df['bollinger_Std'] * 2)
        df['bollinger_Low'] = df['bollinger_Middle'] - (df['bollinger_Std'] * 2)

        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

        df['MACD'] = df['EMA_20'] - df['EMA_50']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']

        delta = df['close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['Prev_Close'] = df['close'].shift(1)
        df['TR'] = df.apply(lambda row: max(row['high'] - row['low'], abs(row['high'] - row['Prev_Close']), abs(row['low'] - row['Prev_Close'])), axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df.drop(columns=['Prev_Close', 'TR'], inplace=True, errors='ignore')

        df.dropna(inplace=True)
        logging.info(f"Finished indicator calculation. Remaining rows: {len(df)}")
    except Exception as e:
        logging.error(f"Error during indicator calculation: {e}")
        return pd.DataFrame()

    return df

# Function to fetch data from the correct URL
def fetch_data(pair_name):
    url = f"https://forex-bestshots-black-meadow-4133.fly.dev/api/FetchPairsData/IndicatorsBarHistoricaldata?currencyPairs={pair_name}&interval=OneHour&barsnumber=30"
    try:
        response = requests.get(url, headers={"accept": "text/plain"})
        response.raise_for_status()
        json_data = response.json()
        logging.info(f"API response for {pair_name}: {json_data}")
        return json_data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data for {pair_name}: {e}")
        raise Exception(f"Data fetching failed for {pair_name}")

# Function for predictions with validation
def predict_next_bar(pair_name, json_data, models_root="Models", window_size=10):
    model_path = os.path.join(models_root, pair_name, "my_lstm_model.keras")
    scaler_features_path = os.path.join(models_root, pair_name, "scaler_features.pkl")
    scaler_target_path = os.path.join(models_root, pair_name, "scaler_target.pkl")

    # Load model and scalers
    try:
        model = load_model(model_path)
        model.compile(optimizer='adam', loss='mean_squared_error')
    except Exception as e:
        logging.error(f"Model loading failed for {pair_name}: {e}")
        return {"pair_name": pair_name, "error": f"Model not found for {pair_name}"}

    try:
        scaler_features = joblib.load(scaler_features_path)
        scaler_target = joblib.load(scaler_target_path)
    except Exception as e:
        logging.error(f"Scaler loading failed for {pair_name}: {e}")
        return {"pair_name": pair_name, "error": f"Scalers not found for {pair_name}"}

    # Prepare the DataFrame
    df = pd.DataFrame(json_data)
    logging.info(f"Raw API data for {pair_name}: {df}")
    if df.empty:
        return {"pair_name": pair_name, "error": "No data available from API"}

    df.rename(columns=rename_map, inplace=True)
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    df.sort_values(by='dateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Calculate indicators for all rows
    df = calculate_indicators(df)
    if df.empty:
        logging.error(f"DataFrame is empty after indicator calculation for {pair_name}")
        return {"pair_name": pair_name, "error": "Insufficient data after preprocessing"}

    feature_cols = [
        "SMA_20", "SMA_50", "EMA_20", "EMA_50",
        "RSI", "MACD", "MACD_Signal", "MACD_Diff",
        "bollinger_High", "bollinger_Low", "ATR",
        "open", "close", "high", "low", "volume"
    ]

    if len(df) < window_size:
        logging.error(f"Not enough data after preprocessing for prediction. Required: {window_size}, Found: {len(df)}")
        return {"pair_name": pair_name, "error": f"Not enough data for prediction. Required: {window_size}, Found: {len(df)}"}

    # Use only the last 10 rows for prediction
    feature_data = df[feature_cols].tail(window_size).values
    feature_data_scaled = scaler_features.transform(feature_data)
    seq_x = np.array([feature_data_scaled])

    try:
        pred_scaled = model.predict(seq_x)
        pred = scaler_target.inverse_transform(pred_scaled)
    except Exception as e:
        logging.error(f"Prediction failed for {pair_name}: {e}")
        return {"pair_name": pair_name, "error": "Prediction failed"}

    open_pred, close_pred, high_pred, low_pred = map(float, pred[0])
    direction = "Buy" if close_pred > open_pred else "Sell"

    # Determine pips multiplier based on currency pair
    pips_multiplier = 100 if "JPY" in pair_name else 10000

    # Calculate TP and SL pips
    tp_pips = abs(close_pred - open_pred) * pips_multiplier
    if direction == "Buy":
        sl_pips = abs(open_pred - low_pred) * pips_multiplier
    else:  # direction == "Sell"
        sl_pips = abs(high_pred - open_pred) * pips_multiplier

    # Calculate risk-reward ratio
    if sl_pips > 0:  # Avoid division by zero
        reward_ratio = tp_pips / sl_pips
        risk_reward = f"1:{round(reward_ratio, 2)}"
    else:
        risk_reward = "Undefined"

    return {
        "pair_name": pair_name,
        "direction": direction,
        "open_pred": round(open_pred, 5),
        "close_pred": round(close_pred, 5),
        "high_pred": round(high_pred, 5),
        "low_pred": round(low_pred, 5),
        "tp_pips": round(tp_pips, 2),
        "sl_pips": round(sl_pips, 2),
        "risk_reward": risk_reward
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
            prediction = predict_next_bar(pair_name, json_data)

            # Filter by risk-reward ratio >= 1
            if "risk_reward" in prediction and ":" in prediction["risk_reward"]:
                risk, reward = map(float, prediction["risk_reward"].split(":"))
                if reward >= 3:
                    results.append(prediction)
        except Exception as e:
            logging.error(f"Error processing {pair_name}: {e}")

    if not results:
        return jsonify({"message": "No valid predictions met the conditions."}), 200

    return jsonify(results)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
