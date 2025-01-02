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

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns for {pair_name}: {', '.join(missing_cols)}")
        return {"pair_name": pair_name, "error": f"Missing columns: {', '.join(missing_cols)}"}

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

    # Validate prediction direction with indicators
    latest_row = df.iloc[-1]  # Get the last row of processed data

    if not (latest_row['bollinger_Low'] <= close_pred <= latest_row['bollinger_High']):
        return {
            "pair_name": pair_name,
            "error": f"Prediction outside Bollinger Bands for {pair_name}",
            "predicted_direction": direction,
            "indicators_suggestion": "Invalid Bollinger Band alignment"
        }

    if direction == "Buy" and latest_row['MACD'] <= latest_row['MACD_Signal']:
        return {
            "pair_name": pair_name,
            "error": f"MACD validation failed for Buy on {pair_name}",
            "predicted_direction": direction,
            "indicators_suggestion": "MACD does not confirm Buy"
        }

    if direction == "Sell" and latest_row['MACD'] >= latest_row['MACD_Signal']:
        return {
            "pair_name": pair_name,
            "error": f"MACD validation failed for Sell on {pair_name}",
            "predicted_direction": direction,
            "indicators_suggestion": "MACD does not confirm Sell"
        }

    if direction == "Buy" and latest_row['RSI'] > 70:
        return {
            "pair_name": pair_name,
            "error": f"RSI indicates overbought for Buy on {pair_name}",
            "predicted_direction": direction,
            "indicators_suggestion": "RSI does not confirm Buy"
        }

    if direction == "Sell" and latest_row['RSI'] < 30:
        return {
            "pair_name": pair_name,
            "error": f"RSI indicates oversold for Sell on {pair_name}",
            "predicted_direction": direction,
            "indicators_suggestion": "RSI does not confirm Sell"
        }

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
            prediction = predict_next_bar(pair_name, json_data)
            
            # Add only matched predictions
            if "error" not in prediction:
                results.append(prediction)
        except Exception as e:
            logging.error(f"Error processing {pair_name}: {e}")

    if not results:
        return jsonify({"message": "No valid predictions met the conditions."}), 200

    return jsonify(results)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
