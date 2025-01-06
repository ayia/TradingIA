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

rename_map = {
    "smA_20": "SMA_20",
    "smA_50": "SMA_50",
    "emA_20": "EMA_20",
    "emA_50": "EMA_50",
    "macD_Signal": "MACD_Signal",
    "macD_Diff": "MACD_Diff"
}



def fetch_data(pair_name):
    url = f"https://forex-bestshots-black-meadow-4133.fly.dev/api/FetchPairsData/IndicatorsBarHistoricaldata?currencyPairs={pair_name}&interval=OneHour&barsnumber=10"
    try:
        response = requests.get(url, headers={"accept": "text/plain"})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur lors de la récupération des données pour {pair_name} : {e}")
        raise Exception(f"Échec de la récupération des données pour {pair_name}")

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

    # Ensure enough data is available for prediction
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
        sl_pips = max(abs(open_pred - low_pred) * pips_multiplier, 1)  # Ensure sl_pips is not zero
    else:  # direction == "Sell"
        sl_pips = max(abs(high_pred - open_pred) * pips_multiplier, 1)  # Ensure sl_pips is not zero

    # Calculate risk-reward ratio
    reward_ratio = tp_pips / sl_pips
    if reward_ratio >= 1.5:  # Define a valid risk-reward threshold
        risk_reward = f"1:{round(reward_ratio, 2)}"
    else:
        risk_reward = "1:1"  # Default to a conservative ratio if below threshold

    logging.info(f"{pair_name} Prediction - Direction: {direction}, TP Pips: {tp_pips}, SL Pips: {sl_pips}, Risk-Reward: {risk_reward}")

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
    pair_names = [
        "AUDJPY", "AUDUSD", "CHFJPY", "EURCAD", "EURCHF", "EURGBP", "EURJPY", 
        "EURUSD", "GBPCHF", "GBPJPY", "GBPUSD", "NZDJPY", "NZDUSD", "USDCAD", 
        "USDCHF", "USDJPY", "AUDCAD", "AUDCHF", "AUDNZD", "CADCHF", "CADJPY", 
        "EURAUD", "EURNZD", "GBPAUD", "GBPCAD", "GBPNZD", "NZDCAD", "NZDCHF"
    ]
    results = []
    for pair_name in pair_names:
        try:
            json_data = fetch_data(pair_name)
            prediction = predict_next_bar(pair_name, json_data)
            if "error" not in prediction and ":" in prediction.get("risk_reward", ""):
                risk, reward = map(float, prediction["risk_reward"].split(":"))
                if (reward >= 2 and reward < 5):
                    prediction["reward"] = reward  # Add reward to the prediction dict for sorting
                    results.append(prediction)
        except Exception as e:
            logging.error(f"Erreur pour {pair_name} : {e}")

    if not results:
        return jsonify({"message": "Aucune prédiction valide."}), 200

    # Sort the results by reward in descending order
    results = sorted(results, key=lambda x: x["reward"], reverse=True)

    return jsonify(results)

@app.route('/predictCtreader', methods=['POST'])
def predictCtreader():
    response = predict()
    if isinstance(response, tuple):
        return response[0], response[1]
    results = response.json if hasattr(response, 'json') else response
    if "message" in results:
        return "No valid predictions met the conditions."
    formatted_results = [
        f"{result['pair_name']}|{result['direction']}|{result['tp_pips']}|{result['sl_pips']}"
        for result in results
    ]
    return "@".join(formatted_results)


@app.route('/predict_pairCtreder', methods=['POST'])
def predict_pairCtreder():
    """
    Predict the next bar for a specific pair given its data.
    """
    try:
        # Parse the request JSON
        request_data = request.get_json()
        if not request_data or 'pair_name' not in request_data or 'json_data' not in request_data:
            return jsonify({"error": "Invalid request. 'pair_name' and 'data' are required."}), 400

        pair_name = request_data['pair_name']
        json_data = request_data['json_data']

        # Call the predict_next_bar function
        prediction = predict_next_bar(pair_name, json_data)

        # Return the prediction result
        if "error" in prediction:
            return jsonify({"error": prediction["error"]}), 400
        return jsonify(prediction), 200

    except Exception as e:
        logging.error(f"Error during prediction for pair {request_data.get('pair_name', 'unknown')}: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)