from flask import Flask, request, jsonify
import tensorflow as tf
import os
import numpy as np
import joblib
import requests

# Initialize Flask app
app = Flask(__name__)

# Directory containing models
MODEL_DIR = "./Trained.models"

# Load model function
def load_model(pair_name):
    model_path = os.path.join(MODEL_DIR, pair_name, "lstm_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for pair: {pair_name}")
    return tf.keras.models.load_model(model_path)

# Preprocess data function
def preprocess_data(data, scaler_path):
    scaler_file = os.path.join(scaler_path, "scaler_X.pkl")
    if not os.path.exists(scaler_file):
        raise FileNotFoundError("Scaler not found for preprocessing.")
    scaler = joblib.load(scaler_file)
    return scaler.transform(data)

# Endpoint: /predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        pair_name = data.get("pair_name")
        input_data = data.get("data")

        if not input_data or len(input_data) < 10:
            return jsonify({"error": "Insufficient data for prediction"}), 400

        model = load_model(pair_name)
        scaler_path = os.path.join(MODEL_DIR, pair_name)
        data_scaled = preprocess_data(np.array(input_data), scaler_path)

        predictions = model.predict(np.expand_dims(data_scaled, axis=0))

        scaler_y_path = os.path.join(scaler_path, "scaler_y.pkl")
        if not os.path.exists(scaler_y_path):
            raise FileNotFoundError("Scaler Y not found for inverse transformation.")
        scaler_y = joblib.load(scaler_y_path)
        predictions_original = scaler_y.inverse_transform(predictions)

        open_price = float(predictions_original[0][0])
        close_price = float(predictions_original[0][1])
        high_price = float(predictions_original[0][2])
        low_price = float(predictions_original[0][3])

        direction = "Bullish" if close_price > open_price else "Bearish"
        atr = abs(data_scaled[-1][7]) if len(data_scaled[0]) > 7 else 0.001
        if atr <= 0:
            return jsonify({"error": "Invalid ATR value."}), 400
            

        if direction == "Bullish":
            tp_price = high_price
            atr_sl = open_price - atr / 10000
            sl_price = atr_sl if atr_sl < open_price else open_price - 0.0001
        else:
            tp_price = low_price
            atr_sl = open_price + atr / 10000
            sl_price = atr_sl if atr_sl > open_price else open_price + 0.0001

        tp_pips = abs(tp_price - open_price) * 10000
        sl_pips = abs(sl_price - open_price) * 10000

        result = {
            "Open": round(open_price, 5),
            "Close": round(close_price, 5),
            "High": round(high_price, 5),
            "Low": round(low_price, 5),
            "Direction": direction,
            "TP Price": round(tp_price, 5),
            "SL Price": round(sl_price, 5),
            "TP Pips": round(tp_pips, 2),
            "SL Pips": round(sl_pips, 2),
            "ATR": round(atr, 5),
        }

        return jsonify({"pair_name": pair_name, "predictions": [result]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint: /fetch_and_predict
@app.route('/fetch_and_predict', methods=['POST'])
def fetch_and_predict():
    try:
        pair_name = request.args.get("pair_name")
        interval = request.args.get("interval", "OneHour")
        limits = int(request.args.get("limits", 20))
        aba = request.args.get("aba", "Lastyear")

        external_api_url = "https://forex-bestshots-black-meadow-4133.fly.dev/api/FetchPairsData/PreparePredictionData"
        params = {"currencyPairs": pair_name, "interval": interval, "limits": limits, "aba": aba}
        response = requests.post(external_api_url, params=params)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch data from external API"}), 500

        data = response.json()
        if not data or "data" not in data or not isinstance(data["data"], list):
            return jsonify({"error": "Invalid data from external API"}), 400

        return predict()

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint: /fetch_and_predict_string
@app.route('/fetch_and_predict_string', methods=['POST'])
def fetch_and_predict_string():
    try:
        pair_name = request.args.get("pair_name")
        interval = request.args.get("interval", "OneHour")
        limits = int(request.args.get("limits", 20))
        aba = request.args.get("aba", "Lastyear")

        external_api_url = "https://forex-bestshots-black-meadow-4133.fly.dev/api/FetchPairsData/PreparePredictionData"
        params = {"currencyPairs": pair_name, "interval": interval, "limits": limits, "aba": aba}
        response = requests.post(external_api_url, params=params)

        if response.status_code != 200:
            return "null"

        data = response.json()
        if not data or "data" not in data or not isinstance(data["data"], list):
            return "null"

        prediction_request = {"pair_name": pair_name, "data": data["data"]}
        prediction_response = predict()
        if prediction_response.status_code != 200:
            return "null"

        prediction_result = prediction_response.get_json()
        if "predictions" in prediction_result and prediction_result["predictions"]:
            prediction = prediction_result["predictions"][0]
            return f"{prediction['Direction']}|{prediction['Open']}|{prediction['TP Pips']}|{prediction['SL Pips']}"
        else:
            return "null"

    except Exception:
        return "null"

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8073)
