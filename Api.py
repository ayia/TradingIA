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
          "Direction": direction,       
           "Open": round(open_price, 5),
            "Close": round(close_price, 5),
            "High": round(high_price, 5),
            "Low": round(low_price, 5),
            "UpPips": round(tp_pips, 2),
            "DownPips": round(sl_pips, 2),
           
        }

        return jsonify({"pair_name": pair_name, "predictions": [result]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8073)
