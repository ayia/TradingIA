import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def predict_next_bar(json_file_path, pair_name="GBPUSD", models_root="Models", window_size=10):
    # 1) Charger le modèle et les scalers
    model_path = os.path.join(models_root, pair_name, "my_lstm_model.h5")
    scaler_features_path = os.path.join(models_root, pair_name, "scaler_features.pkl")
    scaler_target_path   = os.path.join(models_root, pair_name, "scaler_target.pkl")

    model = load_model(model_path)
    scaler_features = joblib.load(scaler_features_path)
    scaler_target   = joblib.load(scaler_target_path)

    # 2) Charger la data JSON à prédire
    df = pd.read_json(json_file_path)
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

    # 3) Prendre la dernière fenêtre (10 derniers jours) pour faire la prédiction
    seq_x = feature_data_scaled[-window_size:]
    seq_x = np.array([seq_x])  # shape: (1, 10, nb_features)

    # 4) Prédiction
    pred_scaled = model.predict(seq_x)  # shape (1, 4)
    pred = scaler_target.inverse_transform(pred_scaled)
    
    # pred = [ [open, close, high, low] ]
    open_pred, close_pred, high_pred, low_pred = pred[0]
    print(f"Prédiction pour {pair_name} :")
    print("  Open :", open_pred)
    print("  Close:", close_pred)
    print("  High :", high_pred)
    print("  Low  :", low_pred)

if __name__ == "__main__":
    # Exemple d'utilisation
    predict_next_bar(
        json_file_path="data.json",
        pair_name="GBPUSD",
        models_root="Models",
        window_size=10
    )
