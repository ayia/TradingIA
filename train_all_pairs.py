import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_lstm_for_file(json_file_path, models_root="Models", window_size=10, horizon=1, epochs=200):
    pair_name = os.path.splitext(os.path.basename(json_file_path))[0]
    print(f"\n[INFO] Processing file: {json_file_path}, pair = {pair_name}")

    df = pd.read_json(json_file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.sort_values(by='DateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_cols = [
        "SMA_20", "SMA_50", "EMA_20", "EMA_50",
        "RSI", "MACD", "MACD_Signal", "MACD_Diff",
        "Bollinger_High", "Bollinger_Low", "ATR",
        "Open", "Close", "High", "Low", "Volume"
    ]
    target_cols = ["Open", "Close", "High", "Low"]

    feature_data = df[feature_cols].values
    target_data = df[target_cols].values

    scaler_features = MinMaxScaler()
    feature_data_scaled = scaler_features.fit_transform(feature_data)

    scaler_target = MinMaxScaler()
    target_data_scaled = scaler_target.fit_transform(target_data)

    X, Y = [], []
    for i in range(len(df) - window_size - horizon + 1):
        seq_x = feature_data_scaled[i:i + window_size]
        seq_y = target_data_scaled[i + window_size:i + window_size + horizon]
        X.append(seq_x)
        Y.append(seq_y[0])

    X = np.array(X)
    Y = np.array(Y)

    if len(X) == 0:
        print(f"[WARNING] Not enough data for sequences for {pair_name}.")
        return

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    model = Sequential()
    model.add(Input(shape=(window_size, X.shape[2])))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='linear'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]
    model.fit(X_train, Y_train, epochs=epochs, batch_size=32, validation_split=0.2, shuffle=True, callbacks=callbacks)

    # Evaluate on test data
    loss = model.evaluate(X_test, Y_test)
    print(f"[INFO] Test loss for {pair_name}: {loss}")

    output_dir = os.path.join(models_root, pair_name)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "my_lstm_model.keras")
    scaler_features_path = os.path.join(output_dir, "scaler_features.pkl")
    scaler_target_path = os.path.join(output_dir, "scaler_target.pkl")

    model.save(model_path)
    joblib.dump(scaler_features, scaler_features_path)
    joblib.dump(scaler_target, scaler_target_path)

    print(f"[INFO] Model saved at: {model_path}")
    print(f"[INFO] Scalers saved at: {scaler_features_path}, {scaler_target_path}")

def main():
    input_folder = "training.Data"
    models_root = "Models"
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    if not json_files:
        print("[ERROR] No .json files found in the 'training.Data' folder.")
        return

    for json_file_path in json_files:
        train_lstm_for_file(
            json_file_path=json_file_path, 
            models_root=models_root,
            window_size=10,
            horizon=1,
            epochs=200
        )

if __name__ == "__main__":
    main()