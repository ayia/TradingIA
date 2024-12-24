import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# ---- Import des callbacks Keras pour gérer l'overfitting ----
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt

def train_lstm_for_file(json_file_path, models_root="Models", window_size=10, horizon=1, epochs=20):
    """
    Entraîne un modèle LSTM pour un fichier JSON donné.
    Enregistre le modèle et les scalers dans un sous-dossier de 'models_root'.
    """

    # --------------------------------------------------------
    # 1) Récupérer le nom de la paire à partir du nom de fichier
    # --------------------------------------------------------
    pair_name = os.path.splitext(os.path.basename(json_file_path))[0]
    print(f"\n[INFO] Traitement du fichier : {json_file_path}, pair = {pair_name}")

    # --------------------------------------------------------
    # 2) Charger les données depuis le JSON
    # --------------------------------------------------------
    df = pd.read_json(json_file_path)

    # --------------------------------------------------------
    # 3) Préparer la série temporelle
    # --------------------------------------------------------
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
    target_data  = df[target_cols].values

    # Normalisation
    scaler_features = MinMaxScaler()
    feature_data_scaled = scaler_features.fit_transform(feature_data)

    scaler_target = MinMaxScaler()
    target_data_scaled = scaler_target.fit_transform(target_data)

    # --------------------------------------------------------
    # 4) Création des séquences (fenêtres glissantes)
    # --------------------------------------------------------
    X, Y = [], []
    for i in range(len(df) - window_size - horizon + 1):
        seq_x = feature_data_scaled[i : i + window_size]
        seq_y = target_data_scaled[i + window_size : i + window_size + horizon]
        X.append(seq_x)
        Y.append(seq_y[0])  # (4,)

    X = np.array(X)
    Y = np.array(Y)

    if len(X) == 0:
        print(f"[WARNING] Pas assez de données pour créer des séquences pour {pair_name}.")
        return

    print(f"[INFO] {pair_name} : X shape = {X.shape}, Y shape = {Y.shape}")

    # --------------------------------------------------------
    # 5) Split Train/Test
    # --------------------------------------------------------
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # --------------------------------------------------------
    # 6) Construction du modèle LSTM
    # --------------------------------------------------------
    model = Sequential()
    model.add(LSTM(64, input_shape=(window_size, X.shape[2]), return_sequences=False))
    model.add(Dropout(0.3))   # Augmentation du dropout
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='linear'))  # 4 sorties : Open, Close, High, Low

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Callbacks pour stopper ou réduire le LR en cas de surapprentissage
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]

    # --------------------------------------------------------
    # 7) Entraînement
    # --------------------------------------------------------
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        shuffle=True,              # shuffle=True pour réduire le risque de surapprentissage
        callbacks=callbacks
    )

    # --------------------------------------------------------
    # 8) Évaluation
    # --------------------------------------------------------
    test_loss = model.evaluate(X_test, Y_test)
    print(f"[INFO] {pair_name} - Test MSE: {test_loss:.6f}")

    # --------------------------------------------------------
    # 9) Sauvegarde du modèle et des scalers
    # --------------------------------------------------------
    output_dir = os.path.join(models_root, pair_name)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "my_lstm_model.h5")
    scaler_features_path = os.path.join(output_dir, "scaler_features.pkl")
    scaler_target_path   = os.path.join(output_dir, "scaler_target.pkl")

    model.save(model_path)
    joblib.dump(scaler_features, scaler_features_path)
    joblib.dump(scaler_target,   scaler_target_path)

    print(f"[INFO] Modèle sauvegardé dans : {model_path}")
    print(f"[INFO] Scalers sauvegardés dans : {scaler_features_path}, {scaler_target_path}")

    # Affichage de la courbe de perte
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{pair_name} - Courbe d'apprentissage")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{pair_name}_learning_curve.png"))
    plt.close()

def main():
    """
    Parcours tous les .json dans le dossier 'training.Data' et entraîne un modèle LSTM 
    pour chaque fichier. Les modèles sont sauvegardés dans 'Models/pair_name'.
    """
    input_folder = "training.Data"  
    models_root = "Models"      

    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    if not json_files:
        print("[ERROR] Aucun fichier .json trouvé dans le dossier 'training.Data'.")
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
