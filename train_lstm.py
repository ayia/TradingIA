import os
import json
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Désactiver le GPU pour forcer l'utilisation du CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

##############################################################################
#                      1) FONCTIONS UTILES (Lecture, Prétraitement, etc.)
##############################################################################

def load_json_file(file_path):
    """Charge un fichier JSON et retourne un DataFrame."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if df.isnull().values.any():
            print(f"Warning: Missing values found in {file_path}. Imputing with mean.")
            for col in df.columns:
                if df[col].dtype != object:  # Impute only numerical columns
                    df[col].fillna(df[col].mean(), inplace=True)
        df.dropna(subset=['dateTime'], inplace=True)  # Supprime les lignes sans dateTime
        df.reset_index(drop=True, inplace=True)
        return df
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None

def create_sequences(features, targets, sequence_length=10):
    """Crée des séquences pour l'entraînement du modèle LSTM."""
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(targets[i+sequence_length])
    return np.array(X), np.array(y)

def build_optimized_lstm_model(input_shape, output_shape):
    """Construit un modèle LSTM optimisé."""
    inputs = Input(shape=input_shape, name='LSTM_Input')
    x = LSTM(256, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(128, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(output_shape)(x)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def save_best_model_info(pair_dir, sequence_length, batch_size, success_rate_01, success_rate_005, success_rate_001):
    """Sauvegarde les informations du meilleur modèle."""
    info_path = os.path.join(pair_dir, "best_model_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"sequence_length: {sequence_length}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"Pourcentage de prédictions réussies (tolérance de 0.1%): {success_rate_01:.2f}%\n")
        f.write(f"Pourcentage de prédictions réussies (tolérance de 0.05%): {success_rate_005:.2f}%\n")
        f.write(f"Pourcentage de prédictions réussies (tolérance de 0.01%): {success_rate_001:.2f}%\n")

##############################################################################
#     2) FONCTION PRINCIPALE POUR ENTRAÎNER UN MODÈLE SUR UN FICHIER JSON
##############################################################################

def train_model_for_pair(args):
    """Entraîne un modèle LSTM pour une paire de trading spécifique."""
    # Récupérer les arguments
    json_file, output_dir, sequence_length, test_ratio, batch_size, epochs = args
    
    base_name = os.path.basename(json_file)
    pair_name, ext = os.path.splitext(base_name)
    pair_dir = os.path.join(output_dir, pair_name)
    os.makedirs(pair_dir, exist_ok=True)
    
    df = load_json_file(json_file)
    if df is None or df.empty:
        print(f"Fichier {json_file} vide ou invalide. Entraînement annulé.")
        return
    
    df.sort_values(by='dateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Liste des indicateurs techniques (ajoutez ou retirez selon vos besoins)
    feature_columns = [
        "SMA_20", "SMA_50", "EMA_20", "EMA_50", "RSI",
        "MACD", "MACD_Signal", "MACD_Diff",
        "bollinger_High", "bollinger_Low", "ATR",
        "open", "close", "high", "low",
        "Momentum",
        "Stochastic_K", "Stochastic_D",
        "ADX",
        "CCI",
        "WilliamsR",
        "Hour", "DayOfWeek",
        "Close_Lag1", "Close_Lag2",
        "SMA_20_Slope",
        "Trend",
        "IsDoji", "IsEngulfing", "IsHammer", "IsHangingMan"
    ]
    target_columns = ["open", "close", "high", "low"]

    # Vérification robuste des colonnes
    missing_cols = set(feature_columns + target_columns + ["dateTime"]) - set(df.columns)
    if missing_cols:
        print(f"Error: Missing columns in {json_file}: {missing_cols}. Training aborted.")
        return

    features = df[feature_columns].astype('float32').values
    targets = df[target_columns].astype('float32').values
    n = len(df)
    split_index = int(n * (1 - test_ratio))
    features_train = features[:split_index]
    features_test = features[split_index:]
    targets_train = targets[:split_index]
    targets_test = targets[split_index:]
    
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    scaler_X.fit(features_train)
    scaler_y.fit(targets_train)
    X_train_scaled = scaler_X.transform(features_train)
    X_test_scaled = scaler_X.transform(features_test)
    y_train_scaled = scaler_y.transform(targets_train)
    y_test_scaled = scaler_y.transform(targets_test)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)
    
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        print(f"Données insuffisantes pour la paire {pair_name} (sequence_length={sequence_length}).")
        return
    
    print(f"Entraînement sur {pair_name} :")
    print(f" - X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
    print(f" - X_test_seq: {X_test_seq.shape},  y_test_seq: {y_test_seq.shape}")
    
    input_shape = (sequence_length, X_train_seq.shape[2])
    output_shape = y_train_seq.shape[1]
    model = build_optimized_lstm_model(input_shape, output_shape)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint(os.path.join(pair_dir, 'best_model.keras'), monitor='val_loss', save_best_only=True, mode='min')
    
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    loss_test = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    print(f"Perte de test ({pair_name}) : {loss_test}")
    
    y_pred_scaled = model.predict(X_test_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test_seq)
    
    success_rates = {}
    for tolerance in [0.001, 0.0005, 0.0001]:
        successful_predictions = 0
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                if abs(y_pred[i][j] - y_true[i][j]) / (abs(y_true[i][j]) + 1e-9) <= tolerance:
                    successful_predictions += 1
        total_predictions = len(y_true) * len(y_true[0])
        success_rate = (successful_predictions / total_predictions) * 100
        success_rates[tolerance] = success_rate
        print(f"Pourcentage de prédictions réussies (tolérance de {tolerance*100}%) : {success_rate:.2f}%")
    
    best_model_info_path = os.path.join(pair_dir, "best_model_info.txt")
    if os.path.exists(best_model_info_path):
        with open(best_model_info_path, 'r') as f:
            lines = f.readlines()
            try:
                best_success_rate_01 = float(lines[2].split(":")[1].strip().replace('%', ''))
            except (IndexError, ValueError):
                best_success_rate_01 = 0
    else:
        best_success_rate_01 = 0
    
    if success_rates[0.001] > best_success_rate_01:
        model_path = os.path.join(pair_dir, "lstm_model.keras")
        model.save(model_path)
        print(f"Modèle sauvegardé : {model_path}")
        save_best_model_info(pair_dir, sequence_length, batch_size, success_rates[0.001], success_rates[0.0005], success_rates[0.0001])
        
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Perte Entraînement')
        plt.plot(history.history['val_loss'], label='Perte Validation')
        plt.title(f'Historique de l\'entraînement - {pair_name}')
        plt.xlabel('Époque')
        plt.ylabel('Perte (MSE)')
        plt.legend()
        plot_path = os.path.join(pair_dir, "training_history.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Courbe d'entraînement sauvegardée : {plot_path}")
        
        predictions_csv = os.path.join(pair_dir, "test_predictions.csv")
        pred_df = pd.DataFrame(np.concatenate([y_pred, y_true], axis=1),
                               columns=[
                                   "pred_open", "pred_close", "pred_high", "pred_low",
                                   "true_open", "true_close", "true_high", "true_low"
                               ])
        pred_df.to_csv(predictions_csv, index=False)
        print(f"Fichier de prédictions sauvegardé : {predictions_csv}")
        
        plt.figure(figsize=(12, 6))
        plt.plot(abs(y_pred - y_true).mean(axis=1), label='Erreur moyenne', color='purple', linewidth=1.5)
        plt.title(f'Erreurs des prédictions - {pair_name}')
        plt.xlabel('Pas de temps')
        plt.ylabel('Erreur absolue moyenne')
        plt.legend()
        error_plot_path = os.path.join(pair_dir, "error_plot.png")
        plt.savefig(error_plot_path)
        plt.close()
        print(f"Graphique des erreurs sauvegardé : {error_plot_path}")
    else:
        print(f"Le modèle actuel n'est pas meilleur que le modèle précédent. Aucune sauvegarde effectuée.")
    print("--------------------------------------------------------\n")

##############################################################################
#     3) SCRIPT PRINCIPAL : BOUCLE SUR CHAQUE FICHIER JSON
##############################################################################

def main():
    training_data_path = './training.Data'
    models_folder = './models'
    sequence_lengths = [15, 20, 30]
    test_ratio = 0.2
    batch_sizes = [32, 64, 128]
    epochs = 100

    json_files = glob.glob(os.path.join(training_data_path, '*.json'))
    if not json_files:
        print("Aucun fichier JSON trouvé dans training.Data.")
        return

    tasks = []
    for json_file in json_files:
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                tasks.append((json_file, models_folder, seq_len, test_ratio, batch_size, epochs))

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(train_model_for_pair, tasks)

##############################################################################
#     4) EXÉCUTION DU SCRIPT
##############################################################################

if __name__ == '__main__':
    main()
