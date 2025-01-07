import os
import json
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')  # Pour environnement sans interface graphique
import matplotlib.pyplot as plt

##############################################################################
#                      1) FONCTIONS UTILES (Lecture, Prétraitement, etc.)
##############################################################################

def load_json_file(file_path):
    """
    Charge un fichier JSON unique, le convertit en DataFrame,
    et supprime les lignes contenant des valeurs manquantes.
    Retourne un DataFrame pandas, ou None en cas d'erreur.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df.dropna(inplace=True)  # Supprime les lignes vides
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {file_path}: {e}")
        return None

def create_sequences(features, targets, sequence_length=10):
    """
    Crée des séquences (X, y) pour l'entraînement d'un modèle RNN/LSTM.
    - features: np.array de forme (N, nb_features)
    - targets: np.array de forme (N, nb_cibles)
    - sequence_length: longueur de la séquence temporelle

    Retourne X, y tels que :
      X.shape = (nombre_sequences, sequence_length, nb_features)
      y.shape = (nombre_sequences, nb_cibles)
    """
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(targets[i+sequence_length])  # Valeur "future"
    return np.array(X), np.array(y)

def build_improved_lstm_model(input_shape, output_shape):
    """
    Construit un modèle LSTM amélioré avec quatre couches LSTM et plus de régularisation.
    input_shape = (sequence_length, nb_features)
    output_shape = nb_targets
    """
    inputs = Input(shape=input_shape, name='LSTM_Input')
    x = LSTM(256, return_sequences=True)(inputs)
    x = Dropout(0.4)(x)  # Régularisation
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.4)(x)  # Régularisation
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.4)(x)  # Régularisation
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.4)(x)  # Régularisation
    outputs = Dense(output_shape, name='Output')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def model_exists(pair_dir):
    """
    Vérifie si le modèle existe déjà dans le dossier de la paire.
    """
    model_path = os.path.join(pair_dir, "lstm_model.keras")
    return os.path.exists(model_path)

def save_best_model_info(pair_dir, sequence_length, batch_size, success_rate_01, success_rate_005, success_rate_001):
    """
    Sauvegarde les informations du meilleur modèle dans un fichier texte.
    """
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

def train_model_for_pair(json_file, output_dir='models', sequence_length=10, 
                         test_ratio=0.2, batch_size=32, epochs=100):
    # 1) Extraction du nom de la paire depuis le nom de fichier
    base_name = os.path.basename(json_file)
    pair_name, ext = os.path.splitext(base_name)

    # 2) Vérifier si le modèle existe déjà
    pair_dir = os.path.join(output_dir, pair_name)
    os.makedirs(pair_dir, exist_ok=True)

    # 3) Lecture des données
    df = load_json_file(json_file)
    if df is None or df.empty:
        print(f"Fichier {json_file} vide ou invalide. Entraînement annulé.")
        return
    
    df.sort_values(by='dateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 4) Définir les features et les cibles
    feature_columns = [
       "SMA_20", "SMA_50", "EMA_20", "EMA_50", "RSI",
    "MACD", "MACD_Signal", "MACD_Diff",
    "bollinger_High", "bollinger_Low", "ATR",  # ATR est un indicateur de volatilité
    "open", "close", "high", "low",  # Prix OHLC
    "Momentum",  # Momentum
    "Stochastic_K", "Stochastic_D",  # Indicateurs stochastiques
    "ADX",  # Average Directional Index
    "CCI",  # Commodity Channel Index
    "WilliamsR",  # Williams %R
    "Hour", "DayOfWeek",  # Informations temporelles
    "Close_Lag1", "Close_Lag2",  # Valeurs décalées du prix de clôture
    "SMA_20_Slope",  # Pente de la SMA_20
    "Trend",  # Indicateur de tendance
    "IsDoji", "IsEngulfing", "IsHammer", "IsHangingMan"  # Motifs de chandeliers japonais
    ]
    target_columns = ["open", "close", "high", "low"]

    # Vérifier que toutes les colonnes sont présentes
    for col in feature_columns + target_columns + ["dateTime"]:
        if col not in df.columns:
            print(f"Colonne {col} manquante dans {json_file} ! Entraînement annulé.")
            return

    # Conversion en numpy
    features = df[feature_columns].astype('float32').values
    targets = df[target_columns].astype('float32').values
    
    # 5) Division Train/Test
    n = len(df)
    split_index = int(n * (1 - test_ratio))
    features_train = features[:split_index]
    features_test = features[split_index:]
    targets_train = targets[:split_index]
    targets_test = targets[split_index:]
    
    # 6) Normalisation
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(features_train)
    scaler_y.fit(targets_train)
    
    X_train_scaled = scaler_X.transform(features_train)
    X_test_scaled = scaler_X.transform(features_test)
    y_train_scaled = scaler_y.transform(targets_train)
    y_test_scaled = scaler_y.transform(targets_test)
    
    # 7) Création des séquences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        print(f"Données insuffisantes pour la paire {pair_name} (sequence_length={sequence_length}).")
        return

    print(f"Entraînement sur {pair_name} :")
    print(f" - X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
    print(f" - X_test_seq: {X_test_seq.shape},  y_test_seq: {y_test_seq.shape}")
    
    # 8) Construction du modèle amélioré
    input_shape = (sequence_length, X_train_seq.shape[2])
    output_shape = y_train_seq.shape[1]
    model = build_improved_lstm_model(input_shape, output_shape)

    # Callback d'arrêt anticipé
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 9) Entraînement
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 10) Évaluation sur le test
    loss_test = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    print(f"Perte de test ({pair_name}) : {loss_test}")

    # 11) Optionnel : prédictions de test
    y_pred_scaled = model.predict(X_test_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test_seq)
    
    # 12) Calcul du pourcentage de prédictions réussies avec différentes tolérances
    success_rates = {}
    for tolerance in [0.001, 0.0005, 0.0001]:  # 0.1%, 0.05%, 0.01%
        successful_predictions = 0
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                if abs(y_pred[i][j] - y_true[i][j]) / y_true[i][j] <= tolerance:
                    successful_predictions += 1

        total_predictions = len(y_true) * len(y_true[0])
        success_rate = (successful_predictions / total_predictions) * 100
        success_rates[tolerance] = success_rate

        print(f"Pourcentage de prédictions réussies (tolérance de {tolerance*100}%) : {success_rate:.2f}%")
    
    # 13) Vérifier si ce modèle est le meilleur
    best_model_info_path = os.path.join(pair_dir, "best_model_info.txt")
    if os.path.exists(best_model_info_path):
        with open(best_model_info_path, 'r') as f:
            lines = f.readlines()
            best_success_rate_01 = float(lines[2].split(":")[1].strip().replace('%', ''))
    else:
        best_success_rate_01 = 0  # Si aucun modèle n'a été sauvegardé auparavant

    # Si le modèle actuel est meilleur, sauvegarder le modèle et les informations
    if success_rates[0.001] > best_success_rate_01:
        # Sauvegarder le modèle
        model_path = os.path.join(pair_dir, "lstm_model.keras")
        model.save(model_path)
        print(f"Modèle sauvegardé : {model_path}")

        # Sauvegarder les informations du meilleur modèle
        save_best_model_info(pair_dir, sequence_length, batch_size, success_rates[0.001], success_rates[0.0005], success_rates[0.0001])

        # Sauvegarder la courbe d'entraînement
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

        # Sauvegarder les prédictions
        predictions_csv = os.path.join(pair_dir, "test_predictions.csv")
        pred_df = pd.DataFrame(np.concatenate([y_pred, y_true], axis=1),
                               columns=[
                                   "pred_open", "pred_close", "pred_high", "pred_low",
                                   "true_open", "true_close", "true_high", "true_low"
                               ])
        pred_df.to_csv(predictions_csv, index=False)
        print(f"Fichier de prédictions sauvegardé : {predictions_csv}")

        # Sauvegarder le graphique des erreurs
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
#         3) SCRIPT PRINCIPAL : BOUCLE SUR CHAQUE FICHIER JSON
##############################################################################

def main():
    # Chemin du dossier contenant les fichiers JSON de training
    training_data_path = './training.Data'
    # Dossier parent de sortie
    models_folder = './models'

    # Paramètres communs
    sequence_lengths = [10, 15, 20]  # Testez différentes valeurs
    test_ratio = 0.2
    batch_sizes = [16, 32, 64]  # Testez différentes tailles de batch
    epochs = 100  # Augmentez le nombre d'époques

    # Recherche de tous les fichiers .json dans training.Data
    json_files = glob.glob(os.path.join(training_data_path, '*.json'))
    if not json_files:
        print("Aucun fichier JSON trouvé dans training.Data.")
        return
    
    for json_file in json_files:
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                print(f"\nEntraînement avec sequence_length = {seq_len}, batch_size = {batch_size}")
                train_model_for_pair(
                    json_file,
                    output_dir=models_folder,
                    sequence_length=seq_len,
                    test_ratio=test_ratio,
                    batch_size=batch_size,
                    epochs=epochs
                )

if __name__ == '__main__':
    main()