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


def build_lstm_model(input_shape, output_shape):
    """
    Construit un modèle LSTM simple avec deux couches LSTM et dropout.
    input_shape = (sequence_length, nb_features)
    output_shape = nb_targets
    """
    inputs = Input(shape=input_shape, name='LSTM_Input')
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(output_shape, name='Output')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


##############################################################################
#     2) FONCTION PRINCIPALE POUR ENTRAÎNER UN MODÈLE SUR UN FICHIER JSON
##############################################################################

def train_model_for_pair(json_file, output_dir='models', sequence_length=10, 
                         test_ratio=0.2, batch_size=32, epochs=50):
    """
    Entraîne un modèle LSTM pour la paire correspondant à 'json_file'.
    Sauvegarde le modèle et la courbe d'entraînement dans un dossier 
    'models/PairName' (ex: models/AUDUSD).

    :param json_file: Chemin du fichier JSON (ex: ./training.Data/AUDUSD.json)
    :param output_dir: Dossier parent où sauvegarder les modèles (ex: 'models')
    :param sequence_length: Nombre de pas de temps pour chaque séquence
    :param test_ratio: Proportion de données pour le test (ex: 0.2 = 20%)
    :param batch_size: Taille de lot pour l'entraînement
    :param epochs: Nombre d'époques
    """
    # 1) Extraction du nom de la paire depuis le nom de fichier
    #    Par ex. "AUDUSD" depuis "AUDUSD.json"
    base_name = os.path.basename(json_file)          # AUDUSD.json
    pair_name, ext = os.path.splitext(base_name)     # pair_name="AUDUSD", ext=".json"

    # 2) Lecture des données
    df = load_json_file(json_file)
    if df is None or df.empty:
        print(f"Fichier {json_file} vide ou invalide. Entraînement annulé.")
        return
    
    df.sort_values(by='dateTime', inplace=True)  # Tri temporel si nécessaire
    df.reset_index(drop=True, inplace=True)
    
    # 3) Définir les features et les cibles
    feature_columns = [
        "SMA_20", "SMA_50", "EMA_20", "EMA_50", "RSI",
        "MACD", "MACD_Signal", "MACD_Diff",
        "bollinger_High", "bollinger_Low", "ATR",
        "open", "close", "high", "low", "volume"
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
    
    # 4) Division Train/Test
    n = len(df)
    split_index = int(n * (1 - test_ratio))  # ex: 80% train / 20% test
    features_train = features[:split_index]
    features_test = features[split_index:]
    targets_train = targets[:split_index]
    targets_test = targets[split_index:]
    
    # 5) Normalisation
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(features_train)  # Ajustement sur la partie train
    scaler_y.fit(targets_train)
    
    X_train_scaled = scaler_X.transform(features_train)
    X_test_scaled = scaler_X.transform(features_test)
    y_train_scaled = scaler_y.transform(targets_train)
    y_test_scaled = scaler_y.transform(targets_test)
    
    # 6) Création des séquences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

    # Vérification qu'on a assez de données séquencées
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        print(f"Données insuffisantes pour la paire {pair_name} (sequence_length={sequence_length}).")
        return

    print(f"Entraînement sur {pair_name} :")
    print(f" - X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
    print(f" - X_test_seq: {X_test_seq.shape},  y_test_seq: {y_test_seq.shape}")
    
    # 7) Construction du modèle
    input_shape = (sequence_length, X_train_seq.shape[2])  # (10, nb_features)
    output_shape = y_train_seq.shape[1]                    # 4 (OHLC)
    model = build_lstm_model(input_shape, output_shape)

    # Callback d'arrêt anticipé
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 8) Entraînement
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 9) Création du dossier de sortie "models/pairName" s'il n'existe pas
    pair_dir = os.path.join(output_dir, pair_name)
    os.makedirs(pair_dir, exist_ok=True)

    # 10) Sauvegarde du modèle
    model_path = os.path.join(pair_dir, "lstm_model.keras")
    model.save(model_path)
    print(f"Modèle sauvegardé : {model_path}")
    
    # 11) Courbe d’entraînement
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
    
    # 12) Évaluation sur le test
    loss_test = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    print(f"Perte de test ({pair_name}) : {loss_test}")

    # 13) Optionnel : prédictions de test
    y_pred_scaled = model.predict(X_test_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test_seq)
    
    # On peut enregistrer y_pred vs y_true dans un CSV
    # Format : columns = [pred_open, pred_close, pred_high, pred_low,
    #                     true_open, true_close, true_high, true_low]
    predictions_csv = os.path.join(pair_dir, "test_predictions.csv")
    pred_df = pd.DataFrame(np.concatenate([y_pred, y_true], axis=1),
                           columns=[
                               "pred_open", "pred_close", "pred_high", "pred_low",
                               "true_open", "true_close", "true_high", "true_low"
                           ])
    pred_df.to_csv(predictions_csv, index=False)
    print(f"Fichier de prédictions sauvegardé : {predictions_csv}")
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
    sequence_length = 10
    test_ratio = 0.2
    batch_size = 32
    epochs = 50

    # Recherche de tous les fichiers .json dans training.Data
    json_files = glob.glob(os.path.join(training_data_path, '*.json'))
    if not json_files:
        print("Aucun fichier JSON trouvé dans training.Data.")
        return
    
    for json_file in json_files:
        train_model_for_pair(
            json_file,
            output_dir=models_folder,
            sequence_length=sequence_length,
            test_ratio=test_ratio,
            batch_size=batch_size,
            epochs=epochs
        )


if __name__ == '__main__':
    main()
