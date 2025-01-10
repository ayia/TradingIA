from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import os
import numpy as np
import joblib  # Utiliser joblib pour charger les scalers

# Initialiser FastAPI
app = FastAPI()

# Chemin vers le dossier contenant les modèles
MODEL_DIR = "./Trained.models"

# Classe pour définir la structure des requêtes
class PredictionRequest(BaseModel):
    pair_name: str
    data: list

# Fonction pour charger un modèle basé sur le nom de la paire
def load_model(pair_name):
    model_path = os.path.join(MODEL_DIR, pair_name, "lstm_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé pour la paire : {pair_name}")
    return tf.keras.models.load_model(model_path)

# Fonction pour prétraiter les données avec joblib
def preprocess_data(data, scaler_path):
    scaler_file = os.path.join(scaler_path, "scaler_X.pkl")
    if not os.path.exists(scaler_file):
        raise FileNotFoundError("Scaler introuvable pour le prétraitement.")

    # Charger le scaler avec joblib
    scaler = joblib.load(scaler_file)
    
    # Appliquer la transformation
    data_scaled = scaler.transform(data)
    return data_scaled

# Point de terminaison pour les prédictions
@app.post("/predict")
async def predict(request: PredictionRequest):
    pair_name = request.pair_name
    data = request.data

    # Vérification des données d'entrée
    if not data or len(data) < 10:
        raise HTTPException(status_code=400, detail="Données insuffisantes pour la prédiction")

    try:
        # Charger le modèle
        model = load_model(pair_name)

        # Charger le scaler et prétraiter les données
        scaler_path = os.path.join(MODEL_DIR, pair_name)
        data_scaled = preprocess_data(np.array(data), scaler_path)

        # Générer des prédictions
        predictions = model.predict(np.expand_dims(data_scaled, axis=0))

        # Post-traitement pour revenir aux échelles originales
        scaler_y_path = os.path.join(scaler_path, "scaler_y.pkl")
        if not os.path.exists(scaler_y_path):
            raise FileNotFoundError("Scaler Y introuvable pour l'inverse transformation.")
        
        scaler_y = joblib.load(scaler_y_path)
        predictions_original = scaler_y.inverse_transform(predictions)

        # Extraire les valeurs
        open_price = float(predictions_original[0][0])
        close_price = float(predictions_original[0][1])
        high_price = float(predictions_original[0][2])
        low_price = float(predictions_original[0][3])

        # Déterminer la direction
        direction = "Bullish" if close_price > open_price else "Bearish"

        # Calculer TP et SL (ajustez les pips selon vos besoins)
        tp_pips = 0.0010  # Exemple : 10 pips
        sl_pips = 0.0010  # Exemple : 10 pips

        if direction == "Bullish":
            tp_pips = abs(high_price - open_price) * 10000 
            sl_pips=abs(open_price - low_price) * 1000
           
        else:
            tp_pips = abs(open_price - low_price) * 10000
            sl_pips=abs(high_price - open_price) * 1000

       
        # Formater la réponse avec les nouvelles informations
        formatted_predictions = {
            "Open": round(open_price, 5),
            "Close": round(close_price, 5),
            "High": round(high_price, 5),
            "Low": round(low_price, 5),
            "Direction": direction,
            "TP Pips": round(tp_pips, 2),
            "SL Pips": round(sl_pips, 2)
        }

        return {"pair_name": pair_name, "predictions": [formatted_predictions]}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

# Lancer l'API si le script est exécuté directement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
