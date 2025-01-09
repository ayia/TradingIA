from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import os
import json
import numpy as np
from sklearn.preprocessing import RobustScaler

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
    model_path = os.path.join(MODEL_DIR, pair_name, "best_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé pour la paire : {pair_name}")
    return tf.keras.models.load_model(model_path)

# Fonction pour prétraiter les données
def preprocess_data(data, scaler_path):
    scaler = RobustScaler()
    scaler_file = os.path.join(scaler_path, "scaler_X.pkl")
    if not os.path.exists(scaler_file):
        raise FileNotFoundError("Scaler introuvable pour le prétraitement.")
    
    with open(scaler_file, "rb") as f:
        scaler = json.load(f)
    
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
        with open(scaler_y_path, "rb") as f:
            scaler_y = json.load(f)
        predictions_original = scaler_y.inverse_transform(predictions)

        return {"pair_name": pair_name, "predictions": predictions_original.tolist()}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

# Lancer l'API si le script est exécuté directement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
