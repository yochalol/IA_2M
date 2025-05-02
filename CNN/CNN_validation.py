import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import pickle

# Charger le dataset de validation depuis le CSV
df_val = pd.read_csv('CNN_landmarks_validation.csv')  # Remplace le nom du fichier si besoin

# Séparer les features et les labels
X_val = df_val.drop(columns=['label']).values
y_val = df_val['label'].values

# Charger le modèle et les objets de prétraitement
model = load_model('modele_CNN.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Encodage des labels du dataset de validation
y_val_encoded = label_encoder.transform(y_val)

# Normalisation des features
X_val_scaled = scaler.transform(X_val)

# Prédiction
y_pred = model.predict(X_val_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

# Rapport de classification
print("Résultats sur le jeu de validation :")
print(classification_report(y_val_encoded, y_pred_classes, target_names=label_encoder.classes_))
