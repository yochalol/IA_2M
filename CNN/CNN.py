import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Charger le dataset
df = pd.read_csv('../CNN/CNN_landmarks_dataset.csv', header=0)
#df = pd.read_csv('../CNN/mini_landmarks_dataset.csv', header=0)


# Séparer les labels (première colonne) et les features (autres colonnes)
labels = df.iloc[:, 0].values  # Label : première colonne
features = df.iloc[:, 1:].values  # Données numériques : colonnes restantes

# Convertir les labels en format numérique (encodage des labels)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Normalisation des features
scaler = MinMaxScaler()
X = scaler.fit_transform(features)

# Création du modèle de réseau de neurones
model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Sauvegarde du modèle
model.save("modele_CNN.h5")

# Sauvegarde du LabelEncoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Sauvegarde du scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Modèle, LabelEncoder et scaler sauvegardés.")
