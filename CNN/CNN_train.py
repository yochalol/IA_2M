import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import pickle

# Charger le dataset depuis le CSV
df = pd.read_csv('CNN_landmarks_train.csv')

# Séparer les features et les labels
X_train = df.drop(columns=['label']).values  # Toutes les colonnes sauf 'label'
y_train = df['label'].values  # La colonne 'label'

# Encoder les labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

# Normaliser les features (les landmarks)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Création du modèle
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Sauvegarde du modèle, LabelEncoder et scaler
model.save("modele_CNN.h5")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Modèle, LabelEncoder et scaler sauvegardés.")
