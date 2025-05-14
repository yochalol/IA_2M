import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import pickle


df_val = pd.read_csv('CNN_landmarks_validation.csv')

X_val = df_val.drop(columns=['label']).values
y_val = df_val['label'].values

model = load_model('modele_CNN.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

y_val_encoded = label_encoder.transform(y_val)
X_val_scaled = scaler.transform(X_val)

y_pred = model.predict(X_val_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

print("Résultats sur le jeu de validation :")
print(classification_report(y_val_encoded, y_pred_classes, target_names=label_encoder.classes_))
