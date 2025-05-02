import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import pickle

# Charger le dataset de test depuis le CSV
df_test = pd.read_csv('CNN_landmarks_test.csv')

# Séparer les features et les labels
X_test = df_test.drop(columns=['label']).values
y_test = df_test['label'].values

# Charger le modèle et les objets de prétraitement
model = load_model('modele_CNN.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 1. Encodage des labels du test
y_test_encoded = label_encoder.transform(y_test)

# 2. Normalisation des features (landmarks)
X_test_scaled = scaler.transform(X_test)

# 3. Prédictions sur le jeu de test
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calcul du rapport de classification (sans l'accuracy)
report = classification_report(y_test_encoded, y_pred_classes, target_names=label_encoder.classes_, output_dict=True)

# Convertir en DataFrame
report_df = pd.DataFrame(report).transpose()

# Supprimer la ligne "accuracy" du tableau
report_df = report_df.drop('accuracy', axis=0)

# Arrondir tous les résultats à 4 chiffres après la virgule
report_df = report_df.round(4)

# Calcul de l'accuracy globale en pourcentage avec 4 chiffres significatifs
accuracy = np.sum(y_pred_classes == y_test_encoded) / len(y_test_encoded) * 100
accuracy = round(accuracy, 4)

# Affichage du tableau des résultats sans l'accuracy
print("Tableau des résultats sur le jeu de test :")
print(report_df)

# Afficher l'accuracy en pourcentage séparément
print("\nAccuracy sur l'ensemble du jeu de test : {:.4f}%".format(accuracy))
