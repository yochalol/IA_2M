import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 📂 Charger le dataset
data_path = "./Datasets/Landmarks/landmarks_dataset.csv"
df = pd.read_csv(data_path)

# 📌 Séparer les labels et les features
X = df.iloc[:, 1:].values  # Toutes les colonnes sauf la première (features numériques)
y = df.iloc[:, 0].values   # Première colonne (label)

# 🔄 Convertir les labels en valeurs numériques (0,1,2)
le = LabelEncoder()
y = le.fit_transform(y)  # Ex: "scissors" → 0, "rock" → 1, "paper" → 2

# 🔀 Séparer en train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌲 Initialiser et entraîner le modèle Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, "random_forest_model.pkl")          # Sauvegarde du modèle
joblib.dump(le, "label_encoder.pkl")                 # 🔄 Sauvegarde du label encoder


# 📊 Prédire et évaluer
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Précision du modèle : {accuracy:.2f}")
