import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Chargement des datasets
df_train = pd.read_csv("../Datasets/Landmarks/landmarks_dataset_V2_train.csv")
df_val = pd.read_csv("../Datasets/Landmarks/landmarks_dataset_V2_validation.csv")
df_test = pd.read_csv("../Datasets/Landmarks/landmarks_dataset_V2.csv")  # optionnel

# Fusionner train + val (ou les garder séparés)
df_all = pd.concat([df_train, df_val], ignore_index=True)

# Séparer features et labels
X = df_all.drop("label", axis=1)
y = df_all["label"]

# 🔁 Optionnel : Encoder les labels en entiers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # ex: "rock" => 2

# Séparer en train / validation (si tu veux)
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 🔁 Tu peux aussi utiliser directement df_train et df_val séparés si tu préfères





# entrainement du modèle

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# evaluer le modèle *
y_pred = model.predict(X_val)

# Affichage du rapport de classification
print(classification_report(y_val, y_pred, target_names=le.classes_))


#tester  avec test
# Préparer les features test
X_test = df_test.drop("label", axis=1)
y_test = le.transform(df_test["label"])

# Prédictions
y_pred_test = model.predict(X_test)

# Résultats
print("Résultats sur le jeu de test :")
print(classification_report(y_test, y_pred_test, target_names=le.classes_))


import pickle
from sklearn.ensemble import RandomForestClassifier

# Entraînement du modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sauvegarde du modèle
with open("../RF/random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)
