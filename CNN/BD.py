import pandas as pd
import numpy as np

# Exemple de données générées
labels = ['rock', 'scissors', 'paper']
dataset = []

for _ in range(100):  # Générer 100 échantillons
    label = np.random.choice(labels)
    landmarks = np.random.rand(21, 3).flatten()  # 21 points avec 3 coordonnées (x, y, z) par point
    dataset.append([label] + list(landmarks))  # Ajouter l'étiquette et les coordonnées

# Créer un DataFrame avec les bonnes colonnes
columns = ['label'] + [f'x{i}_y{i}_z{i}' for i in range(1, 64)]  # 21 points de repère
df = pd.DataFrame(dataset, columns=columns)

# Sauvegarder dans un fichier CSV
df.to_csv('mini_landmarks_dataset.csv', index=False)

print(df.head())
