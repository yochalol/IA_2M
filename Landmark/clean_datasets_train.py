import pandas as pd
import csv

# 📁 Chemin vers le fichier original
input_file = "../Datasets/Landmarks/landmarks_dataset_train.csv"
output_file = "../Datasets/Landmarks/landmarks_dataset_V2_train.csv"

# 🧼 Création d’un nouveau DataFrame nettoyé
cleaned_data = []
with open(input_file, "r") as f:
    reader = csv.reader(f)
    for index, row in enumerate(reader):
        if index == 0:
            # Sauter l'en-tête
            continue

        label = row[0]
        values = row[1:]

        # Si les coordonnées sont bien séparées (63 valeurs)
        if len(values) == 63:
            cleaned_data.append([label] + values)
        else:
            print(f"⚠️ Ligne {index+1} ignorée, {len(values)} valeurs au lieu de 63")

# 🏷️ Générer les colonnes x0, y0, z0, ..., x20, y20, z20
columns = ["label"] + [f"{coord}{i}" for i in range(21) for coord in ["x", "y", "z"]]

# 💾 Sauvegarder dans un nouveau CSV propre
df_clean = pd.DataFrame(cleaned_data, columns=columns)
df_clean.to_csv(output_file, index=False)

print(f"✅ Fichier propre enregistré ici : {output_file}")
