import pandas as pd
import csv


input_file = "../Datasets/Landmarks/landmarks_dataset.csv"
output_file = "../Datasets/Landmarks/landmarks_dataset_V2.csv"


cleaned_data = []
with open(input_file, "r") as f:
    reader = csv.reader(f)
    for index, row in enumerate(reader):
        if index == 0:
            continue

        label = row[0]
        values = row[1:]

        if len(values) == 63:
            cleaned_data.append([label] + values)
        else:
            print(f"⚠️ Ligne {index+1} ignorée, {len(values)} valeurs au lieu de 63")

columns = ["label"] + [f"{coord}{i}" for i in range(21) for coord in ["x", "y", "z"]]

df_clean = pd.DataFrame(cleaned_data, columns=columns)
df_clean.to_csv(output_file, index=False)

print(f"✅ Fichier propre enregistré ici : {output_file}")
