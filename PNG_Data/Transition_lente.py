import os
import cv2
import numpy as np
import pandas as pd

# 📂 Dossier contenant les images
image_base_path = "../Datasets/Rock-Paper-Scissors/train/"

labels = ["rock", "paper", "scissors"]

# 📏 Paramètres de transformation
IMG_SIZE = 300  # Taille des images (300x300 pixels)
csv_data = []

# 📸 Parcours des images et conversion en vecteur de pixels
for label in labels:
    image_folder = os.path.join(image_base_path, label)

    for image_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, image_name)

        # Charger l’image avec OpenCV
        img = cv2.imread(img_path)

        # Vérifier si l’image est bien chargée
        if img is None:
            print(f"❌ Impossible de charger {img_path}")
            continue

        # Convertir en RGB (si nécessaire)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Redimensionner l’image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normaliser les pixels entre 0 et 1
        img = img.astype("float32") / 255.0

        # Aplatir l’image en une seule ligne (1D)
        img_flat = img.flatten().tolist()

        # Ajouter les données sous forme de ligne (nom, label, pixels)
        csv_data.append([image_name, label] + img_flat)

# 📂 Convertir en DataFrame
columns = ["filename", "label"] + [f"pixel_{i}" for i in range(IMG_SIZE * IMG_SIZE * 3)]
df = pd.DataFrame(csv_data, columns=columns)

# 📂 Dossier où sauvegarder le CSV
output_csv_path = "../Datasets/Landmarks/image_data_pas_utile.csv"

# Sauvegarder les données en CSV
df.to_csv(output_csv_path, index=False)

print(f"✅ Toutes les images ont été converties et enregistrées dans {output_csv_path} !")
