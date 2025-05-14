import os
import cv2
import numpy as np
import pandas as pd

image_base_path = "../Datasets/Rock-Paper-Scissors/train/"

labels = ["rock", "paper", "scissors"]

IMG_SIZE = 300
csv_data = []

for label in labels:
    image_folder = os.path.join(image_base_path, label)

    for image_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, image_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Impossible de charger {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = img.astype("float32") / 255.0

        img_flat = img.flatten().tolist()

        csv_data.append([image_name, label] + img_flat)


columns = ["filename", "label"] + [f"pixel_{i}" for i in range(IMG_SIZE * IMG_SIZE * 3)]
df = pd.DataFrame(csv_data, columns=columns)

output_csv_path = "../Datasets/Landmarks/image_data_pas_utile.csv"

df.to_csv(output_csv_path, index=False)

print(f"✅ Toutes les images ont été converties et enregistrées dans {output_csv_path} !")
