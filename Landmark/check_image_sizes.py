import os
import cv2

# 📂 Base du dataset (modifie si besoin)
dataset_base_path = "../Datasets/Rock-Paper-Scissors"

# Catégories d'étiquettes
labels = ["rock", "paper", "scissors"]

# Dictionnaire pour stocker les tailles
image_sizes = {}

# 🔁 Parcours tous les sous-dossiers (train, validation, test, etc.)
for root, dirs, files in os.walk(dataset_base_path):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(root, file)

            # Vérifier si le fichier appartient à une des catégories
            if not any(label in img_path.lower() for label in labels):
                continue  # On ignore les fichiers hors catégories

            img = cv2.imread(img_path)

            if img is None:
                print(f"❌ Impossible de charger {img_path}")
                continue

            # Taille de l’image
            height, width, _ = img.shape
            size = (width, height)

            # Stockage
            if size not in image_sizes:
                image_sizes[size] = 0
            image_sizes[size] += 1

# ✅ Résultat final
print("\n📌 Tailles d'images trouvées dans le dataset :")
for size, count in image_sizes.items():
    print(f"📏 {size} pixels → {count} image(s)")
