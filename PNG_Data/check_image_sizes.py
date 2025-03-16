import os
import cv2

# 📂 Dossier contenant les images (modifie le chemin si besoin)
image_base_path = "../Datasets/Rock-Paper-Scissors/train/"

labels = ["rock", "paper", "scissors"]
image_sizes = {}

# 📏 Vérifier la taille de toutes les images
for label in labels:
    image_folder = os.path.join(image_base_path, label)

    for image_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, image_name)

        # Charger l’image
        img = cv2.imread(img_path)

        # Vérifier si l’image est bien chargée
        if img is None:
            print(f"❌ Impossible de charger {img_path}")
            continue

        # Récupérer la taille
        height, width, _ = img.shape
        size = (width, height)

        # Stocker les tailles uniques
        if size not in image_sizes:
            image_sizes[size] = 0
        image_sizes[size] += 1

# 📊 Afficher les tailles d’images trouvées
print("📌 Tailles d'images dans le dataset :")
for size, count in image_sizes.items():
    print(f"📏 {size} pixels → {count} images")
