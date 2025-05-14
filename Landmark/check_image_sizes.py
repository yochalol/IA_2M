import os
import cv2

dataset_base_path = "../Datasets/Rock-Paper-Scissors"


labels = ["rock", "paper", "scissors"]

image_sizes = {}
for root, dirs, files in os.walk(dataset_base_path):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(root, file)

            if not any(label in img_path.lower() for label in labels):
                continue

            img = cv2.imread(img_path)

            if img is None:
                print(f"❌ Impossible de charger {img_path}")
                continue

            height, width, _ = img.shape
            size = (width, height)

            if size not in image_sizes:
                image_sizes[size] = 0
            image_sizes[size] += 1

print("\n📌 Tailles d'images trouvées dans le dataset :")
for size, count in image_sizes.items():
    print(f"📏 {size} pixels → {count} image(s)")
