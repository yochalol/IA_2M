import cv2
import mediapipe as mp
import os
import csv

# Définir les chemins des dossiers
dossier_images = "../Datasets/Rock-Paper-Scissors/test"  # Modifier avec le chemin correct
dossier_sortie = "../Datasets/Landmarks"
os.makedirs(dossier_sortie, exist_ok=True)

# Fichier CSV pour enregistrer les coordonnées des landmarks
fichier_csv = os.path.join(dossier_sortie, "landmarks_dataset.csv")

# Initialisation de la détection des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Liste des catégories (paper, rock, scissors)
categories = ["paper", "rock", "scissors"]

# Ouvrir le fichier CSV en mode écriture
with open(fichier_csv, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Écriture de l'en-tête CSV
    header = ["label"] + [f"x{i}, y{i}, z{i}" for i in range(21)]
    writer.writerow(header)

    for categorie in categories:
        dossier_categorie = os.path.join(dossier_images, categorie)
        dossier_sortie_categorie = os.path.join(dossier_sortie, categorie)
        os.makedirs(dossier_sortie_categorie, exist_ok=True)

        fichiers_images = [f for f in os.listdir(dossier_categorie) if f.endswith((".png", ".jpg", ".jpeg"))]

        for fichier in fichiers_images:
            chemin_image = os.path.join(dossier_categorie, fichier)
            image = cv2.imread(chemin_image)
            if image is None:
                print(f"Erreur de chargement de l'image : {fichier}")
                continue

            # Conversion en RGB pour MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)

            # Si une main est détectée
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dessiner les landmarks sur l'image
                    mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extraire les coordonnées des 21 landmarks
                    landmarks = [categorie]  # Ajouter la catégorie en début de ligne
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])  # Ajouter (x, y, z)

                    # Écriture dans le fichier CSV
                    writer.writerow(landmarks)

            # Sauvegarde de l'image annotée
            chemin_sortie = os.path.join(dossier_sortie_categorie, fichier)
            cv2.imwrite(chemin_sortie, image)
            print(f"Image traitée et enregistrée : {chemin_sortie}")

print("Traitement terminé. Les landmarks ont été enregistrés dans :", fichier_csv)
