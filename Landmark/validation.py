import cv2
import mediapipe as mp
import os
import csv


dossier_images = "../Datasets/Rock-Paper-Scissors/validation"
dossier_sortie = "../Datasets/Landmarks"
os.makedirs(dossier_sortie, exist_ok=True)

fichier_csv = os.path.join(dossier_sortie, "landmarks_dataset_validation.csv")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

with open(fichier_csv, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    header = ["label"] + [f"x{i}, y{i}, z{i}" for i in range(21)]
    writer.writerow(header)

    fichiers_images = [f for f in os.listdir(dossier_images) if f.endswith((".png", ".jpg", ".jpeg"))]

    for fichier in fichiers_images:
        chemin_image = os.path.join(dossier_images, fichier)
        image = cv2.imread(chemin_image)
        if image is None:
            print(f"Erreur de chargement de l'image : {fichier}")
            continue

        fichier_lower = fichier.lower()
        if "rock" in fichier_lower:
            categorie = "rock"
        elif "paper" in fichier_lower:
            categorie = "paper"
        elif "scissors" in fichier_lower:
            categorie = "scissors"
        else:
            print(f"Catégorie inconnue pour le fichier : {fichier}")
            continue

        dossier_sortie_categorie = os.path.join(dossier_sortie, categorie)
        os.makedirs(dossier_sortie_categorie, exist_ok=True)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [categorie]
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                writer.writerow(landmarks)

        chemin_sortie = os.path.join(dossier_sortie_categorie, fichier)
        cv2.imwrite(chemin_sortie, image)
        print(f"Image traitée et enregistrée : {chemin_sortie}")

print("✅ Traitement terminé. Les landmarks ont été enregistrés dans :", fichier_csv)
