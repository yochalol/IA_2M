import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

DATASET_DIR = "Datasets/Landmarks/"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

CSV_FILE = os.path.join(DATASET_DIR, "landmarks_dataset.csv")

if not os.path.exists(CSV_FILE):
    cols = ["label"] + [f"x{i}, y{i}, z{i}" for i in range(21)]
    pd.DataFrame(columns=cols).to_csv(CSV_FILE, index=False)

gestures = {
    "1": "pierre",
    "2": "feuille",
    "3": "ciseaux"
}

print("Appuie sur : \n[1] pour capturer PIERRE\n[2] pour capturer FEUILLE\n[3] pour capturer CISEAUX\n[Q] pour quitter.")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            cv2.putText(frame, "Press 1 (pierre), 2 (feuille) or 3 to save (siceaux), and q for leave", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("1") or key == ord("2") or key == ord("3"):
                label = gestures[chr(key)]
                new_data = pd.DataFrame([[label] + landmarks])
                new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)
                print(f"Données sauvegardées pour : {label}")

    cv2.imshow("Capture Data", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
