import cv2
import mediapipe as mp
import numpy as np
import joblib

# Charger le modèle Random Forest
model = joblib.load("random_forest_model.pkl")

# Initialisation MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Mapping des classes prédictes
label_map = {
    0: "scissors",
    1: "rock",
    2: "paper"
}

# Démarrer la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Conversion BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraire les 63 coordonnées
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Prédire si on a bien les 63 données
            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]
                gesture = label_map[prediction]

                # Afficher le résultat
                cv2.putText(frame, f"Geste : {gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Affichage de la vidéo
    cv2.imshow("Détection Pierre Feuille Ciseaux", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
