import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model

# Charger le modèle et les encodeurs
model = load_model("modele_CNN.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialiser MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture vidéo
cap = cv2.VideoCapture(0)

print("Appuie sur 'q' pour quitter.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip horizontal pour l'effet miroir
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extraire les landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Vérifier si on a bien 21 points * 3 coords = 63
            if len(landmarks) == 63:
                # Normaliser et prédire
                input_data = scaler.transform([landmarks])
                prediction = model.predict(input_data)
                predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

                # Afficher la prédiction sur l'image
                cv2.putText(frame, f"Geste : {predicted_label}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Dessiner les landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Afficher la frame
    cv2.imshow("Détection de gestes - ESC pour quitter", frame)

    # Condition de sortie avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
