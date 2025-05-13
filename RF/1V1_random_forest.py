import cv2
import mediapipe as mp
import numpy as np
import pickle

# Charger le modèle Random Forest
with open("../random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Labels (tu peux les ajuster selon ton encodage)
labels = ["paper", "rock", "scissors"]

# Détection des mains avec MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Fonction de prédiction
def predict_gesture(landmarks):
    if len(landmarks) == 63:
        prediction = model.predict([landmarks])[0]
        return labels[prediction]
    return None

# Détermination du gagnant
def determine_winner(p1, p2):
    if p1 == p2:
        return "Egalite"
    wins = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
    return "Joueur Droit" if wins[p1] == p2 else "Joueur Gauche"

# Capture de la webcam
cap = cv2.VideoCapture(0)
print("Appuie sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    height, width, _ = frame.shape

    gestures = []

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            gesture = predict_gesture(landmark_list)
            gestures.append(gesture)

            if gesture:
                cv2.putText(frame, f"Joueur {i+1} : {gesture}", (10, 40 + i * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afficher le gagnant si deux mains détectées
    if len(gestures) == 2 and all(gestures):
        winner = determine_winner(gestures[0], gestures[1])
        cv2.putText(frame, f"Gagnant : {winner}", (10, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.imshow("Pierre-Feuille-Ciseaux - 1v1", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
