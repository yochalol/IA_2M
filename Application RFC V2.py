import cv2
import mediapipe as mp
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Charger le modèle
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# MediaPipe init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Labels des classes
labels = ["paper", "rock", "scissors"]

# Setup matplotlib
plt.ion()
fig, ax = plt.subplots()
bar_container = ax.bar(labels, [0, 0, 0], color=["skyblue", "lightcoral", "lightgreen"])
ax.set_ylim(0, 100)
ax.set_ylabel("Probabilité (%)")
plt.title("Prédiction temps réel")

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    height, width, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            if len(landmark_list) == 63:
                prediction = model.predict_proba([landmark_list])[0]

                # Texte dans l'image
                y_offset = 10
                for i, label in enumerate(labels):
                    percent = int(prediction[i] * 100)
                    text = f"{label}: {percent}%"
                    cv2.putText(frame, text, (10, height - y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30

                # MAJ du graphe matplotlib
                for rect, prob in zip(bar_container, prediction):
                    rect.set_height(prob * 100)
                fig.canvas.draw()
                fig.canvas.flush_events()

    cv2.imshow("Rock Paper Scissors - Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
