import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

cap.set(3, 1280)  # Largeur
cap.set(4, 720)   # Hauteur

last_gesture_time = None
gesture_left = "Aucune main"
gesture_right = "Aucune main"
round_result = ""
waiting_for_input = False

def detecter_geste(hand_landmarks):
    fingers = []
    tips = [4, 8, 12, 16, 20]
    base = [0, 5, 9, 13, 17]

    for i in range(5):
        y_tip = hand_landmarks.landmark[tips[i]].y
        y_base = hand_landmarks.landmark[base[i]].y
        x_tip = hand_landmarks.landmark[tips[i]].x
        x_base = hand_landmarks.landmark[base[i]].x

        distance = ((x_tip - x_base) ** 2 + (y_tip - y_base) ** 2) ** 0.5

        if y_tip > y_base:
            fingers.append(0)
        elif y_tip < y_base - 0.07:
            fingers.append(1)
        else:
            fingers.append(0.5)

    if fingers == [0, 0, 0, 0, 0]:
        return "Pierre"
    elif sum(fingers) >= 4.5:
        return "Feuille"
    elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
        return "Ciseaux"

    return "Inconnu"


def determine_winner(gesture_left, gesture_right):
    if gesture_left == gesture_right:
        return "Egalite"
    elif (gesture_left == "Pierre" and gesture_right == "Ciseaux") or \
            (gesture_left == "Ciseaux" and gesture_right == "Feuille") or \
            (gesture_left == "Feuille" and gesture_right == "Pierre"):
        return "Gagnant: Main Gauche"
    else:
        return "Gagnant: Main Droite"


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    mid_x = width // 2
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)


    hands_positions = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detecter_geste(hand_landmarks)
            avg_x = sum([lm.x for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
            hands_positions.append((avg_x, gesture))

    hands_positions.sort(key=lambda x: x[0])

    gesture_left = "Aucune main"
    gesture_right = "Aucune main"

    if len(hands_positions) > 0:
        gesture_left = hands_positions[0][1]
    if len(hands_positions) > 1:
        gesture_right = hands_positions[1][1]


    if last_gesture_time is None:
        last_gesture_time = time.time()
    elif time.time() - last_gesture_time > 5:
        round_result = determine_winner(gesture_left, gesture_right)
        cv2.putText(frame, f"Resultat: {round_result}", (50, height // 2),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

        cv2.putText(frame, " Nouveau round 'N'", (50, height - 150),cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            last_gesture_time = time.time()
            gesture_left = "Aucune main"
            gesture_right = "Aucune main"
            round_result = ""

    cv2.rectangle(frame, (30, height - 90), (mid_x - 30, height - 40), (0, 0, 0), -1)  # Fond noir pour Joueur 1
    cv2.rectangle(frame, (mid_x + 30, height - 90), (width - 30, height - 40), (0, 0, 0), -1)  # Fond noir pour Joueur 2

    cv2.putText(frame, f"Main Gauche: {gesture_left}", (50, height - 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 150, 255),3)  # Orange
    cv2.putText(frame, f"Main Droite: {gesture_right}", (mid_x + 50, height - 50), cv2.FONT_HERSHEY_COMPLEX, 0.7,(50, 255, 50), 3)  # Vert clair

    cv2.imshow("Detection des mains - 2 Joueurs", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

