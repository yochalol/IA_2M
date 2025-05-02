
''' Modèle de identification, pierre, feuille, ciseaux '''

import cv2
import mediapipe as mp



# Initialisation de la détection des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

# Définir une taille plus grande pour la fenêtre
cap.set(3, 1280)  # Largeur
cap.set(4, 720)   # Hauteur


# Fonction pour détecter les gestes d'une main
def detecter_geste(hand_landmarks):
    fingers = []
    tips = [4, 8, 12, 16, 20]  # Indices des extrémités des doigts
    base = [0, 5, 9, 13, 17]   # Base des doigts (poignet et articulations)

    for i in range(5):
        y_tip = hand_landmarks.landmark[tips[i]].y
        y_base = hand_landmarks.landmark[base[i]].y
        x_tip = hand_landmarks.landmark[tips[i]].x
        x_base = hand_landmarks.landmark[base[i]].x

        # Vérifier si le bout du doigt est très proche de la paume (Pierre)
        distance = ((x_tip - x_base) ** 2 + (y_tip - y_base) ** 2) ** 0.5

        if y_tip > y_base:  # Le doigt est replié si l'extrémité est plus basse que la base
            fingers.append(0)
        elif y_tip < y_base - 0.07:  # Doigt levé (Feuille ou Ciseaux)
            fingers.append(1)
        else:
            fingers.append(0.5)

    # Reconnaissance des gestes
    if fingers == [0, 0, 0, 0, 0]:  # Tous les doigts sont repliés
        return "Pierre"
    elif sum(fingers) >= 4.5:  # Tous les doigts levés
        return "Feuille"
    elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
        return "Ciseaux"

    return "Inconnu"


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Diviser l'écran en deux parties
    height, width, _ = frame.shape
    mid_x = width // 2  # Milieu de l'écran
    frame = cv2.flip(frame, 1)  # ✅ Inversion horizontale (corrige l'effet miroir)

    # Conversion en RGB pour MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Initialisation des gestes des deux mains
    gesture_left = "Aucune main"
    gesture_right = "Aucune main"

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detecter_geste(hand_landmarks)

            # Déterminer si c'est la main gauche ou droite
            hand_label = results.multi_handedness[idx].classification[0].label

            if hand_label == "Left":
                gesture_left = gesture
            else:
                gesture_right = gesture

    # Affichage des résultats
    # ✅ Ajout d'un fond noir sous les textes pour améliorer la lisibilité
    cv2.rectangle(frame, (30, height - 90), (mid_x - 30, height - 40), (0, 0, 0), -1)  # Fond noir pour Joueur 1
    cv2.rectangle(frame, (mid_x + 30, height - 90), (width - 30, height - 40), (0, 0, 0), -1)  # Fond noir pour Joueur 2

    # ✅ Affichage des résultats avec une police plus lisible et des couleurs améliorées
    cv2.putText(frame, f"Main Droite: {gesture_left}", (50, height - 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 150, 255),
                3)  # Orange
    cv2.putText(frame, f"Main Gauche: {gesture_right}", (mid_x + 50, height - 50), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                (50, 255, 50), 3)  # Vert clair

    # Afficher la vidéo
    #cv2.imshow("Détection des mains - Correction miroir", frame)
    cv2.imshow("Détection des mains - 2 Joueurs", frame)

    # Quitter avec 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
