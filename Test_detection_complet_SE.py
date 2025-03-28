import cv2
import mediapipe as mp
import time

# Initialisation de la détection des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

# Définir une taille plus grande pour la fenêtre
cap.set(3, 1280)  # Largeur
cap.set(4, 720)   # Hauteur

# Initialisation des variables
last_gesture_time = None
gesture_left = "Aucune main"
gesture_right = "Aucune main"
round_result = ""
waiting_for_input = False


# Fonction pour détecter les gestes d'une main
def detecter_geste(hand_landmarks):
    fingers = []
    tips = [4, 8, 12, 16, 20]  # Indices des extrémités des doigts
    base = [0, 5, 9, 13, 17]  # Base des doigts (poignet et articulations)

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


# Fonction pour déterminer le gagnant
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

    # Diviser l'écran en deux parties
    height, width, _ = frame.shape
    mid_x = width // 2  # Milieu de l'écran
    frame = cv2.flip(frame, 1)  # Inversion horizontale (corrige l'effet miroir)

    # Conversion en RGB pour MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Liste pour stocker les mains avec leur position moyenne en X
    hands_positions = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detecter_geste(hand_landmarks)
            # Calculer la position moyenne en X de la main
            avg_x = sum([lm.x for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
            hands_positions.append((avg_x, gesture))  # Stocker la position X et le geste

    # Trier les mains de gauche à droite
    hands_positions.sort(key=lambda x: x[0])

    # Réinitialisation des gestes
    gesture_left = "Aucune main"
    gesture_right = "Aucune main"

    # Attribution des gestes selon la position
    if len(hands_positions) > 0:
        gesture_left = hands_positions[0][1]  # Main la plus à gauche
    if len(hands_positions) > 1:
        gesture_right = hands_positions[1][1]  # Main la plus à droite


    # Si 5 secondes sont écoulées depuis le début du round, afficher les résultats
    if last_gesture_time is None:
        last_gesture_time = time.time()
    elif time.time() - last_gesture_time > 5:  # Attendre 5 secondes
        round_result = determine_winner(gesture_left, gesture_right)
        cv2.putText(frame, f"Resultat: {round_result}", (50, height // 2),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

        # Afficher un message pour passer au prochain round
        cv2.putText(frame, " Nouveau round 'N'", (50, height - 150),cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        # Attendre que l'utilisateur appuie sur 'N' pour recommencer un autre round
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):  # Code ASCII pour la touche 'n'
            last_gesture_time = time.time()  # Redémarrer le timer pour un nouveau round
            gesture_left = "Aucune main"
            gesture_right = "Aucune main"
            round_result = ""

    # Affichage des résultats pendant le round
    cv2.rectangle(frame, (30, height - 90), (mid_x - 30, height - 40), (0, 0, 0), -1)  # Fond noir pour Joueur 1
    cv2.rectangle(frame, (mid_x + 30, height - 90), (width - 30, height - 40), (0, 0, 0), -1)  # Fond noir pour Joueur 2

    cv2.putText(frame, f"Main Gauche: {gesture_left}", (50, height - 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 150, 255),3)  # Orange
    cv2.putText(frame, f"Main Droite: {gesture_right}", (mid_x + 50, height - 50), cv2.FONT_HERSHEY_COMPLEX, 0.7,(50, 255, 50), 3)  # Vert clair

    # Afficher la vidéo
    cv2.imshow("Detection des mains - 2 Joueurs", frame)

    # Quitter avec 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

