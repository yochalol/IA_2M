import threading
import mediapipe as mp
import random
import time
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

######################## Chargement des modèles ################################

# Chargement modèle CNN
model_cnn = load_model("./CNN/modele_CNN.h5")
with open("./CNN/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("./CNN/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


cnn_labels = label_encoder.classes_


# Chargement modèle Random Forest
with open("random_forest_model.pkl", "rb") as f:
    model_rf = pickle.load(f)

labels_rf = ["paper", "rock", "scissors"]


######################## INITIALISATION ################################

# Variable globale pour la caméra
cap = None
jeu_en_cours = False  # Global flag
jeu_1v1_en_cours = False
jeu_cnn_en_cours = False

video_en_cours = False
webcam_window = None


def capture_video():
    global video_en_cours
    if video_en_cours:
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)

            label_video.imgtk = imgtk
            label_video.configure(image=imgtk)

        label_video.after(10, capture_video)

def ouvrir_camera():
    global cap, video_en_cours
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        video_en_cours = True
        print("Caméra allumée.")
        capture_video()
    else:
        print("Erreur : Impossible d'ouvrir la caméra.")

def afficher_frame_dans_label(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((640, 480), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=img)
    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)

def predict_gesture_cnn(landmarks):
    if len(landmarks) == 63:
        scaled = scaler.transform([landmarks])
        prediction = model_cnn.predict(scaled, verbose=0)
    for bar, prob in zip(bars, prediction[0]):
        bar.set_height(prob * 100)
    canvas.draw()
    return cnn_labels[np.argmax(prediction)]

def predict_gesture_rf(landmarks):
    if len(landmarks) == 63:
        prediction = model_rf.predict([landmarks])[0]
        return labels_rf[prediction]
    return None

############################ Jeu: Jeu contre IA #############################

def lancer_jeu_contre_IA():
    global jeu_en_cours
    stop_tous_les_jeux()
    #eteindre_camera()
    time.sleep(0.2)
    jeu_en_cours = True
    threading.Thread(target=jeu_pierre_feuille_ciseaux).start()

def jeu_pierre_feuille_ciseaux():
    global jeu_en_cours
    global cap
    jeu_en_cours = True

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    last_gesture_time = None
    gesture_left = "Aucune main"
    gesture_right = "Aucune main"
    round_result = ""
    ia_choice_made = False
    player_gesture = None
    start_time = None

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

    def ia_choice(player_gesture):
        if player_gesture == "Pierre":
            return "Feuille"
        elif player_gesture == "Feuille":
            return "Ciseaux"
        elif player_gesture == "Ciseaux":
            return "Pierre"
        return random.choice(["Pierre", "Feuille", "Ciseaux"])

    def determine_winner(gesture_left, gesture_right):
        if gesture_left == gesture_right:
            return "Egalite"
        elif (gesture_left == "Pierre" and gesture_right == "Ciseaux") or \
             (gesture_left == "Ciseaux" and gesture_right == "Feuille") or \
             (gesture_left == "Feuille" and gesture_right == "Pierre"):
            return "Gagnant: Joueur"
        else:
            return "Gagnant: IA"


    while cap.isOpened() and jeu_en_cours:
        if not jeu_en_cours:
            break
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

        if len(hands_positions) > 0:
            gesture_left = hands_positions[0][1]
            if player_gesture is None:
                player_gesture = gesture_left

        if player_gesture and start_time is None:
            start_time = time.time()

        if player_gesture and start_time and time.time() - start_time >= 1 and not ia_choice_made:
            gesture_right = ia_choice(player_gesture)
            ia_choice_made = True

        if last_gesture_time is None:
            last_gesture_time = time.time()
        elif time.time() - last_gesture_time > 3:
            round_result = determine_winner(gesture_left, gesture_right)
            cv2.putText(frame, f"Resultat: {round_result}", (50, height // 2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(frame, " Nouveau round 'N'", (50, height - 150), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                last_gesture_time = time.time()
                gesture_left = "Aucune main"
                gesture_right = "Aucune main"
                round_result = ""
                ia_choice_made = False
                player_gesture = None
                start_time = None

        cv2.rectangle(frame, (30, height - 90), (mid_x - 30, height - 40), (0, 0, 0), -1)
        cv2.rectangle(frame, (mid_x + 30, height - 90), (width - 30, height - 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Joueur: {gesture_left}", (50, height - 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 150, 255), 3)
        cv2.putText(frame, f"IA: {gesture_right}", (mid_x + 50, height - 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (50, 255, 50), 3)

        cv2.imshow("Jeu Contre IA", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Jeu Contre IA", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyWindow("Jeu Contre IA")




############################## Jeu: 1 Vs 1 ###################################

def lancer_jeu_1v1():
    global jeu_1v1_en_cours
    stop_tous_les_jeux()
    #eteindre_camera()
    time.sleep(0.2)
    jeu_1v1_en_cours = True
    threading.Thread(target=jeu_1v1).start()

def jeu_1v1():
    global jeu_1v1_en_cours
    global cap
    jeu_1v1_en_cours = True

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Largeur
    cap.set(4, 720)  # Hauteur

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

        # Reconnaissance des gestes
        if fingers == [0, 0, 0, 0, 0]:  # Les doigts sont repliés
            return "Pierre"
        elif sum(fingers) >= 4.5:  # Les doigts levés
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

    while cap.isOpened() and jeu_1v1_en_cours:
        if not jeu_1v1_en_cours:
            break
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        mid_x = width // 2
        frame = cv2.flip(frame, 1)

        # Conversion en RGB pour MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hands_positions = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detecter_geste(hand_landmarks)
                avg_x = sum([lm.x for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
                hands_positions.append((avg_x, gesture))

        # Trier les mains de gauche à droite
        hands_positions.sort(key=lambda x: x[0])
        gesture_left = "Aucune main"
        gesture_right = "Aucune main"

        if len(hands_positions) > 0:
            gesture_left = hands_positions[0][1]  # Main la plus à gauche
        if len(hands_positions) > 1:
            gesture_right = hands_positions[1][1]  # Main la plus à droite

        if last_gesture_time is None:
            last_gesture_time = time.time()
        elif time.time() - last_gesture_time > 5:  # Attendre 5 secondes
            round_result = determine_winner(gesture_left, gesture_right)
            cv2.putText(frame, f"Resultat: {round_result}", (50, height // 2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                        3)
            cv2.putText(frame, " Nouveau round 'N'", (50, height - 150), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):  # Code ASCII pour la touche 'n'
                last_gesture_time = time.time()  # Redémarrer le timer pour un nouveau round
                gesture_left = "Aucune main"
                gesture_right = "Aucune main"
                round_result = ""

        # Affichage des résultats pendant le round
        cv2.rectangle(frame, (30, height - 90), (mid_x - 30, height - 40), (0, 0, 0), -1)  # Fond noir pour Joueur 1
        cv2.rectangle(frame, (mid_x + 30, height - 90), (width - 30, height - 40), (0, 0, 0),-1)  # Fond noir pour Joueur 2

        cv2.putText(frame, f"Main Gauche: {gesture_left}", (50, height - 50), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 150, 255), 3)  # Orange
        cv2.putText(frame, f"Main Droite: {gesture_right}", (mid_x + 50, height - 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (50, 255, 50), 3)  # Vert clair

        cv2.imshow("Jeu 1vs1", frame)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty("Jeu 1vs1", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyWindow("Jeu 1vs1")


####################### 1vs1 CNN ##############################

def lancer_jeu_cnn():
    global jeu_cnn_en_cours
    stop_tous_les_jeux()
    #eteindre_camera()
    time.sleep(0.2)
    jeu_cnn_en_cours = True
    threading.Thread(target=jeu_cnn_1v1).start()


def jeu_cnn_1v1():
    global cap
    global jeu_cnn_en_cours

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    gestures = []

    def determine_winner_cnn(p1, p2):
        if p1 == p2:
            return "Egalite"
        wins = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
        return "Joueur Droit" if wins[p1] == p2 else "Joueur Gauche"


    while cap.isOpened() and jeu_cnn_en_cours:
        if not jeu_cnn_en_cours:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        height, width, _ = frame.shape
        gestures.clear()

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])
                gesture = predict_gesture_cnn(landmark_list)
                gestures.append(gesture)
                if gesture:
                    cv2.putText(frame, f"Joueur {i+1}: {gesture}", (10, 30 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if len(gestures) == 2 and all(gestures):
            winner = determine_winner_cnn(gestures[0], gestures[1])
            cv2.putText(frame, f"Gagnant: {winner}", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        cv2.imshow("Jeu CNN", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty("Jeu CNN", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyWindow("Jeu CNN")



def ouvrir_camera_et_graphique_CNN():
    global cap, video_en_cours
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        video_en_cours = True
        print("Caméra avec graphique allumée.")
        update_camera_et_graphique_CNN()
    else:
        print("Erreur : Impossible d'ouvrir la caméra.")

def update_camera_et_graphique_CNN():
    global video_en_cours
    if not video_en_cours:
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7).process(rgb)

    height, width, _ = frame.shape

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        landmark_list = []
        for lm in hand_landmarks.landmark:
            landmark_list.extend([lm.x, lm.y, lm.z])

        if len(landmark_list) == 63:
            scaled = scaler.transform([landmark_list])
            prediction = model_cnn.predict(scaled, verbose=0)[0]

            # MAJ du graphe matplotlib
            for bar, prob in zip(bars, prediction):
                bar.set_height(prob * 100)
            canvas.draw()

            # Ajout des pourcentages sur l'image
            y_offset = 30
            for i, label in enumerate(["paper", "rock", "scissors"]):
                percent = int(prediction[i] * 100)
                text = f"{label}: {percent}%"
                cv2.putText(frame, text, (10, height - y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

    # Affichage dans Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((400, 400), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=img)
    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)

    label_video.after(20, update_camera_et_graphique_CNN)

##################### 1 Vs 1 RF #############################

def lancer_jeu_rf():
    global jeu_rf_en_cours
    stop_tous_les_jeux()
    time.sleep(0.2)
    jeu_rf_en_cours = True
    threading.Thread(target=jeu_rf).start()
def jeu_rf():
    global jeu_rf_en_cours
    global cap
    jeu_rf_en_cours = True

    # Initialisation de MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Largeur
    cap.set(4, 720)  # Hauteur

    last_gesture_time = None
    gesture_left = "Aucune main"
    gesture_right = "Aucune main"
    round_result = ""

    # Fonction pour déterminer le gagnant
    def determine_winner_rf(p1, p2):
        if p1 == p2:
            return "Egalite"
        wins = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
        return "Gagnant: Joueur Gauche" if wins[p1] == p2 else "Gagnant: Joueur Droit"

    while cap.isOpened() and jeu_rf_en_cours:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        mid_x = width // 2
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        gestures = []

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])

                # Prédiction du geste du joueur avec Random Forest
                gesture = predict_gesture_rf(landmark_list)  # Utilise la fonction de prédiction RF
                gestures.append(gesture)

                if gesture:
                    cv2.putText(frame, f"Joueur {i+1} : {gesture}", (10, 40 + i * 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Afficher le gagnant si deux mains détectées
        if len(gestures) == 2 and all(gestures):
            winner = determine_winner_rf(gestures[0], gestures[1])
            cv2.putText(frame, f"Gagnant : {winner}", (10, height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        # Affichage du cadre
        cv2.imshow("Jeu 1v1 Random Forest", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow("Jeu 1v1 Random Forest")




def ouvrir_camera_et_graphique_RF():
    global cap, video_en_cours
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        video_en_cours = True
        print("Caméra avec graphique allumée.")
        update_camera_et_graphique_RF()
    else:
        print("Erreur : Impossible d'ouvrir la caméra.")

def update_camera_et_graphique_RF():
    global video_en_cours
    if not video_en_cours:
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7).process(rgb)

    height, width, _ = frame.shape

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        landmark_list = []
        for lm in hand_landmarks.landmark:
            landmark_list.extend([lm.x, lm.y, lm.z])

        if len(landmark_list) == 63:
            scaled = scaler.transform([landmark_list])
            prediction = model_cnn.predict(scaled, verbose=0)[0]

            # MAJ du graphe matplotlib
            for bar, prob in zip(bars, prediction):
                bar.set_height(prob * 100)
            canvas.draw()

            # Ajout des pourcentages sur l'image
            y_offset = 30
            for i, label in enumerate(["paper", "rock", "scissors"]):
                percent = int(prediction[i] * 100)
                text = f"{label}: {percent}%"
                cv2.putText(frame, text, (10, height - y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

    # Affichage dans Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((400, 400), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=img)
    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)

    label_video.after(20, update_camera_et_graphique_RF)



######################## FERMETURE #############################

def eteindre_jeu_contre_IA():
    global jeu_en_cours
    jeu_en_cours = False  # Cela fera sortir la boucle while dans le thread du jeu
    print("Jeu arrêté.")

def eteindre_jeu_1v1():
    global jeu_1v1_en_cours
    jeu_1v1_en_cours = False
    print("Jeu 1v1 arrêté.")

def eteindre_jeu_CNN():
    global jeu_cnn_en_cours
    jeu_cnn_en_cours = False
    print("Jeu CNN arrêté.")

def eteindre_camera():
    global cap, video_en_cours, webcam_window
    if cap and cap.isOpened():
        cap.release()
        video_en_cours = False  # Arrêter le flux vidéo
        print("Caméra éteinte.")
        if webcam_window:  # Si la fenêtre webcam existe
            webcam_window.destroy()  # Fermer la fenêtre de la webcam
            webcam_window = None  # Réinitialiser la variable
    else:
        print("La caméra n'était pas allumée.")


# Mofifier la fonction pour fermer le graphique et le Weebcame du modele de diane
def eteindre_camera_et_graphique():
    global cap, video_en_cours
    if cap and cap.isOpened():
        cap.release()
        print("Caméra avec graphique éteinte.")
    else:
        print("La caméra était déjà éteinte.")
    video_en_cours = False
    for bar in bars: # Réinitialiser les barres du graphique à 0
        bar.set_height(0)
    canvas.draw()
    # Nettoyer l'affichage de la vidéo dans Tkinter
    label_video.configure(image='')  # Vider l'image


def stop_tous_les_jeux():
    global jeu_en_cours, jeu_1v1_en_cours, jeu_cnn_en_cours, jeu_rf_en_cours
    jeu_en_cours = False
    jeu_1v1_en_cours = False
    jeu_cnn_en_cours = False
    jeu_rf_en_cours = False
    for bar in bars:
        bar.set_height(0)
    canvas.draw()
    print("Tous les jeux sont arrêtés.")


def quitter():
    global cap, video_en_cours
    stop_tous_les_jeux()
    if cap and cap.isOpened():
        cap.release()
        print("Caméra libérée.")
    video_en_cours = False
    cv2.destroyAllWindows()
    fenetre.destroy()


###################### Tkinter #####################

# Créer une fenêtre Tkinter
fenetre = tk.Tk()
fenetre.title("Webcam")
fenetre.geometry("1000x600")

# Création cadre pour la webcam + autres éléments
frame = tk.Frame(fenetre)
frame.pack(padx=20, pady=20)

# Titre de l'interface
titre = tk.Label(frame, text=" Bienvenue dans notre Interface IA_2M", font=("Arial", 20))
titre.pack(pady=10)


###### FRAME GAUCHE ######
frame_gauche = tk.Frame(fenetre, width=200)
frame_gauche.pack(side="left", fill="y", padx=10, pady=10)


###### FRAME DROITE ######
frame_droite = tk.Frame(fenetre)
frame_droite.pack(side="right", expand=True, fill="both", padx=10, pady=10)

label_video = tk.Label(frame_droite, bg="black", width=400, height=400)
label_video.pack(side="left", padx=10, pady=10)



###################### GRAPHIQUE MATPLOLIB ######################

# Créer la figure matplotlib
fig, ax = plt.subplots(figsize=(4, 2))
bars = ax.bar(["paper", "rock", "scissors"], [0, 0, 0], color=["skyblue", "lightcoral", "lightgreen"])

ax.set_ylim(0, 100)
ax.set_ylabel("Probabilité (%)")
ax.set_title("Prédiction temps réel")

canvas = FigureCanvasTkAgg(fig, master=frame_droite)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side="right", padx=10, pady=10)


###################### Bouton #####################

#Titre: MODES DE JEU
label_section_jeux = tk.Label(frame_gauche, text="Modes de jeu", font=("Arial", 12, "bold"))
label_section_jeux.pack(pady=(20, 0))

#button_ouvrir_fenetre = tk.Button(frame_gauche, text="Ouvrir Webcam ", command=ouvrir_camera)
#button_ouvrir_fenetre.pack(pady=5)

#button_eteindre = tk.Button(frame_gauche, text="Éteindre Webcam", command=eteindre_camera)
#button_eteindre.pack(pady=5)

button_jeu_contre_IA = tk.Button(frame_gauche, text="Lancer Contre IA ", command=lancer_jeu_contre_IA )
button_jeu_contre_IA.pack(pady=5)

#button_eteindre_jeu_contre_IA = tk.Button(frame_gauche, text="Éteindre Contre IA", command=eteindre_jeu_contre_IA)
#button_eteindre_jeu_contre_IA.pack(pady=5)

button_jeu_1v1 = tk.Button(frame_gauche, text="Lancer 1 vs 1", command=lancer_jeu_1v1)
button_jeu_1v1.pack(pady=5)

#button_eteindre_1v1 = tk.Button(frame_gauche, text="Éteindre 1 vs 1", command=eteindre_jeu_1v1)
#button_eteindre_1v1.pack(pady=5)

button_jeu_cnn = tk.Button(frame_gauche, text="Lancer Jeu CNN", command=lancer_jeu_cnn)
button_jeu_cnn.pack(pady=5)


button_jeu_rf = tk.Button(frame_gauche, text="Lancer Jeu RF", command=lancer_jeu_rf)
button_jeu_rf.pack(pady=5)


#button_eteindre_CNN = tk.Button(frame_gauche, text="Éteindre CNN", command=eteindre_jeu_CNN)
#button_eteindre_CNN.pack(pady=5)

button_stop_jeux = tk.Button(frame_gauche, text="Stop Tous les Jeux", command=stop_tous_les_jeux)
button_stop_jeux.pack(pady=5)

button_quitter = tk.Button(frame_gauche, text="Quitter", command=quitter)
button_quitter.pack(pady=5)

#Titre: CAMERA ET GRAPHIQUE
label_section_graphique = tk.Label(frame_gauche, text="Caméra et Graphique", font=("Arial", 12, "bold"))
label_section_graphique.pack(pady=(10, 0))  # Espacement haut uniquement

button_graphique_webcam = tk.Button(frame_gauche, text="Webcam + Graphique = Modèle CNN ", command=ouvrir_camera_et_graphique_CNN)
button_graphique_webcam.pack(pady=5)

button_graphique_webcam = tk.Button(frame_gauche, text="Webcam + Graphique = Modèle RF ", command=ouvrir_camera_et_graphique_RF)
button_graphique_webcam.pack(pady=5)

## il faudrait que ça eteigne les graphique des deux modèles
button_eteindre_graphique = tk.Button(frame_gauche, text="Éteindre Graphiques + Caméras", command=eteindre_camera_et_graphique)
button_eteindre_graphique.pack(pady=5)









######################## FIN ##############################

# Ouverture fenêtre
fenetre.protocol("WM_DELETE_WINDOW", quitter)
fenetre.mainloop()

# Libérer la caméra lorsque la fenêtre principale est fermée
if cap:
    cap.release()


