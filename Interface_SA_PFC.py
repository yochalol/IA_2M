import threading
import mediapipe as mp
import random
import time
import tkinter as tk
import cv2
from PIL import Image, ImageTk



######################## INITIALISATION ################################

# Variable globale pour la caméra
cap = None
jeu_en_cours = False  # Global flag
jeu_1v1_en_cours = False
video_en_cours = False
webcam_window = None


def capture_video(new_window_label):
    global video_en_cours
    if video_en_cours:  # Vérifie si la caméra est allumée
        ret, frame = cap.read()  # Capture une image
        if ret:
            # Convertir l'image BGR (OpenCV) en RGB (Tkinter)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Redimensionner l'image pour qu'elle corresponde à la taille du label
            img = img.resize((640, 480), Image.Resampling.LANCZOS)

            # Convertir l'image pour Tkinter
            imgtk = ImageTk.PhotoImage(image=img)

            # Mettre à jour l'image du label dans la nouvelle fenêtre
            new_window_label.imgtk = imgtk
            new_window_label.configure(image=imgtk)

        # Rappeler cette fonction toutes les 10ms pour un flux vidéo continu
        new_window_label.after(10, capture_video, new_window_label)


def ouvrir_nouvelle_fenetre():
    global cap, video_en_cours, webcam_window
    cap = cv2.VideoCapture(0)  # Initialiser la caméra
    if cap.isOpened():
        video_en_cours = True  # Indiquer que la caméra est allumée
        print("Caméra allumée.")
        # Créer une nouvelle fenêtre pour la webcam
        webcam_window = tk.Toplevel()  # Créer une nouvelle fenêtre
        webcam_window.title("Webcam en dehors de l'interface")
        webcam_window.geometry("640x480")  # Taille de la nouvelle fenêtre

        # Créer un label pour afficher l'image de la webcam dans la nouvelle fenêtre
        new_window_label = tk.Label(webcam_window)
        new_window_label.pack()

        # Démarrer la capture vidéo dans la nouvelle fenêtre
        capture_video(new_window_label)
    else:
        print("Erreur : Impossible d'ouvrir la caméra.")




############################ Jeu: Jeu contre IA #############################"

def lancer_jeu_contre_IA():
    threading.Thread(target=jeu_pierre_feuille_ciseaux).start()

def jeu_pierre_feuille_ciseaux():
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

    global jeu_en_cours
    jeu_en_cours = True
    while cap.isOpened() and jeu_en_cours:
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
        cv2.imshow("Detection des mains - Pierre Feuille Ciseaux", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



############################## Jeu: 1 Vs 1 ###################################

def lancer_jeu_1v1():
    threading.Thread(target=jeu_1v1).start()

def jeu_1v1():
    global jeu_1v1_en_cours
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

        cv2.imshow("Detection des mains - 2 Joueurs", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()





######################## FERMETURE #############################


def eteindre_jeu_contre_IA():
    global jeu_en_cours
    jeu_en_cours = False  # Cela fera sortir la boucle while dans le thread du jeu
    print("Jeu arrêté.")


def eteindre_jeu_1v1():
    global jeu_1v1_en_cours
    jeu_1v1_en_cours = False
    print("Jeu 1v1 arrêté.")


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


def quitter():
    global cap
    if cap and cap.isOpened():
        cap.release()
        print("Caméra libérée.")
    else:
        print("Aucune caméra à libérer.")
    fenetre.quit()




###################### Tkinter #####################

# Créer une fenêtre Tkinter
fenetre = tk.Tk()
fenetre.title("Webcam")
fenetre.geometry("800x600")

# Création cadre pour la webcam + autres éléments
frame = tk.Frame(fenetre)
frame.pack(padx=20, pady=20)

# Titre de l'interface
titre = tk.Label(frame, text=" Bienvenue dans notre Interface IA_2M", font=("Arial", 20))
titre.pack(pady=10)



###################### Bouton #####################

button_eteindre = tk.Button(frame, text="Éteindre Webcam", command=eteindre_camera)
button_eteindre.pack(pady=10)

button_ouvrir_fenetre = tk.Button(frame, text="Ouvrir Webcam ", command=ouvrir_nouvelle_fenetre)
button_ouvrir_fenetre.pack(pady=10)

button_jeu_contre_IA = tk.Button(frame, text="Lancer Contre IA ", command=lancer_jeu_contre_IA )
button_jeu_contre_IA.pack(pady=10)

button_eteindre_jeu_contre_IA = tk.Button(frame, text="Éteindre Contre IA", command=eteindre_jeu_contre_IA)
button_eteindre_jeu_contre_IA.pack(pady=10)

button_jeu_1v1 = tk.Button(frame, text="Lancer 1 vs 1", command=lancer_jeu_1v1)
button_jeu_1v1.pack(pady=10)

button_eteindre_1v1 = tk.Button(frame, text="Éteindre 1 vs 1", command=eteindre_jeu_1v1)
button_eteindre_1v1.pack(pady=10)

button_quitter = tk.Button(frame, text="Quitter", command=quitter)
button_quitter.pack(pady=10)



######################## FIN ##############################

# Ouverture fenêtre
fenetre.mainloop()

# Libérer la caméra lorsque la fenêtre principale est fermée
if cap:
    cap.release()


