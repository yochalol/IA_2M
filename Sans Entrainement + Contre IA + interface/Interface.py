
import tkinter as tk
import cv2
from PIL import Image, ImageTk

# Variable globale pour la caméra
cap = None

# Variable pour savoir si la caméra est allumée
video_en_cours = False


# Variable pour la fenêtre de la webcam
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




# Fonction pour quitter l'application et fermer la caméra
def quitter():
    cap.release()  # Libérer la caméra
    fenetre.quit()  # Fermer la fenêtre Tkinter




def eteindre_camera():
    global cap, video_en_cours, webcam_window
    if cap and cap.isOpened():
        cap.release()  # Libérer la caméra
        video_en_cours = False  # Arrêter le flux vidéo
        print("Caméra éteinte.")
        if webcam_window:  # Si la fenêtre webcam existe
            webcam_window.destroy()  # Fermer la fenêtre de la webcam
            webcam_window = None  # Réinitialiser la variable
    else:
        print("La caméra n'était pas allumée.")





# Créer une fenêtre Tkinter
fenetre = tk.Tk()
fenetre.title("Webcam")
fenetre.geometry("800x600")

# Créer un cadre pour la webcam et les autres éléments
frame = tk.Frame(fenetre)
frame.pack(padx=20, pady=20)

# Titre de l'interface
titre = tk.Label(frame, text=" Bienvenue dans notre Interface IA_2M", font=("Arial", 20))
titre.pack(pady=10)



## Bouton ##

# Ajouter un bouton "Quitter"
button_quitter = tk.Button(frame, text="Quitter", command=quitter)
button_quitter.pack(side="left", padx=20, pady=20)



button_eteindre = tk.Button(frame, text="Éteindre Webcam", command=eteindre_camera)
button_eteindre.pack(side="bottom", padx=20, pady=20)


# Ajouter un bouton pour ouvrir la nouvelle fenêtre avec la webcam
button_ouvrir_fenetre = tk.Button(frame, text="Ouvrir Webcam ", command=ouvrir_nouvelle_fenetre)
button_ouvrir_fenetre.pack(pady=20)




# Lancer la fenêtre Tkinter
fenetre.mainloop()

# Libérer la caméra lorsque la fenêtre principale est fermée
if cap:
    cap.release()












'''
# Variable pour suivre si la vidéo est en cours
video_en_cours = True

def toggle_video():
    global video_en_cours
    if video_en_cours:
        video_en_cours = False
        print("Capture vidéo arrêtée.")
    else:
        video_en_cours = True
        print("Capture vidéo démarrée.")
        capture_video()

# Créer un bouton pour démarrer/arrêter la vidéo
button = tk.Button(frame, text="Démarrer/Arrêter la vidéo", command=toggle_video)
button.pack(pady=20)



########

# Ajouter un bouton à l'interface
def bouton_click():
    print("Le bouton a été cliqué")


#button_test = tk.Button(frame, text="Test", command=bouton_click)
#button_test.pack(side="left", padx=20, pady=20)
'''

