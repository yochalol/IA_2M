
import tkinter as tk
import cv2
from PIL import Image, ImageTk

cap = None
video_en_cours = False
webcam_window = None

def capture_video(new_window_label):
    global video_en_cours
    if video_en_cours:
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            img = img.resize((640, 480), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)

            new_window_label.imgtk = imgtk
            new_window_label.configure(image=imgtk)

        new_window_label.after(10, capture_video, new_window_label)


def ouvrir_nouvelle_fenetre():
    global cap, video_en_cours, webcam_window
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        video_en_cours = True
        print("Caméra allumée.")
        webcam_window = tk.Toplevel()
        webcam_window.title("Webcam en dehors de l'interface")
        webcam_window.geometry("640x480")

        new_window_label = tk.Label(webcam_window)
        new_window_label.pack()

        capture_video(new_window_label)
    else:
        print("Erreur : Impossible d'ouvrir la caméra.")

def quitter():
    cap.release()
    fenetre.quit()




def eteindre_camera():
    global cap, video_en_cours, webcam_window
    if cap and cap.isOpened():
        cap.release()
        video_en_cours = False
        print("Caméra éteinte.")
        if webcam_window:
            webcam_window.destroy()
            webcam_window = None
    else:
        print("La caméra n'était pas allumée.")



fenetre = tk.Tk()
fenetre.title("Webcam")
fenetre.geometry("800x600")

frame = tk.Frame(fenetre)
frame.pack(padx=20, pady=20)

titre = tk.Label(frame, text=" Bienvenue dans notre Interface IA_2M", font=("Arial", 20))
titre.pack(pady=10)


button_quitter = tk.Button(frame, text="Quitter", command=quitter)
button_quitter.pack(side="left", padx=20, pady=20)

button_eteindre = tk.Button(frame, text="Éteindre Webcam", command=eteindre_camera)
button_eteindre.pack(side="bottom", padx=20, pady=20)

button_ouvrir_fenetre = tk.Button(frame, text="Ouvrir Webcam ", command=ouvrir_nouvelle_fenetre)
button_ouvrir_fenetre.pack(pady=20)

fenetre.mainloop()

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

button = tk.Button(frame, text="Démarrer/Arrêter la vidéo", command=toggle_video)
button.pack(pady=20)



########

def bouton_click():
    print("Le bouton a été cliqué")


#button_test = tk.Button(frame, text="Test", command=bouton_click)
#button_test.pack(side="left", padx=20, pady=20)
'''

