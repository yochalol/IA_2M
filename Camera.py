# Ouvrir la caméra (0 pour la webcam par défaut)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra")
    exit()

while True:
    # Lire une frame (image) de la caméra
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de récupérer l'image")
        break

    # Afficher l'image capturée
    cv2.imshow("Flux vidéo", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
