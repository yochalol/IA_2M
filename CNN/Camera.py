import cv2
print("hey")
print("Your OpenCV version : " + cv2.__version__)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de récupérer l'image")
        break

    cv2.imshow("Flux vidéo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
