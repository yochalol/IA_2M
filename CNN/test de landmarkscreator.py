import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        if len(landmarks) == 63:
            return landmarks
    return None

def create_csv_from_images(folder_path, output_csv_path):
    data = []
    labels = []

    label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
    total, skipped = 0, 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)

            label = None
            for gesture in label_map:
                if gesture in filename.lower():
                    label = gesture
                    break

            if label is not None:
                landmarks = extract_landmarks(image_path)
                if landmarks is not None and len(landmarks) == 63:
                    data.append([label_map[label]] + landmarks)
                    total += 1
                else:
                    skipped += 1

    print(f"{total} images traitées avec succès.")
    print(f"{skipped} images ignorées (pas de main détectée ou nombre de landmarks incorrect).")

    columns = ['label'] + [f"x{i}, y{i}, z{i}" for i in range(21)]

    df = pd.DataFrame(data, columns=columns)

    df.to_csv(output_csv_path, index=False)
    print(f"CSV enregistré : {output_csv_path}")


train_folder = '../CNN/Rock-Paper-Scissors/train'
test_folder = '../CNN/Rock-Paper-Scissors/test'
validation_folder = '../CNN/Rock-Paper-Scissors/validation'

train_csv_path = 'CNN_landmarks_train.csv'
create_csv_from_images(train_folder, train_csv_path)

test_csv_path = 'CNN_landmarks_test.csv'
create_csv_from_images(test_folder, test_csv_path)

validation_csv_path = 'CNN_landmarks_validation.csv'
create_csv_from_images(validation_folder, validation_csv_path)
