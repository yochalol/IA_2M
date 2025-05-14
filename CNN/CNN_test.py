import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import pickle


df_test = pd.read_csv('CNN_landmarks_test.csv')

X_test = df_test.drop(columns=['label']).values
y_test = df_test['label'].values

model = load_model('modele_CNN.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

y_test_encoded = label_encoder.transform(y_test)
X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(y_test_encoded, y_pred_classes, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df = report_df.drop('accuracy', axis=0)
report_df = report_df.round(4)

accuracy = np.sum(y_pred_classes == y_test_encoded) / len(y_test_encoded) * 100
accuracy = round(accuracy, 4)

print("Tableau des résultats sur le jeu de test :")
print(report_df)
print("\nAccuracy sur l'ensemble du jeu de test : {:.4f}%".format(accuracy))
