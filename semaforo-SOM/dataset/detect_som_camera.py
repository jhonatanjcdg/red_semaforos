import cv2
import numpy as np
from minisom import MiniSom
from utils import extract_rgb_features
import pickle

with open("som_trained.pkl", "rb") as f:
    som, label_map = pickle.load(f)

def map_to_state(label):
    return {
        'red': '🚨 PELIGRO',
        'yellow': '⚠️ ADVERTENCIA',
        'green': '✅ SEGURO'
    }.get(label, 'Desconocido')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[100:200, 100:200]  # Región central
    features = extract_rgb_features(roi)
    winner = som.winner(features)
    label = label_map.get(winner, "unknown")

    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), 2)
    cv2.putText(frame, map_to_state(label), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Detección SOM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
