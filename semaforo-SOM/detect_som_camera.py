import cv2
import numpy as np
from minisom import MiniSom
from utils import extract_rgb_features
import pickle

# 1) Cargar SOM y label_map
with open("model/som_trained.pkl", "rb") as f:
    som, label_map = pickle.load(f)

def map_to_state(label):
    """
    Traduce la etiqueta de color a su estado:
      - 'red'    → '🚨 PELIGRO'
      - 'yellow' → '⚠️ ADVERTENCIA'
      - 'green'  → '✅ SEGURO'
    """
    return {
        'red': '🚨 PELIGRO',
        'yellow': '⚠️ ADVERTENCIA',
        'green': '✅ SEGURO'
    }.get(label, 'Desconocido')

# 2) Inicializar captura de cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Iniciando detección por SOM. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Definimos una ROI en el centro (ajusta según tu cámara y posición del semáforo)
    h, w, _ = frame.shape
    roi_size = 100
    x1 = w//2 - roi_size//2
    y1 = h//2 - roi_size//2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    roi = frame[y1:y2, x1:x2]
    features = extract_rgb_features(roi)

    # 3) Obtener neurona ganadora y etiqueta previsional
    winner = som.winner(features)
    label = label_map.get(winner, "unknown")

    # 4) Mostrar la ROI y el resultado en pantalla
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    estado = map_to_state(label)
    cv2.putText(frame, estado, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Opcional: mostrar neurona ganadora en consola para diagnóstico
    # print(f"Winner neuron: {winner}, Label predicho: {label}")

    cv2.imshow("Detección SOM Semáforo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
