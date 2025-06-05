import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo
model = load_model("traffic_light_model.h5")
classes = ['Peligro', 'Advertencia', 'Seguro']  # Rojo, Amarillo, Verde

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesar la imagen
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predecir
    prediction = model.predict(img)[0]
    label = classes[np.argmax(prediction)]

    # Mostrar resultado
    cv2.putText(frame, f"Estado: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Detección de Semáforo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
