import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Configuración de rutas y clases
RAW_DIR = 'data/raw'
OUTPUT_NPZ = 'data/semaforo_dataset.npz'
CLASSES = {'red': 0, 'yellow': 1, 'green': 2}

def compute_hsv_means(img_path):
    frame = cv2.imread(img_path)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # máscaras para rojo (dos rangos), amarillo y verde
    m_r = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255])) \
        + cv2.inRange(hsv, np.array([160,100,100]), np.array([179,255,255]))
    m_y = cv2.inRange(hsv, np.array([15,100,100]), np.array([35,255,255]))
    m_g = cv2.inRange(hsv, np.array([40,100,100]), np.array([90,255,255]))
    mask = m_r | m_y | m_g
    h = cv2.mean(hsv[:,:,0], mask=mask)[0]
    s = cv2.mean(hsv[:,:,1], mask=mask)[0]
    v = cv2.mean(hsv[:,:,2], mask=mask)[0]
    return [h, s, v]

def main():
    X, y = [], []
    for color, label in CLASSES.items():
        folder = os.path.join(RAW_DIR, color)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            feats = compute_hsv_means(path)
            X.append(feats)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez(OUTPUT_NPZ,
             X_train=X_train, y_train=y_train,
             X_test=X_test,   y_test=y_test)
    print(f"Dataset guardado en {OUTPUT_NPZ}")

if __name__ == '__main__':
    main()
