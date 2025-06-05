import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_rgb_features(img):
    """
    Extrae el color dominante de la imagen usando KMeans (1 clúster).
    Devuelve el RGB normalizado: array de 3 floats entre 0 y 1.
    """
    img = cv2.resize(img, (32, 32))
    pixels = img.reshape(-1, 3).astype(np.float32)

    # KMeans para encontrar el color dominante
    kmeans = KMeans(n_clusters=1, n_init=10, random_state=0)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color / 255.0

def load_dataset(base_path):
    """
    Carga todas las imágenes de las subcarpetas de base_path.
    Retorna:
        data: numpy array de shape (N, 3) con características RGB
        labels: lista de strings con la etiqueta de cada muestra
    """
    data = []
    labels = []
    for label in os.listdir(base_path):
        path = os.path.join(base_path, label)
        if not os.path.isdir(path):
            continue
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            features = extract_rgb_features(img)
            data.append(features)
            labels.append(label)
    if len(data) == 0:
        raise ValueError(f"No se encontraron imágenes en {base_path}")
    return np.array(data), labels
