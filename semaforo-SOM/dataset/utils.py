import os
import cv2
import numpy as np

def extract_rgb_features(img):
    img = cv2.resize(img, (32, 32))
    avg_color = np.mean(img.reshape(-1, 3), axis=0)
    return avg_color / 255.0

def load_dataset(base_path):
    data = []
    labels = []
    for label in os.listdir(base_path):
        path = os.path.join(base_path, label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                features = extract_rgb_features(img)
                data.append(features)
                labels.append(label)
    return np.array(data), labels
