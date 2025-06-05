import os
import numpy as np
from minisom import MiniSom
from utils import extract_rgb_features, load_dataset
import pickle

data, labels = load_dataset('dataset')
colors = {'red': 0, 'yellow': 1, 'green': 2}

# Inicializar SOM
som = MiniSom(5, 5, 3, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, 100)

# Asociar neuronas con clases
label_map = {}
for i, x in enumerate(data):
    winner = som.winner(x)
    if winner not in label_map:
        label_map[winner] = labels[i]

# Guardar modelo y mapa
with open("som_trained.pkl", "wb") as f:
    pickle.dump((som, label_map), f)

print("✅ Modelo SOM entrenado y guardado.")
