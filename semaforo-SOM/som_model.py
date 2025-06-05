import os
import numpy as np
from minisom import MiniSom
from utils import extract_rgb_features, load_dataset
import pickle
from collections import Counter

# Ruta donde están las carpetas 'red', 'yellow' y 'green'
DATASET_PATH = 'dataset'

# 1) Cargamos datos y etiquetas
data, labels = load_dataset(DATASET_PATH)
print(f"Total muestras cargadas: {len(data)}")

# 2) Convertimos etiquetas de texto a índices numéricos (opcional, pero útil para conteo)
unique_labels = sorted(list(set(labels)))  # e.g. ['green', 'red', 'yellow']
label_to_index = {lab: idx for idx, lab in enumerate(unique_labels)}
index_to_label = {idx: lab for lab, idx in label_to_index.items()}

numeric_labels = np.array([label_to_index[lab] for lab in labels])

# 3) Creamos y entrenamos el SOM más grande
som_width = 10
som_height = 10
input_len = data.shape[1]  # 3 (RGB)
som = MiniSom(som_width, som_height, input_len, sigma=1.0, learning_rate=0.5, random_seed=0)

print("Inicializando pesos del SOM aleatoriamente...")
som.random_weights_init(data)

print("Entrenando SOM (500 iteraciones)...")
som.train_random(data, 500)

# 4) Construir un mapa de etiquetas: para cada neurona, ver qué etiquetas ganó durante entrenamiento
label_map = {}  # {(i,j): etiqueta_mayoritaria}

# Recorrer cada muestra de entrenamiento para ver qué neurona gana
winner_counts = {}  # {(i,j): [lista_de_indices_de_etiquetas]}
for i, vector in enumerate(data):
    winner = som.winner(vector)
    if winner not in winner_counts:
        winner_counts[winner] = []
    winner_counts[winner].append(numeric_labels[i])

# Para cada neurona con muestras, asignar la etiqueta mayoritaria
for neuron_coord, label_list in winner_counts.items():
    most_common_idx, _ = Counter(label_list).most_common(1)[0]
    label_map[neuron_coord] = index_to_label[most_common_idx]

# 5) Guardar el modelo SOM y el mapa de etiquetas
os.makedirs('model', exist_ok=True)
with open("model/som_trained.pkl", "wb") as f:
    pickle.dump((som, label_map), f)

print("✅ Modelo SOM entrenado y guardado en 'model/som_trained.pkl'.")
print(f"Neuronas con etiquetas asignadas: {len(label_map)} de {som_width*som_height}")
