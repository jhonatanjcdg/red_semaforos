import numpy as np
import cv2

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for pattern in patterns:
            p = np.array(pattern).reshape(-1, 1)
            self.weights += np.dot(p, p.T)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, steps=10):
        pattern = np.array(pattern)
        for _ in range(steps):
            pattern = np.sign(np.dot(self.weights, pattern))
            pattern[pattern == 0] = 1
        return pattern

def print_pattern(pattern):
    size = int(np.sqrt(len(pattern)))
    for i in range(size):
        row = pattern[i * size:(i + 1) * size]
        print(" ".join("□" if x == 1 else "■" for x in row))
    print()

def preprocess_image(image_path, output_size=(5, 5)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")

    print("Imagen original:")
    cv2.imshow("Imagen Original", img)
    cv2.waitKey(0)

    # Redimensionar imagen
    img_resized = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)
    enlarged = cv2.resize(img_resized, (200, 200), interpolation=cv2.INTER_NEAREST)
    print("Imagen redimensionada:")
    cv2.imshow("Imagen Redimensionada", enlarged)
    cv2.waitKey(0)

    # Convertir a HSV
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # Máscaras de color
    mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_yellow = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    mask_green = cv2.inRange(hsv, (40, 70, 70), (80, 255, 255))

    # Contar activaciones
    red_count = np.sum(mask_red > 0)
    yellow_count = np.sum(mask_yellow > 0)
    green_count = np.sum(mask_green > 0)

    # Elegir el color predominante
    active_color = max((red_count, 'red'), (yellow_count, 'yellow'), (green_count, 'green'))[1]

    # Mostrar máscara activa
    if active_color == 'red':
        mask = mask_red
    elif active_color == 'yellow':
        mask = mask_yellow
    else:
        mask = mask_green

    print(f"Máscara del color detectado ({active_color.upper()}):")
    cv2.imshow("Máscara de Color", cv2.resize(mask, (200, 200), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)

    # Crear patrón binarizado
    pattern = np.where(mask > 0, 1, -1).astype(int)

    # Mostrar el patrón binario como imagen (visualización)
    pattern_visual = np.where(pattern == 1, 255, 0).astype(np.uint8).reshape(output_size)
    print("Patrón binarizado")
    cv2.imshow("Patrón Binarizado", cv2.resize(pattern_visual, (200, 200), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)

    return pattern.flatten(), active_color

patterns = [
    [  # Rojo
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
       -1,-1,-1,-1,-1,-1,
       -1,-1,-1,-1,-1,-1,
       -1,-1,-1,-1,-1,-1,
       -1,-1,-1,-1,-1,-1
    ],
    [  # Amarillo
       -1,-1,-1,-1,-1, -1,
       -1,-1,-1,-1,-1, -1,
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
       -1,-1,-1,-1,-1, -1,
       -1,-1,-1,-1,-1, -1
    ],
    [  # Verde
       -1,-1,-1,-1,-1, -1,
       -1,-1,-1,-1,-1, -1,
        -1,-1,-1,-1,-1, -1,
       -1,-1,-1,-1,-1, -1,
       1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1
    ]
]

named_patterns = {
    "rojo": patterns[0],
    "amarillo": patterns[1],
    "verde": patterns[2]
}

def match_pattern(output, named_patterns):
    best_match = None
    best_score = -1
    for name, ref_pattern in named_patterns.items():
        score = np.sum(np.array(ref_pattern) == output)
        if score > best_score:
            best_score = score
            best_match = name
    
    if best_score < len(output) * 0.7:
        return "desconocido", best_score
    return best_match, best_score

def process_roi(roi, output_size=(6, 6)):
    # Redimensionar imagen
    img_resized = cv2.resize(roi, output_size, interpolation=cv2.INTER_AREA)
    
    # Convertir a HSV
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # Máscaras de color
    mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_yellow = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    mask_green = cv2.inRange(hsv, (40, 70, 70), (80, 255, 255))

    # Contar activaciones
    red_count = np.sum(mask_red > 0)
    yellow_count = np.sum(mask_yellow > 0)
    green_count = np.sum(mask_green > 0)

    # Elegir el color predominante
    active_color = max((red_count, 'red'), (yellow_count, 'yellow'), (green_count, 'green'))[1]

    # Mostrar máscara activa
    if active_color == 'red':
        mask = mask_red
    elif active_color == 'yellow':
        mask = mask_yellow
    else:
        mask = mask_green

    # Crear patrón binarizado
    pattern = np.where(mask > 0, 1, -1).astype(int)
    
    return pattern.flatten(), active_color

def capture_and_detect():
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("No se pudo abrir la cámara")

    # Dimensiones del rectángulo de detección (más alto que ancho, como un semáforo)
    rect_width = 150
    rect_height = 300
    screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calcular posición del rectángulo (centrado)
    x1 = (screen_width - rect_width) // 2
    y1 = (screen_height - rect_height) // 2
    x2 = x1 + rect_width
    y2 = y1 + rect_height

    # Inicializar la red de Hopfield
    hopfield_net = HopfieldNetwork(size=36)
    hopfield_net.train(patterns)
    print("Red de Hopfield inicializada y entrenada con los patrones base")

    print("Presiona 'q' para salir")
    print("Coloca una imagen de semáforo dentro del rectángulo")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Dibujar el rectángulo de detección
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Obtener la región de interés (ROI)
        roi = frame[y1:y2, x1:x2]
        
        # Procesar la ROI si no está vacía
        if roi.size > 0:
            try:
                # Procesar directamente el ROI
                input_pattern, detected_color = process_roi(roi, output_size=(6, 6))
                
                # Recuperar el patrón usando la red de Hopfield
                output_pattern = hopfield_net.recall(input_pattern)
                
                # Clasificar el patrón
                result_name, score = match_pattern(output_pattern, named_patterns)
                
                # Elegir color del texto según el resultado
                text_color = (0, 255, 0)  # Verde por defecto
                if result_name == "desconocido":
                    text_color = (0, 0, 255)  # Rojo para desconocido
                
                # Mostrar el resultado en la pantalla
                cv2.putText(frame, f"Estado: {result_name.upper()}", 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.9, text_color, 2)
                
                # Mostrar la ROI procesada
                roi_resized = cv2.resize(roi, (rect_width, rect_height))
                cv2.imshow("ROI", roi_resized)
                
                # Mostrar el patrón recuperado por la red
                pattern_visual = np.where(output_pattern == 1, 255, 0).astype(np.uint8).reshape(6, 6)
                pattern_visual = cv2.resize(pattern_visual, (200, 200), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Patrón Recuperado por Hopfield", pattern_visual)
                
                # Mostrar información de coincidencia
                cv2.putText(frame, f"Coincidencia: {score}/36", 
                          (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, text_color, 2)
                
            except Exception as e:
                print(f"Error al procesar la imagen: {e}")

        # Mostrar el frame
        cv2.imshow("Detección de Semáforo", frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

def main():
    # === Entrenamiento ===
    hopfield_net = HopfieldNetwork(size=36)
    hopfield_net.train(patterns)

    # Preguntar al usuario qué modo quiere usar
    print("Selecciona el modo de operación:")
    print("1. Procesar imagen desde archivo")
    print("2. Usar cámara web")
    choice = input("Ingresa tu elección (1 o 2): ")

    if choice == "1":
        # === Cargar imagen ===
        image_path = "./data/verde.png"  # Ajustar ruta de prueba

        try:
            input_pattern, detected_color = preprocess_image(image_path, output_size=(6, 6))
            print("\nPatrón ingresado (desde imagen):")
            print_pattern(input_pattern)

            # === Recuperación desde la red ===
            output_pattern = hopfield_net.recall(input_pattern)

            print("Patrón recuperado por la red:")
            print_pattern(output_pattern)

            # Clasificación del patrón recuperado
            result_name, score = match_pattern(output_pattern, named_patterns)
            print(f"Clasificación según la red de Hopfield: {result_name.upper()} (coincidencia: {score} de {len(output_pattern)})")

        except Exception as e:
            print(f"Error al procesar la imagen: {e}")
        finally:
            cv2.destroyAllWindows()
    
    elif choice == "2":
        capture_and_detect()
    
    else:
        print("Opción no válida")

if __name__ == "__main__":
    main()
