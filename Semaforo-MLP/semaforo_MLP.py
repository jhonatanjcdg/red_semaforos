import cv2
import numpy as np
import pickle
import argparse

# --- Parámetros configurables ---
MODEL_FILE    = 'mlp_semaforo.pkl'
LEARNING_RATE = 0.01
EPOCHS        = 1000
HIDDEN_UNITS  = 16

# --- Clase MLP desde cero con NumPy ---
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    def _softmax(self, z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def _one_hot(self, y, C):
        m = y.shape[0]
        one = np.zeros((m, C))
        one[np.arange(m), y] = 1
        return one

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self._softmax(self.Z2)
        return self.A2

    def compute_loss(self, Y_hat, Y):
        m = Y.shape[0]
        return -np.sum(Y * np.log(Y_hat + 1e-8)) / m

    def backward(self, X, Y):
        m = X.shape[0]
        dZ2 = self.A2 - Y
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (1 - np.tanh(self.Z1)**2)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # update
        self.W2 -= LEARNING_RATE * dW2
        self.b2 -= LEARNING_RATE * db2
        self.W1 -= LEARNING_RATE * dW1
        self.b1 -= LEARNING_RATE * db1

    def fit(self, X, y):
        Y = self._one_hot(y, self.b2.shape[1])
        for epoch in range(EPOCHS):
            Y_hat = self.forward(X)
            loss = self.compute_loss(Y_hat, Y)
            self.backward(X, Y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, loss: {loss:.4f}")

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X):
        return self.forward(X)

class TrafficLightDetector:
    
    def __init__(self):
        self.mlp = MLP(input_dim=3, hidden_dim=HIDDEN_UNITS, output_dim=3)
        self.is_trained = False
        self.labels = ['ROJO', 'AMARILLO', 'VERDE']
        self.status_messages = ['PELIGRO CRUZAR', 'ADVERTENCIA', 'SEGURO CRUZAR']
        
        # Entrenar la red con datos sintéticos
        self.train_network()
    
    def generate_training_data(self):
        """Generar datos de entrenamiento sintéticos basados en colores RGB"""
        np.random.seed(42)
        
        # Datos para ROJO (clase 0)
        red_data = []
        for _ in range(200):
            r = np.random.uniform(150, 255)  # Rojo alto
            g = np.random.uniform(0, 80)     # Verde bajo
            b = np.random.uniform(0, 80)     # Azul bajo
            red_data.append([r, g, b])
            
        # Datos para AMARILLO (clase 1)
        yellow_data = []
        for _ in range(200):
            r = np.random.uniform(200, 255)  # Rojo alto
            g = np.random.uniform(180, 255)  # Verde alto
            b = np.random.uniform(0, 100)    # Azul bajo
            yellow_data.append([r, g, b])
            
        # Datos para VERDE (clase 2)
        green_data = []
        for _ in range(200):
            r = np.random.uniform(0, 100)    # Rojo bajo
            g = np.random.uniform(150, 255)  # Verde alto
            b = np.random.uniform(0, 150)    # Azul bajo-medio
            green_data.append([r, g, b])
        
        # Combinar datos
        X = np.array(red_data + yellow_data + green_data)
        y = np.array([0]*200 + [1]*200 + [2]*200)
        
        return X, y
    
    def train_network(self):
        """Entrenar la red MLP"""
        print("Entrenando red MLP...")
        X, y = self.generate_training_data()
        
        # Normalizar datos
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        X_normalized = (X - self.X_mean) / self.X_std
        
        # Entrenar la red
        self.mlp.fit(X_normalized, y)
        self.is_trained = True
        
        # Evaluar precisión
        predictions = self.mlp.predict(X_normalized)
        accuracy = np.mean(predictions == y)
        print(f"Precisión del entrenamiento: {accuracy:.2%}")
    
    def detect_circles(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar blur para mejorar detección
        gray_blurred = cv2.medianBlur(gray, 5)
        
        # Detectar círculos
        circles = cv2.HoughCircles(
            gray_blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=30,
            param1=50, 
            param2=30, 
            minRadius=15, 
            maxRadius=150
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return circles
        return None
    
    def extract_circle_color(self, image, circle):
        """Extraer color promedio de un círculo específico"""
        x, y, r = circle
        
        # Crear máscara circular
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Convertir a RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extraer píxeles dentro del círculo
        circle_pixels = rgb_image[mask > 0]
        
        if len(circle_pixels) > 0:
            # Calcular color promedio
            avg_color = np.mean(circle_pixels, axis=0)
            return avg_color
        
        return None
    
    def predict_traffic_light(self, image):
        """Predecir el estado del semáforo detectando círculos"""
        if not self.is_trained:
            return None, None, None
        
        # Detectar círculos
        circles = self.detect_circles(image)
        
        if circles is None or len(circles) == 0:
            # Si no hay círculos, usar color promedio de toda la imagen
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            small_image = cv2.resize(rgb_image, (100, 100))
            color_features = np.mean(small_image.reshape(-1, 3), axis=0)
            circle_info = None
        else:
            # Tomar el círculo más grande
            largest_circle = max(circles, key=lambda c: c[2])
            color_features = self.extract_circle_color(image, largest_circle)
            circle_info = tuple(largest_circle)
            
            if color_features is None:
                return None, None, None
        
        # Normalizar características
        normalized_features = (color_features - self.X_mean) / self.X_std
        
        # Predecir
        probabilities = self.mlp.predict_proba(normalized_features.reshape(1, -1))[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence, circle_info
    
    def run_camera(self):
        """Ejecutar detección en tiempo real con cámara"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        print("Detector de semáforos MLP iniciado")
        print("Presiona 'q' para salir")
        print("-" * 40)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error al capturar frame")
                    break
                
                # Realizar predicción
                prediction, confidence, circle_info = self.predict_traffic_light(frame)
                
                # Dibujar círculo detectado si existe
                if circle_info:
                    x, y, r = circle_info
                    cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 3)
                    # Mostrar coordenadas del centroide
                    cv2.putText(frame, f"Centro: ({x}, {y})", (x-50, y-r-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Mostrar información en pantalla
                if prediction is not None:
                    status = f"{self.labels[prediction]} - {self.status_messages[prediction]}"
                    confidence_text = f"Confianza: {confidence:.1%}"
                    
                    # Agregar texto al frame
                    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                    cv2.putText(frame, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 255), 1)
                    cv2.putText(frame, "MLP Network", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 0), 1)
                    
                    # Mostrar en consola
                    if circle_info:
                        x, y, r = circle_info
                        print(f"\rDetección: {status} ({confidence_text}) - Centro: ({x}, {y}), Radio: {r}", 
                              end="", flush=True)
                    else:
                        print(f"\rDetección: {status} ({confidence_text}) - Sin círculo detectado", 
                              end="", flush=True)
                
                # Mostrar frame
                cv2.imshow('Detector de Semáforos - MLP', frame)
                
                # Salir con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrumpido por el usuario")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nCámara cerrada")
    
    def test_with_image(self, image_path):
        """Probar detector con una imagen específica"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: No se pudo cargar la imagen {image_path}")
                return
            
            prediction, confidence, circle_info = self.predict_traffic_light(image)
            
            if prediction is not None:
                print(f"Imagen: {image_path}")
                print(f"Predicción: {self.labels[prediction]}")
                print(f"Estado: {self.status_messages[prediction]}")
                print(f"Confianza: {confidence:.1%}")
                
                # Dibujar círculo si se detectó
                if circle_info:
                    x, y, r = circle_info
                    cv2.circle(image, (x, y), r, (0, 255, 255), 2)
                    cv2.circle(image, (x, y), 2, (0, 255, 255), 3)
                    print(f"Círculo detectado - Centro: ({x}, {y}), Radio: {r}")
                else:
                    print("No se detectó círculo - usando color promedio de imagen")
                
                # Mostrar imagen
                cv2.imshow('Resultado MLP', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No se pudo realizar predicción")
                
        except Exception as e:
            print(f"Error al procesar imagen: {str(e)}")
    
    def save_model(self, filename=MODEL_FILE):
        """Guardar el modelo entrenado"""
        model_data = {
            'mlp': self.mlp,
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'is_trained': self.is_trained
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Modelo guardado en {filename}")
    
    def load_model(self, filename=MODEL_FILE):
        """Cargar modelo pre-entrenado"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.mlp = model_data['mlp']
            self.X_mean = model_data['X_mean']
            self.X_std = model_data['X_std']
            self.is_trained = model_data['is_trained']
            print(f"Modelo cargado desde {filename}")
            return True
        except FileNotFoundError:
            print(f"No se encontró el archivo {filename}")
            return False
        except Exception as e:
            print(f"Error al cargar modelo: {str(e)}")
            return False

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Detector de Semáforos con MLP y detección de círculos')
    parser.add_argument('--mode', choices=['train', 'detect', 'test'], required=True,
                       help='Modo de operación')
    parser.add_argument('--image', type=str, help='Ruta de imagen para modo test')
    parser.add_argument('--save', action='store_true', help='Guardar modelo después de entrenar')
    parser.add_argument('--load', action='store_true', help='Cargar modelo existente')
    
    args = parser.parse_args()
    
    detector = TrafficLightDetector()
    
    if args.load:
        if not detector.load_model():
            print("Usando modelo recién entrenado...")
    
    if args.mode == 'train':
        print("Entrenando modelo...")
        detector.train_network()
        if args.save:
            detector.save_model()
    
    elif args.mode == 'detect':
        print("Iniciando detección en tiempo real...")
        detector.run_camera()
    
    elif args.mode == 'test':
        if args.image:
            detector.test_with_image(args.image)
        else:
            print("Especifica la ruta de la imagen con --image")

if __name__ == '__main__':
    main()