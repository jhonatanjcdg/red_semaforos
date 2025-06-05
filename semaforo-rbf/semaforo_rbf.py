import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances

class RBFNetwork:
    """Red Neuronal de Base Radial para clasificación de colores de semáforo"""
    
    def __init__(self, num_centers=10, sigma=1.0):
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None
        self.kmeans = KMeans(n_clusters=num_centers, random_state=42)
        
    def _rbf_kernel(self, X, centers):
        """Función de base radial (Gaussiana)"""
        distances = euclidean_distances(X, centers)
        return np.exp(-distances**2 / (2 * self.sigma**2))
    
    def fit(self, X, y):
        """Entrenar la red RBF"""
        # Encontrar centros usando K-means
        self.centers = self.kmeans.fit(X).cluster_centers_
        
        # Calcular activaciones RBF
        rbf_activations = self._rbf_kernel(X, self.centers)
        
        # Añadir bias
        rbf_activations_bias = np.column_stack([rbf_activations, np.ones(len(X))])
        
        # Calcular pesos usando mínimos cuadrados
        self.weights = np.linalg.pinv(rbf_activations_bias) @ y
        
    def predict(self, X):
        """Predecir usando la red RBF"""
        if self.centers is None or self.weights is None:
            raise ValueError("La red debe ser entrenada primero")
            
        rbf_activations = self._rbf_kernel(X, self.centers)
        rbf_activations_bias = np.column_stack([rbf_activations, np.ones(len(X))])
        
        return rbf_activations_bias @ self.weights
    
    def predict_proba(self, X):
        """Predecir probabilidades"""
        raw_output = self.predict(X)
        # Aplicar softmax para obtener probabilidades
        exp_output = np.exp(raw_output - np.max(raw_output, axis=1, keepdims=True))
        return exp_output / np.sum(exp_output, axis=1, keepdims=True)

class TrafficLightDetector:
    """Detector de semáforos usando red RBF - Versión básica"""
    
    def __init__(self):
        self.rbf_network = RBFNetwork(num_centers=15, sigma=2.0)
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
        
        # Convertir a one-hot encoding
        y_onehot = np.zeros((len(y), 3))
        y_onehot[np.arange(len(y)), y] = 1
        
        return X, y_onehot, y
    
    def train_network(self):
        """Entrenar la red RBF"""
        print("Entrenando red RBF...")
        X, y_onehot, y = self.generate_training_data()
        
        # Normalizar datos
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        X_normalized = (X - self.X_mean) / self.X_std
        
        # Entrenar la red
        self.rbf_network.fit(X_normalized, y_onehot)
        self.is_trained = True
        
        # Evaluar precisión
        predictions = self.rbf_network.predict(X_normalized)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_classes == y)
        print(f"Precisión del entrenamiento: {accuracy:.2%}")
        
    def extract_color_features(self, image):
        """Extraer características de color dominante de la imagen"""
        # Convertir a RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionar para procesamiento más rápido
        small_image = cv2.resize(rgb_image, (100, 100))
        
        # Obtener color promedio
        avg_color = np.mean(small_image.reshape(-1, 3), axis=0)
        
        # Detectar regiones circulares (semáforos)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=10, maxRadius=100)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Tomar el círculo más grande
            if len(circles) > 0:
                largest_circle = max(circles, key=lambda x: x[2])
                x, y, r = largest_circle
                
                # Extraer región del círculo
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                
                # Calcular color promedio en la región circular
                masked_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
                circle_pixels = masked_image[mask > 0]
                
                if len(circle_pixels) > 0:
                    avg_color = np.mean(circle_pixels, axis=0)
                    
                return avg_color, (x, y, r)
        
        return avg_color, None
    
    def predict_traffic_light(self, image):
        """Predecir el estado del semáforo"""
        if not self.is_trained:
            return None, None, None
            
        # Extraer características de color
        color_features, circle_info = self.extract_color_features(image)
        
        # Normalizar características
        normalized_features = (color_features - self.X_mean) / self.X_std
        
        # Predecir
        probabilities = self.rbf_network.predict_proba(normalized_features.reshape(1, -1))[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence, circle_info
    
    def run_camera(self):
        """Ejecutar detección en tiempo real con cámara"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        print("Detector de semáforos iniciado")
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
                
                # Mostrar información en pantalla
                if prediction is not None:
                    status = f"{self.labels[prediction]} - {self.status_messages[prediction]}"
                    confidence_text = f"Confianza: {confidence:.1%}"
                    
                    # Agregar texto al frame
                    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                    cv2.putText(frame, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 255), 1)
                    
                    # Mostrar en consola
                    print(f"\rDetección: {status} ({confidence_text})", end="", flush=True)
                
                # Mostrar frame
                cv2.imshow('Detector de Semáforos - RBF', frame)
                
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
                
                # Mostrar imagen
                cv2.imshow('Resultado', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No se pudo realizar predicción")
                
        except Exception as e:
            print(f"Error al procesar imagen: {str(e)}")

def main():
    """Función principal"""
    detector = TrafficLightDetector()
    
    print("\n=== DETECTOR DE SEMÁFOROS CON RED RBF ===")
    print("Opciones:")
    print("1. Usar cámara en tiempo real")
    print("2. Probar con imagen específica")
    print("3. Salir")
    
    while True:
        try:
            opcion = input("\nSelecciona una opción (1-3): ").strip()
            
            if opcion == '1':
                detector.run_camera()
            elif opcion == '2':
                ruta = input("Ingresa la ruta de la imagen: ").strip()
                detector.test_with_image(ruta)
            elif opcion == '3':
                print("¡Hasta luego!")
                break
            else:
                print("Opción no válida. Intenta de nuevo.")
                
        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()