import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from config.settings import Config

class ClassicalML:
    """Классические методы машинного обучения"""
    
    def __init__(self):
        self.feature_extractor = None
    
    def color_based_segmentation(self, image, n_clusters=5):
        """Сегментация на основе цветов с K-means"""
        # Преобразование в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Изменение формы для кластеризации
        pixels = image_rgb.reshape(-1, 3)
        
        # K-means кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Создание сегментированного изображения
        segmented = kmeans.cluster_centers_[labels].reshape(image_rgb.shape).astype(np.uint8)
        segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
        
        return segmented_bgr, labels.reshape(image.shape[:2]), kmeans.cluster_centers_
    
    def edge_based_segmentation(self, image):
        """Сегментация на основе границ"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Детекция границ
        edges = cv2.Canny(gray, 100, 200)
        
        # Морфологические операции для улучшения границ
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Нахождение контуров
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Создание маски
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, -1)
        
        return mask, contours
    
    def texture_analysis(self, image):
        """Анализ текстур с помощью GLCM-like features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Вычисление градиентов
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        # Вычисление характеристик текстур
        texture_features = {
            'gradient_magnitude_mean': np.mean(np.sqrt(sobelx**2 + sobely**2)),
            'gradient_magnitude_std': np.std(np.sqrt(sobelx**2 + sobely**2)),
            'energy': np.mean(gray**2),
            'entropy': -np.sum((np.histogram(gray, 256, [0, 256])[0] / gray.size) * 
                              np.log2(np.histogram(gray, 256, [0, 256])[0] / gray.size + 1e-8))
        }
        
        return texture_features
    
    def face_detection(self, image):
        """Детекция лиц с помощью Haar каскадов"""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Детекция лиц
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        result_image = image.copy()
        detections = []
        
        for i, (x, y, w, h) in enumerate(faces):
            detections.append({
                'class': 'Face',
                'confidence': 0.9,
                'bbox': [int(x), int(y), int(x+w), int(y+h)]
            })
            
            # Рисуем bounding box для лица
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Детекция глаз в области лица
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(result_image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
        
        return {
            'detections': detections,
            'processed_image': result_image,
            'model_name': 'Haar Cascade'
        }

class ImageClassifier:
    """Классификатор изображений на классических признаках"""
    
    def __init__(self):
        self.classifier = None
        self.feature_names = None
    
    def extract_advanced_features(self, image):
        """Извлечение расширенных признаков"""
        features = {}
        
        # Цветовые признаки
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features.update({
            'hue_mean': np.mean(hsv[:,:,0]),
            'saturation_mean': np.mean(hsv[:,:,1]),
            'value_mean': np.mean(hsv[:,:,2]),
            'hue_std': np.std(hsv[:,:,0]),
            'saturation_std': np.std(hsv[:,:,1]),
            'value_std': np.std(hsv[:,:,2])
        })
        
        # Текстурные признаки
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features.update({
            'intensity_mean': np.mean(gray),
            'intensity_std': np.std(gray),
            'contrast': np.max(gray) - np.min(gray)
        })
        
        # Признаки формы (через моменты)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            
            features.update({
                'area': cv2.contourArea(largest_contour),
                'perimeter': cv2.arcLength(largest_contour, True),
                'circularity': (4 * np.pi * features['area']) / (features['perimeter'] ** 2) if features['perimeter'] > 0 else 0
            })
        
        return features
    
    def train(self, features, labels, classifier_type='random_forest'):
        """Обучение классификатора"""
        if classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier_type == 'svm':
            self.classifier = SVC(probability=True, random_state=42)
        
        self.classifier.fit(features, labels)
        
        # Сохранение модели
        model_path = os.path.join(Config.MODEL_DIR, 'classical_classifier.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
    
    def predict(self, image):
        """Предсказание класса изображения"""
        if self.classifier is None:
            raise ValueError("Classifier not trained")
        
        features = self.extract_advanced_features(image)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        prediction = self.classifier.predict(feature_vector)[0]
        probability = self.classifier.predict_proba(feature_vector)[0]
        
        return prediction, probability