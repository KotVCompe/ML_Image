import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

class ImageProcessor:
    @staticmethod
    def load_image(image_path):
        """Загрузка изображения с проверкой"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return image
    
    @staticmethod
    def resize_image(image, size):
        """Изменение размера изображения с сохранением пропорций"""
        h, w = image.shape[:2]
        target_w, target_h = size
        
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Добавляем паддинг
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded
    
    @staticmethod
    def draw_detections(image, detections, colors=None):
        """Отрисовка bounding boxes и labels"""
        result = image.copy()
        
        if colors is None:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), 
                     (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            label = det['class']
            confidence = det['confidence']
            
            color = colors[i % len(colors)]
            
            # Рисуем bounding box
            cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Рисуем фон для текста
            label_text = f"{label}: {confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(result, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]), 
                         color, -1)
            
            # Рисуем текст
            cv2.putText(result, label_text, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result
    
    @staticmethod
    def apply_color_mask(image, mask, color=(0, 255, 0), alpha=0.5):
        """Наложение цветовой маски на изображение"""
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        result = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        return result
    
    @staticmethod
    def extract_features(image):
        """Извлечение простых признаков из изображения"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = {
            'mean_intensity': np.mean(gray),
            'std_intensity': np.std(gray),
            'entropy': ImageProcessor.calculate_entropy(gray),
            'edges': np.sum(cv2.Canny(gray, 100, 200) > 0),
            'histogram': cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        }
        
        return features
    
    @staticmethod
    def calculate_entropy(image):
        """Вычисление энтропии изображения"""
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram[histogram > 0]
        histogram = histogram / histogram.sum()
        
        return -np.sum(histogram * np.log2(histogram))

class FeatureExtractor:
    """Класс для извлечения расширенных признаков"""
    
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.sift = cv2.SIFT_create() if hasattr(cv2, 'SIFT_create') else None
    
    def extract_orb_features(self, image):
        """Извлечение ORB features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def extract_color_moments(self, image):
        """Вычисление цветовых моментов"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        moments = {
            'hue_mean': np.mean(hsv[:,:,0]),
            'saturation_mean': np.mean(hsv[:,:,1]),
            'value_mean': np.mean(hsv[:,:,2]),
            'hue_std': np.std(hsv[:,:,0]),
            'saturation_std': np.std(hsv[:,:,1]),
            'value_std': np.std(hsv[:,:,2])
        }
        
        return moments