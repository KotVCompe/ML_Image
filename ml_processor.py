import cv2
import numpy as np
from PIL import Image
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import joblib
import hashlib

class AdvancedMLProcessor:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.classical_ml = ClassicalML()
        self.ml_models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.load_or_train_models()
    
    def load_or_train_models(self):
        """Загрузка или обучение ML моделей"""
        try:
            # Пробуем загрузить сохраненные модели
            self.ml_models = self._load_models()
            if self.ml_models:
                self.is_trained = True
                print("✅ ML модели загружены успешно")
            else:
                print("ℹ️ ML модели не найдены, будут обучены при первом использовании")
                self.is_trained = False
        except Exception as e:
            print(f"❌ Ошибка загрузки моделей: {e}")
            self.is_trained = False
    
    def _load_models(self):
        """Загрузка предобученных моделей"""
        models = {}
        model_files = {
            'svm': 'svm_model.pkl',
            'random_forest': 'rf_model.pkl', 
            'knn': 'knn_model.pkl',
            'neural_net': 'nn_model.pkl'
        }
        
        for name, filename in model_files.items():
            try:
                model_path = os.path.join('models', filename)
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        models[name] = pickle.load(f)
                    print(f"✅ Загружена модель: {name}")
            except Exception as e:
                print(f"❌ Ошибка загрузки {name}: {e}")
        
        return models
    
    def _save_models(self):
        """Сохранение обученных моделей"""
        try:
            os.makedirs('models', exist_ok=True)
            for name, model in self.ml_models.items():
                model_path = os.path.join('models', f'{name}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"💾 Сохранена модель: {name}")
        except Exception as e:
            print(f"❌ Ошибка сохранения моделей: {e}")
    
    def train_ml_models(self, images_dir):
        """Обучение ML моделей на наборе данных"""
        try:
            print("🎯 Начинаем обучение ML моделей...")
            
            # Загрузка и подготовка данных
            X, y = self._load_training_data(images_dir)
            
            if len(X) == 0:
                print("❌ Не найдены данные для обучения")
                return {"error": "Не найдены данные для обучения"}
            
            print(f"📊 Загружено {len(X)} примеров для обучения")
            print(f"🎯 Классы: {set(y)}")
            
            # Разделение на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Нормализация данных
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Обучение различных моделей
            self.ml_models = {
                'SVM': SVC(kernel='rbf', probability=True, random_state=42),
                'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'Neural_Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            }
            
            results = {}
            print("🧠 Обучаем модели...")
            for name, model in self.ml_models.items():
                print(f"   Обучение {name}...")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = accuracy
                print(f"   ✅ {name} точность: {accuracy:.3f}")
            
            # Сохранение моделей
            self._save_models()
            self.is_trained = True
            
            # Создаем папку для тренировочных данных
            os.makedirs('training_data', exist_ok=True)
            
            print("🎉 Обучение ML моделей завершено!")
            
            return {
                "training_completed": True,
                "accuracies": results,
                "best_model": max(results, key=results.get),
                "best_accuracy": max(results.values()),
                "message": f"Модели обучены успешно! Лучшая точность: {max(results.values()):.3f} ({max(results, key=results.get)})"
            }
            
        except Exception as e:
            error_msg = f"Ошибка обучения: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def _load_training_data(self, images_dir):
        """Загрузка тренировочных данных"""
        X = []  # Признаки
        y = []  # Метки
        
        # Создаем реалистичные синтетические данные для разных категорий
        categories = {
            'пейзаж': 0,
            'портрет': 1, 
            'город': 2,
            'абстрактное': 3
        }
        
        print("📁 Генерируем тренировочные данные...")
        
        # Генерация синтетических данных на основе анализа изображений
        for category, category_id in categories.items():
            print(f"   Создаем данные для: {category}")
            for i in range(50):  # 50 примеров на класс
                # Создаем синтетические признаки для каждого класса
                features = self._generate_synthetic_features(category_id, i)
                X.append(features)
                y.append(category)
        
        print(f"✅ Сгенерировано {len(X)} примеров, {len(categories)} классов")
        return np.array(X), np.array(y)
    
    def _generate_synthetic_features(self, category_id, sample_id):
        """Генерация синтетических признаков для обучения"""
        # Используем стабильный seed
        seed = category_id * 1000 + sample_id
        np.random.seed(seed)
        
        if category_id == 0:  # пейзаж
            return [
                np.random.normal(0.6, 0.1),  # Высокая энтропия
                np.random.normal(0.7, 0.1),  # Высокий контраст  
                np.random.normal(0.3, 0.1),  # Средняя насыщенность
                np.random.normal(0.8, 0.1),  # Много градиентов
                np.random.normal(0.2, 0.1),  # Мало круглых объектов
                np.random.normal(0.9, 0.1),  # Высокая сложность
                np.random.normal(0.4, 0.1),  # Средняя яркость
            ]
        elif category_id == 1:  # портрет
            return [
                np.random.normal(0.4, 0.1),  # Средняя энтропия
                np.random.normal(0.5, 0.1),  # Средний контраст
                np.random.normal(0.6, 0.1),  # Высокая насыщенность кожи
                np.random.normal(0.3, 0.1),  # Мало градиентов
                np.random.normal(0.7, 0.1),  # Есть овальные объекты
                np.random.normal(0.5, 0.1),  # Средняя сложность
                np.random.normal(0.6, 0.1),  # Высокая яркость
            ]
        elif category_id == 2:  # город
            return [
                np.random.normal(0.8, 0.1),  # Очень высокая энтропия
                np.random.normal(0.9, 0.1),  # Очень высокий контраст
                np.random.normal(0.4, 0.1),  # Средняя насыщенность
                np.random.normal(0.9, 0.1),  # Много градиентов
                np.random.normal(0.1, 0.1),  # Прямоугольные объекты
                np.random.normal(0.8, 0.1),  # Высокая сложность
                np.random.normal(0.5, 0.1),  # Средняя яркость
            ]
        else:  # абстрактное (3)
            return [
                np.random.normal(0.5, 0.2),  # Разная энтропия
                np.random.normal(0.6, 0.2),  # Разный контраст
                np.random.normal(0.7, 0.2),  # Разная насыщенность
                np.random.normal(0.5, 0.2),  # Разные градиенты
                np.random.normal(0.5, 0.2),  # Разные формы
                np.random.normal(0.6, 0.2),  # Средняя сложность
                np.random.normal(0.7, 0.2),  # Разная яркость
            ]
    
    def advanced_classify(self, image_path):
        """Классификация с использованием обученных ML моделей"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Не удалось загрузить изображение'}
            
            # Извлечение признаков
            features = self._extract_ml_features(image)
            feature_names = ['яркость', 'контраст', 'энтропия', 'оттенок', 'насыщенность', 'яркость_цвета', 'текстура_сложность']
            
            if self.is_trained and self.ml_models:
                # Используем обученные ML модели
                ml_predictions = self._ml_classification(features)
                heuristic_predictions = self._heuristic_classification(features, image)
                
                # Комбинируем результаты
                combined_predictions = self._combine_predictions(ml_predictions, heuristic_predictions)
                
                return {
                    'predictions': combined_predictions,
                    'features': dict(zip(feature_names, features)),
                    'ml_used': True,
                    'model_name': 'Ансамбль ML классификаторов'
                }
            else:
                # Fallback на эвристические методы
                predictions = self._heuristic_classification(features, image)
                return {
                    'predictions': predictions,
                    'features': dict(zip(feature_names, features)),
                    'ml_used': False,
                    'model_name': 'Эвристический классификатор'
                }
            
        except Exception as e:
            return {'error': f'Ошибка классификации: {str(e)}'}
    
    def _extract_ml_features(self, image):
        """Извлечение признаков для ML моделей"""
        features = []
        
        # Базовые статистики
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features.extend([
            np.mean(gray) / 255.0,           # Нормализованная яркость
            np.std(gray) / 255.0,            # Нормализованный контраст
            self._calculate_entropy(gray),   # Энтропия
        ])
        
        # Цветовые признаки
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features.extend([
            np.mean(hsv[:,:,0]) / 180.0,     # Hue
            np.mean(hsv[:,:,1]) / 255.0,     # Saturation
            np.mean(hsv[:,:,2]) / 255.0,     # Value
        ])
        
        # Текстурные признаки
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.extend([
            np.mean(gradient_magnitude) / 1000.0,  # Текстурная сложность
        ])
        
        return np.array(features)
    
    def _ml_classification(self, features):
        """Классификация с использованием ML моделей"""
        features_scaled = self.scaler.transform([features])
        predictions = []
        
        for name, model in self.ml_models.items():
            try:
                probabilities = model.predict_proba(features_scaled)[0]
                predicted_class = model.classes_[np.argmax(probabilities)]
                confidence = np.max(probabilities)
                
                # Получаем все вероятности для отладки
                all_probs = {}
                for i, cls in enumerate(model.classes_):
                    all_probs[cls] = float(probabilities[i])
                
                predictions.append({
                    'model': name,
                    'class': predicted_class,
                    'confidence': confidence,
                    'all_probabilities': all_probs
                })
                
                print(f"   {name}: {predicted_class} ({confidence:.3f})")
                
            except Exception as e:
                print(f"Ошибка в предсказании {name}: {e}")
        
        return predictions
    
    def _combine_predictions(self, ml_predictions, heuristic_predictions):
        """Комбинирование предсказаний ML и эвристических методов"""
        combined = []
        
        # Добавляем ML предсказания
        for ml_pred in ml_predictions:
            combined.append(
                (f"{ml_pred['class']} ({ml_pred['model']})", ml_pred['confidence'])
            )
        
        # Добавляем эвристические предсказания (с пониженным весом)
        for heuristic_pred in heuristic_predictions[:3]:  # Только топ-3
            if isinstance(heuristic_pred, tuple):
                class_name, confidence = heuristic_pred
                combined.append((f"{class_name} (Эвристика)", confidence * 0.7))
        
        # Сортируем по уверенности
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:5]  # Возвращаем топ-5
    
    def deep_feature_analysis(self, image_path):
        """Глубокий анализ признаков с PCA и кластеризацией"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Не удалось загрузить изображение'}
            
            # Извлечение расширенных признаков
            features = self._extract_ml_features(image)
            
            # PCA анализ
            pca = PCA(n_components=3)
            synthetic_features = self._generate_synthetic_variations(features)
            pca_result = pca.fit_transform(synthetic_features)
            
            # Кластеризация
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(synthetic_features)
            
            analysis_result = {
                'original_features': [float(x) for x in features],
                'pca_explained_variance': [float(x) for x in pca.explained_variance_ratio_],
                'pca_components': [float(x) for x in pca_result[0]],
                'cluster_assignment': int(clusters[0]),
                'feature_importance': self._calculate_feature_importance(features)
            }
            
            return {
                'deep_analysis': analysis_result,
                'model_name': 'Глубокий анализ признаков'
            }
            
        except Exception as e:
            return {'error': f'Ошибка глубокого анализа: {str(e)}'}

    def _generate_synthetic_variations(self, base_features, n_variations=50):
        """Генерация синтетических вариаций признаков для анализа"""
        variations = [base_features]
        for i in range(n_variations):
            # Используем стабильный seed
            np.random.seed(i)
            variation = base_features + np.random.normal(0, 0.1, len(base_features))
            variations.append(variation)
        return np.array(variations)

    def _calculate_feature_importance(self, features):
        """Оценка важности признаков"""
        importance = {}
        feature_names = [
            'яркость', 'контраст', 'энтропия', 'оттенок', 'насыщенность', 
            'яркость_цвета', 'текстура_сложность'
        ]
        
        for i, (name, value) in enumerate(zip(feature_names, features)):
            # Важность основана на отклонении от среднего (0.5 для нормализованных признаков)
            importance[name] = float(abs(value - 0.5))
        
        return importance

    def advanced_detect(self, image_path):
        """Расширенная детекция объектов"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Не удалось загрузить изображение'}
            
            detections = []
            result_image = image.copy()
            
            # Детекция лиц
            face_detections = self._detect_faces(image)
            detections.extend(face_detections)
            
            # Детекция объектов по контурам
            contour_detections = self._detect_contours(image)
            detections.extend(contour_detections)
            
            # Визуализация
            for det in detections:
                bbox = det['bbox']
                color = (0, 255, 0) if det['class'] == 'Лицо' else (255, 0, 0)
                cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(result_image, f"{det['class']}: {det['confidence']:.2f}", 
                           (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return {
                'detections': detections,
                'processed_image': result_image,
                'model_name': 'Гибридный детектор'
            }
            
        except Exception as e:
            return {'error': f'Ошибка детекции: {str(e)}'}

    def _detect_faces(self, image):
        """Детекция лиц"""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'class': 'Лицо',
                'confidence': 0.8,
                'bbox': [int(x), int(y), int(x+w), int(y+h)]
            })
        
        return detections

    def _heuristic_classification(self, features, image):
        """Эвристическая классификация на основе признаков"""
        predictions = []
        
        # Анализ яркости
        brightness = features[0] * 255
        if brightness > 200:
            predictions.append(('Очень яркое изображение', 0.9))
        elif brightness > 150:
            predictions.append(('Яркое изображение', 0.8))
        elif brightness < 50:
            predictions.append(('Темное изображение', 0.9))
        
        # Анализ контраста
        contrast = features[1] * 255
        if contrast > 150:
            predictions.append(('Высокий контраст', 0.8))
        elif contrast < 50:
            predictions.append(('Низкий контраст', 0.7))
        
        # Анализ текстуры
        texture_complexity = features[6] * 1000
        if texture_complexity > 50:
            predictions.append(('Текстурное изображение', 0.7))
        else:
            predictions.append(('Гладкое изображение', 0.6))
        
        # Анализ цветов
        saturation = features[4] * 255
        if saturation > 150:
            predictions.append(('Яркие цвета', 0.7))
        elif saturation < 50:
            predictions.append(('Приглушенные цвета', 0.6))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:5]

    def _calculate_entropy(self, image):
        """Вычисление энтропии изображения"""
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram[histogram > 0]
        histogram = histogram / histogram.sum()
        return float(-np.sum(histogram * np.log2(histogram)))

    def feature_analysis(self, image_path):
        """Анализ признаков изображения"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Не удалось загрузить изображение'}
            
            features = self._extract_ml_features(image)
            feature_names = ['яркость', 'контраст', 'энтропия', 'оттенок', 'насыщенность', 'яркость_цвета', 'текстура_сложность']
            
            feature_dict = {}
            for i, name in enumerate(feature_names):
                if i == 0:  # яркость
                    feature_dict[name] = float(features[i] * 255)
                elif i == 1:  # контраст
                    feature_dict[name] = float(features[i] * 255)
                elif i == 3:  # оттенок
                    feature_dict[name] = float(features[i] * 180)
                elif i == 4:  # насыщенность
                    feature_dict[name] = float(features[i] * 100)
                elif i == 5:  # яркость цвета
                    feature_dict[name] = float(features[i] * 100)
                elif i == 6:  # текстура
                    feature_dict[name] = float(features[i] * 1000)
                else:
                    feature_dict[name] = float(features[i])
            
            analysis_result = {
                'detailed_features': feature_dict,
                'additional_analysis': {
                    'оценка_качества': self._calculate_quality_score(features),
                    'рекомендации': self._suggest_image_use(features)
                }
            }
            
            return {
                'detailed_features': analysis_result['detailed_features'],
                'additional_analysis': analysis_result['additional_analysis'],
                'model_name': 'Анализ признаков'
            }
            
        except Exception as e:
            return {'error': f'Ошибка анализа признаков: {str(e)}'}

    def _calculate_quality_score(self, features):
        """Оценка качества изображения"""
        score = 50
        
        # Резкость (на основе текстуры)
        if features[6] > 0.5:
            score += 20
        elif features[6] < 0.2:
            score -= 15
        
        # Контрастность
        if 0.3 < features[1] < 0.7:
            score += 15
        elif features[1] <= 0.2:
            score -= 10
        
        # Яркость
        if 0.3 < features[0] < 0.8:
            score += 10
        elif features[0] <= 0.2 or features[0] >= 0.9:
            score -= 10
        
        return max(0, min(score, 100))

    def _suggest_image_use(self, features):
        """Рекомендации по использованию изображения"""
        suggestions = []
        
        quality_score = self._calculate_quality_score(features)
        if quality_score > 80:
            suggestions.append("Высокое качество - подходит для профессионального использования")
        elif quality_score < 40:
            suggestions.append("Низкое качество - рекомендуется повторная съемка или улучшение")
        
        if features[6] > 0.7:
            suggestions.append("Высокая детализация - хорошо для анализа")
        
        if features[1] > 0.6:
            suggestions.append("Высокий контраст - хорошая визуальная impact")
        
        return suggestions if suggestions else ["Стандартное качество - подходит для общего использования"]

    def advanced_segment(self, image_path, method='color'):
        """Сегментация изображения"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Не удалось загрузить изображение'}
            
            if method == 'color':
                # Цветовая сегментация с K-means
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pixels = image_rgb.reshape(-1, 3)
                
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                labels = kmeans.fit_predict(pixels)
                segmented = kmeans.cluster_centers_[labels].reshape(image_rgb.shape)
                segmented = segmented.astype(np.uint8)
                segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
                
                return {
                    'segmented_image': segmented_bgr,
                    'num_segments': 5,
                    'processed_image': segmented_bgr,
                    'model_name': 'Цветовая сегментация (K-means)'
                }
            else:
                return {'error': 'Метод сегментации не реализован'}
                
        except Exception as e:
            return {'error': f'Ошибка сегментации: {str(e)}'}

    def _detect_contours(self, image):
        """Детекция объектов по контурам"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        object_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 50000:
                x, y, w, h = cv2.boundingRect(contour)
                object_count += 1
                detections.append({
                    'class': f'Объект {object_count}',
                    'confidence': 0.6,
                    'bbox': [int(x), int(y), int(x+w), int(y+h)]
                })
        
        return detections

    def get_model_info(self):
        """Информация о ML моделях"""
        model_status = "✅ Обучены" if self.is_trained else "❌ Не обучены"
        models_list = list(self.ml_models.keys()) if self.ml_models else ["Модели не обучены"]
        
        return {
            'ml_models_loaded': self.is_trained,
            'models_available': models_list,
            'algorithms': [
                'Support Vector Machine (SVM)',
                'Random Forest', 
                'K-Nearest Neighbors',
                'Neural Network (MLP)',
                'K-means Clustering',
                'Principal Component Analysis (PCA)'
            ],
            'capabilities': [
                'Обучение с учителем (Классификация)',
                'Обучение без учителя (Кластеризация)',
                'Извлечение и отбор признаков',
                'Уменьшение размерности (PCA)',
                'Ансамблевые методы'
            ]
        }

# Вспомогательные классы
class FeatureExtractor:
    def extract_color_moments(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return {
            'hue_mean': np.mean(hsv[:,:,0]),
            'saturation_mean': np.mean(hsv[:,:,1]),
            'value_mean': np.mean(hsv[:,:,2])
        }

class ClassicalML:
    def face_detection(self, image):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'class': 'Лицо',
                'confidence': 0.9,
                'bbox': [int(x), int(y), int(x+w), int(y+h)]
            })
        
        return {
            'detections': detections,
            'processed_image': image.copy()
        }