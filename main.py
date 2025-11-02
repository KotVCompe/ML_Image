import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QMainWindow, QApplication, QLabel, 
                           QPushButton, QVBoxLayout, QHBoxLayout, 
                           QWidget, QFileDialog, QMessageBox,
                           QListWidget, QProgressBar, QComboBox,
                           QTabWidget, QGroupBox, QTextEdit,
                           QSplitter, QTreeWidget, QTreeWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont
import cv2
import numpy as np
import json
from datetime import datetime

from ml_processor import AdvancedMLProcessor

class MLThread(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    
    def __init__(self, image_path, operation, processor, method=None):
        super().__init__()
        self.image_path = image_path
        self.operation = operation
        self.processor = processor
        self.method = method
        
    def run(self):
        try:
            self.status.emit(f"Запуск {self.operation}...")
            
            if self.operation == "classify":
                result = self.processor.advanced_classify(self.image_path)
            elif self.operation == "detect":
                result = self.processor.advanced_detect(self.image_path)
            elif self.operation == "segment":
                result = self.processor.advanced_segment(self.image_path, self.method)
            elif self.operation == "analyze":
                result = self.processor.feature_analysis(self.image_path)
            elif self.operation == "deep_analyze":
                result = self.processor.deep_feature_analysis(self.image_path)
            elif self.operation == "train_models":
                result = self.processor.train_ml_models("training_data")
            else:
                result = {"error": "Неизвестная операция"}
            
            self.status.emit("Обработка завершена")
            self.finished.emit(result)
            
        except Exception as e:
            self.finished.emit({"error": str(e)})

class AdvancedMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.processor = AdvancedMLProcessor()
        self.results_history = []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Работа ML с изображениям")
        self.setGeometry(100, 100, 1600, 900)
        
        # Центральный виджет с табами
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # Создаем табы
        self.create_analysis_tab()
        self.create_ml_tab()
        self.create_model_info_tab()
        
        # Статус бар
        self.statusBar().showMessage("Готов к работе")
        
    def create_analysis_tab(self):
        """Вкладка для анализа одного изображения"""
        analysis_tab = QWidget()
        layout = QHBoxLayout()
        
        # Левая панель - управление
        left_panel = self.create_control_panel()
        
        # Правая панель - результаты
        right_panel = self.create_results_panel()
        
        # Разделитель
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1200])
        
        layout.addWidget(splitter)
        analysis_tab.setLayout(layout)
        self.tab_widget.addTab(analysis_tab, "📊 Анализ изображений")
        
    def create_control_panel(self):
        """Панель управления"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Группа загрузки изображения
        load_group = QGroupBox("Загрузка изображения")
        load_layout = QVBoxLayout()
        
        self.btn_load = QPushButton("📁 Загрузить изображение")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_load.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        load_layout.addWidget(self.btn_load)
        
        self.lbl_image_info = QLabel("Изображение не загружено")
        self.lbl_image_info.setStyleSheet("color: #666; padding: 5px;")
        load_layout.addWidget(self.lbl_image_info)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Группа операций
        ops_group = QGroupBox("ML операции")
        ops_layout = QVBoxLayout()
        
        self.btn_classify = QPushButton("Классифицировать")
        self.btn_classify.clicked.connect(lambda: self.run_ml_operation("classify"))
        self.btn_classify.setEnabled(False)
        self.btn_classify.setStyleSheet("QPushButton { padding: 8px; }")
        ops_layout.addWidget(self.btn_classify)
        
        self.btn_detect = QPushButton("Детектировать объекты")
        self.btn_detect.clicked.connect(lambda: self.run_ml_operation("detect"))
        self.btn_detect.setEnabled(False)
        self.btn_detect.setStyleSheet("QPushButton { padding: 8px; }")
        ops_layout.addWidget(self.btn_detect)
        
        self.btn_segment = QPushButton("Сегментировать")
        self.btn_segment.clicked.connect(lambda: self.run_ml_operation("segment"))
        self.btn_segment.setEnabled(False)
        self.btn_segment.setStyleSheet("QPushButton { padding: 8px; }")
        ops_layout.addWidget(self.btn_segment)
        
        # Выбор метода сегментации
        seg_method_layout = QHBoxLayout()
        seg_method_layout.addWidget(QLabel("Метод:"))
        self.seg_combo = QComboBox()
        self.seg_combo.addItems(["Цветовая", "Текстурная", "Комбинированная"])
        seg_method_layout.addWidget(self.seg_combo)
        ops_layout.addLayout(seg_method_layout)
        
        self.btn_analyze = QPushButton("Анализ признаков")
        self.btn_analyze.clicked.connect(lambda: self.run_ml_operation("analyze"))
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setStyleSheet("QPushButton { padding: 8px; }")
        ops_layout.addWidget(self.btn_analyze)
        
        self.btn_deep_analyze = QPushButton("Глубокий анализ")
        self.btn_deep_analyze.clicked.connect(lambda: self.run_ml_operation("deep_analyze"))
        self.btn_deep_analyze.setEnabled(False)
        self.btn_deep_analyze.setStyleSheet("QPushButton { padding: 8px; }")
        ops_layout.addWidget(self.btn_deep_analyze)
        
        ops_group.setLayout(ops_layout)
        layout.addWidget(ops_group)
        
        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel("Готов")
        self.lbl_status.setStyleSheet("color: #2196F3; font-weight: bold; padding: 5px;")
        layout.addWidget(self.lbl_status)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
        
    def create_ml_tab(self):
        """Вкладка машинного обучения"""
        ml_tab = QWidget()
        layout = QVBoxLayout()
        
        # Группа обучения моделей
        train_group = QGroupBox("Обучение ML моделей")
        train_layout = QVBoxLayout()
        
        train_info = QLabel(
            "Обучите модели машинного обучения для улучшения классификации.\n"
            "Модели будут обучены на синтетических данных и сохранены для последующего использования."
        )
        train_info.setWordWrap(True)
        train_info.setStyleSheet("color: #666; padding: 10px;")
        train_layout.addWidget(train_info)
        
        self.btn_train = QPushButton("Обучить ML модели")
        self.btn_train.clicked.connect(lambda: self.run_ml_operation("train_models"))
        self.btn_train.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        train_layout.addWidget(self.btn_train)
        
        self.train_status = QLabel("Модели не обучены")
        self.train_status.setStyleSheet("padding: 5px;")
        train_layout.addWidget(self.train_status)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # Группа информации о ML
        info_group = QGroupBox("Информация о машинном обучении")
        info_layout = QVBoxLayout()
        
        ml_info_text = QTextEdit()
        ml_info_text.setReadOnly(True)
        ml_info_text.setHtml("""
        <h3>Используемые алгоритмы ML:</h3>
        <ul>
        <li><b>Support Vector Machine (SVM)</b> - для классификации изображений</li>
        <li><b>Random Forest</b> - ансамблевый метод на основе деревьев</li>
        <li><b>K-Nearest Neighbors</b> - метрический классификатор</li>
        <li><b>Neural Network (MLP)</b> - многослойный перцептрон</li>
        <li><b>K-means Clustering</b> - кластеризация без учителя</li>
        <li><b>PCA</b> - анализ главных компонент</li>
        </ul>
        
        <h3>Извлекаемые признаки:</h3>
        <ul>
        <li>Яркость и контрастность</li>
        <li>Цветовые характеристики (HSV)</li>
        <li>Текстурные особенности</li>
        <li>Геометрические параметры</li>
        <li>Энтропия и сложность</li>
        </ul>
        """)
        info_layout.addWidget(ml_info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        ml_tab.setLayout(layout)
        self.tab_widget.addTab(ml_tab, "Машинное обучение")
        
    def create_results_panel(self):
        """Панель результатов"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Вкладки результатов
        results_tabs = QTabWidget()
        
        # Вкладка изображений
        images_tab = QWidget()
        images_layout = QHBoxLayout()
        
        original_group = QGroupBox("Исходное изображение")
        original_layout = QVBoxLayout()
        self.lbl_original = QLabel("Изображение не загружено")
        self.lbl_original.setAlignment(Qt.AlignCenter)
        self.lbl_original.setStyleSheet("border: 2px solid #ccc; background-color: #f8f9fa; min-height: 300px;")
        original_layout.addWidget(self.lbl_original)
        original_group.setLayout(original_layout)
        
        processed_group = QGroupBox("Обработанное изображение")
        processed_layout = QVBoxLayout()
        self.lbl_processed = QLabel("Результат появится здесь")
        self.lbl_processed.setAlignment(Qt.AlignCenter)
        self.lbl_processed.setStyleSheet("border: 2px solid #ccc; background-color: #f8f9fa; min-height: 300px;")
        processed_layout.addWidget(self.lbl_processed)
        processed_group.setLayout(processed_layout)
        
        images_layout.addWidget(original_group)
        images_layout.addWidget(processed_group)
        images_tab.setLayout(images_layout)
        results_tabs.addTab(images_tab, "Изображения")
        
        # Вкладка результатов
        results_text_tab = QWidget()
        results_text_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_text_layout.addWidget(self.results_text)
        
        results_text_tab.setLayout(results_text_layout)
        results_tabs.addTab(results_text_tab, "Результаты")
        
        # Вкладка истории
        history_tab = QWidget()
        history_layout = QVBoxLayout()
        
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.show_historical_result)
        history_layout.addWidget(self.history_list)
        
        history_tab.setLayout(history_layout)
        results_tabs.addTab(history_tab, "История")
        
        layout.addWidget(results_tabs)
        panel.setLayout(layout)
        return panel
        
    def create_model_info_tab(self):
        """Вкладка информации о моделях"""
        info_tab = QWidget()
        layout = QVBoxLayout()
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        
        # Получаем информацию о моделях
        model_info = self.processor.get_model_info()
        info_text.append("=== ИНФОРМАЦИЯ О СИСТЕМЕ ===")
        info_text.append(f"\nМодели ML загружены: {'Да' if model_info['ml_models_loaded'] else 'Нет'}")
        
        if model_info['models_available']:
            info_text.append(f"\nДоступные модели: {', '.join(model_info['models_available'])}")
        else:
            info_text.append("\nДоступные модели: Модели не обучены")
        
        info_text.append("\n=== АЛГОРИТМЫ МАШИННОГО ОБУЧЕНИЯ ===")
        for algorithm in model_info['algorithms']:
            info_text.append(f"• {algorithm}")
        
        info_text.append("\n=== ВОЗМОЖНОСТИ СИСТЕМЫ ===")
        for capability in model_info['capabilities']:
            info_text.append(f"• {capability}")
        
        layout.addWidget(info_text)
        self.tab_widget.addTab(info_tab, "🔧 Информация о системе")
        
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", 
            "Изображения (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            self.current_image = file_path
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_original.setPixmap(scaled_pixmap)
            self.lbl_processed.clear()
            self.lbl_processed.setText("Обработанное изображение появится здесь")
            
            # Обновляем информацию
            file_size = os.path.getsize(file_path) / 1024  # KB
            image = cv2.imread(file_path)
            if image is not None:
                height, width = image.shape[:2]
                channels = image.shape[2] if len(image.shape) > 2 else 1
                
                info_text = f"""Файл: {os.path.basename(file_path)}
Размер: {file_size:.1f} KB
Разрешение: {width} x {height}
Каналы: {channels}"""
            else:
                info_text = f"Файл: {os.path.basename(file_path)}\nРазмер: {file_size:.1f} KB\nОшибка: Не удалось загрузить изображение"
            
            self.lbl_image_info.setText(info_text)
            
            # Активируем кнопки
            self.set_buttons_enabled(True)
            
            self.results_text.clear()
            
    def run_ml_operation(self, operation):
        if not self.current_image and operation != "train_models":
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение!")
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        if operation == "train_models":
            self.lbl_status.setText("Обучение ML моделей...")
        else:
            self.lbl_status.setText(f"Выполнение {operation}...")
        
        # Для сегментации передаем метод
        method = None
        if operation == "segment":
            method_map = {"Цветовая": "color", "Текстурная": "texture", "Комбинированная": "combined"}
            method = method_map.get(self.seg_combo.currentText(), "color")
        
        # Запускаем ML обработку в отдельном потоке
        self.ml_thread = MLThread(self.current_image, operation, self.processor, method)
        self.ml_thread.finished.connect(self.on_ml_finished)
        self.ml_thread.status.connect(self.lbl_status.setText)
        self.ml_thread.start()
        
        self.set_buttons_enabled(False)
        
    def on_ml_finished(self, result):
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        self.lbl_status.setText("Готов")
        
        if "error" in result:
            QMessageBox.critical(self, "Ошибка", result["error"])
            return
            
        # Обновляем статус обучения если нужно
        if "training_completed" in result and result["training_completed"]:
            self.train_status.setText("Модели успешно обучены!")
            self.train_status.setStyleSheet("color: green; font-weight: bold;")
            
        # Сохраняем в историю
        self.save_to_history(result)
        
        # Отображаем результаты
        self.display_results(result)
        
        # Показываем обработанное изображение
        if "processed_image" in result:
            self.display_processed_image(result["processed_image"])
            
    def display_results(self, result):
        """Отображение результатов в текстовом виде"""
        self.results_text.clear()
        self.results_text.append("=== РЕЗУЛЬТАТЫ АНАЛИЗА ===\n")
        self.results_text.append(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Информация о использовании ML
        if result.get('ml_used'):
            self.results_text.append("Использовано машинное обучение\n")
        elif 'ml_used' in result:
            self.results_text.append("ℹИспользованы эвристические методы\n")
        
        if "predictions" in result:
            self.results_text.append("\n--- РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ ---")
            for pred in result["predictions"]:
                if isinstance(pred, tuple) and len(pred) == 2:
                    class_name, confidence = pred
                    self.results_text.append(f"• {class_name}: {confidence:.1%}")
                elif isinstance(pred, dict):
                    class_name = pred.get('class', 'Неизвестно')
                    confidence = pred.get('confidence', 0.0)
                    model_name = pred.get('model', '')
                    if model_name:
                        self.results_text.append(f"• {class_name} ({model_name}): {confidence:.1%}")
                    else:
                        self.results_text.append(f"• {class_name}: {confidence:.1%}")
                
        if "detections" in result:
            self.results_text.append(f"\n--- ДЕТЕКЦИЯ ОБЪЕКТОВ ({len(result['detections'])} найдено) ---")
            for det in result["detections"]:
                if isinstance(det, dict):
                    bbox = det.get('bbox', [])
                    bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]" if len(bbox) == 4 else "[]"
                    self.results_text.append(
                        f"• {det.get('class', 'Объект')}: {det.get('confidence', 0.0):.1%} {bbox_str}"
                    )
                
        if "num_segments" in result:
            self.results_text.append(f"\n--- РЕЗУЛЬТАТЫ СЕГМЕНТАЦИИ ---")
            self.results_text.append(f"Количество сегментов: {result['num_segments']}")
            
        if "detailed_features" in result:
            self.results_text.append(f"\n--- АНАЛИЗ ПРИЗНАКОВ ---")
            features = result['detailed_features']
            if isinstance(features, dict):
                for key, value in features.items():
                    if isinstance(value, (int, float)):
                        self.results_text.append(f"• {key}: {value:.2f}")
            
        if "additional_analysis" in result:
            self.results_text.append(f"\n--- ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ---")
            analysis = result['additional_analysis']
            if isinstance(analysis, dict):
                for key, value in analysis.items():
                    if isinstance(value, list):
                        self.results_text.append(f"• {key}:")
                        for item in value:
                            self.results_text.append(f"  └ {item}")
                    else:
                        self.results_text.append(f"• {key}: {value}")
            
        if "deep_analysis" in result:
            self.results_text.append(f"\n--- ГЛУБОКИЙ АНАЛИЗ ПРИЗНАКОВ ---")
            analysis = result['deep_analysis']
            if isinstance(analysis, dict):
                self.results_text.append(f"• Объясненная дисперсия (PCA): {analysis.get('pca_explained_variance', [])}")
                self.results_text.append(f"• Кластер: {analysis.get('cluster_assignment', 'N/A')}")
                self.results_text.append(f"• Важность признаков:")
                importance = analysis.get('feature_importance', {})
                for feature, imp in importance.items():
                    self.results_text.append(f"  └ {feature}: {imp:.3f}")
            
        if "model_name" in result:
            self.results_text.append(f"\nИспользованная модель: {result['model_name']}")
            
    def display_processed_image(self, image_array):
        """Отображение обработанного изображения"""
        if isinstance(image_array, np.ndarray):
            height, width = image_array.shape[:2]
            
            if len(image_array.shape) == 3:
                bytes_per_line = 3 * width
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                bytes_per_line = width
                q_img = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_processed.setPixmap(scaled_pixmap)
            
    def save_to_history(self, result):
        """Сохранение результатов в историю"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        operation = "Неизвестно"
        
        if "predictions" in result:
            operation = "Классификация"
        elif "detections" in result:
            operation = f"Детекция ({len(result['detections'])} объектов)"
        elif "num_segments" in result:
            operation = f"Сегментация ({result['num_segments']} сегментов)"
        elif "detailed_features" in result:
            operation = "Анализ признаков"
        elif "deep_analysis" in result:
            operation = "Глубокий анализ"
        elif "training_completed" in result:
            operation = "Обучение моделей"
            
        history_item = f"{timestamp} - {operation}"
        self.history_list.addItem(history_item)
        self.results_history.append(result)
        
    def show_historical_result(self, item):
        """Показать выбранный результат из истории"""
        index = self.history_list.row(item)
        if 0 <= index < len(self.results_history):
            result = self.results_history[index]
            self.display_results(result)
            
            # Показываем обработанное изображение если есть
            if "processed_image" in result:
                self.display_processed_image(result["processed_image"])
            
    def set_buttons_enabled(self, enabled):
        has_image = self.current_image is not None
        self.btn_classify.setEnabled(enabled and has_image)
        self.btn_detect.setEnabled(enabled and has_image)
        self.btn_segment.setEnabled(enabled and has_image)
        self.btn_analyze.setEnabled(enabled and has_image)
        self.btn_deep_analyze.setEnabled(enabled and has_image)
        self.btn_train.setEnabled(enabled)  # Обучение не требует изображения

def main():
    app = QApplication(sys.argv)
    
    # Устанавливаем стиль
    app.setStyle('Fusion')
    
    # Настройка шрифта
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = AdvancedMainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()