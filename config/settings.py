import os
import platform

class Config:
    # Пути для моделей
    MODEL_DIR = "models"
    DATA_DIR = "data"
    CACHE_DIR = "cache"
    
    # Настройки обработки
    IMAGE_SIZE = (640, 640)
    CONFIDENCE_THRESHOLD = 0.25
    MAX_DETECTIONS = 100
    
    # URL для скачивания ONNX моделей
    MODEL_URLS = {
        "yolov8n": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
        "resnet50": "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx"
    }
    
    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

Config.setup_directories()