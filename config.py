# config.py
MODEL_PATH = "yolov8n.pt"
DEFAULT_APP_NAME = "Chrome"
REGION_DIVISIONS = 4  # 可配置是否要动态调整

# 模型配置
MODELS_FOLDER = "models"  # 本地模型文件夹路径
DEFAULT_MODEL = "yolov8n.pt"  # 默认模型文件名

# 支持的模型文件扩展名
SUPPORTED_MODEL_EXTENSIONS = [".pt", ".pth", ".onnx", ".engine"]