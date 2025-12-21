# Configuracoes - APLICATIVO USG FINAL
# =====================================
# 100% Python/OpenCV - ZERO conversao de imagem

# Fonte de video
VIDEO_SOURCE = "AIRPLAY"  # "AIRPLAY", "SHM", 0, 1, 2

# Resolucao
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# ROI (Region of Interest)
USE_ROI = True
ROI_PRESET = "butterfly"  # "butterfly", "clarius", "custom"
ROI_CUSTOM = (0.12, 0.08, 0.76, 0.84)

# IA
YOLO_MODEL_PATH = "models/best.pt"
AI_CONFIDENCE = 0.5
AI_FRAME_SKIP = 3
USE_MPS = True  # GPU Metal do Mac

# Interface
WINDOW_NAME = "USG FINAL"
