# =============================================================================
# CONFIGURACOES - USG FLOW
# =============================================================================
# Aplicativo de Ultrassom Otimizado
# Baseado em Butterfly iQ3, Clarius HD3, SUPRA/ITKPOCUS
# =============================================================================

# =============================================================================
# FONTE DE VIDEO / CAPTURA
# =============================================================================
# Opcoes:
#   "AIRPLAY"  - Espelhamento iPhone via QuickTime/AirPlay (RECOMENDADO)
#   0, 1, 2    - Cameras USB/Webcam
#   "video.mp4"- Arquivo de video para teste
# =============================================================================

VIDEO_SOURCE = "AIRPLAY"
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FPS = 30

# =============================================================================
# QUALIDADE DA IMAGEM
# =============================================================================
# PRESERVE_ORIGINAL_IMAGE:
#   True  = Mostra imagem ORIGINAL (cores fieis ao iPhone) - RECOMENDADO
#   False = Aplica CLAHE/enhance (altera contraste)
#
# KEEP_ORIGINAL_RESOLUTION:
#   True  = Mantem resolucao EXATA (pode cortar se nao caber)
#   False = Redimensiona para caber na janela - RECOMENDADO para performance
#
# USE_LANCZOS:
#   True  = Usa INTER_LANCZOS4 para resize (melhor qualidade, mais lento)
#   False = Usa INTER_LINEAR (mais rapido) - RECOMENDADO para performance
# =============================================================================

PRESERVE_ORIGINAL_IMAGE = True
KEEP_ORIGINAL_RESOLUTION = True   # NAO MEXE NA IMAGEM
USE_LANCZOS = True                # Caso precise resize, usa melhor qualidade

# =============================================================================
# ROI (Region of Interest)
# =============================================================================
# USE_ROI = True para capturar apenas a area da imagem de ultrassom
# Isso melhora performance e remove bordas/UI do app do iPhone
#
# ROI_PRESET:
#   "butterfly" - Otimizado para Butterfly iQ
#   "clarius"   - Otimizado para Clarius
#   "custom"    - Usa ROI_CUSTOM abaixo
# =============================================================================

USE_ROI = False
ROI_PRESET = "butterfly"
ROI_CUSTOM = (0.12, 0.08, 0.76, 0.84)

# =============================================================================
# INTELIGENCIA ARTIFICIAL
# =============================================================================
# YOLO_MODEL_PATH: Caminho para modelo de deteccao de agulha
# AI_CONFIDENCE: Limiar de confianca (0.0 a 1.0)
# AI_FRAME_SKIP: Processar IA a cada N frames (maior = mais leve)
#   1 = todos os frames (mais preciso, mais CPU)
#   3 = balanceado - RECOMENDADO
#   5 = mais leve, menos responsivo
#
# USE_MPS: Usar GPU do Mac (Metal) para IA - MUITO mais rapido
# =============================================================================

YOLO_MODEL_PATH = "models/best.pt"
AI_CONFIDENCE = 0.5
AI_FRAME_SKIP = 3
USE_MPS = True

# =============================================================================
# INTERFACE / JANELA
# =============================================================================
# WINDOW_WIDTH/HEIGHT: Tamanho da janela
# FULLSCREEN: Abre em tela cheia
# SHOW_SIDEBAR: Mostra sidebar por padrao
# UI_SIMPLE: True = UI mais rapida
# =============================================================================

WINDOW_NAME = "USG FLOW"
WINDOW_WIDTH = 1700      # Largura (imagem + sidebar 200px)
WINDOW_HEIGHT = 950      # Altura (imagem + header 50px)
FULLSCREEN = False       # True = abre em fullscreen
SHOW_SIDEBAR = True

# Performance UI
UI_SIMPLE = True         # True = UI otimizada

# =============================================================================
# SCREENSHOT / GRAVACAO
# =============================================================================
# SCREENSHOT_FORMAT: PNG (lossless) ou BMP (raw)
#
# VIDEO_QUALITY: Qualidade da gravacao
#   "lossless" = Frames PNG individuais (perfeito, arquivos ENORMES)
#   "high"     = H.264 alta qualidade (RECOMENDADO)
#   "medium"   = H.264 media qualidade
#   "light"    = MJPG leve
# =============================================================================

SCREENSHOT_FORMAT = "PNG"
VIDEO_QUALITY = "high"
CAPTURES_DIR = "captures"
