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
# CALIBRAÇÃO DE ESCALA (FASE 3 - NEEDLE PILOT)
# =============================================================================
# Sistema para converter pixels em milímetros reais.
# Permite medições precisas de profundidade da agulha.
#
# ULTRASOUND_DEPTH_MM: Profundidade total da imagem em mm
#   - Ajustar conforme configuração do probe/ultrassom
#   - Valores típicos: 40mm (superficial), 80mm (médio), 120mm (profundo)
#
# CALIBRATION_MODE:
#   "auto"   - Calcula baseado na altura da imagem e profundidade (RECOMENDADO)
#   "manual" - Usa valor fixo de MM_PER_PIXEL
#
# MM_PER_PIXEL: Valor manual em mm/pixel (usado se mode="manual")
#
# IMAGE_TOP_OFFSET: Pixels do topo que não fazem parte da imagem US
#   - Útil se houver header/barra no topo da imagem do iPhone
# =============================================================================

ULTRASOUND_DEPTH_MM = 80      # Profundidade em mm (ajustar conforme probe)
CALIBRATION_MODE = "auto"     # "auto" ou "manual"
MM_PER_PIXEL = 0.3            # Usado apenas se CALIBRATION_MODE = "manual"
IMAGE_TOP_OFFSET = 0          # Pixels do topo a ignorar

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


# =============================================================================
# NERVE TRACK v2.0 - SEGMENTAÇÃO E TRACKING DE NERVOS
# =============================================================================
# Sistema avançado baseado em pesquisa de:
# - Clarius Median Nerve AI
# - GE Healthcare cNerve
# - ScanNav Anatomy PNB
# - DeepNerve (U-Net + ConvLSTM)
# - segmentation_models.pytorch
# =============================================================================

# Modelo de Segmentação
NERVE_MODEL_PATH = "models/nerve_segmentation/nerve_unetpp_effb4.pt"
NERVE_MODEL_ENCODER = "efficientnet-b4"  # timm encoder
NERVE_MODEL_INPUT_SIZE = (384, 384)       # Resolução do modelo
NERVE_CONFIDENCE = 0.5                     # Limiar de confiança
NERVE_USE_TEMPORAL = True                  # ConvLSTM para consistência temporal

# Tracking Temporal (Anti-Flicker)
NERVE_KALMAN_ENABLED = True               # Kalman Filter para suavização
NERVE_KALMAN_PROCESS_NOISE = 0.01         # Ruído do processo (menor = mais suave)
NERVE_KALMAN_MEASUREMENT_NOISE = 0.1      # Ruído da medição
NERVE_TRACKING_MAX_DISTANCE = 50          # Distância máxima para associação (pixels)
NERVE_TRACKING_MIN_HITS = 3               # Frames mínimos para confirmar estrutura
NERVE_TRACKING_MAX_AGE = 10               # Frames sem detecção antes de remover

# Classificação de Estruturas (Nervo/Artéria/Veia)
NERVE_CLASSIFIER_ENABLED = True           # Ativar classificação automática
NERVE_CLASSIFIER_USE_DOPPLER = False      # Se Doppler disponível
NERVE_CLASSIFIER_HISTORY_SIZE = 10        # Frames para análise de pulsatilidade

# Medição CSA (Cross-Sectional Area)
NERVE_CSA_ENABLED = True                  # Medição automática de área
NERVE_CSA_MIN_AREA_MM2 = 2.0              # Área mínima para ser nervo válido
NERVE_CSA_MAX_AREA_MM2 = 100.0            # Área máxima (evitar falsos positivos)

# Visual Premium
NERVE_VISUAL_MODE = "premium"             # "simple" ou "premium"
NERVE_SHOW_UNCERTAINTY = True             # Mostrar halo de incerteza
NERVE_SHOW_TRAJECTORY = True              # Mostrar trajetória do nervo
NERVE_SHOW_ANATOMY_LABELS = True          # Labels anatômicos detalhados
NERVE_DANGER_ZONE_ALERT = True            # Alertas visuais para zonas de perigo

# Cores por Tipo de Estrutura (BGR)
NERVE_COLOR_NERVE = (0, 255, 255)         # Amarelo
NERVE_COLOR_ARTERY = (0, 0, 255)          # Vermelho
NERVE_COLOR_VEIN = (255, 0, 0)            # Azul
NERVE_COLOR_FASCIA = (128, 128, 128)      # Cinza
NERVE_COLOR_MUSCLE = (128, 0, 128)        # Roxo
NERVE_COLOR_BONE = (255, 255, 255)        # Branco
NERVE_COLOR_PLEURA = (0, 165, 255)        # Laranja
NERVE_COLOR_DANGER = (0, 0, 255)          # Vermelho (zonas de perigo)

# Performance
NERVE_FRAME_SKIP = 2                      # Processar a cada N frames
NERVE_USE_MPS = True                      # Usar GPU Apple Metal
NERVE_BATCH_SIZE = 1                      # Batch size para inferência
NERVE_NUM_WORKERS = 2                     # Workers para data loading

# =============================================================================
# VALIDACAO DE CONFIGURACAO
# =============================================================================

def validate_config():
    """
    Valida todas as configuracoes e retorna lista de erros/avisos.
    Retorna: (is_valid: bool, errors: list, warnings: list)
    """
    import os

    errors = []
    warnings = []

    # VIDEO_SOURCE
    if isinstance(VIDEO_SOURCE, str) and VIDEO_SOURCE not in ("AIRPLAY",):
        if not os.path.exists(VIDEO_SOURCE):
            warnings.append(f"VIDEO_SOURCE '{VIDEO_SOURCE}' nao existe (arquivo de video)")
    elif isinstance(VIDEO_SOURCE, int) and VIDEO_SOURCE < 0:
        errors.append(f"VIDEO_SOURCE invalido: {VIDEO_SOURCE}")

    # Resolucao
    if FRAME_WIDTH < 320 or FRAME_WIDTH > 4096:
        warnings.append(f"FRAME_WIDTH={FRAME_WIDTH} fora do range recomendado (320-4096)")
    if FRAME_HEIGHT < 240 or FRAME_HEIGHT > 2160:
        warnings.append(f"FRAME_HEIGHT={FRAME_HEIGHT} fora do range recomendado (240-2160)")
    if FPS < 1 or FPS > 120:
        warnings.append(f"FPS={FPS} fora do range recomendado (1-120)")

    # AI
    if AI_CONFIDENCE < 0.0 or AI_CONFIDENCE > 1.0:
        errors.append(f"AI_CONFIDENCE={AI_CONFIDENCE} deve estar entre 0.0 e 1.0")
    if AI_FRAME_SKIP < 1 or AI_FRAME_SKIP > 30:
        warnings.append(f"AI_FRAME_SKIP={AI_FRAME_SKIP} fora do range recomendado (1-30)")

    # YOLO Model
    if YOLO_MODEL_PATH and not os.path.exists(YOLO_MODEL_PATH):
        warnings.append(f"YOLO_MODEL_PATH '{YOLO_MODEL_PATH}' nao encontrado (AI usara fallback)")

    # Calibracao
    if ULTRASOUND_DEPTH_MM < 10 or ULTRASOUND_DEPTH_MM > 300:
        warnings.append(f"ULTRASOUND_DEPTH_MM={ULTRASOUND_DEPTH_MM} fora do range tipico (10-300mm)")
    if CALIBRATION_MODE not in ("auto", "manual"):
        errors.append(f"CALIBRATION_MODE '{CALIBRATION_MODE}' invalido (use 'auto' ou 'manual')")
    if MM_PER_PIXEL <= 0 or MM_PER_PIXEL > 2.0:
        warnings.append(f"MM_PER_PIXEL={MM_PER_PIXEL} fora do range tipico (0.01-2.0)")
    if IMAGE_TOP_OFFSET < 0:
        errors.append(f"IMAGE_TOP_OFFSET={IMAGE_TOP_OFFSET} nao pode ser negativo")

    # ROI
    if USE_ROI:
        if ROI_PRESET not in ("butterfly", "clarius", "custom"):
            errors.append(f"ROI_PRESET '{ROI_PRESET}' invalido")
        if ROI_PRESET == "custom":
            if len(ROI_CUSTOM) != 4:
                errors.append("ROI_CUSTOM deve ter 4 valores (x, y, w, h)")
            else:
                for val in ROI_CUSTOM:
                    if not (0.0 <= val <= 1.0):
                        errors.append(f"ROI_CUSTOM valores devem estar entre 0.0 e 1.0")
                        break

    # Screenshot/Video
    if SCREENSHOT_FORMAT not in ("PNG", "BMP", "JPEG", "JPG"):
        warnings.append(f"SCREENSHOT_FORMAT '{SCREENSHOT_FORMAT}' nao reconhecido")
    if VIDEO_QUALITY not in ("lossless", "high", "medium", "light"):
        warnings.append(f"VIDEO_QUALITY '{VIDEO_QUALITY}' nao reconhecido")

    # Diretorio de capturas
    if CAPTURES_DIR:
        try:
            os.makedirs(CAPTURES_DIR, exist_ok=True)
        except Exception as e:
            warnings.append(f"Nao foi possivel criar CAPTURES_DIR: {e}")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def print_config_status():
    """Imprime status da validacao de configuracao."""
    is_valid, errors, warnings = validate_config()

    print("\n" + "=" * 50)
    print("  CONFIG STATUS")
    print("=" * 50)

    if is_valid and not warnings:
        print("  [OK] Todas configuracoes validas")
    else:
        if errors:
            print("  [ERRO] Erros encontrados:")
            for e in errors:
                print(f"    - {e}")
        if warnings:
            print("  [AVISO] Avisos:")
            for w in warnings:
                print(f"    - {w}")

    print("=" * 50 + "\n")
    return is_valid
