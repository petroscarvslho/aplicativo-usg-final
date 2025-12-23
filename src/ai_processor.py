"""
AI Processor - Processamento de IA Otimizado para Ultrassom
============================================================
Baseado em tecnicas de: Butterfly iQ3, Clarius, SUPRA, EchoNet
Otimizacoes: MPS/GPU, Lazy Loading, Frame Skip, Resolution Scale,
             Temporal Smoothing, ROI Processing
"""

import torch
import cv2
import numpy as np
import time
import os
import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Deque
from collections import deque
import config
from src.ui_utils import (
    Theme, draw_panel, draw_status_indicator, draw_progress_bar,
    draw_labeled_value, draw_structure_list, draw_scan_quality,
    draw_warning_banner, draw_legend, FONT, FONT_SCALE_LABEL, FONT_SCALE_SMALL
)

# ═══════════════════════════════════════════════════════════════════════════════
# KALMAN FILTER para Needle Tracking (Fase 1 - Otimização Premium)
# ═══════════════════════════════════════════════════════════════════════════════
from filterpy.kalman import KalmanFilter

# ═══════════════════════════════════════════════════════════════════════════════
# NERVE TRACK v2.0 PREMIUM - Sistema Avançado de Detecção de Nervos
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from nerve_track import (
        NerveTrackSystem,
        create_nerve_track_system,
        NerveTracker,
        StructureClassifier,
        NerveIdentifier,
        PremiumVisualRenderer,
        get_block_config,
        get_all_block_ids,
        ALL_NERVE_BLOCKS,
        BLOCK_NAMES,
        HAS_MODEL as NERVE_HAS_MODEL,
    )
    NERVE_TRACK_AVAILABLE = True
    logger.info("NERVE TRACK v2.0 carregado com sucesso")
except ImportError as e:
    NERVE_TRACK_AVAILABLE = False
    logger.warning(f"NERVE TRACK v2.0 não disponível: {e}")

# Atlas Educacional (funciona SEM modelo treinado)
try:
    from nerve_track import (
        EducationalAtlas,
        create_educational_atlas,
        HAS_ATLAS as NERVE_HAS_ATLAS,
    )
    EDUCATIONAL_ATLAS_AVAILABLE = True
    logger.info("Atlas Educacional carregado com sucesso")
except ImportError:
    EDUCATIONAL_ATLAS_AVAILABLE = False
    logger.info("Atlas Educacional não disponível")

# ═══════════════════════════════════════════════════════════════════════════════
# RANSAC para Detecção Robusta de Agulha (Fase 2 - Otimização Premium)
# ═══════════════════════════════════════════════════════════════════════════════
from sklearn.linear_model import RANSACRegressor

# ═══════════════════════════════════════════════════════════════════════════════
# VASST CNN - Detecção de Centroide de Agulha (Fase 4 - Otimização Premium)
# ═══════════════════════════════════════════════════════════════════════════════
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger('USG_FLOW.AI')


class VASSTPyTorch(nn.Module):
    """
    VASST CNN para detecção de centroide de agulha em PyTorch.

    Baseado no paper: "CNN-based Needle Localization for Ultrasound-Guided Interventions"
    Repositório original: https://github.com/VASST/AECAI.CNN-US-Needle-Segmentation

    Arquitetura:
    - 5 camadas Conv2D com LeakyReLU e MaxPool
    - Flatten + 3 camadas Dense
    - Saída: coordenadas normalizadas (y, x) em [0, 1] (compatível via metadata)

    Input: Imagem grayscale (1, H, W)
    Output: Coordenadas (y, x) do centroide da agulha
    """

    def __init__(self, input_shape: Tuple[int, int] = (256, 256)):
        super(VASSTPyTorch, self).__init__()

        self.input_shape = input_shape

        # Bloco de convolução
        self.conv0 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0)

        self.pool = nn.MaxPool2d(2, 2)
        self.leaky = nn.LeakyReLU(0.01)

        # Calcular tamanho do flatten
        self._calculate_flatten_size()

        # Camadas densas
        self.fc0 = nn.Linear(self.flatten_size, 1024)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 16)
        self.output = nn.Linear(16, 2)  # (x, y)

        # Metadados de label (default do treinamento atual)
        self.label_order = "yx"
        self.label_scale = "0_1"

    def _calculate_flatten_size(self):
        """Calcula o tamanho após todas as convoluções."""
        x = torch.zeros(1, 1, self.input_shape[0], self.input_shape[1])
        x = self.pool(self.leaky(self.conv0(x)))
        x = self.pool(self.leaky(self.conv1(x)))
        x = self.pool(self.leaky(self.conv2(x)))
        x = self.pool(self.leaky(self.conv3(x)))
        x = self.pool(self.leaky(self.conv4(x)))
        self.flatten_size = x.view(1, -1).size(1)

    def forward(self, x):
        # Convoluções
        x = self.pool(self.leaky(self.conv0(x)))
        x = self.pool(self.leaky(self.conv1(x)))
        x = self.pool(self.leaky(self.conv2(x)))
        x = self.pool(self.leaky(self.conv3(x)))
        x = self.pool(self.leaky(self.conv4(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Camadas densas
        x = self.leaky(self.fc0(x))
        x = self.leaky(self.fc1(x))
        x = self.leaky(self.fc2(x))
        x = self.output(x)  # Linear activation

        return x

    def predict_centroid(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Prediz o centroide da agulha em uma imagem.

        Args:
            image: Imagem grayscale (H, W) ou BGR (H, W, 3)

        Returns:
            (x, y): Coordenadas do centroide em pixels
        """
        self.eval()

        # Converter para grayscale se necessário
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = image.shape

        # Resize para input_shape
        resized = cv2.resize(image, self.input_shape)

        # Normalizar para [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Converter para tensor
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = self.forward(tensor)

        vals = output[0].cpu().numpy()
        if getattr(self, "label_order", "yx") == "xy":
            x_norm, y_norm = vals
        else:
            y_norm, x_norm = vals

        scale = getattr(self, "label_scale", "auto")
        if scale not in {"0_1", "neg1_1"}:
            scale = "auto"

        if scale == "auto":
            scale = "neg1_1" if (x_norm < 0 or y_norm < 0) else "0_1"

        if scale == "neg1_1":
            x = (x_norm + 1) / 2 * w
            y = (y_norm + 1) / 2 * h
        else:
            x = x_norm * w
            y = y_norm * h

        x = float(np.clip(x, 0, w - 1))
        y = float(np.clip(y, 0, h - 1))

        return x, y


class TemporalSmoother:
    """Suavizacao temporal para reducao de jitter nas deteccoes."""

    def __init__(self, alpha=0.3, max_history=10):
        self.alpha = alpha  # Peso do frame atual (0.3 = 30% atual, 70% historico)
        self.history = deque(maxlen=max_history)
        self.last_detections = None

    def smooth(self, detections):
        """Aplica exponential moving average nas deteccoes."""
        if not detections:
            return detections

        if self.last_detections is None:
            self.last_detections = detections
            return detections

        smoothed = []
        for curr in detections:
            # Encontrar deteccao mais proxima no historico
            best_match = None
            min_dist = float('inf')

            for prev in self.last_detections:
                if curr.get('class_id') == prev.get('class_id'):
                    # Distancia entre centros
                    cx1, cy1 = curr.get('center', (0, 0))
                    cx2, cy2 = prev.get('center', (0, 0))
                    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_match = prev

            if best_match and min_dist < 100:  # Threshold de matching
                # Interpolar posicao
                smoothed_bbox = []
                for c, p in zip(curr.get('bbox', []), best_match.get('bbox', [])):
                    smoothed_bbox.append(self.alpha * c + (1 - self.alpha) * p)
                curr['bbox'] = smoothed_bbox

            smoothed.append(curr)

        self.last_detections = smoothed
        return smoothed


class SmartFrameSkipper:
    """Frame skip inteligente baseado em movimento."""

    def __init__(self, base_skip=3, motion_threshold=15.0):
        self.base_skip = base_skip  # Skip padrao
        self.motion_threshold = motion_threshold
        self.skip_count = 0
        self.last_frame = None
        self.last_result = None

    def should_process(self, frame):
        """Decide se deve processar este frame."""
        self.skip_count += 1

        if self.last_frame is None:
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True

        # Calcular movimento
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Downscale para calculo rapido
        small_curr = cv2.resize(gray, (160, 90))
        small_prev = cv2.resize(self.last_frame, (160, 90))

        # Diferenca absoluta media
        motion = np.mean(cv2.absdiff(small_curr, small_prev))

        # Atualizar frame anterior
        self.last_frame = gray

        # Alto movimento = processar imediatamente
        if motion > self.motion_threshold:
            self.skip_count = 0
            return True

        # Baixo movimento = pode pular mais frames
        max_skip = self.base_skip + 2 if motion < self.motion_threshold / 2 else self.base_skip

        if self.skip_count >= max_skip:
            self.skip_count = 0
            return True

        return False

    def get_last_result(self):
        return self.last_result

    def set_last_result(self, result):
        self.last_result = result


class AIProcessor:
    """
    Processador de IA otimizado para ultrassom.
    Suporta todos os 11 modos com lazy loading.
    """

    # Mapeamento de modos para configuracoes especificas
    MODE_CONFIG = {
        'needle': {
            'model_type': 'yolo',
            'model_path': config.YOLO_MODEL_PATH,
            'resolution_scale': 0.75,  # Processar em 75% da resolucao
            'frame_skip': 2,
            'roi': None,  # Full frame
        },
        'nerve': {
            'model_type': 'unet',
            'model_path': 'models/unet_nerve.pt',
            'resolution_scale': 0.5,  # Processar em 50%
            'frame_skip': 3,
            'roi': 'center_70',  # Centro 70%
        },
        'cardiac': {
            'model_type': 'echonet',
            'model_path': 'models/echonet.pt',
            'resolution_scale': 1.0,  # Full resolution
            'frame_skip': 1,  # Todo frame (critico)
            'roi': None,
        },
        'fast': {
            'model_type': 'yolo',
            'model_path': 'models/fast_detector.pt',
            'resolution_scale': 0.75,
            'frame_skip': 2,
            'roi': None,
        },
        'segment': {  # ANATOMIA
            'model_type': 'usfm',
            'model_path': 'models/usfm_segment.pt',
            'resolution_scale': 0.5,
            'frame_skip': 3,
            'roi': None,
        },
        'm_mode': {
            'model_type': 'custom',
            'model_path': None,  # Processamento CV
            'resolution_scale': 1.0,
            'frame_skip': 1,
            'roi': None,
        },
        'color': {  # COLOR DOPPLER
            'model_type': 'doppler',
            'model_path': None,  # Simulacao
            'resolution_scale': 1.0,
            'frame_skip': 1,
            'roi': None,
        },
        'power': {  # POWER DOPPLER
            'model_type': 'doppler',
            'model_path': None,
            'resolution_scale': 1.0,
            'frame_skip': 1,
            'roi': None,
        },
        'b_lines': {  # PULMAO
            'model_type': 'custom',
            'model_path': None,
            'resolution_scale': 0.75,
            'frame_skip': 3,
            'roi': None,
        },
        'bladder': {  # BEXIGA
            'model_type': 'unet',
            'model_path': 'models/bladder_seg.pt',
            'resolution_scale': 0.5,
            'frame_skip': 3,
            'roi': 'center_80',
        },
    }

    def __init__(self):
        # Device (prioriza MPS/GPU)
        if config.USE_MPS and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info("Usando MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info("Usando CUDA GPU")
        else:
            self.device = torch.device('cpu')
            logger.info("Usando CPU")

        # Modelos carregados (lazy loading)
        self.models = {}
        self.active_mode = None

        # Otimizadores
        self.frame_skipper = SmartFrameSkipper(base_skip=config.AI_FRAME_SKIP)
        self.temporal_smoother = TemporalSmoother(alpha=0.3)

        # Historico para visualizacoes
        self.needle_history = deque(maxlen=15)
        self.b_line_count = 0
        self.cardiac_ef = None
        self.bladder_volume = None

        # Metricas de performance
        self.last_inference_time = 0
        self.inference_count = 0

        # ═══════════════════════════════════════
        # PRE-INICIALIZAR VARIAVEIS DE TODOS OS MODOS
        # (evita hasattr checks a cada frame)
        # ═══════════════════════════════════════

        # Cardiac AI
        self.cardiac_history = deque(maxlen=60)
        self.cardiac_edv = None
        self.cardiac_esv = None
        self.cardiac_gls = None
        self.cardiac_hr = None
        self.cardiac_phase = 'diastole'
        self.cardiac_cycle_frames = []
        self.cardiac_last_peak = 0
        self.cardiac_view = 'A4C'  # A4C, PLAX, PSAX, A2C
        self.cardiac_view_confidence = 0.0
        self.cardiac_lv_aspect_ratio = 1.0

        # FAST Protocol
        self.fast_windows = {
            'RUQ': {'status': 'pending', 'fluid': False, 'score': 0, 'time_spent': 0},
            'LUQ': {'status': 'pending', 'fluid': False, 'score': 0, 'time_spent': 0},
            'PELV': {'status': 'pending', 'fluid': False, 'score': 0, 'time_spent': 0},
            'CARD': {'status': 'pending', 'fluid': False, 'score': 0, 'time_spent': 0},
        }
        self.fast_current_window = 'RUQ'
        self.fast_start_time = None
        self.fast_fluid_regions = []
        self.fast_window_history = {key: deque(maxlen=8) for key in self.fast_windows}

        # FAST Auto-Navegacao
        self.fast_window_order = ['RUQ', 'LUQ', 'PELV', 'CARD']
        self.fast_window_start_time = None  # Quando iniciou a janela atual
        self.fast_auto_advance = True       # Avancar automaticamente quando confirmar
        self.fast_stability_time = 0.0      # Tempo com score estavel
        self.fast_stability_threshold = 2.5 # Segundos de estabilidade para confirmar
        self.fast_min_scan_time = 3.0       # Tempo minimo de scan por janela
        self.fast_last_stable_score = 0.0   # Ultimo score estavel
        self.fast_confirmed_pending = False # Confirmacao pendente (piscar)

        # Anatomy AI
        self.anatomy_structures = []
        self.anatomy_measurements = []

        # M-Mode
        self.mmode_cursor_x = 320  # Default, sera ajustado
        self.mmode_buffer = []
        self.mmode_timeline_height = 160
        self.mmode_measurements = []

        # Color Doppler
        self.doppler_prf = 4000
        self.doppler_scale = 50
        self.doppler_aliasing_detected = False
        self.doppler_prev_frame = None
        self.doppler_flow_history = []

        # Power Doppler
        self.power_sensitivity = 75
        self.power_prev_frame = None

        # Lung AI
        self.lung_b_line_history = deque(maxlen=30)
        self.lung_a_lines_detected = False
        self.lung_pleura_sliding = True
        self.lung_consolidation = False
        self.lung_pleura_history = deque(maxlen=12)
        self.lung_b_line_density_history = deque(maxlen=30)

        # ═══════════════════════════════════════════════════════════════════════
        # LUNG ZONES - Protocolo BLUE (8 zonas)
        # ═══════════════════════════════════════════════════════════════════════
        self.lung_zone_order = ['R1', 'R2', 'R3', 'R4', 'L1', 'L2', 'L3', 'L4']
        self.lung_zone_names = {
            'R1': 'Right Upper Ant',
            'R2': 'Right Lower Ant',
            'R3': 'Right Lateral',
            'R4': 'Right Posterior',
            'L1': 'Left Upper Ant',
            'L2': 'Left Lower Ant',
            'L3': 'Left Lateral',
            'L4': 'Left Posterior',
        }
        self.lung_zones = {zone: {
            'status': 'pending',  # pending, scanning, checked
            'b_lines': 0,
            'a_lines': False,
            'sliding': True,
            'profile': 'A',  # A or B
            'time_spent': 0.0,
        } for zone in self.lung_zone_order}
        self.lung_current_zone = 'R1'
        self.lung_zone_start_time = None
        self.lung_zone_min_scan_time = 2.0  # Segundos mínimos por zona
        self.lung_exam_complete = False
        self.lung_exam_start_time = None

        # Bladder AI
        self.bladder_history = deque(maxlen=30)
        self.bladder_d1 = None
        self.bladder_d2 = None
        self.bladder_d3 = None
        self.bladder_pvr_mode = False
        self.bladder_pre_void = None
        self.bladder_view = None
        self.bladder_quality = 0.0
        self.bladder_last_update = 0.0
        self.bladder_view_data = {
            'transverse': {'major': None, 'minor': None, 't': 0.0},
            'sagittal': {'major': None, 'minor': None, 't': 0.0},
        }
        self.bladder_view_history = {
            'transverse': deque(maxlen=8),
            'sagittal': deque(maxlen=8),
        }

        # ═══════════════════════════════════════════════════════════════════════
        # NERVE TRACK v2.0 PREMIUM - Sistema Avançado de Detecção de Nervos
        # ═══════════════════════════════════════════════════════════════════════
        # Sistema completo com:
        # - U-Net++ + CBAM + ConvLSTM para segmentação
        # - Kalman Filter para tracking temporal
        # - Classificador Nervo/Artéria/Veia
        # - Identificador de nervos específicos por bloqueio
        # - Medição automática de CSA
        # - Visual premium por tipo de bloqueio
        # ═══════════════════════════════════════════════════════════════════════
        self.nerve_track_system = None
        self.nerve_block_id = None  # Tipo de bloqueio selecionado
        self.nerve_track_available = False

        if NERVE_TRACK_AVAILABLE:
            try:
                # Calcular mm_per_pixel baseado na calibração
                mm_per_pixel = config.MM_PER_PIXEL if hasattr(config, 'MM_PER_PIXEL') else 0.1

                self.nerve_track_system = create_nerve_track_system(
                    block_id=None,  # Será definido pelo usuário
                    mm_per_pixel=mm_per_pixel,
                    use_model=True,
                    use_temporal=True,
                    device='auto'
                )
                self.nerve_track_available = True
                logger.info("NERVE TRACK v2.0 inicializado com sucesso")
            except Exception as e:
                logger.warning(f"Erro ao inicializar NERVE TRACK v2.0: {e}")
                self.nerve_track_available = False

        # Lista de bloqueios disponíveis (para UI)
        self.available_nerve_blocks = {}
        if NERVE_TRACK_AVAILABLE:
            try:
                self.available_nerve_blocks = BLOCK_NAMES
            except:
                pass

        # ═══════════════════════════════════════════════════════════════════════
        # ATLAS EDUCACIONAL - Funciona SEM modelo treinado
        # ═══════════════════════════════════════════════════════════════════════
        self.educational_atlas = None
        self.atlas_mode = "side"  # "side", "overlay", "full"

        if EDUCATIONAL_ATLAS_AVAILABLE:
            try:
                mm_per_pixel = config.MM_PER_PIXEL if hasattr(config, 'MM_PER_PIXEL') else 0.1
                self.educational_atlas = create_educational_atlas(mm_per_pixel=mm_per_pixel)
                logger.info("Atlas Educacional inicializado")
            except Exception as e:
                logger.warning(f"Erro ao inicializar Atlas Educacional: {e}")

        # ═══════════════════════════════════════════════════════════════════════
        # KALMAN FILTER - Needle Tracking Premium (Fase 1)
        # ═══════════════════════════════════════════════════════════════════════
        # Estado: [x, y, vx, vy, angle, v_angle]
        # - x, y: posição da ponta da agulha (pixels)
        # - vx, vy: velocidade da ponta (pixels/frame)
        # - angle: ângulo da agulha (graus)
        # - v_angle: velocidade angular (graus/frame)
        # ═══════════════════════════════════════════════════════════════════════
        self.needle_kalman = self._create_needle_kalman()
        self.kalman_initialized = False
        self.frames_without_detection = 0
        self.max_prediction_frames = 10  # Máximo de frames para predizer sem detecção
        self.kalman_confidence = 0.0  # Confiança atual do Kalman
        self.needle_tracking_mode = 'searching'  # 'searching', 'tracking', 'predicting'

        # ═══════════════════════════════════════════════════════════════════════
        # SISTEMA DE CALIBRAÇÃO - Escala mm/pixel (Fase 3)
        # ═══════════════════════════════════════════════════════════════════════
        # Converte pixels em milímetros reais baseado na profundidade do US
        # ═══════════════════════════════════════════════════════════════════════
        self.calibration = {
            'mode': config.CALIBRATION_MODE,
            'depth_mm': config.ULTRASOUND_DEPTH_MM,
            'mm_per_pixel': config.MM_PER_PIXEL,
            'top_offset': config.IMAGE_TOP_OFFSET,
            'image_height': None,  # Será definido no primeiro frame
            'is_calibrated': False,
        }

        # ═══════════════════════════════════════════════════════════════════════
        # VASST CNN - Detecção de Centroide (Fase 4)
        # ═══════════════════════════════════════════════════════════════════════
        # Modelo CNN para refinar a posição do centroide da agulha
        # Especialmente útil para agulhas out-of-plane (perpendiculares)
        # ═══════════════════════════════════════════════════════════════════════
        self.vasst_model = None
        self.vasst_available = False
        self.vasst_weights_path = 'models/vasst_needle.pt'
        self.use_vasst_refinement = True  # Usar refinamento CV quando VASST não disponível
        self._try_load_vasst()

        logger.info("AI Processor inicializado com otimizações: Kalman + RANSAC + Calibração + VASST")

    # ═══════════════════════════════════════════════════════════════════════════════
    # KALMAN FILTER - Métodos (Fase 1 - Otimização Premium)
    # ═══════════════════════════════════════════════════════════════════════════════

    def _create_needle_kalman(self) -> KalmanFilter:
        """
        Cria Kalman Filter para tracking de agulha.

        O Kalman Filter é um algoritmo que:
        1. SUAVIZA posições ruidosas (agulha para de tremer)
        2. PREDIZ posição quando a detecção falha (agulha "some")
        3. USA FÍSICA do movimento para estimar posições futuras

        Estado (6 variáveis):
        - x, y: posição da ponta da agulha em pixels
        - vx, vy: velocidade da ponta em pixels/frame
        - angle: ângulo da agulha em graus
        - v_angle: velocidade angular em graus/frame

        Medição (3 variáveis):
        - x, y: posição detectada
        - angle: ângulo detectado

        Returns:
            KalmanFilter configurado para needle tracking
        """
        kf = KalmanFilter(dim_x=6, dim_z=3)

        # ═══════════════════════════════════════════════════════════════════════
        # MATRIZ DE TRANSIÇÃO DE ESTADO (F)
        # ═══════════════════════════════════════════════════════════════════════
        # Define como o estado evolui de um frame para o próximo
        # Assume modelo de velocidade constante
        dt = 1.0 / 30.0  # ~30 FPS (ajustar se necessário)

        kf.F = np.array([
            [1, 0, dt, 0,  0,  0 ],   # x_new = x + vx * dt
            [0, 1, 0,  dt, 0,  0 ],   # y_new = y + vy * dt
            [0, 0, 1,  0,  0,  0 ],   # vx_new = vx (velocidade constante)
            [0, 0, 0,  1,  0,  0 ],   # vy_new = vy
            [0, 0, 0,  0,  1,  dt],   # angle_new = angle + v_angle * dt
            [0, 0, 0,  0,  0,  1 ],   # v_angle_new = v_angle
        ])

        # ═══════════════════════════════════════════════════════════════════════
        # MATRIZ DE MEDIÇÃO (H)
        # ═══════════════════════════════════════════════════════════════════════
        # Define quais variáveis de estado conseguimos medir diretamente
        # Medimos: x, y, angle (não medimos velocidades diretamente)

        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],   # Medimos x
            [0, 1, 0, 0, 0, 0],   # Medimos y
            [0, 0, 0, 0, 1, 0],   # Medimos angle
        ])

        # ═══════════════════════════════════════════════════════════════════════
        # COVARIÂNCIA DO PROCESSO (Q)
        # ═══════════════════════════════════════════════════════════════════════
        # Representa a incerteza no modelo de movimento
        # Valores maiores = modelo menos confiável = responde mais rápido a mudanças

        kf.Q = np.eye(6)
        kf.Q[0, 0] = 0.1    # Incerteza em x (posição muda pouco sozinha)
        kf.Q[1, 1] = 0.1    # Incerteza em y
        kf.Q[2, 2] = 1.0    # Incerteza em vx (velocidade pode mudar)
        kf.Q[3, 3] = 1.0    # Incerteza em vy
        kf.Q[4, 4] = 0.5    # Incerteza em angle
        kf.Q[5, 5] = 0.5    # Incerteza em v_angle

        # ═══════════════════════════════════════════════════════════════════════
        # COVARIÂNCIA DA MEDIÇÃO (R)
        # ═══════════════════════════════════════════════════════════════════════
        # Representa a incerteza nas detecções
        # Valores maiores = detecção menos confiável = suaviza mais

        kf.R = np.array([
            [5.0, 0,   0  ],   # Incerteza em x: ~5 pixels
            [0,   5.0, 0  ],   # Incerteza em y: ~5 pixels
            [0,   0,   3.0],   # Incerteza em angle: ~3 graus
        ])

        # ═══════════════════════════════════════════════════════════════════════
        # COVARIÂNCIA INICIAL (P)
        # ═══════════════════════════════════════════════════════════════════════
        # Alta incerteza inicial (não sabemos onde a agulha está)

        kf.P *= 1000

        # Estado inicial zerado (será sobrescrito na primeira detecção)
        kf.x = np.zeros((6, 1))

        return kf

    def _update_needle_kalman(self, detected: bool, tip_x: float, tip_y: float,
                               angle: float) -> Tuple[Optional[float], Optional[float],
                                                       Optional[float], float]:
        """
        Atualiza o Kalman Filter com nova medição ou faz predição.

        Este método é o CORAÇÃO do sistema de tracking:
        1. Se detectou agulha → atualiza Kalman com medição → retorna posição suavizada
        2. Se NÃO detectou → usa Kalman para PREDIZER posição → retorna predição

        Args:
            detected: True se a agulha foi detectada neste frame
            tip_x: coordenada X da ponta detectada (ignorado se detected=False)
            tip_y: coordenada Y da ponta detectada (ignorado se detected=False)
            angle: ângulo da agulha em graus (ignorado se detected=False)

        Returns:
            Tupla (x, y, angle, confidence):
            - x, y: posição suavizada ou predita (None se não há tracking)
            - angle: ângulo suavizado ou predito
            - confidence: confiança de 0.0 a 1.0
        """
        if detected:
            # ═══════════════════════════════════════════════════════════════════
            # CASO 1: Agulha detectada → Atualizar Kalman
            # ═══════════════════════════════════════════════════════════════════

            # Resetar contador de frames sem detecção
            self.frames_without_detection = 0
            self.needle_tracking_mode = 'tracking'

            if not self.kalman_initialized:
                # Primeira detecção - inicializar estado
                self.needle_kalman.x = np.array([
                    [tip_x],    # x
                    [tip_y],    # y
                    [0.0],      # vx (velocidade inicial zero)
                    [0.0],      # vy
                    [angle],    # angle
                    [0.0],      # v_angle
                ])
                self.kalman_initialized = True
                self.kalman_confidence = 1.0
                logger.debug(f"Kalman inicializado em ({tip_x:.1f}, {tip_y:.1f})")
                return tip_x, tip_y, angle, 1.0

            # Passo 1: Predição (baseado no modelo de movimento)
            self.needle_kalman.predict()

            # Passo 2: Atualização com a medição
            z = np.array([[tip_x], [tip_y], [angle]])
            self.needle_kalman.update(z)

            # Extrair estado suavizado
            state = self.needle_kalman.x
            smooth_x = float(state[0, 0])
            smooth_y = float(state[1, 0])
            smooth_angle = float(state[4, 0])

            self.kalman_confidence = 1.0
            return smooth_x, smooth_y, smooth_angle, 1.0

        else:
            # ═══════════════════════════════════════════════════════════════════
            # CASO 2: Agulha NÃO detectada → Usar predição
            # ═══════════════════════════════════════════════════════════════════

            self.frames_without_detection += 1

            if not self.kalman_initialized:
                # Nunca detectamos nada - não há o que predizer
                self.needle_tracking_mode = 'searching'
                self.kalman_confidence = 0.0
                return None, None, None, 0.0

            if self.frames_without_detection > self.max_prediction_frames:
                # Muitos frames sem detecção - perder tracking
                # (agulha provavelmente saiu do campo de visão)
                self.kalman_initialized = False
                self.needle_tracking_mode = 'searching'
                self.kalman_confidence = 0.0
                logger.debug("Kalman: tracking perdido após muitos frames sem detecção")
                return None, None, None, 0.0

            # Ainda dentro do limite - fazer predição
            self.needle_tracking_mode = 'predicting'

            # Apenas predição (sem update porque não temos medição)
            self.needle_kalman.predict()

            # Confiança decai linearmente com o tempo sem detecção
            self.kalman_confidence = 1.0 - (self.frames_without_detection /
                                             self.max_prediction_frames)

            # Extrair estado predito
            state = self.needle_kalman.x
            pred_x = float(state[0, 0])
            pred_y = float(state[1, 0])
            pred_angle = float(state[4, 0])

            return pred_x, pred_y, pred_angle, self.kalman_confidence

    def _reset_needle_kalman(self) -> None:
        """
        Reseta o Kalman Filter para estado inicial.
        Útil quando o usuário quer reiniciar o tracking.
        """
        self.needle_kalman = self._create_needle_kalman()
        self.kalman_initialized = False
        self.frames_without_detection = 0
        self.kalman_confidence = 0.0
        self.needle_tracking_mode = 'searching'
        logger.debug("Kalman Filter resetado")

    # ═══════════════════════════════════════════════════════════════════════════════
    # RANSAC - FASE 2: Detecção Robusta de Linha (Remove Outliers)
    # ═══════════════════════════════════════════════════════════════════════════════

    def _ransac_fit_needle(self, points: List[Tuple[int, int]],
                           min_samples: int = 3) -> Optional[Tuple[int, int, int, int, int]]:
        """
        Usa RANSAC para encontrar a melhor linha entre pontos detectados.

        O RANSAC (Random Sample Consensus) é robusto a outliers - ignora
        detecções falsas e encontra a linha que melhor representa a agulha.

        Args:
            points: lista de (x, y) dos pontos detectados por Hough
            min_samples: mínimo de pontos para considerar válido

        Returns:
            (x1, y1, x2, y2, inliers_count): linha ajustada e quantidade de inliers
            ou None se não conseguir ajustar
        """
        if len(points) < min_samples:
            return None

        # Separar X e Y
        X = np.array([p[0] for p in points]).reshape(-1, 1)
        y = np.array([p[1] for p in points])

        try:
            # RANSAC para regressão linear
            ransac = RANSACRegressor(
                min_samples=min_samples,
                residual_threshold=10.0,  # pixels de tolerância
                max_trials=100,
                random_state=42  # Reprodutibilidade
            )
            ransac.fit(X, y)

            # Obter inliers (pontos que fazem parte da linha)
            inlier_mask = ransac.inlier_mask_
            inliers_count = int(np.sum(inlier_mask))

            if inliers_count < min_samples:
                return None

            # Calcular extremos da linha usando apenas inliers
            X_inliers = X[inlier_mask]
            x_min, x_max = float(X_inliers.min()), float(X_inliers.max())

            # Prever Y nos extremos
            y_min = float(ransac.predict([[x_min]])[0])
            y_max = float(ransac.predict([[x_max]])[0])

            return (int(x_min), int(y_min), int(x_max), int(y_max), inliers_count)

        except Exception as e:
            logger.debug(f"RANSAC falhou: {e}")
            return None

    def _detect_needle_enhanced(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int, int, float]]:
        """
        Detecção de agulha melhorada com RANSAC.

        Pipeline:
        1. Melhora contraste com CLAHE
        2. Detecta edges com Canny
        3. Encontra linhas candidatas com HoughLinesP
        4. Filtra por ângulo típico de agulha
        5. Coleta pontos ao longo das linhas
        6. Usa RANSAC para encontrar a melhor linha

        Args:
            frame: Frame BGR para processar

        Returns:
            (x1, y1, x2, y2, inliers, confidence) ou None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar CLAHE para melhorar contraste em regiões escuras
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Detectar edges
        edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)

        # Encontrar linhas candidatas
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,  # Mais sensível
            minLineLength=40,
            maxLineGap=15
        )

        if lines is None:
            return None

        # Coletar pontos de linhas com ângulo típico de agulha (15-75° ou 105-165°)
        needle_points = []
        line_scores = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            # Filtrar por ângulo típico de agulha
            if 15 < angle < 75 or 105 < angle < 165:
                # Score baseado em comprimento e proximidade de 45°
                score = length * (1 - abs(angle - 45) / 45)
                line_scores.append(score)

                # Adicionar pontos ao longo da linha para RANSAC
                num_points = max(5, int(length / 10))
                for t in np.linspace(0, 1, num_points):
                    px = int(x1 + t * (x2 - x1))
                    py = int(y1 + t * (y2 - y1))
                    needle_points.append((px, py))

        if len(needle_points) < 5:
            return None

        # Aplicar RANSAC
        result = self._ransac_fit_needle(needle_points)

        if result is None:
            return None

        x1, y1, x2, y2, inliers = result

        # Calcular confiança baseada em inliers e qualidade das linhas originais
        conf = min(0.95, inliers / 20)  # Mais inliers = mais confiança
        if line_scores:
            avg_score = np.mean(line_scores)
            conf = conf * min(1.0, avg_score / 100)

        return (x1, y1, x2, y2, inliers, conf)

    # ═══════════════════════════════════════════════════════════════════════════════
    # CALIBRAÇÃO - FASE 3: Conversão pixel ↔ milímetros
    # ═══════════════════════════════════════════════════════════════════════════════

    def _calibrate_scale(self, frame_height: int) -> float:
        """
        Calcula a escala mm/pixel baseado na configuração.

        Em modo "auto", calcula baseado na altura do frame e profundidade configurada.
        Em modo "manual", usa o valor fixo de mm_per_pixel.

        Args:
            frame_height: altura do frame em pixels

        Returns:
            mm_per_pixel: escala calculada
        """
        self.calibration['image_height'] = frame_height

        if self.calibration['mode'] == 'auto':
            # Altura útil (descontando offset do topo)
            useful_height = frame_height - self.calibration['top_offset']

            if useful_height > 0:
                # Calcular escala: mm_total / pixels_totais
                mm_per_pixel = self.calibration['depth_mm'] / useful_height
                self.calibration['mm_per_pixel'] = mm_per_pixel
                self.calibration['is_calibrated'] = True
            else:
                # Fallback se altura inválida
                self.calibration['mm_per_pixel'] = config.MM_PER_PIXEL
                self.calibration['is_calibrated'] = False

        else:
            # Modo manual - usa valor fixo
            self.calibration['mm_per_pixel'] = config.MM_PER_PIXEL
            self.calibration['is_calibrated'] = True

        return self.calibration['mm_per_pixel']

    def _pixel_to_mm(self, pixel_y: int) -> float:
        """
        Converte posição Y em pixels para profundidade em mm.

        Args:
            pixel_y: posição Y em pixels (do topo da imagem)

        Returns:
            depth_mm: profundidade em milímetros
        """
        adjusted_y = pixel_y - self.calibration['top_offset']
        return max(0, adjusted_y * self.calibration['mm_per_pixel'])

    def _mm_to_pixel(self, depth_mm: float) -> int:
        """
        Converte profundidade em mm para posição Y em pixels.

        Args:
            depth_mm: profundidade em milímetros

        Returns:
            pixel_y: posição Y em pixels
        """
        if self.calibration['mm_per_pixel'] > 0:
            return int(depth_mm / self.calibration['mm_per_pixel']) + self.calibration['top_offset']
        return 0

    def set_ultrasound_depth(self, depth_mm: int) -> None:
        """
        Define a profundidade do ultrassom (chamado pelos atalhos [ e ]).

        Args:
            depth_mm: profundidade total em milímetros
        """
        self.calibration['depth_mm'] = max(10, min(300, depth_mm))
        # Forçar recalibração no próximo frame
        if self.calibration['image_height']:
            self._calibrate_scale(self.calibration['image_height'])
        logger.info(f"Profundidade ajustada para {self.calibration['depth_mm']}mm")

    def get_calibration_info(self) -> Dict[str, Any]:
        """
        Retorna informações de calibração para exibição.

        Returns:
            Dict com informações de calibração
        """
        return {
            'depth_mm': self.calibration['depth_mm'],
            'mm_per_pixel': self.calibration['mm_per_pixel'],
            'mode': self.calibration['mode'],
            'is_calibrated': self.calibration['is_calibrated'],
        }

    # ═══════════════════════════════════════════════════════════════════════════════
    # VASST CNN - FASE 4: Refinamento de Centroide
    # ═══════════════════════════════════════════════════════════════════════════════

    def _try_load_vasst(self) -> None:
        """
        Tenta carregar o modelo VASST CNN.
        Se não houver weights disponíveis, marca como indisponível.
        """
        if os.path.exists(self.vasst_weights_path):
            try:
                self.vasst_model = VASSTPyTorch()
                state = torch.load(self.vasst_weights_path, map_location=self.device)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state_dict = state["model_state_dict"]
                    meta = state.get("meta", {})
                else:
                    state_dict = state
                    meta = {}

                # Metadata via arquivo .meta.json (tem prioridade)
                meta_path = Path(self.vasst_weights_path).with_suffix(".meta.json")
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                    except Exception:
                        pass

                self.vasst_model.load_state_dict(state_dict)
                self.vasst_model.label_scale = meta.get("label_scale", self.vasst_model.label_scale)
                self.vasst_model.label_order = meta.get("label_order", self.vasst_model.label_order)
                self.vasst_model.to(self.device)
                self.vasst_model.eval()
                self.vasst_available = True
                logger.info("VASST CNN carregado com sucesso")
            except Exception as e:
                logger.warning(f"Falha ao carregar VASST: {e}")
                self.vasst_available = False
        else:
            logger.info("VASST weights não encontrados - usando refinamento CV")
            self.vasst_available = False

    def _vasst_predict_centroid(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Usa VASST CNN para predizer o centroide da agulha.

        Args:
            frame: Frame BGR

        Returns:
            (x, y) do centroide ou None se não conseguir
        """
        if self.vasst_available and self.vasst_model is not None:
            try:
                x, y = self.vasst_model.predict_centroid(frame)
                return (x, y)
            except Exception as e:
                logger.debug(f"VASST prediction failed: {e}")
                return None
        return None

    def _cv_refine_centroid(self, frame: np.ndarray, initial_tip: Tuple[int, int],
                            search_radius: int = 30) -> Tuple[int, int, float]:
        """
        Refina a posição do centroide usando técnicas CV avançadas.

        Este método é usado como fallback quando VASST não está disponível.
        Analisa a região ao redor da ponta detectada para encontrar
        o ponto mais provável de ser o centroide real.

        Técnicas utilizadas:
        1. CLAHE para melhor contraste
        2. Gradiente de Sobel para detectar bordas
        3. Análise de intensidade local
        4. Ponderação por distância do ponto inicial

        Args:
            frame: Frame BGR
            initial_tip: Posição inicial da ponta (x, y)
            search_radius: Raio de busca em pixels

        Returns:
            (x, y, confidence): Centroide refinado e confiança
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Aplicar CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Calcular gradientes
        grad_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Definir região de busca
        x0, y0 = initial_tip
        x_min = max(0, x0 - search_radius)
        x_max = min(w, x0 + search_radius)
        y_min = max(0, y0 - search_radius)
        y_max = min(h, y0 + search_radius)

        # Extrair região
        roi_grad = grad_mag[y_min:y_max, x_min:x_max]
        roi_intensity = enhanced[y_min:y_max, x_min:x_max]

        if roi_grad.size == 0:
            return initial_tip[0], initial_tip[1], 0.5

        # Criar mapa de pesos
        # Combinar: alto gradiente + alta intensidade + proximidade do centro
        roi_h, roi_w = roi_grad.shape
        cy, cx = roi_h // 2, roi_w // 2
        y_coords, x_coords = np.ogrid[:roi_h, :roi_w]
        distance_map = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        distance_weight = 1 - (distance_map / (search_radius * 1.5))
        distance_weight = np.clip(distance_weight, 0, 1)

        # Normalizar gradiente e intensidade
        grad_norm = roi_grad / (roi_grad.max() + 1e-6)
        intensity_norm = roi_intensity / 255.0

        # Score combinado
        # Para agulhas: alto gradiente (borda) e alta intensidade (reflexão)
        score_map = (0.5 * grad_norm + 0.3 * intensity_norm + 0.2 * distance_weight)

        # Encontrar máximo
        max_idx = np.unravel_index(np.argmax(score_map), score_map.shape)
        refined_y = y_min + max_idx[0]
        refined_x = x_min + max_idx[1]

        # Calcular confiança baseada no score máximo
        confidence = float(np.max(score_map))

        return refined_x, refined_y, confidence

    def _refine_needle_position(self, frame: np.ndarray, detected_tip: Tuple[int, int],
                                 detected_line: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int, float]:
        """
        Refina a posição da ponta da agulha usando VASST ou CV.

        Este é o método principal que decide qual técnica usar:
        1. Se VASST disponível → usa CNN
        2. Senão → usa refinamento CV

        Args:
            frame: Frame BGR
            detected_tip: Ponta detectada (x, y)
            detected_line: Linha detectada (x1, y1, x2, y2) ou None

        Returns:
            (x, y, confidence): Posição refinada e confiança
        """
        # Tentar VASST primeiro
        if self.vasst_available:
            vasst_result = self._vasst_predict_centroid(frame)
            if vasst_result is not None:
                # VASST retorna o centroide - precisamos estimar a ponta
                # Para isso, usamos a direção da linha detectada
                cx, cy = vasst_result

                if detected_line is not None:
                    x1, y1, x2, y2 = detected_line
                    dx, dy = x2 - x1, y2 - y1
                    length = np.sqrt(dx**2 + dy**2)

                    if length > 0:
                        # Mover do centroide para a ponta
                        ux, uy = dx / length, dy / length
                        tip_x = cx + ux * length * 0.4  # Aproximar ponta
                        tip_y = cy + uy * length * 0.4

                        return int(tip_x), int(tip_y), 0.95

                return int(cx), int(cy), 0.85

        # Fallback: refinamento CV
        if self.use_vasst_refinement:
            refined_x, refined_y, conf = self._cv_refine_centroid(frame, detected_tip)
            return refined_x, refined_y, conf

        # Sem refinamento
        return detected_tip[0], detected_tip[1], 0.7

    def set_mode(self, mode: Optional[str]) -> None:
        """Define modo ativo e carrega modelo se necessario."""
        if mode == self.active_mode:
            return

        old_mode = self.active_mode
        self.active_mode = mode

        # Resetar frame skipper
        self.frame_skipper.skip_count = 0
        self.frame_skipper.last_result = None
        self.frame_skipper.last_frame = None

        # Resetar historicos do modo anterior
        self._reset_mode_state(old_mode)

        # Carregar modelo se necessario
        if mode and mode not in self.models:
            self._load_model(mode)

        logger.debug(f"Modo AI: {mode or 'Desligado'}")

    def _reset_mode_state(self, old_mode):
        """Limpa estado do modo anterior para evitar artefatos."""
        # Reset geral
        self.needle_history.clear()

        # Reset especifico por modo
        if old_mode == 'cardiac':
            self.cardiac_history.clear()
            self.cardiac_ef = None
            self.cardiac_edv = None
            self.cardiac_esv = None
            self.cardiac_gls = None
            self.cardiac_hr = None
            self.cardiac_phase = 'diastole'
            self.cardiac_cycle_frames.clear()
            self.cardiac_last_peak = 0

        elif old_mode == 'fast':
            for key in self.fast_windows:
                self.fast_windows[key] = {'status': 'pending', 'fluid': False, 'score': 0, 'time_spent': 0}
            self.fast_current_window = 'RUQ'
            self.fast_start_time = None
            self.fast_fluid_regions.clear()
            self.fast_window_history = {key: deque(maxlen=8) for key in self.fast_windows}
            # Reset auto-navegacao
            self.fast_window_start_time = None
            self.fast_stability_time = 0.0
            self.fast_last_stable_score = 0.0
            self.fast_confirmed_pending = False

        elif old_mode == 'm_mode':
            self.mmode_buffer.clear()
            self.mmode_measurements.clear()

        elif old_mode == 'color':
            self.doppler_prev_frame = None
            self.doppler_flow_history.clear()
            self.doppler_aliasing_detected = False

        elif old_mode == 'power':
            self.power_prev_frame = None

        elif old_mode == 'b_lines':
            self.lung_b_line_history.clear()
            self.lung_a_lines_detected = False
            self.lung_pleura_sliding = True
            self.lung_consolidation = False
            self.lung_pleura_history.clear()
            self.lung_b_line_density_history.clear()

        elif old_mode == 'bladder':
            self.bladder_history.clear()
            self.bladder_d1 = None
            self.bladder_d2 = None
            self.bladder_d3 = None
            self.bladder_volume = None
            self.bladder_view = None
            self.bladder_quality = 0.0
            self.bladder_last_update = 0.0
            self.bladder_view_data = {
                'transverse': {'major': None, 'minor': None, 't': 0.0},
                'sagittal': {'major': None, 'minor': None, 't': 0.0},
            }
            self.bladder_view_history = {
                'transverse': deque(maxlen=8),
                'sagittal': deque(maxlen=8),
            }

    def _load_model(self, mode):
        """Carrega modelo sob demanda (lazy loading)."""
        config_mode = self.MODE_CONFIG.get(mode, {})
        model_type = config_mode.get('model_type')
        model_path = config_mode.get('model_path')

        try:
            if model_type == 'yolo':
                self._load_yolo(mode, model_path)
            elif model_type == 'unet':
                self._load_unet(mode, model_path)
            elif model_type == 'echonet':
                self._load_echonet(mode, model_path)
            elif model_type == 'usfm':
                self._load_usfm(mode, model_path)
            else:
                # Modos sem modelo (CV puro)
                self.models[mode] = None
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {mode}: {e}")
            self.models[mode] = None

    def _load_yolo(self, mode, path):
        """Carrega modelo YOLO."""
        if not os.path.exists(path):
            logger.warning(f"Modelo YOLO nao encontrado: {path}")
            self.models[mode] = None
            return

        from ultralytics import YOLO
        model = YOLO(path)

        # Configurar device
        if str(self.device) != 'cpu':
            model.to(self.device)

        self.models[mode] = model
        logger.info(f"YOLO carregado: {path}")

    def _load_unet(self, mode, path):
        """Carrega modelo U-Net."""
        try:
            import segmentation_models_pytorch as smp

            model = smp.Unet(
                encoder_name="resnet34",  # Mais leve que resnet50
                encoder_weights=None,
                in_channels=1,
                classes=3,
                activation='sigmoid'
            ).to(self.device)

            if path and os.path.exists(path):
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"U-Net carregado: {path}")
            else:
                print(f"U-Net inicializado sem pesos pre-treinados")

            model.eval()
            self.models[mode] = model

        except ImportError:
            print("segmentation_models_pytorch nao instalado")
            self.models[mode] = None

    def _load_echonet(self, mode, path):
        """Carrega modelo EchoNet para ecocardiografia."""
        if not path or not os.path.exists(path):
            print(f"EchoNet nao encontrado: {path}")
            self.models[mode] = None
            return

        try:
            import torchvision.models as models

            # Criar modelo EchoNet
            class EchoNet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    resnet = models.resnet18(weights=None)
                    self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    self.bn1 = resnet.bn1
                    self.relu = resnet.relu
                    self.maxpool = resnet.maxpool
                    self.layer1 = resnet.layer1
                    self.layer2 = resnet.layer2
                    self.layer3 = resnet.layer3
                    self.layer4 = resnet.layer4
                    self.avgpool = resnet.avgpool
                    self.fc = torch.nn.Linear(512, 3)

                def forward(self, x):
                    x = self.conv1(x)
                    x = self.bn1(x)
                    x = self.relu(x)
                    x = self.maxpool(x)
                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.layer4(x)
                    x = self.avgpool(x)
                    x = torch.flatten(x, 1)
                    x = self.fc(x)
                    return x

            model = EchoNet().to(self.device)
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            self.models[mode] = model
            print(f"EchoNet carregado: {path}")

        except Exception as e:
            print(f"Erro ao carregar EchoNet: {e}")
            self.models[mode] = None

    def _load_usfm(self, mode, path):
        """Carrega modelo USFM (Universal Ultrasound Foundation Model)."""
        if not path or not os.path.exists(path):
            print(f"USFM nao encontrado: {path}")
            self.models[mode] = None
            return

        try:
            import segmentation_models_pytorch as smp

            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=1,
                classes=5,  # múltiplas estruturas anatômicas
                activation='sigmoid'
            ).to(self.device)

            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            self.models[mode] = model
            print(f"USFM carregado: {path}")

        except ImportError:
            print("segmentation_models_pytorch nao instalado")
            self.models[mode] = None
        except Exception as e:
            print(f"Erro ao carregar USFM: {e}")
            self.models[mode] = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Processa frame com IA.

        Retorna o frame original em caso de erro para manter estabilidade.
        """
        if not self.active_mode:
            return frame

        # Validar frame de entrada
        if frame is None or frame.size == 0:
            return frame

        # Frame skip inteligente
        try:
            if not self.frame_skipper.should_process(frame):
                # Reusar resultado anterior
                last = self.frame_skipper.get_last_result()
                if last is not None:
                    return last
                return frame
        except Exception:
            pass  # Continuar processando se frame skipper falhar

        start_time = time.time()

        try:
            # Processar baseado no modo
            result = self._process_mode(frame)

            # Validar resultado
            if result is None or result.size == 0:
                result = frame

        except torch.cuda.OutOfMemoryError:
            # GPU sem memoria - tentar liberar e retornar frame original
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result = frame
            self._error_count = getattr(self, '_error_count', 0) + 1
            if self._error_count <= 3:  # Evitar flood de logs
                logger.warning("GPU OOM - retornando frame original")

        except Exception as e:
            # Qualquer outro erro - retornar frame original
            result = frame
            self._error_count = getattr(self, '_error_count', 0) + 1
            if self._error_count <= 3:
                logger.error(f"Erro ({type(e).__name__}): {e}")

        # Salvar resultado para reuso
        try:
            self.frame_skipper.set_last_result(result)
        except Exception:
            pass

        # Metricas
        self.last_inference_time = (time.time() - start_time) * 1000
        self.inference_count += 1

        return result

    def _process_mode(self, frame):
        """Processa frame de acordo com o modo ativo."""
        mode = self.active_mode

        if mode == 'needle':
            return self._process_needle(frame)
        elif mode == 'nerve':
            return self._process_nerve(frame)
        elif mode == 'cardiac':
            return self._process_cardiac(frame)
        elif mode == 'fast':
            return self._process_fast(frame)
        elif mode == 'segment':
            return self._process_segment(frame)
        elif mode == 'm_mode':
            return self._process_mmode(frame)
        elif mode == 'color':
            return self._process_color_doppler(frame)
        elif mode == 'power':
            return self._process_power_doppler(frame)
        elif mode == 'b_lines':
            return self._process_blines(frame)
        elif mode == 'bladder':
            return self._process_bladder(frame)

        return frame

    def _get_roi(self, frame, roi_type):
        """Extrai ROI do frame."""
        h, w = frame.shape[:2]

        if roi_type == 'center_70':
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)
            return frame[margin_y:h-margin_y, margin_x:w-margin_x], (margin_x, margin_y)
        elif roi_type == 'center_80':
            margin_x = int(w * 0.10)
            margin_y = int(h * 0.10)
            return frame[margin_y:h-margin_y, margin_x:w-margin_x], (margin_x, margin_y)

        return frame, (0, 0)

    def _scale_frame(self, frame, scale):
        """Reduz resolucao para processamento."""
        if scale >= 1.0:
            return frame, 1.0

        h, w = frame.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return scaled, scale

    def _auto_gain_and_quality(self, gray, roi=None):
        """Aplica auto gain (apenas para processamento) e calcula score de qualidade."""
        h, w = gray.shape[:2]
        if roi:
            x0, y0, x1, y1 = roi
            x0 = max(0, min(w - 1, x0))
            x1 = max(1, min(w, x1))
            y0 = max(0, min(h - 1, y0))
            y1 = max(1, min(h, y1))
            region = gray[y0:y1, x0:x1]
        else:
            region = gray

        if region.size == 0:
            region = gray

        # Reduzir custo em regioes grandes
        if region.shape[0] > 360 or region.shape[1] > 360:
            region = cv2.resize(region, (min(360, region.shape[1]), min(360, region.shape[0])),
                                interpolation=cv2.INTER_AREA)

        mean = float(np.mean(region))
        p10, p90 = np.percentile(region, [10, 90])
        contrast = float(p90 - p10)
        sharpness = float(cv2.Laplacian(region, cv2.CV_64F).var())
        dark_pct = float(np.mean(region < 20))
        bright_pct = float(np.mean(region > 235))

        contrast_score = np.clip((contrast - 30.0) / 80.0, 0.0, 1.0)
        sharpness_score = np.clip((sharpness - 50.0) / 200.0, 0.0, 1.0)
        exposure_score = 1.0 - np.clip(abs(mean - 110.0) / 110.0, 0.0, 1.0)
        saturation_penalty = np.clip((dark_pct + bright_pct) / 0.2, 0.0, 1.0)

        quality = 100.0 * (
            0.35 * contrast_score
            + 0.35 * sharpness_score
            + 0.2 * exposure_score
            + 0.1 * (1.0 - saturation_penalty)
        )
        quality = float(np.clip(quality, 0.0, 100.0))

        gain = 110.0 / (mean + 1e-6)
        gain = float(np.clip(gain, 0.75, 1.35))
        tuned = np.clip(gray.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        info = {
            'gain': gain,
            'quality': quality,
        }
        return tuned, info

    def _draw_auto_gain_status(self, output, x, y, gain_info, color=(180, 180, 180)):
        if not gain_info:
            return
        gain_pct = int((gain_info['gain'] - 1.0) * 100)
        quality = int(gain_info['quality'])
        text = f"AUTO GAIN {gain_pct:+d}%  SCAN Q {quality}%"
        cv2.putText(output, text, (x, y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.32, color, 1, cv2.LINE_AA)

    # =========================================================================
    # PROCESSADORES POR MODO
    # =========================================================================

    def _process_needle(self, frame):
        """
        Needle Pilot Premium - Sistema avançado de guia de agulha.

        NEEDLE PILOT v3.1 PREMIUM - TODAS AS OTIMIZAÇÕES IMPLEMENTADAS:
        ═══════════════════════════════════════════════════════════════════════
        ✅ FASE 1 - Kalman Filter: suavização e predição de posição
        ✅ FASE 1 - Tracking contínuo quando agulha some temporariamente
        ✅ FASE 1 - Visual premium com indicador de modo (TRACKING/PREDICTING)
        ✅ FASE 2 - RANSAC: detecção robusta de linha (remove outliers)
        ✅ FASE 2 - CLAHE: melhora contraste em regiões escuras
        ✅ FASE 2 - Confiança baseada em inliers do RANSAC
        ✅ FASE 3 - Calibração: conversão pixel ↔ mm real
        ✅ FASE 3 - Escala lateral com marcações calibradas
        ✅ FASE 3 - Profundidade ajustável com atalhos [ e ]
        ✅ FASE 4 - VASST CNN: refinamento de centroide (pronto para weights)
        ✅ FASE 4 - Fallback CV: refinamento por gradiente quando CNN indisponível
        ✅ FASE 5 - Elipse de incerteza (covariância do Kalman)
        ✅ FASE 5 - Info de calibração no painel
        ✅ FASE 5 - Trail com cores por confiança
        ═══════════════════════════════════════════════════════════════════════
        """
        model = self.models.get('needle')
        output = frame.copy()
        h, w = frame.shape[:2]

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 0: CALIBRAÇÃO (FASE 3)
        # ═══════════════════════════════════════════════════════════════════════
        self._calibrate_scale(h)

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 1: DETECÇÃO (YOLO ou CV)
        # ═══════════════════════════════════════════════════════════════════════
        raw_detected = False
        raw_tip_x, raw_tip_y = 0, 0
        raw_angle = 0
        raw_confidence = 0
        raw_line = None  # (x1, y1, x2, y2)

        if model:
            # Detecção com YOLO
            config_mode = self.MODE_CONFIG['needle']
            scaled, scale = self._scale_frame(frame, config_mode['resolution_scale'])
            results = model(scaled, verbose=False, conf=config.AI_CONFIDENCE)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                    raw_confidence = float(box.conf[0])
                    raw_detected = True

                    # Determinar ponta (ponto mais profundo)
                    raw_tip_x, raw_tip_y = (x2, y2) if y2 > y1 else (x1, y1)
                    raw_line = (x1, y1, x2, y2)

                    # Calcular ângulo
                    dx, dy = x2 - x1, y2 - y1
                    if dx != 0 or dy != 0:
                        raw_angle = np.arctan2(dy, dx) * 180 / np.pi
        else:
            # ═══════════════════════════════════════════════════════════════════════
            # Fallback CV com RANSAC (FASE 2) - Detecção robusta
            # ═══════════════════════════════════════════════════════════════════════
            ransac_result = self._detect_needle_enhanced(frame)

            if ransac_result is not None:
                x1, y1, x2, y2, inliers, ransac_conf = ransac_result
                raw_detected = True
                raw_confidence = ransac_conf
                raw_tip_x, raw_tip_y = (x2, y2) if y2 > y1 else (x1, y1)
                raw_line = (x1, y1, x2, y2)
                dx, dy = x2 - x1, y2 - y1
                raw_angle = np.arctan2(dy, dx) * 180 / np.pi
                logger.debug(f"RANSAC: {inliers} inliers, conf={ransac_conf:.2f}")

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 1.5: REFINAMENTO VASST/CV (FASE 4)
        # ═══════════════════════════════════════════════════════════════════════
        if raw_detected and self.use_vasst_refinement:
            refined_x, refined_y, refine_conf = self._refine_needle_position(
                frame, (raw_tip_x, raw_tip_y), raw_line
            )
            # Usar posição refinada se confiança for boa
            if refine_conf > 0.3:
                raw_tip_x, raw_tip_y = refined_x, refined_y
                # Ponderar confiança original com refinamento
                raw_confidence = raw_confidence * 0.7 + refine_conf * 0.3
                logger.debug(f"VASST/CV refined: ({refined_x}, {refined_y}), conf={refine_conf:.2f}")

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 2: KALMAN FILTER (Suavização e Predição)
        # ═══════════════════════════════════════════════════════════════════════
        smooth_x, smooth_y, smooth_angle, kalman_conf = self._update_needle_kalman(
            raw_detected, raw_tip_x, raw_tip_y, raw_angle
        )

        # Determinar se temos tracking válido (detecção OU predição)
        has_tracking = smooth_x is not None

        # Variáveis finais para visualização
        needle_tip = None
        needle_angle = None
        needle_depth = None
        final_confidence = 0

        if has_tracking:
            needle_tip = (int(smooth_x), int(smooth_y))
            needle_angle = smooth_angle
            # FASE 3: Usar calibração real em vez de valor fixo
            needle_depth = self._pixel_to_mm(int(smooth_y))
            final_confidence = kalman_conf * (raw_confidence if raw_detected else 0.7)

            # Guardar no histórico com confiança
            if raw_line:
                x1, y1, x2, y2 = raw_line
            else:
                # Usar predição para reconstruir linha
                x1 = int(smooth_x - 50 * np.cos(np.radians(smooth_angle)))
                y1 = int(smooth_y - 50 * np.sin(np.radians(smooth_angle)))
                x2, y2 = int(smooth_x), int(smooth_y)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self.needle_history.append((cx, cy, x1, y1, x2, y2, kalman_conf))

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 3: VISUALIZAÇÃO PREMIUM
        # ═══════════════════════════════════════════════════════════════════════

        if len(self.needle_history) >= 1:
            # Pegar último ponto do histórico
            last_point = self.needle_history[-1]
            if len(last_point) >= 6:
                if len(last_point) >= 7:
                    _, _, x1, y1, x2, y2, point_conf = last_point
                else:
                    _, _, x1, y1, x2, y2 = last_point
                    point_conf = 1.0

                # ═══════════════════════════════════════════════════════════
                # Cor baseada no modo de tracking
                # ═══════════════════════════════════════════════════════════
                if self.needle_tracking_mode == 'tracking':
                    # Verde sólido - detecção confirmada
                    line_color_outer = (0, 80, 0)
                    line_color_mid = (0, 200, 0)
                    line_color_inner = (0, 255, 0)
                    tip_color = (0, 255, 0)
                elif self.needle_tracking_mode == 'predicting':
                    # Amarelo/Laranja - predição (sem detecção atual)
                    line_color_outer = (0, 80, 100)
                    line_color_mid = (0, 180, 220)
                    line_color_inner = (0, 200, 255)
                    tip_color = (0, 200, 255)
                else:
                    # Cinza - searching
                    line_color_outer = (40, 40, 40)
                    line_color_mid = (80, 80, 80)
                    line_color_inner = (120, 120, 120)
                    tip_color = (100, 100, 100)

                # ═══════════════════════════════════════════════════════════
                # Desenhar linha da agulha com glow
                # ═══════════════════════════════════════════════════════════
                cv2.line(output, (x1, y1), (x2, y2), line_color_outer, 7)
                cv2.line(output, (x1, y1), (x2, y2), line_color_mid, 4)
                cv2.line(output, (x1, y1), (x2, y2), line_color_inner, 2)

                # ═══════════════════════════════════════════════════════════
                # Ponta da agulha destacada
                # ═══════════════════════════════════════════════════════════
                if needle_tip:
                    # ═══════════════════════════════════════════════════════
                    # FASE 5: Elipse de Incerteza (baseada na covariância do Kalman)
                    # ═══════════════════════════════════════════════════════
                    if self.kalman_initialized and self.needle_tracking_mode in ['tracking', 'predicting']:
                        try:
                            # Extrair covariância da posição (2x2 superior esquerdo)
                            P = self.needle_kalman.P
                            cov_xy = np.array([[P[0, 0], P[0, 1]], [P[1, 0], P[1, 1]]])

                            # Calcular autovalores e autovetores para elipse
                            eigenvalues, eigenvectors = np.linalg.eig(cov_xy)

                            # Eixos da elipse (2-sigma = ~95% confiança)
                            scale_factor = 2.0 if self.needle_tracking_mode == 'tracking' else 3.0
                            width = int(scale_factor * np.sqrt(max(0.1, abs(eigenvalues[0]))))
                            height = int(scale_factor * np.sqrt(max(0.1, abs(eigenvalues[1]))))

                            # Ângulo da elipse
                            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi

                            # Limitar tamanho para não ficar absurdo
                            width = min(40, max(5, width))
                            height = min(40, max(5, height))

                            # Cor da elipse baseada no modo
                            if self.needle_tracking_mode == 'tracking':
                                ellipse_color = (0, 150, 0)  # Verde suave
                            else:
                                ellipse_color = (0, 130, 180)  # Laranja suave

                            # Desenhar elipse de incerteza (atrás da ponta)
                            cv2.ellipse(output, needle_tip, (width, height),
                                       angle, 0, 360, ellipse_color, 1, cv2.LINE_AA)
                        except Exception:
                            pass  # Ignorar erros de cálculo matricial

                    # Glow externo
                    cv2.circle(output, needle_tip, 12, line_color_outer, 2)
                    # Círculo médio
                    cv2.circle(output, needle_tip, 8, line_color_mid, 2)
                    # Círculo interno preenchido
                    cv2.circle(output, needle_tip, 5, tip_color, -1)
                    # Centro branco
                    cv2.circle(output, needle_tip, 2, (255, 255, 255), -1)

                # ═══════════════════════════════════════════════════════════
                # Projeção de trajetória
                # ═══════════════════════════════════════════════════════════
                dx, dy = x2 - x1, y2 - y1
                if abs(dx) > 5 or abs(dy) > 5:
                    proj_len = 120
                    norm = np.sqrt(dx**2 + dy**2)
                    if norm > 0:
                        ux, uy = dx/norm, dy/norm

                        # Linha de projeção tracejada com fade
                        for i in range(0, proj_len, 10):
                            t1, t2 = i/proj_len, (i+5)/proj_len
                            p1 = (int(x2 + ux * proj_len * t1), int(y2 + uy * proj_len * t1))
                            p2 = (int(x2 + ux * proj_len * t2), int(y2 + uy * proj_len * t2))
                            alpha = 1 - i/proj_len
                            color = (0, int(180*alpha), int(255*alpha))
                            cv2.line(output, p1, p2, color, 2)

                        # Ponto alvo com crosshair
                        target = (int(x2 + ux * proj_len), int(y2 + uy * proj_len))
                        cv2.circle(output, target, 15, (0, 120, 200), 2)
                        cv2.circle(output, target, 8, (0, 180, 255), -1)
                        cv2.circle(output, target, 4, (255, 255, 255), -1)
                        # Crosshair
                        cv2.line(output, (target[0]-12, target[1]), (target[0]+12, target[1]), (255, 255, 255), 1)
                        cv2.line(output, (target[0], target[1]-12), (target[0], target[1]+12), (255, 255, 255), 1)

            # ═══════════════════════════════════════════════════════════════
            # Trail histórico com cores por confiança
            # ═══════════════════════════════════════════════════════════════
            points = list(self.needle_history)
            for i in range(1, min(len(points), 12)):
                if len(points[-i]) >= 2 and len(points[-i-1]) >= 2:
                    p1 = (points[-i-1][0], points[-i-1][1])
                    p2 = (points[-i][0], points[-i][1])

                    # Confiança do ponto (se disponível)
                    if len(points[-i]) >= 7:
                        pt_conf = points[-i][6]
                    else:
                        pt_conf = 1.0

                    # Cor baseada em confiança e idade
                    age_factor = 1 - (i / 12)
                    if pt_conf > 0.7:
                        color = (0, int(100 + 100*age_factor), int(150 + 100*age_factor))
                    else:
                        color = (0, int(100*age_factor), int(180*age_factor + 75))

                    thickness = max(1, int(2 * age_factor))
                    cv2.line(output, p1, p2, color, thickness)

        # ═══════════════════════════════════════════════════════════════════════
        # ETAPA 4: PAINEL LATERAL PREMIUM
        # ═══════════════════════════════════════════════════════════════════════
        panel_w = 170
        panel_x = w - panel_w - 10
        panel_y = 10
        panel_h = 250  # Aumentado para caber calibração (FASE 5)

        # Background do painel com gradiente
        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (15, 25, 15), -1)
        cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)

        # Borda do painel baseada no modo
        if self.needle_tracking_mode == 'tracking':
            border_color = (0, 220, 0)
        elif self.needle_tracking_mode == 'predicting':
            border_color = (0, 180, 255)
        else:
            border_color = (80, 80, 80)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     border_color, 2)

        # ═══════════════════════════════════════════════════════════════════════
        # Título com ícone de status
        # ═══════════════════════════════════════════════════════════════════════
        cv2.putText(output, "NEEDLE PILOT", (panel_x + 12, panel_y + 22),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Versão/modo pequeno (FASE 5 PREMIUM + VASST)
        cv2.putText(output, "v3.1", (panel_x + 110, panel_y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 200, 150), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════════════════════════════════════
        # Status do Tracking (TRACKING / PREDICTING / SEARCHING)
        # ═══════════════════════════════════════════════════════════════════════
        status_y = panel_y + 48

        if self.needle_tracking_mode == 'tracking':
            # LED verde pulsante
            cv2.circle(output, (panel_x + 15, status_y), 6, (0, 255, 0), -1)
            cv2.circle(output, (panel_x + 15, status_y), 8, (0, 200, 0), 1)
            cv2.putText(output, "TRACKING", (panel_x + 28, status_y + 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        elif self.needle_tracking_mode == 'predicting':
            # LED amarelo/laranja
            cv2.circle(output, (panel_x + 15, status_y), 6, (0, 200, 255), -1)
            cv2.circle(output, (panel_x + 15, status_y), 8, (0, 150, 200), 1)
            cv2.putText(output, "PREDICTING", (panel_x + 28, status_y + 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
            # Mostrar frames restantes
            frames_left = self.max_prediction_frames - self.frames_without_detection
            cv2.putText(output, f"({frames_left})", (panel_x + 125, status_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 150, 200), 1, cv2.LINE_AA)
        else:
            # LED cinza
            cv2.circle(output, (panel_x + 15, status_y), 6, (80, 80, 80), -1)
            cv2.circle(output, (panel_x + 15, status_y), 8, (60, 60, 60), 1)
            cv2.putText(output, "SEARCHING", (panel_x + 28, status_y + 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════════════════════════════════════
        # Barra de Confiança (Detection)
        # ═══════════════════════════════════════════════════════════════════════
        conf_y = panel_y + 75
        cv2.putText(output, "DETECT", (panel_x + 10, conf_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (100, 130, 100), 1, cv2.LINE_AA)
        bar_w = panel_w - 70
        bar_x = panel_x + 55
        cv2.rectangle(output, (bar_x, conf_y - 8), (bar_x + bar_w, conf_y + 3), (30, 40, 30), -1)
        fill_w = int(bar_w * raw_confidence)
        conf_color = (0, 255, 0) if raw_confidence > 0.7 else (0, 200, 200) if raw_confidence > 0.4 else (0, 100, 180)
        cv2.rectangle(output, (bar_x, conf_y - 8), (bar_x + fill_w, conf_y + 3), conf_color, -1)
        cv2.putText(output, f"{raw_confidence*100:.0f}%", (bar_x + bar_w + 5, conf_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.28, conf_color, 1, cv2.LINE_AA)

        # ═══════════════════════════════════════════════════════════════════════
        # Barra de Confiança do Kalman (Track)
        # ═══════════════════════════════════════════════════════════════════════
        kalman_y = panel_y + 98
        cv2.putText(output, "TRACK", (panel_x + 10, kalman_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (100, 130, 100), 1, cv2.LINE_AA)
        cv2.rectangle(output, (bar_x, kalman_y - 8), (bar_x + bar_w, kalman_y + 3), (30, 40, 30), -1)
        kalman_fill = int(bar_w * self.kalman_confidence)
        kalman_color = (0, 255, 200) if self.kalman_confidence > 0.7 else (0, 200, 255) if self.kalman_confidence > 0.3 else (80, 80, 80)
        cv2.rectangle(output, (bar_x, kalman_y - 8), (bar_x + kalman_fill, kalman_y + 3), kalman_color, -1)
        cv2.putText(output, f"{self.kalman_confidence*100:.0f}%", (bar_x + bar_w + 5, kalman_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.28, kalman_color, 1, cv2.LINE_AA)

        # ═══════════════════════════════════════════════════════════════════════
        # Ângulo
        # ═══════════════════════════════════════════════════════════════════════
        angle_y = panel_y + 128
        cv2.putText(output, "ANGLE", (panel_x + 10, angle_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 130, 100), 1, cv2.LINE_AA)
        if needle_angle is not None:
            angle_text = f"{abs(needle_angle):.1f}"
            cv2.putText(output, angle_text, (panel_x + 60, angle_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(output, "deg", (panel_x + 115, angle_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 150, 150), 1, cv2.LINE_AA)
        else:
            cv2.putText(output, "---", (panel_x + 75, angle_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════════════════════════════════════
        # Profundidade
        # ═══════════════════════════════════════════════════════════════════════
        depth_y = panel_y + 158
        cv2.putText(output, "DEPTH", (panel_x + 10, depth_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 130, 100), 1, cv2.LINE_AA)
        if needle_depth is not None:
            depth_text = f"{needle_depth:.1f}"
            cv2.putText(output, depth_text, (panel_x + 60, depth_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(output, "mm", (panel_x + 115, depth_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 150, 150), 1, cv2.LINE_AA)
        else:
            cv2.putText(output, "---", (panel_x + 75, depth_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════════════════════════════════════
        # Dica de correção
        # ═══════════════════════════════════════════════════════════════════════
        tip_y = panel_y + 195
        if has_tracking and needle_angle is not None:
            abs_angle = abs(needle_angle)
            if abs_angle < 25:
                tip_text = "↗ STEEPEN"
                tip_color = (0, 180, 255)
            elif abs_angle > 65:
                tip_text = "↘ FLATTEN"
                tip_color = (0, 180, 255)
            else:
                tip_text = "✓ ON TARGET"
                tip_color = (0, 255, 0)
            cv2.putText(output, tip_text, (panel_x + 12, tip_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.42, tip_color, 1, cv2.LINE_AA)

        # ═══════════════════════════════════════════════════════════════════════
        # FASE 5: Info de Calibração e Atalhos
        # ═══════════════════════════════════════════════════════════════════════
        cal_y = panel_y + 225
        total_depth = self.calibration['depth_mm']
        mm_px = self.calibration['mm_per_pixel']

        # Linha de calibração
        cv2.putText(output, f"Scale: {mm_px:.3f}mm/px", (panel_x + 10, cal_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.22, (80, 100, 80), 1, cv2.LINE_AA)

        # Atalhos
        cal_y2 = panel_y + 242
        cv2.putText(output, "[", (panel_x + 10, cal_y2),
                   cv2.FONT_HERSHEY_DUPLEX, 0.25, (100, 130, 100), 1, cv2.LINE_AA)
        cv2.putText(output, f"-10mm", (panel_x + 20, cal_y2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.2, (80, 100, 80), 1, cv2.LINE_AA)
        cv2.putText(output, "]", (panel_x + 70, cal_y2),
                   cv2.FONT_HERSHEY_DUPLEX, 0.25, (100, 130, 100), 1, cv2.LINE_AA)
        cv2.putText(output, f"+10mm", (panel_x + 80, cal_y2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.2, (80, 100, 80), 1, cv2.LINE_AA)
        cv2.putText(output, f"[{int(total_depth)}mm]", (panel_x + 125, cal_y2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.22, (0, 180, 0), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════════════════════════════════════
        # ESCALA DE PROFUNDIDADE (lateral esquerda)
        # ═══════════════════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════════════════
        # ESCALA LATERAL CALIBRADA (FASE 3)
        # ═══════════════════════════════════════════════════════════════════════
        scale_x = 30
        mm_per_pixel = self.calibration['mm_per_pixel']

        # Linha principal da escala
        cv2.line(output, (scale_x, 10), (scale_x, h - 10), (0, 120, 0), 1)

        # Calcular step baseado na profundidade total para ter ~8-10 marcações
        total_depth = self.calibration['depth_mm']
        step_mm = 10 if total_depth <= 100 else 20  # 10mm steps para < 100mm, 20mm para maiores

        step_pixels = int(step_mm / mm_per_pixel) if mm_per_pixel > 0 else 50
        step_pixels = max(20, step_pixels)  # Mínimo 20 pixels entre marcações

        for i in range(0, h, step_pixels):
            depth_mm = self._pixel_to_mm(i)
            # Marcação maior a cada 2x step
            if int(depth_mm) % (step_mm * 2) == 0:
                cv2.line(output, (scale_x - 8, i), (scale_x + 3, i), (0, 180, 0), 1)
                cv2.putText(output, f"{int(depth_mm)}", (scale_x + 6, i + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 180, 0), 1, cv2.LINE_AA)
            else:
                cv2.line(output, (scale_x - 4, i), (scale_x + 2, i), (0, 100, 0), 1)

        # Label "mm" no topo + indicador de calibração
        cv2.putText(output, "mm", (scale_x - 8, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 150, 0), 1, cv2.LINE_AA)

        # Mostrar profundidade total configurada no canto
        cv2.putText(output, f"[{int(total_depth)}mm]", (scale_x - 12, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 120, 0), 1, cv2.LINE_AA)

        return output

    def _cv_needle_enhance(self, frame):
        """Realce de agulha por visao computacional."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteccao de linhas (agulhas sao lineares)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

        output = frame.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Filtrar linhas muito horizontais ou verticais
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if 15 < angle < 75 or 105 < angle < 165:
                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return output

    def _process_nerve(self, frame):
        """
        NERVE TRACK v2.0 PREMIUM
        ========================
        Sistema avançado de detecção e tracking de nervos para ultrassom.

        Características:
        - U-Net++ com EfficientNet-B4 + CBAM + ConvLSTM
        - Kalman Filter para tracking temporal (anti-flicker)
        - Classificação automática Nervo/Artéria/Veia
        - Identificação contextual por tipo de bloqueio (28 tipos)
        - Medição automática de CSA (Cross-Sectional Area)
        - Visual premium com alertas de zonas de perigo
        """
        output = frame.copy()
        h, w = frame.shape[:2]

        # ═══════════════════════════════════════════════════════════════════════
        # USAR NERVE TRACK v2.0 SE DISPONÍVEL
        # ═══════════════════════════════════════════════════════════════════════
        if self.nerve_track_available and self.nerve_track_system is not None:
            try:
                # Processar frame com sistema completo
                result = self.nerve_track_system.process_frame(frame)

                # Se temos visualização renderizada, usar ela
                if result.get('visualization') is not None:
                    return result['visualization']

                # Caso contrário, renderizar manualmente as estruturas identificadas
                identified = result.get('identified', [])
                tracks = result.get('tracks', [])

                # Renderizar estruturas identificadas
                for struct in identified:
                    if struct.contour is not None and len(struct.contour) > 0:
                        # Cor baseada no tipo
                        if struct.is_target:
                            color = (0, 255, 0)  # Verde para alvo
                        elif struct.is_danger_zone:
                            color = (0, 0, 255)  # Vermelho para perigo
                        elif struct.structure_type.value == 'nerve':
                            color = (0, 255, 255)  # Amarelo para nervo
                        elif struct.structure_type.value == 'artery':
                            color = (0, 0, 255)  # Vermelho para artéria
                        elif struct.structure_type.value == 'vein':
                            color = (255, 0, 0)  # Azul para veia
                        else:
                            color = (128, 128, 128)  # Cinza para outros

                        # Overlay
                        overlay = output.copy()
                        cv2.drawContours(overlay, [struct.contour], -1, color, -1)
                        cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)

                        # Contorno
                        thickness = 3 if struct.is_target else 2
                        cv2.drawContours(output, [struct.contour], -1, color, thickness)

                        # Label
                        cx, cy = int(struct.centroid[0]), int(struct.centroid[1])
                        label = f"{struct.abbreviation}"
                        if struct.area_mm2:
                            label += f" {struct.area_mm2:.1f}mm²"

                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
                        cv2.rectangle(output, (cx - tw//2 - 3, cy - th - 8),
                                     (cx + tw//2 + 3, cy - 2), (0, 0, 0), -1)
                        cv2.putText(output, label, (cx - tw//2, cy - 5),
                                   cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1, cv2.LINE_AA)

                # Painel de informações
                output = self._draw_nerve_track_panel(output, identified, w, h)

            except Exception as e:
                logger.warning(f"Erro no NERVE TRACK v2.0: {e}")
                # Fallback para detecção básica
                output = self._process_nerve_fallback(frame)
        else:
            # Fallback se NERVE TRACK não disponível
            output = self._process_nerve_fallback(frame)

        return output

    def _process_nerve_fallback(self, frame):
        """
        Fallback PREMIUM: Deteccao avancada de nervos por CV.
        Usa contornos adaptativos, elipses, echogenicidade e CSA.
        """
        output = frame.copy()
        h, w = frame.shape[:2]
        structures = []

        # ═══════════════════════════════════════════════════════════════════════
        # ATLAS EDUCACIONAL - Usar quando modelo não está treinado
        # ═══════════════════════════════════════════════════════════════════════
        if self.educational_atlas is not None:
            if self.nerve_block_id:
                self.educational_atlas.set_block(self.nerve_block_id)
            self.educational_atlas.update_animation()
            output = self.educational_atlas.render_atlas(output, self.atlas_mode)
            return output

        # ═══════════════════════════════════════════════════════════════════════
        # SCAN Q Gating - Verificar qualidade antes de processar
        # ═══════════════════════════════════════════════════════════════════════
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gain_roi = (int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))
        tuned, gain_info = self._auto_gain_and_quality(gray, roi=gain_roi)
        scan_quality = gain_info.get('quality', 0.5)

        # Se qualidade baixa, mostrar aviso
        if scan_quality < 0.35:
            cv2.putText(output, "LOW QUALITY - ADJUST PROBE", (w//2 - 120, 30),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════════════════════════════════════
        # DETECCAO AVANCADA POR CONTORNOS + ELIPSES
        # ═══════════════════════════════════════════════════════════════════════
        # CLAHE para melhor contraste
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(tuned)

        # Bilateral filter preserva bordas
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Threshold adaptativo para encontrar estruturas hipoecogenicas
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 8
        )

        # Morfologia para limpar
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mm_per_pixel = self.calibration.get('mm_per_pixel', 0.1)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200 or area > 50000:  # Filtrar por area
                continue

            # Calcular propriedades
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Bounding rect e centro
            x, y, bw, bh = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + bw // 2, y + bh // 2

            # Aspect ratio
            aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)

            # Calcular CSA (Cross-Sectional Area) em mm²
            csa_mm2 = area * (mm_per_pixel ** 2)

            # Echogenicidade da região
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_echo = cv2.mean(gray, mask=mask)[0]

            # ═══════════════════════════════════════════════════════════════
            # CLASSIFICACAO POR FEATURES
            # ═══════════════════════════════════════════════════════════════
            # Nervos: hipoecogenicos, forma ovalada, CSA tipico 5-50mm²
            # Arterias: anecogenicas (muito escuro), circulares, pulsateis
            # Veias: anecogenicas, compressiveis, forma irregular

            structure_type = "UNKNOWN"
            color = (128, 128, 128)
            confidence = 0.5

            # Nervo: echogenicidade media-baixa, forma ovalada
            if 40 < mean_echo < 120 and circularity > 0.5 and 1.0 < aspect_ratio < 3.0:
                if 3 < csa_mm2 < 80:  # CSA tipico de nervos perifericos
                    structure_type = "NERVE"
                    color = (0, 255, 255)  # Amarelo
                    confidence = min(0.95, 0.6 + circularity * 0.3)

            # Arteria: muito escuro (anecogenico), circular
            elif mean_echo < 50 and circularity > 0.7:
                structure_type = "ARTERY"
                color = (0, 0, 255)  # Vermelho
                confidence = min(0.9, 0.5 + circularity * 0.4)

            # Veia: escuro, menos circular que arteria
            elif mean_echo < 60 and 0.4 < circularity < 0.8:
                structure_type = "VEIN"
                color = (255, 100, 100)  # Azul
                confidence = 0.6

            # Fascia: hiperecogenico (brilhante), linear
            elif mean_echo > 140 and aspect_ratio > 2.5:
                structure_type = "FASCIA"
                color = (200, 200, 200)  # Cinza claro
                confidence = 0.5

            # Pular estruturas desconhecidas
            if structure_type == "UNKNOWN":
                continue

            # ═══════════════════════════════════════════════════════════════
            # DESENHAR ESTRUTURA
            # ═══════════════════════════════════════════════════════════════
            # Overlay com transparencia
            overlay = output.copy()
            cv2.drawContours(overlay, [contour], -1, color, -1)
            alpha = 0.25 if structure_type != "NERVE" else 0.35
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

            # Contorno
            thickness = 2 if structure_type == "NERVE" else 1
            cv2.drawContours(output, [contour], -1, color, thickness)

            # Elipse se possivel (melhor visualizacao)
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(output, ellipse, color, 1)
                except:
                    pass

            # Label com CSA
            label = f"{structure_type}"
            if structure_type == "NERVE":
                label += f" {csa_mm2:.1f}mm²"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.35, 1)
            label_y = cy - 10 if cy > 30 else cy + bh + 15
            cv2.rectangle(output, (cx - tw//2 - 2, label_y - th - 2),
                         (cx + tw//2 + 2, label_y + 2), (0, 0, 0), -1)
            cv2.putText(output, label, (cx - tw//2, label_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.35, color, 1, cv2.LINE_AA)

            structures.append({
                'type': structure_type,
                'center': (cx, cy),
                'contour': contour,
                'csa_mm2': csa_mm2,
                'confidence': confidence,
                'echogenicity': mean_echo,
                'circularity': circularity,
                'color': color
            })

        # ═══════════════════════════════════════════════════════════════════════
        # PAINEL DE INFORMACOES PREMIUM (usando UI Utils)
        # ═══════════════════════════════════════════════════════════════════════
        nerve_count = sum(1 for s in structures if s['type'] == 'NERVE')
        vessel_count = sum(1 for s in structures if s['type'] in ('ARTERY', 'VEIN'))

        panel_w = 160
        panel_x = w - panel_w - 10
        panel_y = 10
        struct_height = min(len(structures), 4) * 20
        panel_h = 95 + struct_height

        # Painel com header
        content_y = draw_panel(output, panel_x, panel_y, panel_w, panel_h,
                               title="NERVE TRACK", version="CV")

        # Bloqueio selecionado
        if self.nerve_block_id:
            block_name = self.nerve_block_id.replace('_', ' ').title()[:18]
            cv2.putText(output, block_name, (panel_x + 8, content_y),
                       FONT, FONT_SCALE_SMALL, Theme.TEXT_SECONDARY, 1, cv2.LINE_AA)
            content_y += 15

        # Contagem com indicadores
        draw_status_indicator(output, panel_x + 5, content_y + 5, f"Nervos: {nerve_count}",
                             active=nerve_count > 0, color=Theme.NERVE)
        content_y += 18

        draw_status_indicator(output, panel_x + 5, content_y + 5, f"Vasos: {vessel_count}",
                             active=vessel_count > 0, color=Theme.VEIN)
        content_y += 18

        # SCAN Q
        draw_scan_quality(output, panel_x + 8, content_y + 5, scan_quality)
        content_y += 18

        # Lista de estruturas
        if structures:
            draw_structure_list(output, panel_x + 5, content_y, structures, max_items=4)

        return output

    def _draw_nerve_track_panel(self, output, identified, w, h):
        """Desenha painel de informações do NERVE TRACK v2.0."""
        # Painel premium
        panel_w = 180
        panel_x = w - panel_w - 10
        panel_y = 10

        # Calcular altura baseado no conteúdo
        base_height = 90
        struct_height = min(len(identified), 5) * 28
        panel_h = base_height + struct_height

        # Background do painel
        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 25, 30), -1)
        cv2.addWeighted(overlay, 0.9, output, 0.1, 0, output)

        # Borda com gradiente
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 200, 200), 1)
        cv2.rectangle(output, (panel_x + 1, panel_y + 1), (panel_x + panel_w - 1, panel_y + 27), (0, 100, 100), -1)

        # Título
        cv2.putText(output, "NERVE TRACK v2.0", (panel_x + 8, panel_y + 20),
                   cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Nome do bloqueio se selecionado
        if self.nerve_block_id and self.nerve_track_system:
            block_info = self.nerve_track_system.get_block_info()
            if block_info:
                block_name = block_info.name[:20]  # Truncar se muito longo
                cv2.putText(output, block_name, (panel_x + 8, panel_y + 42),
                           cv2.FONT_HERSHEY_DUPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            cv2.putText(output, "Selecione bloqueio...", (panel_x + 8, panel_y + 42),
                       cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 100, 100), 1, cv2.LINE_AA)

        # Linha separadora
        cv2.line(output, (panel_x + 5, panel_y + 50), (panel_x + panel_w - 5, panel_y + 50), (60, 60, 60), 1)

        # Lista de estruturas identificadas
        y_off = panel_y + 65
        for i, struct in enumerate(identified[:5]):
            # Cor baseada no tipo
            if struct.is_target:
                color = (0, 255, 0)
                prefix = "[ALVO]"
            elif struct.is_danger_zone:
                color = (0, 0, 255)
                prefix = "[!]"
            else:
                color = (0, 255, 255)
                prefix = ""

            # Indicador de cor
            cv2.circle(output, (panel_x + 12, y_off), 4, color, -1)

            # Nome
            name = struct.abbreviation or struct.name[:8]
            cv2.putText(output, f"{prefix}{name}", (panel_x + 22, y_off + 4),
                       cv2.FONT_HERSHEY_DUPLEX, 0.32, color, 1, cv2.LINE_AA)

            # CSA/Confiança
            if struct.area_mm2:
                info = f"{struct.area_mm2:.1f}mm²"
            else:
                info = f"{struct.confidence:.0%}"
            cv2.putText(output, info, (panel_x + 110, y_off + 4),
                       cv2.FONT_HERSHEY_DUPLEX, 0.28, (150, 150, 180), 1, cv2.LINE_AA)

            y_off += 28

        # Contagem no rodapé
        nerve_count = sum(1 for s in identified if 'nerve' in str(s.structure_type.value).lower())
        vessel_count = len(identified) - nerve_count
        cv2.putText(output, f"Nervos:{nerve_count} Vasos:{vessel_count}",
                   (panel_x + 8, panel_y + panel_h - 8),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (100, 100, 120), 1, cv2.LINE_AA)

        return output

    def set_nerve_block(self, block_id: str) -> bool:
        """
        Define o tipo de bloqueio nervoso para identificação contextual.

        Args:
            block_id: ID do bloqueio (ex: 'femoral', 'interscalene', 'tap')

        Returns:
            True se bloqueio foi definido com sucesso
        """
        self.nerve_block_id = block_id

        # Atualizar atlas educacional (funciona sempre)
        if self.educational_atlas is not None:
            try:
                self.educational_atlas.set_block(block_id)
                logger.info(f"Atlas atualizado para bloqueio: {block_id}")
            except Exception as e:
                logger.warning(f"Erro ao atualizar atlas: {e}")

        # Atualizar NERVE TRACK system
        if self.nerve_track_available and self.nerve_track_system:
            try:
                self.nerve_track_system.set_block(block_id)
                logger.info(f"NERVE TRACK atualizado para bloqueio: {block_id}")
                return True
            except Exception as e:
                logger.error(f"Erro ao definir bloqueio no NERVE TRACK: {e}")
                return False

        # Retorna True se pelo menos o atlas foi atualizado
        return self.educational_atlas is not None

    def get_available_nerve_blocks(self) -> dict:
        """
        Retorna lista de bloqueios nervosos disponíveis.

        Returns:
            Dict de {block_id: nome_do_bloqueio}
        """
        return self.available_nerve_blocks

    def get_nerve_block_info(self, block_id: str = None) -> dict:
        """
        Retorna informações detalhadas de um bloqueio.

        Args:
            block_id: ID do bloqueio (usa atual se None)

        Returns:
            Dict com informações do bloqueio
        """
        if not NERVE_TRACK_AVAILABLE:
            return {}

        try:
            bid = block_id or self.nerve_block_id
            if bid:
                block = get_block_config(bid)
                if block:
                    return {
                        'id': block.id,
                        'name': block.name,
                        'name_en': block.name_en,
                        'region': block.region.value,
                        'targets': [t.name for t in block.targets],
                        'danger_zones': [d.name for d in block.danger_zones],
                        'probe_type': block.probe_type,
                        'depth_cm': block.depth_cm,
                        'skill_level': block.skill_level,
                    }
        except Exception as e:
            logger.error(f"Erro ao obter info do bloqueio: {e}")

        return {}

    def _cv_nerve_enhance(self, frame):
        """Realce de estruturas nervosas por CV."""
        output = frame.copy()
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Equalizar histograma para melhor contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Detectar circulos (nervos/vasos em corte transversal)
        circles = cv2.HoughCircles(
            enhanced, cv2.HOUGH_GRADIENT, 1, 30,
            param1=50, param2=30, minRadius=10, maxRadius=80
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i, (x, y, r) in enumerate(circles[0, :5]):  # Max 5 circulos
                # Cor baseada no tamanho (nervo=maior=amarelo, vaso=menor=vermelho)
                if r > 40:
                    color = (0, 255, 255)  # Amarelo - provavel nervo
                    label = "Nervo?"
                else:
                    color = (0, 0, 255)  # Vermelho - provavel vaso
                    label = "Vaso?"
                cv2.circle(output, (x, y), r, color, 2)
                cv2.putText(output, label, (x - 20, y - r - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return output

    def _process_cardiac(self, frame):
        """Cardiac AI - Sistema avancado de analise cardiaca estilo EchoNet/Clarius."""
        output = frame.copy()
        h, w = frame.shape[:2]

        model = self.models.get('cardiac')

        # ═══════════════════════════════════════
        # PROCESSAMENTO DE IA / CV
        # ═══════════════════════════════════════

        # Detectar contorno do ventriculo esquerdo
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Regiao de interesse central (onde tipicamente esta o coracao)
        roi_margin_x = int(w * 0.15)
        roi_margin_y = int(h * 0.1)
        roi_coords = (roi_margin_x, roi_margin_y, w - roi_margin_x, h - roi_margin_y)
        tuned, gain_info = self._auto_gain_and_quality(gray, roi=roi_coords)

        # ═══════════════════════════════════════
        # GATING POR SCAN QUALITY (reduz falsos positivos)
        # ═══════════════════════════════════════
        scan_quality = gain_info.get('quality', 50.0) if gain_info else 50.0
        quality_threshold = 35.0
        low_quality_mode = scan_quality < quality_threshold

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(tuned)
        roi = enhanced[roi_margin_y:h-roi_margin_y, roi_margin_x:w-roi_margin_x]

        lv_contour = None
        lv_area = 0
        lv_center = None

        if model:
            try:
                resized = cv2.resize(gray, (224, 224))
                tensor = torch.from_numpy(resized).float() / 255.0
                tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = model(tensor).cpu().numpy()[0]

                # outputs = [EF, ESV, EDV]
                ef = np.clip(outputs[0] * 100, 20, 80)
                self.cardiac_ef = ef
                self.cardiac_esv = outputs[1] * 150  # ml
                self.cardiac_edv = outputs[2] * 150  # ml

            except Exception as e:
                print(f"Erro EchoNet: {e}")

        # Fallback CV: detectar estrutura escura grande (cavidade)
        roi_mean = float(np.mean(roi)) if roi.size else 0.0
        thresh_val = int(np.clip(np.percentile(roi, 20), 20, 80)) if roi.size else 50
        _, thresh = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Encontrar contorno mais provavel (grande, eliptico e centrado)
            best_contour = None
            best_score = 0.0
            roi_area = max(1.0, roi.shape[0] * roi.shape[1])
            roi_cx = roi.shape[1] / 2.0
            roi_cy = roi.shape[0] / 2.0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < roi_area * 0.02 or area > roi_area * 0.6:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0.0

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

                center_dist = ((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2) ** 0.5
                center_score = 1.0 - min(center_dist / (0.6 * max(roi.shape)), 1.0)

                mask = np.zeros_like(roi, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_inside = cv2.mean(roi, mask=mask)[0]
                contrast = max(0.0, roi_mean - mean_inside)

                area_score = min(area / (0.25 * roi_area), 1.0)
                circ_score = min(max((circularity - 0.25) / 0.55, 0.0), 1.0)
                solidity_score = min(max((solidity - 0.65) / 0.35, 0.0), 1.0)
                contrast_score = min(contrast / 50.0, 1.0)

                score = (
                    0.35 * contrast_score
                    + 0.25 * area_score
                    + 0.2 * solidity_score
                    + 0.1 * circ_score
                    + 0.1 * center_score
                )

                if score > best_score:
                    best_score = score
                    best_contour = cnt

            if best_contour is not None:
                # Ajustar coordenadas para frame completo
                best_contour = best_contour + np.array([roi_margin_x, roi_margin_y])
                lv_contour = best_contour
                lv_area = cv2.contourArea(best_contour)

                M = cv2.moments(best_contour)
                if M["m00"] > 0:
                    lv_center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

                # ═══ VIEW DETECTION (baseado no aspect ratio do LV) ═══
                rect = cv2.minAreaRect(best_contour)
                (cx, cy), (lv_w, lv_h), angle = rect
                # Garantir que lv_w é a largura (maior dimensao horizontal)
                if lv_h > lv_w:
                    lv_w, lv_h = lv_h, lv_w

                self.cardiac_lv_aspect_ratio = lv_w / max(lv_h, 1)

                # Classificar view baseado no aspect ratio e posição
                # A4C: LV tipicamente oval horizontal (aspect > 1.3)
                # PLAX: LV mais alongado verticalmente (aspect < 0.8)
                # PSAX: LV circular (aspect ~1.0)
                # A2C: Similar a A4C mas LV mais centrado
                if self.cardiac_lv_aspect_ratio > 1.4:
                    new_view = 'A4C'
                    conf = min((self.cardiac_lv_aspect_ratio - 1.4) / 0.6, 1.0)
                elif self.cardiac_lv_aspect_ratio < 0.75:
                    new_view = 'PLAX'
                    conf = min((0.75 - self.cardiac_lv_aspect_ratio) / 0.25, 1.0)
                elif 0.85 <= self.cardiac_lv_aspect_ratio <= 1.15:
                    new_view = 'PSAX'
                    conf = 1.0 - abs(self.cardiac_lv_aspect_ratio - 1.0) / 0.15
                else:
                    new_view = 'A2C'  # Casos intermediários
                    conf = 0.5

                # Suavização temporal da view
                if new_view == self.cardiac_view:
                    self.cardiac_view_confidence = 0.9 * self.cardiac_view_confidence + 0.1 * conf
                else:
                    if conf > self.cardiac_view_confidence + 0.2:
                        self.cardiac_view = new_view
                        self.cardiac_view_confidence = conf

        # Tracking temporal para detectar ciclo cardiaco
        # Em baixa qualidade, nao atualizar historico
        if not low_quality_mode:
            if lv_area > 0:
                self.cardiac_history.append(lv_area)
            elif self.cardiac_history:
                self.cardiac_history.append(self.cardiac_history[-1])

        # Calcular metricas a partir do historico
        if len(self.cardiac_history) >= 30:
            areas = np.array(self.cardiac_history, dtype=np.float32)
            max_area = float(np.percentile(areas, 90))
            min_area = float(np.percentile(areas, 10))
            max_area = max(max_area, 1.0)
            min_area = max(min_area, 1.0)

            # EF baseado na variacao de area (aproximacao 2D)
            # Simpson simplificado: EF ≈ (EDV - ESV) / EDV
            ef_calc = ((max_area - min_area) / max_area) * 100 if max_area > 0 else 55
            ef_calc = np.clip(ef_calc * 1.2, 25, 80)
            if self.cardiac_ef is None:
                self.cardiac_ef = ef_calc
            else:
                self.cardiac_ef = 0.85 * self.cardiac_ef + 0.15 * ef_calc

            # Volumes estimados (px^2 para ml usando formula elipsoidal)
            px_to_mm = 0.3
            self.cardiac_edv = (max_area * px_to_mm**2) * 0.85 / 1000 * 150  # Escala para ml
            self.cardiac_esv = (min_area * px_to_mm**2) * 0.85 / 1000 * 150

            # Detectar picos para HR
            if len(areas) >= 30:
                # Encontrar picos (diastole = area maxima)
                current_frame = len(self.cardiac_history)
                if lv_area > np.percentile(areas, 90):
                    if current_frame - self.cardiac_last_peak > 15:  # Min 0.5s entre batidas
                        self.cardiac_cycle_frames.append(current_frame)
                        self.cardiac_last_peak = current_frame

                # Calcular HR a partir dos intervalos
                if len(self.cardiac_cycle_frames) >= 3:
                    intervals = np.diff(self.cardiac_cycle_frames[-5:])
                    if len(intervals) > 0:
                        avg_interval = np.mean(intervals)
                        fps = getattr(config, 'FPS', 30)
                        self.cardiac_hr = int(60 * fps / avg_interval)
                        self.cardiac_hr = np.clip(self.cardiac_hr, 40, 150)

            # GLS estimado (tipicamente -15 a -25 em normais)
            if self.cardiac_gls is None:
                strain = -((max_area - min_area) / max_area) * 30
                self.cardiac_gls = np.clip(strain, -25, -8)

            # Fase atual do ciclo
            recent_areas = areas[-10:]
            if len(recent_areas) > 5:
                if np.mean(recent_areas[-3:]) > np.mean(recent_areas[:3]):
                    self.cardiac_phase = 'diastole'
                else:
                    self.cardiac_phase = 'systole'

        # ═══════════════════════════════════════
        # VISUALIZACAO DO VENTRICULO
        # ═══════════════════════════════════════

        if lv_contour is not None:
            # Glow effect no contorno
            cv2.drawContours(output, [lv_contour], -1, (80, 40, 40), 5)
            cv2.drawContours(output, [lv_contour], -1, (150, 80, 80), 3)

            # Cor baseada na fase
            phase_color = (100, 150, 255) if self.cardiac_phase == 'diastole' else (100, 255, 150)
            cv2.drawContours(output, [lv_contour], -1, phase_color, 2)

            # Preenchimento semi-transparente
            overlay = output.copy()
            cv2.drawContours(overlay, [lv_contour], -1, phase_color, -1)
            cv2.addWeighted(overlay, 0.15, output, 0.85, 0, output)

            # Centro com marcador
            if lv_center:
                cv2.circle(output, lv_center, 4, (255, 255, 255), -1)
                cv2.circle(output, lv_center, 8, phase_color, 1)

                # Label LV
                cv2.putText(output, "LV", (lv_center[0] + 12, lv_center[1] + 5),
                           cv2.FONT_HERSHEY_DUPLEX, 0.4, phase_color, 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # PAINEL PRINCIPAL (estilo Clarius)
        # ═══════════════════════════════════════

        panel_w = 170
        panel_x = w - panel_w - 10
        panel_y = 10
        panel_h = 250

        # Painel com header (tema vermelho para cardiac)
        cardiac_border = (180, 100, 100)
        content_y = draw_panel(output, panel_x, panel_y, panel_w, panel_h,
                               title="CARDIAC AI",
                               border_color=cardiac_border,
                               title_color=(255, 150, 150))

        # Indicador de fase (canto superior direito)
        phase_text = "DIASTOLE" if self.cardiac_phase == 'diastole' else "SYSTOLE"
        phase_col = (100, 150, 255) if self.cardiac_phase == 'diastole' else (100, 255, 150)
        cv2.circle(output, (panel_x + panel_w - 15, panel_y + 13), 5, phase_col, -1)

        # ═══ EF ═══
        ef_val = self.cardiac_ef if self.cardiac_ef else 55

        # Cor baseada no EF
        if ef_val >= 55:
            ef_color = Theme.STATUS_OK
            ef_status = "NORMAL"
        elif ef_val >= 40:
            ef_color = Theme.STATUS_WARNING
            ef_status = "LEVE"
        elif ef_val >= 30:
            ef_color = (0, 165, 255)
            ef_status = "MODERADO"
        else:
            ef_color = Theme.STATUS_ERROR
            ef_status = "GRAVE"

        draw_labeled_value(output, panel_x + 10, content_y, "EF", f"{int(ef_val)}%", "",
                          label_color=Theme.TEXT_SECONDARY, value_color=ef_color, label_width=35)
        cv2.putText(output, ef_status, (panel_x + 100, content_y),
                   FONT, FONT_SCALE_SMALL, ef_color, 1, cv2.LINE_AA)
        content_y += 12

        # Barra EF
        draw_progress_bar(output, panel_x + 10, content_y, panel_w - 20, 6,
                         ef_val, max_value=80, color=ef_color, show_markers=[30, 55])
        content_y += 18

        # ═══ VOLUMES ═══
        cv2.putText(output, "VOLUMES", (panel_x + 10, content_y),
                   FONT, FONT_SCALE_SMALL, Theme.TEXT_MUTED, 1, cv2.LINE_AA)
        content_y += 18

        edv_val = self.cardiac_edv if self.cardiac_edv else 120
        esv_val = self.cardiac_esv if self.cardiac_esv else 50

        draw_labeled_value(output, panel_x + 10, content_y, "EDV", f"{int(edv_val)}", "ml",
                          label_color=Theme.TEXT_SECONDARY, value_color=Theme.TEXT_PRIMARY, label_width=35)
        draw_labeled_value(output, panel_x + 85, content_y, "ESV", f"{int(esv_val)}", "ml",
                          label_color=Theme.TEXT_SECONDARY, value_color=Theme.TEXT_PRIMARY, label_width=35)
        content_y += 22

        # ═══ GLS ═══
        gls_val = self.cardiac_gls if self.cardiac_gls else -18
        gls_color = Theme.STATUS_OK if gls_val <= -16 else Theme.STATUS_WARNING if gls_val <= -12 else Theme.STATUS_ERROR
        draw_labeled_value(output, panel_x + 10, content_y, "GLS", f"{gls_val:.1f}%", "",
                          label_color=Theme.TEXT_SECONDARY, value_color=gls_color, label_width=35)
        content_y += 22

        # ═══ HR ═══
        hr_val = self.cardiac_hr if self.cardiac_hr else 72
        hr_color = Theme.STATUS_OK if 60 <= hr_val <= 100 else Theme.STATUS_WARNING
        draw_labeled_value(output, panel_x + 10, content_y, "HR", f"{hr_val}", "bpm",
                          label_color=Theme.TEXT_SECONDARY, value_color=hr_color, label_width=35)
        content_y += 18

        # ═══ MINI ECG WAVEFORM ═══
        ecg_y = panel_y + 190
        ecg_h = 45
        cv2.rectangle(output, (panel_x + 10, ecg_y), (panel_x + panel_w - 10, ecg_y + ecg_h),
                     (30, 30, 40), -1)
        cv2.rectangle(output, (panel_x + 10, ecg_y), (panel_x + panel_w - 10, ecg_y + ecg_h),
                     (60, 60, 80), 1)

        # Desenhar "waveform" baseado no historico de area
        if len(self.cardiac_history) >= 10:
            hist = list(self.cardiac_history)[-50:]
            if len(hist) > 1 and max(hist) > min(hist):
                norm_hist = [(a - min(hist)) / (max(hist) - min(hist)) for a in hist]
                ecg_w = panel_w - 30

                points = []
                for i, val in enumerate(norm_hist):
                    x = panel_x + 15 + int((i / len(norm_hist)) * ecg_w)
                    y = ecg_y + ecg_h - 5 - int(val * (ecg_h - 10))
                    points.append((x, y))

                for i in range(1, len(points)):
                    cv2.line(output, points[i-1], points[i], (100, 200, 255), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # HEADER SUPERIOR ESQUERDO
        # ═══════════════════════════════════════

        cv2.rectangle(output, (10, 10), (160, 45), (40, 30, 30), -1)
        cv2.rectangle(output, (10, 10), (160, 45), (180, 100, 100), 1)
        cv2.putText(output, "ECHO MODE", (20, 33),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(output, phase_text, (95, 33),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, phase_col, 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # VIEW INDICATOR (A4C, PLAX, PSAX, A2C)
        # ═══════════════════════════════════════
        view_y = 52
        view_w = 100
        view_h = 28

        overlay = output.copy()
        cv2.rectangle(overlay, (10, view_y), (10 + view_w, view_y + view_h),
                     (40, 35, 35), -1)
        cv2.addWeighted(overlay, 0.9, output, 0.1, 0, output)
        cv2.rectangle(output, (10, view_y), (10 + view_w, view_y + view_h),
                     (120, 100, 100), 1)

        # View name e descrição
        view_names = {
            'A4C': ('A4C', 'Apical 4-Ch'),
            'A2C': ('A2C', 'Apical 2-Ch'),
            'PLAX': ('PLAX', 'ParaStern'),
            'PSAX': ('PSAX', 'Short Axis'),
        }
        view_abbr, view_desc = view_names.get(self.cardiac_view, ('---', '---'))

        # Cor baseada na confiança
        conf_pct = int(self.cardiac_view_confidence * 100)
        if self.cardiac_view_confidence >= 0.7:
            view_color = (100, 255, 100)  # Verde
        elif self.cardiac_view_confidence >= 0.4:
            view_color = (0, 200, 255)  # Amarelo
        else:
            view_color = (120, 120, 150)  # Cinza

        cv2.putText(output, view_abbr, (18, view_y + 18),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, view_color, 1, cv2.LINE_AA)
        cv2.putText(output, f"({conf_pct}%)", (60, view_y + 18),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (100, 100, 120), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # LEGENDA DE CAMARAS
        # ═══════════════════════════════════════

        legend_x = 10
        legend_y = h - 90
        legend_w = 110
        legend_h = 80

        overlay = output.copy()
        cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h),
                     (30, 30, 40), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
        cv2.rectangle(output, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h),
                     (100, 100, 120), 1)

        cv2.putText(output, "CHAMBERS", (legend_x + 5, legend_y + 15),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 100, 120), 1, cv2.LINE_AA)

        chambers = [
            ("LV", (100, 150, 255), "Left Vent."),
            ("LA", (255, 150, 150), "Left Atrium"),
            ("RV", (150, 255, 150), "Right Vent."),
        ]
        for i, (abbr, color, name) in enumerate(chambers):
            cy = legend_y + 35 + i * 18
            cv2.circle(output, (legend_x + 12, cy - 3), 4, color, -1)
            cv2.putText(output, abbr, (legend_x + 22, cy),
                       cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1, cv2.LINE_AA)
            cv2.putText(output, name, (legend_x + 45, cy),
                       cv2.FONT_HERSHEY_DUPLEX, 0.25, (120, 120, 140), 1, cv2.LINE_AA)

        # Indicador de baixa qualidade
        if low_quality_mode:
            alert_y = 85
            cv2.rectangle(output, (w//2 - 90, alert_y), (w//2 + 90, alert_y + 28),
                         (50, 40, 40), -1)
            cv2.rectangle(output, (w//2 - 90, alert_y), (w//2 + 90, alert_y + 28),
                         (150, 100, 100), 2)
            cv2.putText(output, "LOW QUALITY - ADJUST", (w//2 - 78, alert_y + 20),
                       cv2.FONT_HERSHEY_DUPLEX, 0.38, (220, 180, 180), 1, cv2.LINE_AA)

        self._draw_auto_gain_status(output, 20, 60, gain_info, color=(180, 130, 130))

        return output

    # =========================================================================
    # FAST AUTO-NAVEGACAO - Metodos Auxiliares
    # =========================================================================

    def _fast_get_next_window(self):
        """Retorna a proxima janela na ordem, ou None se todas checadas."""
        current_idx = self.fast_window_order.index(self.fast_current_window)
        for i in range(current_idx + 1, len(self.fast_window_order)):
            key = self.fast_window_order[i]
            if self.fast_windows[key]['status'] != 'checked':
                return key
        # Verificar se ha alguma pendente antes da atual
        for i in range(0, current_idx):
            key = self.fast_window_order[i]
            if self.fast_windows[key]['status'] != 'checked':
                return key
        return None

    def _fast_play_confirm_sound(self):
        """Reproduz som de confirmacao (nao bloqueante)."""
        def play():
            try:
                # macOS - usar afplay com som do sistema
                subprocess.run(
                    ['afplay', '/System/Library/Sounds/Pop.aiff'],
                    capture_output=True,
                    timeout=1
                )
            except Exception:
                pass  # Ignorar erros de audio
        threading.Thread(target=play, daemon=True).start()

    def _fast_play_complete_sound(self):
        """Reproduz som de exame completo."""
        def play():
            try:
                subprocess.run(
                    ['afplay', '/System/Library/Sounds/Glass.aiff'],
                    capture_output=True,
                    timeout=1
                )
            except Exception:
                pass
        threading.Thread(target=play, daemon=True).start()

    def _fast_confirm_current_window(self):
        """Confirma a janela atual e avanca para proxima se auto_advance ativo."""
        current = self.fast_current_window
        now = time.time()

        # Marcar como checked
        self.fast_windows[current]['status'] = 'checked'

        # Registrar tempo gasto
        if self.fast_window_start_time:
            self.fast_windows[current]['time_spent'] = now - self.fast_window_start_time

        # Reset estados de estabilidade
        self.fast_stability_time = 0.0
        self.fast_last_stable_score = 0.0
        self.fast_confirmed_pending = False

        # Verificar se todas as janelas foram checadas
        total_checked = sum(1 for w in self.fast_windows.values() if w['status'] == 'checked')
        if total_checked >= 4:
            # Exame completo!
            self._fast_play_complete_sound()
        else:
            # Janela confirmada
            self._fast_play_confirm_sound()

        # Avancar para proxima janela se auto_advance
        if self.fast_auto_advance:
            next_win = self._fast_get_next_window()
            if next_win:
                self.fast_current_window = next_win
                self.fast_window_start_time = now
                # Limpar historico da nova janela
                self.fast_window_history[next_win].clear()

    def _fast_navigate_to(self, window_key):
        """Navega para uma janela especifica."""
        if window_key in self.fast_windows:
            self.fast_current_window = window_key
            self.fast_window_start_time = time.time()
            self.fast_stability_time = 0.0
            self.fast_last_stable_score = 0.0
            self.fast_confirmed_pending = False
            self.fast_window_history[window_key].clear()

    def _fast_check_stability(self, avg_score, scan_quality, dt):
        """
        Verifica estabilidade do score para auto-confirmacao.
        Retorna True se a janela pode ser confirmada.
        """
        now = time.time()

        # Nao confirmar se qualidade baixa
        if scan_quality < 35:
            self.fast_stability_time = 0.0
            self.fast_confirmed_pending = False
            return False

        # Nao confirmar antes do tempo minimo de scan
        if self.fast_window_start_time:
            elapsed = now - self.fast_window_start_time
            if elapsed < self.fast_min_scan_time:
                return False

        # Verificar se score esta estavel
        score_diff = abs(avg_score - self.fast_last_stable_score)
        if score_diff < 0.1:  # Score variou menos de 10%
            self.fast_stability_time += dt
        else:
            self.fast_stability_time = 0.0
            self.fast_last_stable_score = avg_score

        # Janela pode ser confirmada se estavel por tempo suficiente
        if self.fast_stability_time >= self.fast_stability_threshold:
            self.fast_confirmed_pending = True
            return True

        return False

    # ═══════════════════════════════════════════════════════════════════════
    # LUNG ZONES - Navegação e Confirmação (Protocolo BLUE)
    # ═══════════════════════════════════════════════════════════════════════

    def _lung_get_next_zone(self):
        """Retorna a próxima zona pendente ou None se todas checadas."""
        current_idx = self.lung_zone_order.index(self.lung_current_zone)
        # Procurar próxima pendente
        for i in range(current_idx + 1, len(self.lung_zone_order)):
            zone = self.lung_zone_order[i]
            if self.lung_zones[zone]['status'] != 'checked':
                return zone
        # Procurar desde o início
        for i in range(0, current_idx):
            zone = self.lung_zone_order[i]
            if self.lung_zones[zone]['status'] != 'checked':
                return zone
        return None

    def _lung_confirm_current_zone(self):
        """Confirma a zona atual com os dados coletados."""
        now = time.time()
        zone = self.lung_current_zone
        zone_data = self.lung_zones[zone]

        # Salvar dados da zona
        zone_data['status'] = 'checked'
        zone_data['b_lines'] = self.b_line_count
        zone_data['a_lines'] = self.lung_a_lines_detected
        zone_data['sliding'] = self.lung_pleura_sliding
        zone_data['profile'] = 'B' if self.b_line_count >= 3 else 'A'

        if self.lung_zone_start_time:
            zone_data['time_spent'] = now - self.lung_zone_start_time

        # Feedback sonoro
        self._lung_play_confirm_sound()

        # Verificar se todas as zonas foram checadas
        total_checked = sum(1 for z in self.lung_zones.values() if z['status'] == 'checked')
        if total_checked >= len(self.lung_zones):
            self.lung_exam_complete = True
            self._lung_play_complete_sound()
            return

        # Avançar para próxima zona
        next_zone = self._lung_get_next_zone()
        if next_zone:
            self.lung_current_zone = next_zone
            self.lung_zone_start_time = now
            # Limpar históricos
            self.lung_b_line_history.clear()
            self.lung_b_line_density_history.clear()

    def _lung_navigate_to(self, zone_key):
        """Navega para uma zona específica."""
        if zone_key in self.lung_zones:
            self.lung_current_zone = zone_key
            self.lung_zone_start_time = time.time()
            self.lung_b_line_history.clear()
            self.lung_b_line_density_history.clear()

    def _lung_play_confirm_sound(self):
        """Reproduz som de confirmação de zona."""
        def play():
            try:
                subprocess.run(['afplay', '/System/Library/Sounds/Pop.aiff'],
                             capture_output=True, timeout=2)
            except Exception:
                pass
        threading.Thread(target=play, daemon=True).start()

    def _lung_play_complete_sound(self):
        """Reproduz som de exame completo."""
        def play():
            try:
                subprocess.run(['afplay', '/System/Library/Sounds/Glass.aiff'],
                             capture_output=True, timeout=2)
            except Exception:
                pass
        threading.Thread(target=play, daemon=True).start()

    def _lung_reset_exam(self):
        """Reseta o exame de zonas pulmonares."""
        for zone in self.lung_zones:
            self.lung_zones[zone] = {
                'status': 'pending',
                'b_lines': 0,
                'a_lines': False,
                'sliding': True,
                'profile': 'A',
                'time_spent': 0.0,
            }
        self.lung_current_zone = 'R1'
        self.lung_zone_start_time = None
        self.lung_exam_complete = False
        self.lung_exam_start_time = None
        self.lung_b_line_history.clear()
        self.lung_b_line_density_history.clear()

    def _process_fast(self, frame):
        """FAST Protocol - Sistema avancado de trauma estilo Clarius."""
        output = frame.copy()
        h, w = frame.shape[:2]

        # Inicializar start time do FAST se ainda nao foi
        now = time.time()
        if self.fast_start_time is None:
            self.fast_start_time = now
            self.fast_window_start_time = now
            self._fast_last_frame_time = now

        # Calcular delta time entre frames
        dt = now - getattr(self, '_fast_last_frame_time', now)
        self._fast_last_frame_time = now

        # ═══════════════════════════════════════
        # DETECCAO DE LIQUIDO LIVRE (CV)
        # ═══════════════════════════════════════

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ROI suave por janela (usado para threshold/score, nao restringe detecao)
        roi_map = {
            'RUQ': (0.45, 0.05, 0.5, 0.5),
            'LUQ': (0.05, 0.05, 0.5, 0.5),
            'PELV': (0.2, 0.55, 0.6, 0.4),
            'CARD': (0.2, 0.05, 0.6, 0.45),
        }
        rx, ry, rw, rh = roi_map.get(self.fast_current_window, (0.1, 0.1, 0.8, 0.8))
        x0 = max(0, int(w * rx))
        y0 = max(0, int(h * ry))
        x1 = min(w, int(w * (rx + rw)))
        y1 = min(h, int(h * (ry + rh)))
        tuned, gain_info = self._auto_gain_and_quality(gray, roi=(x0, y0, x1, y1))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(tuned)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)

        roi = enhanced[y0:y1, x0:x1]
        if roi.size == 0:
            roi = enhanced
            x0, y0, x1, y1 = 0, 0, w, h

        roi_mean = float(np.mean(roi))
        thresh_val = int(np.clip(np.percentile(roi, 12), 18, 60))

        # Detectar areas anecoicas (escuras) - threshold adaptativo
        _, dark_thresh = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # Morfologia para limpar
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_thresh = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, kernel)
        dark_thresh = cv2.morphologyEx(dark_thresh, cv2.MORPH_CLOSE, kernel)
        dark_thresh = cv2.GaussianBlur(dark_thresh, (3, 3), 0)
        _, dark_thresh = cv2.threshold(dark_thresh, 127, 255, cv2.THRESH_BINARY)

        # Encontrar contornos de fluido
        contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not hasattr(self, "fast_window_history"):
            self.fast_window_history = {key: deque(maxlen=8) for key in self.fast_windows}

        fluid_detected = False
        self.fast_fluid_regions = []
        candidates = []

        roi_area = max(1, (x1 - x0) * (y1 - y0))
        min_area = max(200, int(roi_area * 0.002))
        max_area = int(roi_area * 0.25)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < 15 or ch < 15:
                continue

            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_inside = cv2.mean(enhanced, mask=mask)[0]
            contrast = max(0.0, roi_mean - mean_inside)
            if contrast < 8:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < 0.45:
                continue

            aspect = max(cw, ch) / (min(cw, ch) + 1)
            cx, cy = x + cw // 2, y + ch // 2
            in_roi = x0 <= cx <= x1 and y0 <= cy <= y1

            area_score = min(area / (0.08 * roi_area), 1.0)
            contrast_score = min(contrast / 40.0, 1.0)
            solidity_score = min(max(solidity - 0.45, 0.0) / 0.55, 1.0)
            aspect_score = min(max(aspect - 1.0, 0.0) / 3.0, 1.0)
            roi_bonus = 1.0 if in_roi else 0.7

            score = (
                0.4 * contrast_score
                + 0.3 * area_score
                + 0.2 * solidity_score
                + 0.1 * aspect_score
            ) * roi_bonus

            candidates.append({
                'contour': cnt,
                'area': area,
                'center': (cx, cy),
                'score': score,
                'bbox': (x, y, cw, ch),
            })

        candidates.sort(key=lambda c: c['score'], reverse=True)
        best_score = candidates[0]['score'] if candidates else 0.0

        history = self.fast_window_history[self.fast_current_window]
        history.append(best_score)
        avg_score = float(np.mean(history)) if history else 0.0

        # ═══════════════════════════════════════
        # GATING POR SCAN QUALITY (reduz falsos positivos)
        # ═══════════════════════════════════════
        scan_quality = gain_info.get('quality', 50.0) if gain_info else 50.0
        quality_threshold = 35.0  # Qualidade minima para considerar deteccoes
        low_quality_mode = scan_quality < quality_threshold

        # So considera fluido se qualidade da imagem for aceitavel
        if low_quality_mode:
            fluid_detected = False  # Ignora deteccoes em imagem de baixa qualidade
        else:
            fluid_detected = avg_score >= 0.35
        if candidates:
            for cand in candidates[:2]:
                if cand['score'] < 0.25:
                    continue
                self.fast_fluid_regions.append({
                    'contour': cand['contour'],
                    'area': cand['area'],
                    'center': cand['center'],
                })

        # Desenhar regioes candidatas
        for region in self.fast_fluid_regions:
            cnt = region['contour']
            x, y, cw, ch = cv2.boundingRect(cnt)
            overlay = output.copy()
            cv2.drawContours(overlay, [cnt], -1, (0, 100, 200), -1)
            cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)

            cv2.drawContours(output, [cnt], -1, (0, 80, 150), 3)
            cv2.drawContours(output, [cnt], -1, (0, 150, 255), 2)

            cv2.putText(output, "FLUID?", (x + cw//2 - 25, y - 8),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

        # Atualizar janela atual
        current_win = self.fast_windows[self.fast_current_window]
        if fluid_detected:
            current_win['fluid'] = True
            current_win['score'] = min(10, int(avg_score * 10))
        else:
            current_win['score'] = max(current_win['score'] - 1, 0)
            if avg_score < 0.2:
                current_win['fluid'] = False

        # ═══════════════════════════════════════
        # AUTO-CONFIRMACAO DE JANELA
        # ═══════════════════════════════════════

        # Verificar se janela atual pode ser confirmada automaticamente
        can_confirm = self._fast_check_stability(avg_score, scan_quality, dt)

        # Calcular tempo na janela atual
        window_elapsed = 0.0
        if self.fast_window_start_time:
            window_elapsed = now - self.fast_window_start_time

        # Progresso para confirmacao (0-100%)
        if window_elapsed < self.fast_min_scan_time:
            confirm_progress = (window_elapsed / self.fast_min_scan_time) * 50
        else:
            confirm_progress = 50 + (self.fast_stability_time / self.fast_stability_threshold) * 50
        confirm_progress = min(100, confirm_progress)

        # Auto-confirmar se pronto e janela ainda pendente
        if can_confirm and current_win['status'] != 'checked':
            self._fast_confirm_current_window()

        # ═══════════════════════════════════════
        # HEADER PREMIUM
        # ═══════════════════════════════════════

        cv2.rectangle(output, (10, 10), (180, 50), (50, 50, 30), -1)
        cv2.rectangle(output, (10, 10), (180, 50), (255, 200, 100), 1)
        cv2.putText(output, "FAST EXAM", (20, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 220, 150), 1, cv2.LINE_AA)

        # Timer
        elapsed = int(time.time() - self.fast_start_time)
        mins, secs = divmod(elapsed, 60)
        cv2.putText(output, f"{mins:02d}:{secs:02d}", (130, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (180, 180, 150), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # PAINEL DE JANELAS (lado direito)
        # ═══════════════════════════════════════

        panel_w = 175
        panel_x = w - panel_w - 10
        panel_y = 10
        panel_h = 200

        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (30, 30, 25), -1)
        cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (255, 200, 100), 1)

        cv2.putText(output, "WINDOWS", (panel_x + 10, panel_y + 20),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (150, 150, 140), 1, cv2.LINE_AA)

        windows_config = [
            ('RUQ', 'Morrison Pouch', 'Hepatorenal'),
            ('LUQ', 'Splenorenal', 'Perisplenic'),
            ('PELV', 'Pelvic', 'Retrovesical'),
            ('CARD', 'Cardiac', 'Pericardial'),
        ]

        for i, (key, name, detail) in enumerate(windows_config):
            win = self.fast_windows[key]
            jy = panel_y + 50 + i * 38

            # Checkbox
            box_size = 16
            box_x = panel_x + 10
            box_y = jy - 8

            is_current = (key == self.fast_current_window)

            if win['status'] == 'checked':
                # Checked - verde ou vermelho baseado em fluido
                if win['fluid']:
                    box_color = (0, 100, 255)  # Vermelho (positivo)
                    text_color = (100, 150, 255)
                else:
                    box_color = (100, 200, 100)  # Verde (negativo)
                    text_color = (150, 255, 150)

                cv2.rectangle(output, (box_x, box_y), (box_x + box_size, box_y + box_size),
                             box_color, -1)
                # Checkmark
                cv2.line(output, (box_x + 3, box_y + 8), (box_x + 7, box_y + 12), (255, 255, 255), 2)
                cv2.line(output, (box_x + 7, box_y + 12), (box_x + 13, box_y + 4), (255, 255, 255), 2)
            else:
                # Pendente
                border_color = (255, 220, 150) if is_current else (100, 100, 100)
                cv2.rectangle(output, (box_x, box_y), (box_x + box_size, box_y + box_size),
                             (50, 50, 60), -1)
                cv2.rectangle(output, (box_x, box_y), (box_x + box_size, box_y + box_size),
                             border_color, 2 if is_current else 1)
                text_color = (255, 220, 150) if is_current else (140, 140, 150)

            # Nome da janela
            cv2.putText(output, name, (box_x + 22, jy + 2),
                       cv2.FONT_HERSHEY_DUPLEX, 0.35, text_color, 1, cv2.LINE_AA)

            # Detalhe
            cv2.putText(output, detail, (box_x + 22, jy + 16),
                       cv2.FONT_HERSHEY_DUPLEX, 0.25, (100, 100, 110), 1, cv2.LINE_AA)

            # Indicador de fluido detectado ou tempo gasto
            if win['fluid']:
                cv2.circle(output, (panel_x + panel_w - 20, jy + 4), 5, (0, 100, 255), -1)
                cv2.putText(output, "+", (panel_x + panel_w - 24, jy + 8),
                           cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            elif win['status'] == 'checked' and win.get('time_spent', 0) > 0:
                # Mostrar tempo gasto na janela
                t = int(win['time_spent'])
                time_str = f"{t}s"
                cv2.putText(output, time_str, (panel_x + panel_w - 30, jy + 8),
                           cv2.FONT_HERSHEY_DUPLEX, 0.25, (120, 150, 120), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # PAINEL DE STATUS (inferior esquerdo)
        # ═══════════════════════════════════════

        status_x = 10
        status_y = h - 100
        status_w = 180
        status_h = 90

        overlay = output.copy()
        cv2.rectangle(overlay, (status_x, status_y), (status_x + status_w, status_y + status_h),
                     (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
        cv2.rectangle(output, (status_x, status_y), (status_x + status_w, status_y + status_h),
                     (100, 100, 100), 1)

        # Calcular status geral
        total_fluid = sum(1 for w in self.fast_windows.values() if w['fluid'])
        total_checked = sum(1 for w in self.fast_windows.values() if w['status'] == 'checked')

        if total_fluid > 0:
            overall_status = "POSITIVE"
            status_color = (0, 100, 255)
            status_icon_color = (0, 150, 255)
        else:
            overall_status = "NEGATIVE"
            status_color = (100, 255, 100)
            status_icon_color = (150, 255, 150)

        cv2.putText(output, "RESULT", (status_x + 10, status_y + 18),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, (120, 120, 130), 1, cv2.LINE_AA)

        # Icone de status
        cv2.circle(output, (status_x + 25, status_y + 45), 12, status_icon_color, -1)
        if total_fluid > 0:
            # X para positivo
            cv2.line(output, (status_x + 19, status_y + 39), (status_x + 31, status_y + 51), (255, 255, 255), 2)
            cv2.line(output, (status_x + 31, status_y + 39), (status_x + 19, status_y + 51), (255, 255, 255), 2)
        else:
            # Check para negativo
            cv2.line(output, (status_x + 18, status_y + 45), (status_x + 23, status_y + 50), (255, 255, 255), 2)
            cv2.line(output, (status_x + 23, status_y + 50), (status_x + 32, status_y + 38), (255, 255, 255), 2)

        cv2.putText(output, overall_status, (status_x + 45, status_y + 50),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, status_color, 1, cv2.LINE_AA)

        # Progresso
        cv2.putText(output, f"Views: {total_checked}/4", (status_x + 10, status_y + 75),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, (140, 140, 150), 1, cv2.LINE_AA)

        # Barra de progresso
        prog_w = status_w - 90
        cv2.rectangle(output, (status_x + 80, status_y + 65), (status_x + 80 + prog_w, status_y + 77),
                     (50, 50, 60), -1)
        fill_w = int(prog_w * total_checked / 4)
        cv2.rectangle(output, (status_x + 80, status_y + 65), (status_x + 80 + fill_w, status_y + 77),
                     (100, 200, 100), -1)

        # ═══════════════════════════════════════
        # GUIA DE JANELA ATUAL (inferior direito)
        # ═══════════════════════════════════════

        guide_w = 175
        guide_x = w - guide_w - 10
        guide_y = h - 115
        guide_h = 105

        overlay = output.copy()
        cv2.rectangle(overlay, (guide_x, guide_y), (guide_x + guide_w, guide_y + guide_h),
                     (30, 40, 40), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)

        # Borda que pisca se confirmando
        if self.fast_confirmed_pending and int(now * 3) % 2 == 0:
            border_color = (100, 255, 200)  # Verde piscando
        else:
            border_color = (100, 150, 150)
        cv2.rectangle(output, (guide_x, guide_y), (guide_x + guide_w, guide_y + guide_h),
                     border_color, 1)

        # Header com timer da janela
        cv2.putText(output, "CURRENT VIEW", (guide_x + 10, guide_y + 18),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 130, 130), 1, cv2.LINE_AA)

        # Timer da janela atual
        win_secs = int(window_elapsed)
        win_mins, win_secs = divmod(win_secs, 60)
        cv2.putText(output, f"{win_mins:01d}:{win_secs:02d}", (guide_x + guide_w - 40, guide_y + 18),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (150, 180, 150), 1, cv2.LINE_AA)

        current_name = {'RUQ': 'RUQ Morrison', 'LUQ': 'LUQ Spleen', 'PELV': 'Pelvic', 'CARD': 'Cardiac'}
        cv2.putText(output, current_name[self.fast_current_window], (guide_x + 10, guide_y + 40),
                   cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 255, 255), 1, cv2.LINE_AA)

        # Instrucao
        tips = {
            'RUQ': 'Look hepatorenal space',
            'LUQ': 'Scan perisplenic area',
            'PELV': 'Check retrovesical',
            'CARD': 'Subxiphoid view'
        }
        cv2.putText(output, tips[self.fast_current_window], (guide_x + 10, guide_y + 58),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (150, 180, 180), 1, cv2.LINE_AA)

        # Barra de progresso para confirmacao
        prog_bar_y = guide_y + 72
        prog_bar_w = guide_w - 20
        cv2.rectangle(output, (guide_x + 10, prog_bar_y), (guide_x + 10 + prog_bar_w, prog_bar_y + 8),
                     (40, 50, 50), -1)
        fill_w = int(prog_bar_w * confirm_progress / 100)
        # Cor baseada no progresso
        if confirm_progress < 50:
            prog_color = (100, 150, 150)  # Cinza - scanning
        elif confirm_progress < 100:
            prog_color = (100, 200, 200)  # Ciano - estabilizando
        else:
            prog_color = (100, 255, 150)  # Verde - pronto
        cv2.rectangle(output, (guide_x + 10, prog_bar_y), (guide_x + 10 + fill_w, prog_bar_y + 8),
                     prog_color, -1)

        # Label de status da confirmacao
        if self.fast_confirmed_pending:
            status_label = "CONFIRMING..."
            status_clr = (100, 255, 200)
        elif window_elapsed < self.fast_min_scan_time:
            status_label = "SCANNING..."
            status_clr = (150, 180, 180)
        elif self.fast_stability_time > 0:
            status_label = "STABILIZING..."
            status_clr = (150, 200, 200)
        else:
            status_label = "HOLD STEADY"
            status_clr = (150, 180, 180)
        cv2.putText(output, status_label, (guide_x + 10, guide_y + 95),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, status_clr, 1, cv2.LINE_AA)

        # Atalho para confirmar manualmente
        cv2.putText(output, "[SPACE] confirm", (guide_x + 85, guide_y + 95),
                   cv2.FONT_HERSHEY_DUPLEX, 0.22, (100, 120, 120), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # INDICADORES DE FLUIDO (overlay central)
        # ═══════════════════════════════════════

        if fluid_detected and len(self.fast_fluid_regions) > 0:
            # Alerta de fluido detectado
            alert_y = 60
            cv2.rectangle(output, (w//2 - 80, alert_y), (w//2 + 80, alert_y + 28),
                         (0, 80, 150), -1)
            cv2.rectangle(output, (w//2 - 80, alert_y), (w//2 + 80, alert_y + 28),
                         (0, 150, 255), 2)
            cv2.putText(output, "FLUID DETECTED", (w//2 - 65, alert_y + 20),
                       cv2.FONT_HERSHEY_DUPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA)
        elif low_quality_mode:
            # Alerta de baixa qualidade - deteccoes suspensas
            alert_y = 60
            cv2.rectangle(output, (w//2 - 90, alert_y), (w//2 + 90, alert_y + 28),
                         (40, 40, 80), -1)
            cv2.rectangle(output, (w//2 - 90, alert_y), (w//2 + 90, alert_y + 28),
                         (100, 100, 180), 2)
            cv2.putText(output, "LOW QUALITY - ADJUST", (w//2 - 78, alert_y + 20),
                       cv2.FONT_HERSHEY_DUPLEX, 0.38, (180, 180, 220), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # EXAM COMPLETE (quando todas as janelas checadas)
        # ═══════════════════════════════════════

        if total_checked >= 4:
            # Banner de exame completo
            banner_y = h // 2 - 30
            banner_h = 60

            overlay = output.copy()
            cv2.rectangle(overlay, (w//2 - 120, banner_y), (w//2 + 120, banner_y + banner_h),
                         (20, 80, 20), -1)
            cv2.addWeighted(overlay, 0.9, output, 0.1, 0, output)

            # Borda pulsante
            pulse = int(abs(np.sin(now * 2)) * 255)
            cv2.rectangle(output, (w//2 - 120, banner_y), (w//2 + 120, banner_y + banner_h),
                         (100, pulse, 100), 3)

            cv2.putText(output, "EXAM COMPLETE", (w//2 - 85, banner_y + 25),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (150, 255, 150), 2, cv2.LINE_AA)

            # Resultado final
            if total_fluid > 0:
                result_text = f"POSITIVE ({total_fluid} views)"
                result_color = (100, 150, 255)
            else:
                result_text = "NEGATIVE"
                result_color = (150, 255, 150)
            cv2.putText(output, result_text, (w//2 - 55, banner_y + 48),
                       cv2.FONT_HERSHEY_DUPLEX, 0.45, result_color, 1, cv2.LINE_AA)

        self._draw_auto_gain_status(output, 20, 60, gain_info, color=(200, 190, 130))

        return output

    def _process_segment(self, frame):
        """Anatomy AI - Sistema avancado de segmentacao anatomica estilo Clarius."""
        output = frame.copy()
        h, w = frame.shape[:2]

        model = self.models.get('segment')

        structures_detected = []

        # ═══════════════════════════════════════
        # CONFIGURACAO DE ESTRUTURAS
        # ═══════════════════════════════════════

        structure_config = [
            {'name': 'MUSCLE', 'color': (100, 200, 100), 'color_dark': (50, 100, 50), 'icon': 'M'},
            {'name': 'BONE', 'color': (255, 255, 255), 'color_dark': (180, 180, 180), 'icon': 'B'},
            {'name': 'FLUID', 'color': (255, 150, 100), 'color_dark': (180, 100, 60), 'icon': 'F'},
            {'name': 'VESSEL', 'color': (100, 100, 255), 'color_dark': (60, 60, 180), 'icon': 'V'},
            {'name': 'FAT', 'color': (200, 200, 100), 'color_dark': (140, 140, 60), 'icon': 'A'},
        ]

        if model:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (224, 224))
                tensor = torch.from_numpy(resized).float() / 255.0
                tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    masks = model(tensor).cpu().numpy()[0]

                for i in range(min(5, masks.shape[0])):
                    mask = cv2.resize(masks[i], (w, h))
                    binary = (mask > 0.5).astype(np.uint8)

                    if binary.any():
                        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if area > 300:
                                cfg = structure_config[i]
                                M = cv2.moments(cnt)
                                if M["m00"] > 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])

                                    # Calcular dimensoes
                                    rect = cv2.minAreaRect(cnt)
                                    (_, _), (rw, rh), _ = rect

                                    structures_detected.append({
                                        'type': cfg['name'],
                                        'color': cfg['color'],
                                        'color_dark': cfg['color_dark'],
                                        'center': (cx, cy),
                                        'area': area,
                                        'width': rw * 0.3,  # mm
                                        'height': rh * 0.3,  # mm
                                        'contour': cnt
                                    })

            except Exception as e:
                print(f"Erro USFM: {e}")

        # Fallback CV: detectar estruturas baseado em textura e intensidade
        if not structures_detected:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Detectar diferentes faixas de intensidade
            intensity_ranges = [
                (180, 255, 'BONE', (255, 255, 255), (180, 180, 180)),    # Muito brilhante
                (100, 150, 'MUSCLE', (100, 200, 100), (50, 100, 50)),    # Medio
                (0, 40, 'FLUID', (255, 150, 100), (180, 100, 60)),       # Escuro
            ]

            for low, high, name, color, color_dark in intensity_ranges:
                mask = cv2.inRange(enhanced, low, high)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 500:
                        M = cv2.moments(cnt)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            rect = cv2.minAreaRect(cnt)
                            (_, _), (rw, rh), _ = rect

                            structures_detected.append({
                                'type': name,
                                'color': color,
                                'color_dark': color_dark,
                                'center': (cx, cy),
                                'area': area,
                                'width': rw * 0.3,
                                'height': rh * 0.3,
                                'contour': cnt
                            })

        # Limitar a 10 estruturas mais relevantes
        structures_detected = sorted(structures_detected, key=lambda x: x['area'], reverse=True)[:10]
        self.anatomy_structures = structures_detected

        # ═══════════════════════════════════════
        # VISUALIZACAO DAS ESTRUTURAS
        # ═══════════════════════════════════════

        for s in structures_detected:
            # Preenchimento semi-transparente
            overlay = output.copy()
            cv2.drawContours(overlay, [s['contour']], -1, s['color_dark'], -1)
            cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)

            # Contorno com glow
            cv2.drawContours(output, [s['contour']], -1, s['color_dark'], 3)
            cv2.drawContours(output, [s['contour']], -1, s['color'], 1)

            # Label com background
            label = s['type']
            cx, cy = s['center']
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.35, 1)
            cv2.rectangle(output, (cx - tw//2 - 3, cy - th - 3), (cx + tw//2 + 3, cy + 3),
                         (20, 20, 30), -1)
            cv2.putText(output, label, (cx - tw//2, cy),
                       cv2.FONT_HERSHEY_DUPLEX, 0.35, s['color'], 1, cv2.LINE_AA)

            # Dimensoes abaixo
            dim_text = f"{s['width']:.1f}x{s['height']:.1f}mm"
            cv2.putText(output, dim_text, (cx - 25, cy + 18),
                       cv2.FONT_HERSHEY_DUPLEX, 0.25, (150, 150, 180), 1, cv2.LINE_AA)

        # Desenhar linhas de medicao entre estruturas proximas
        self.anatomy_measurements = []
        for i, s1 in enumerate(structures_detected[:5]):
            for s2 in structures_detected[i+1:5]:
                d = np.sqrt((s1['center'][0]-s2['center'][0])**2 + (s1['center'][1]-s2['center'][1])**2)
                if d < 200:  # Estruturas proximas
                    dist_mm = d * 0.3
                    mid = ((s1['center'][0]+s2['center'][0])//2, (s1['center'][1]+s2['center'][1])//2)

                    # Linha tracejada
                    pts = [s1['center'], s2['center']]
                    cv2.line(output, pts[0], pts[1], (100, 100, 120), 1, cv2.LINE_AA)

                    # Medicao
                    cv2.rectangle(output, (mid[0]-20, mid[1]-10), (mid[0]+25, mid[1]+4), (20, 20, 30), -1)
                    cv2.putText(output, f"{dist_mm:.1f}mm", (mid[0]-18, mid[1]),
                               cv2.FONT_HERSHEY_DUPLEX, 0.3, (180, 180, 220), 1, cv2.LINE_AA)

                    self.anatomy_measurements.append({
                        'from': s1['type'],
                        'to': s2['type'],
                        'distance': dist_mm
                    })

        # ═══════════════════════════════════════
        # PAINEL LATERAL (estilo Clarius)
        # ═══════════════════════════════════════

        panel_w = 165
        panel_x = w - panel_w - 10
        panel_y = 10
        panel_h = 40 + len(structures_detected[:6]) * 32 + 20
        panel_h = min(panel_h, h - 100)

        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (30, 25, 35), -1)
        cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (200, 100, 200), 1)

        cv2.putText(output, "ANATOMY AI", (panel_x + 10, panel_y + 22),
                   cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 150, 255), 1, cv2.LINE_AA)

        # Lista de estruturas
        y_off = panel_y + 45
        for s in structures_detected[:6]:
            cv2.circle(output, (panel_x + 12, y_off - 3), 5, s['color'], -1)
            cv2.putText(output, s['type'], (panel_x + 22, y_off),
                       cv2.FONT_HERSHEY_DUPLEX, 0.32, s['color'], 1, cv2.LINE_AA)

            # Area
            area_mm2 = s['area'] * 0.09  # px^2 to mm^2
            cv2.putText(output, f"{area_mm2:.0f}mm2", (panel_x + 90, y_off),
                       cv2.FONT_HERSHEY_DUPLEX, 0.28, (140, 140, 160), 1, cv2.LINE_AA)
            y_off += 28

        # Contagem total
        cv2.putText(output, f"Total: {len(structures_detected)} structures",
                   (panel_x + 10, panel_y + panel_h - 10),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (120, 100, 140), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # HEADER PREMIUM
        # ═══════════════════════════════════════

        cv2.rectangle(output, (10, 10), (160, 45), (45, 30, 50), -1)
        cv2.rectangle(output, (10, 10), (160, 45), (200, 100, 200), 1)
        cv2.putText(output, "SEGMENT MODE", (20, 32),
                   cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 180, 255), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # ESCALA E LEGENDA
        # ═══════════════════════════════════════

        # Escala lateral
        scale_x = 20
        for i in range(0, h, 50):
            cv2.line(output, (scale_x - 3, i), (scale_x + 3, i), (150, 100, 150), 1)
            if i % 100 == 0:
                cv2.putText(output, f"{int(i*0.3)}", (scale_x + 5, i + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, (150, 100, 150), 1)
        cv2.line(output, (scale_x, 0), (scale_x, h), (100, 80, 100), 1)

        # Legenda de cores (inferior esquerdo)
        legend_x = 10
        legend_y = h - 85
        legend_w = 120
        legend_h = 75

        overlay = output.copy()
        cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h),
                     (30, 30, 40), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
        cv2.rectangle(output, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h),
                     (100, 80, 120), 1)

        cv2.putText(output, "LEGEND", (legend_x + 5, legend_y + 14),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 100, 120), 1, cv2.LINE_AA)

        legend_items = [
            ('MUSCLE', (100, 200, 100)),
            ('BONE', (255, 255, 255)),
            ('FLUID', (255, 150, 100)),
            ('VESSEL', (100, 100, 255)),
        ]
        for i, (name, color) in enumerate(legend_items):
            ly = legend_y + 30 + i * 12
            cv2.circle(output, (legend_x + 10, ly - 2), 3, color, -1)
            cv2.putText(output, name, (legend_x + 18, ly),
                       cv2.FONT_HERSHEY_DUPLEX, 0.25, color, 1, cv2.LINE_AA)

        return output

    def _process_mmode(self, frame):
        """M-Mode - Sistema avancado de analise temporal estilo Clarius."""
        output = frame.copy()
        h, w = frame.shape[:2]

        # Ajustar cursor para dimensoes atuais se necessario
        if self.mmode_cursor_x <= 0 or self.mmode_cursor_x >= w:
            self.mmode_cursor_x = w // 2
        if self.mmode_timeline_height <= 0:
            self.mmode_timeline_height = h // 3

        # ═══════════════════════════════════════
        # LINHA DE AMOSTRAGEM (cursor movel)
        # ═══════════════════════════════════════

        cursor_x = self.mmode_cursor_x

        # Glow effect no cursor
        cv2.line(output, (cursor_x, 0), (cursor_x, h - self.mmode_timeline_height - 10),
                (0, 100, 100), 3)
        cv2.line(output, (cursor_x, 0), (cursor_x, h - self.mmode_timeline_height - 10),
                (0, 200, 200), 2)
        cv2.line(output, (cursor_x, 0), (cursor_x, h - self.mmode_timeline_height - 10),
                (0, 255, 255), 1)

        # Marcador superior
        cv2.circle(output, (cursor_x, 5), 6, (0, 200, 200), -1)
        cv2.circle(output, (cursor_x, 5), 8, (0, 255, 255), 1)

        # ═══════════════════════════════════════
        # CAPTURAR LINHA PARA TIMELINE
        # ═══════════════════════════════════════

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Capturar coluna no cursor
        if 0 < cursor_x < w:
            column = gray[:h - self.mmode_timeline_height - 10, cursor_x]
            self.mmode_buffer.append(column)

            # Limitar buffer (ultimos N frames)
            max_buffer = 200
            if len(self.mmode_buffer) > max_buffer:
                self.mmode_buffer.pop(0)

        # ═══════════════════════════════════════
        # RENDERIZAR TIMELINE M-MODE
        # ═══════════════════════════════════════

        timeline_y = h - self.mmode_timeline_height
        timeline_h = self.mmode_timeline_height - 10

        # Background da timeline
        cv2.rectangle(output, (10, timeline_y), (w - 10, h - 10), (20, 20, 25), -1)
        cv2.rectangle(output, (10, timeline_y), (w - 10, h - 10), (0, 150, 150), 1)

        # Desenhar sweep M-Mode
        if len(self.mmode_buffer) >= 2:
            timeline_w = w - 30
            n_cols = min(len(self.mmode_buffer), timeline_w)

            # Criar imagem da timeline
            mmode_img = np.zeros((timeline_h - 20, timeline_w), dtype=np.uint8)

            for i in range(n_cols):
                buf_idx = len(self.mmode_buffer) - n_cols + i
                if buf_idx >= 0:
                    col = self.mmode_buffer[buf_idx]
                    # Redimensionar coluna para altura da timeline
                    resized_col = cv2.resize(col.reshape(-1, 1), (1, timeline_h - 20))
                    mmode_img[:, i] = resized_col.flatten()

            # Aplicar colormap para visual premium
            mmode_colored = cv2.applyColorMap(mmode_img, cv2.COLORMAP_BONE)

            # Colar na output
            output[timeline_y + 10:timeline_y + timeline_h - 10, 15:15 + timeline_w] = mmode_colored

            # Linha de tempo (cursor atual)
            sweep_x = 15 + n_cols - 1
            cv2.line(output, (sweep_x, timeline_y + 5), (sweep_x, h - 15), (0, 255, 255), 1)

        # Label da timeline
        cv2.putText(output, "M-MODE SWEEP", (20, timeline_y + 18),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, (0, 200, 200), 1, cv2.LINE_AA)

        # Escala de tempo
        for i in range(0, w - 30, 50):
            time_ms = i * 33  # ~30fps = 33ms/frame
            tx = 15 + i
            cv2.line(output, (tx, h - 15), (tx, h - 20), (0, 150, 150), 1)
            if i % 100 == 0:
                cv2.putText(output, f"{time_ms}ms", (tx - 15, h - 5),
                           cv2.FONT_HERSHEY_DUPLEX, 0.2, (100, 150, 150), 1)

        # ═══════════════════════════════════════
        # PAINEL DE MEDICOES (lado direito)
        # ═══════════════════════════════════════

        panel_w = 150
        panel_x = w - panel_w - 10
        panel_y = 10
        panel_h = 160

        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (25, 30, 25), -1)
        cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (0, 200, 200), 1)

        cv2.putText(output, "M-MODE", (panel_x + 10, panel_y + 22),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Metricas de exemplo (baseadas em analise temporal)
        metrics_y = panel_y + 50

        # Calcular metricas do buffer
        if len(self.mmode_buffer) > 30:
            recent = np.array(self.mmode_buffer[-30:])

            # Encontrar estruturas moveis (variancia ao longo do tempo)
            variance = np.var(recent, axis=0)
            max_var_idx = np.argmax(variance)

            # Amplitude do movimento
            amplitude = np.max(recent[:, max_var_idx]) - np.min(recent[:, max_var_idx])
            amplitude_mm = amplitude * 0.3

            # Frequencia (contar cruzamentos do meio)
            signal = recent[:, max_var_idx]
            mean_val = np.mean(signal)
            crossings = np.where(np.diff(np.sign(signal - mean_val)))[0]
            freq_hz = len(crossings) / 2 * 30 / len(recent)  # Ciclos por segundo

            cv2.putText(output, "AMPLITUDE", (panel_x + 10, metrics_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.3, (120, 140, 120), 1, cv2.LINE_AA)
            cv2.putText(output, f"{amplitude_mm:.1f} mm", (panel_x + 10, metrics_y + 18),
                       cv2.FONT_HERSHEY_DUPLEX, 0.45, (150, 255, 150), 1, cv2.LINE_AA)

            cv2.putText(output, "FREQUENCY", (panel_x + 10, metrics_y + 45),
                       cv2.FONT_HERSHEY_DUPLEX, 0.3, (120, 140, 120), 1, cv2.LINE_AA)
            cv2.putText(output, f"{freq_hz:.1f} Hz", (panel_x + 10, metrics_y + 63),
                       cv2.FONT_HERSHEY_DUPLEX, 0.45, (150, 255, 150), 1, cv2.LINE_AA)

            # Velocidade estimada
            velocity = amplitude_mm * freq_hz * 2  # mm/s
            cv2.putText(output, "VELOCITY", (panel_x + 10, metrics_y + 90),
                       cv2.FONT_HERSHEY_DUPLEX, 0.3, (120, 140, 120), 1, cv2.LINE_AA)
            cv2.putText(output, f"{velocity:.1f} mm/s", (panel_x + 10, metrics_y + 108),
                       cv2.FONT_HERSHEY_DUPLEX, 0.45, (150, 255, 150), 1, cv2.LINE_AA)
        else:
            cv2.putText(output, "Acquiring...", (panel_x + 10, metrics_y + 30),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, (100, 120, 100), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # HEADER PREMIUM
        # ═══════════════════════════════════════

        cv2.rectangle(output, (10, 10), (140, 45), (30, 40, 30), -1)
        cv2.rectangle(output, (10, 10), (140, 45), (0, 200, 200), 1)
        cv2.putText(output, "M-MODE", (20, 33),
                   cv2.FONT_HERSHEY_DUPLEX, 0.55, (100, 255, 255), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # ESCALA DE PROFUNDIDADE
        # ═══════════════════════════════════════

        scale_x = 25
        scan_h = h - self.mmode_timeline_height - 10
        for i in range(0, scan_h, 40):
            depth_mm = i * 0.3
            cv2.line(output, (scale_x - 4, i), (scale_x + 4, i), (0, 150, 150), 1)
            if i % 80 == 0:
                cv2.putText(output, f"{int(depth_mm)}", (scale_x + 6, i + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 150, 150), 1)
        cv2.line(output, (scale_x, 0), (scale_x, scan_h), (0, 100, 100), 1)

        # Instrucao de uso
        cv2.putText(output, "Use <- -> to move cursor", (w//2 - 80, 55),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 150, 150), 1, cv2.LINE_AA)

        return output

    def _process_color_doppler(self, frame):
        """Color Doppler - Sistema avancado de fluxo sanguineo estilo Clarius."""
        output = frame.copy()
        h, w = frame.shape[:2]

        # ═══════════════════════════════════════
        # DEFINIR ROI DOPPLER (box central)
        # ═══════════════════════════════════════

        roi_margin = 0.2
        roi_x1 = int(w * roi_margin)
        roi_y1 = int(h * roi_margin)
        roi_x2 = int(w * (1 - roi_margin))
        roi_y2 = int(h * (1 - roi_margin))

        # Desenhar box do ROI Doppler
        cv2.rectangle(output, (roi_x1, roi_y1), (roi_x2, roi_y2), (200, 200, 50), 1)

        # ═══════════════════════════════════════
        # PROCESSAMENTO DOPPLER (CV simulado)
        # ═══════════════════════════════════════

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]

        # Detectar movimento usando diferenca temporal
        if self.doppler_prev_frame is None:
            self.doppler_prev_frame = roi_gray.copy()

        # Calcular fluxo optico simples
        flow_magnitude = cv2.absdiff(roi_gray, self.doppler_prev_frame)
        self.doppler_prev_frame = roi_gray.copy()

        # Threshold para detectar movimento significativo
        _, flow_mask = cv2.threshold(flow_magnitude, 15, 255, cv2.THRESH_BINARY)

        # Morfologia para limpar
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        flow_mask = cv2.morphologyEx(flow_mask, cv2.MORPH_OPEN, kernel)
        flow_mask = cv2.morphologyEx(flow_mask, cv2.MORPH_CLOSE, kernel)
        flow_mask = cv2.dilate(flow_mask, kernel, iterations=2)

        # Criar mapa de velocidade Doppler (simulado)
        # Velocidade baseada na intensidade do movimento
        velocity_map = flow_magnitude.astype(np.float32) / 255.0

        # Criar mapa de cores Doppler
        # Vermelho = fluxo para cima (away), Azul = fluxo para baixo (toward)
        doppler_colored = np.zeros((roi_y2 - roi_y1, roi_x2 - roi_x1, 3), dtype=np.uint8)

        # Simular direcao baseada na posicao vertical (simplificado)
        for y in range(velocity_map.shape[0]):
            for x in range(velocity_map.shape[1]):
                if flow_mask[y, x] > 0:
                    vel = velocity_map[y, x]
                    # Direcao baseada em gradiente local
                    if y > 0 and roi_gray[y, x] > roi_gray[y-1, x]:
                        # Fluxo "para cima" = vermelho
                        doppler_colored[y, x] = (0, 0, min(255, int(200 * vel + 55)))
                    else:
                        # Fluxo "para baixo" = azul
                        doppler_colored[y, x] = (min(255, int(200 * vel + 55)), 0, 0)

        # Detectar aliasing (velocidades muito altas)
        max_velocity = np.max(velocity_map[flow_mask > 0]) if np.any(flow_mask > 0) else 0
        self.doppler_aliasing_detected = max_velocity > 0.8

        # Aplicar overlay Doppler
        roi_output = output[roi_y1:roi_y2, roi_x1:roi_x2]

        # Blend onde ha fluxo
        alpha = 0.7
        mask_3ch = cv2.cvtColor(flow_mask, cv2.COLOR_GRAY2BGR) / 255.0
        roi_output = (roi_output * (1 - mask_3ch * alpha) + doppler_colored * mask_3ch * alpha).astype(np.uint8)
        output[roi_y1:roi_y2, roi_x1:roi_x2] = roi_output

        # ═══════════════════════════════════════
        # ESCALA DE VELOCIDADE (lado direito)
        # ═══════════════════════════════════════

        scale_x = w - 45
        scale_y1 = roi_y1 + 20
        scale_y2 = roi_y2 - 20
        scale_h = scale_y2 - scale_y1

        # Background da escala
        cv2.rectangle(output, (scale_x - 5, scale_y1 - 10), (scale_x + 30, scale_y2 + 25),
                     (30, 30, 35), -1)

        # Gradiente de cores (vermelho -> preto -> azul)
        for i in range(scale_h):
            ratio = i / scale_h
            if ratio < 0.5:
                # Vermelho (velocidade positiva)
                intensity = int((0.5 - ratio) * 2 * 255)
                color = (0, 0, intensity)
            else:
                # Azul (velocidade negativa)
                intensity = int((ratio - 0.5) * 2 * 255)
                color = (intensity, 0, 0)
            cv2.line(output, (scale_x, scale_y1 + i), (scale_x + 15, scale_y1 + i), color, 1)

        # Labels de velocidade
        cv2.putText(output, f"+{self.doppler_scale}", (scale_x - 3, scale_y1 - 2),
                   cv2.FONT_HERSHEY_DUPLEX, 0.25, (150, 150, 200), 1, cv2.LINE_AA)
        cv2.putText(output, "0", (scale_x + 3, scale_y1 + scale_h // 2 + 4),
                   cv2.FONT_HERSHEY_DUPLEX, 0.25, (150, 150, 200), 1, cv2.LINE_AA)
        cv2.putText(output, f"-{self.doppler_scale}", (scale_x - 3, scale_y2 + 12),
                   cv2.FONT_HERSHEY_DUPLEX, 0.25, (150, 150, 200), 1, cv2.LINE_AA)
        cv2.putText(output, "cm/s", (scale_x - 2, scale_y2 + 22),
                   cv2.FONT_HERSHEY_DUPLEX, 0.2, (100, 100, 130), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # PAINEL DOPPLER (superior direito)
        # ═══════════════════════════════════════

        panel_w = 150
        panel_x = w - panel_w - 10
        panel_y = 10
        panel_h = 120

        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (30, 25, 35), -1)
        cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (100, 100, 255), 1)

        cv2.putText(output, "COLOR DOPPLER", (panel_x + 8, panel_y + 20),
                   cv2.FONT_HERSHEY_DUPLEX, 0.38, (150, 150, 255), 1, cv2.LINE_AA)

        # PRF
        cv2.putText(output, "PRF", (panel_x + 10, panel_y + 45),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 100, 130), 1, cv2.LINE_AA)
        cv2.putText(output, f"{self.doppler_prf} Hz", (panel_x + 50, panel_y + 45),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, (180, 180, 220), 1, cv2.LINE_AA)

        # Scale
        cv2.putText(output, "SCALE", (panel_x + 10, panel_y + 65),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 100, 130), 1, cv2.LINE_AA)
        cv2.putText(output, f"{self.doppler_scale} cm/s", (panel_x + 50, panel_y + 65),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, (180, 180, 220), 1, cv2.LINE_AA)

        # Aliasing warning
        if self.doppler_aliasing_detected:
            cv2.rectangle(output, (panel_x + 5, panel_y + 80), (panel_x + panel_w - 5, panel_y + 100),
                         (0, 80, 150), -1)
            cv2.putText(output, "ALIASING!", (panel_x + 30, panel_y + 94),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, (100, 200, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(output, "No Aliasing", (panel_x + 10, panel_y + 90),
                       cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 150, 100), 1, cv2.LINE_AA)

        # Flow status
        flow_detected = np.sum(flow_mask > 0) > 100
        status_color = (100, 200, 255) if flow_detected else (80, 80, 100)
        cv2.circle(output, (panel_x + 15, panel_y + 108), 4, status_color, -1)
        cv2.putText(output, "FLOW" if flow_detected else "NO FLOW", (panel_x + 25, panel_y + 112),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, status_color, 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # HEADER PREMIUM
        # ═══════════════════════════════════════

        cv2.rectangle(output, (10, 10), (160, 45), (40, 30, 50), -1)
        cv2.rectangle(output, (10, 10), (160, 45), (100, 100, 255), 1)
        cv2.putText(output, "CFM MODE", (20, 33),
                   cv2.FONT_HERSHEY_DUPLEX, 0.55, (180, 180, 255), 1, cv2.LINE_AA)

        # Legenda de cores
        legend_y = h - 40
        cv2.rectangle(output, (10, legend_y), (160, h - 10), (30, 30, 40), -1)
        cv2.rectangle(output, (10, legend_y), (160, h - 10), (100, 100, 120), 1)

        cv2.circle(output, (25, legend_y + 12), 6, (0, 0, 200), -1)
        cv2.putText(output, "Away", (35, legend_y + 16),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (150, 150, 200), 1, cv2.LINE_AA)

        cv2.circle(output, (90, legend_y + 12), 6, (200, 0, 0), -1)
        cv2.putText(output, "Toward", (100, legend_y + 16),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (150, 150, 200), 1, cv2.LINE_AA)

        return output

    def _process_power_doppler(self, frame):
        """Power Doppler - Sistema avancado de alta sensibilidade vascular."""
        output = frame.copy()
        h, w = frame.shape[:2]

        # ═══════════════════════════════════════
        # DEFINIR ROI
        # ═══════════════════════════════════════

        roi_margin = 0.15
        roi_x1 = int(w * roi_margin)
        roi_y1 = int(h * roi_margin)
        roi_x2 = int(w * (1 - roi_margin))
        roi_y2 = int(h * (1 - roi_margin))

        cv2.rectangle(output, (roi_x1, roi_y1), (roi_x2, roi_y2), (50, 150, 200), 1)

        # ═══════════════════════════════════════
        # PROCESSAMENTO POWER DOPPLER
        # ═══════════════════════════════════════

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]

        if self.power_prev_frame is None:
            self.power_prev_frame = roi_gray.copy()

        # Detectar movimento (mais sensivel que color doppler)
        flow_magnitude = cv2.absdiff(roi_gray, self.power_prev_frame)
        self.power_prev_frame = roi_gray.copy()

        # Threshold baixo para alta sensibilidade
        sensitivity_threshold = int((100 - self.power_sensitivity) / 10) + 5
        _, flow_mask = cv2.threshold(flow_magnitude, sensitivity_threshold, 255, cv2.THRESH_BINARY)

        # Morfologia suave
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        flow_mask = cv2.morphologyEx(flow_mask, cv2.MORPH_CLOSE, kernel)
        flow_mask = cv2.dilate(flow_mask, kernel, iterations=3)

        # Criar mapa de potencia (laranja/amarelo - sem direcao)
        power_colored = np.zeros((roi_y2 - roi_y1, roi_x2 - roi_x1, 3), dtype=np.uint8)

        # Normalizar magnitude
        power_map = flow_magnitude.astype(np.float32)
        max_power = np.max(power_map) if np.max(power_map) > 0 else 1
        power_map = power_map / max_power

        # Aplicar colormap laranja/amarelo
        for y in range(power_map.shape[0]):
            for x in range(power_map.shape[1]):
                if flow_mask[y, x] > 0:
                    intensity = power_map[y, x]
                    # Laranja para amarelo baseado na intensidade
                    r = min(255, int(200 + 55 * intensity))
                    g = min(255, int(100 + 155 * intensity))
                    b = 0
                    power_colored[y, x] = (b, g, r)

        # Aplicar overlay
        roi_output = output[roi_y1:roi_y2, roi_x1:roi_x2]
        mask_3ch = cv2.cvtColor(flow_mask, cv2.COLOR_GRAY2BGR) / 255.0
        alpha = 0.75
        roi_output = (roi_output * (1 - mask_3ch * alpha) + power_colored * mask_3ch * alpha).astype(np.uint8)
        output[roi_y1:roi_y2, roi_x1:roi_x2] = roi_output

        # ═══════════════════════════════════════
        # ESCALA DE POTENCIA
        # ═══════════════════════════════════════

        scale_x = w - 40
        scale_y1 = roi_y1 + 30
        scale_y2 = roi_y2 - 30
        scale_h = scale_y2 - scale_y1

        cv2.rectangle(output, (scale_x - 5, scale_y1 - 10), (scale_x + 25, scale_y2 + 20),
                     (30, 30, 35), -1)

        # Gradiente de potencia (preto -> laranja -> amarelo)
        for i in range(scale_h):
            ratio = 1 - (i / scale_h)  # Invertido (mais brilhante em cima)
            r = min(255, int(200 + 55 * ratio))
            g = min(255, int(100 * ratio + 80 * ratio))
            b = 0
            color = (b, g, r)
            cv2.line(output, (scale_x, scale_y1 + i), (scale_x + 12, scale_y1 + i), color, 1)

        cv2.putText(output, "HIGH", (scale_x - 2, scale_y1 - 2),
                   cv2.FONT_HERSHEY_DUPLEX, 0.22, (200, 200, 150), 1, cv2.LINE_AA)
        cv2.putText(output, "LOW", (scale_x - 2, scale_y2 + 12),
                   cv2.FONT_HERSHEY_DUPLEX, 0.22, (150, 150, 100), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # PAINEL POWER DOPPLER
        # ═══════════════════════════════════════

        panel_w = 155
        panel_x = w - panel_w - 10
        panel_y = 10
        panel_h = 110

        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (35, 30, 25), -1)
        cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (100, 180, 255), 1)

        cv2.putText(output, "POWER DOPPLER", (panel_x + 8, panel_y + 20),
                   cv2.FONT_HERSHEY_DUPLEX, 0.38, (150, 200, 255), 1, cv2.LINE_AA)

        # Sensitivity
        cv2.putText(output, "SENSITIVITY", (panel_x + 10, panel_y + 45),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (120, 130, 100), 1, cv2.LINE_AA)
        cv2.putText(output, f"{self.power_sensitivity}%", (panel_x + 90, panel_y + 45),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, (200, 220, 180), 1, cv2.LINE_AA)

        # Barra de sensibilidade
        bar_w = panel_w - 20
        cv2.rectangle(output, (panel_x + 10, panel_y + 55), (panel_x + 10 + bar_w, panel_y + 63),
                     (40, 40, 50), -1)
        fill_w = int(bar_w * self.power_sensitivity / 100)
        cv2.rectangle(output, (panel_x + 10, panel_y + 55), (panel_x + 10 + fill_w, panel_y + 63),
                     (100, 180, 255), -1)

        # Vascular detection count
        vessel_pixels = np.sum(flow_mask > 0)
        vessel_area = vessel_pixels * 0.09  # mm²
        cv2.putText(output, "VASCULAR AREA", (panel_x + 10, panel_y + 82),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (120, 130, 100), 1, cv2.LINE_AA)
        cv2.putText(output, f"{vessel_area:.1f} mm2", (panel_x + 10, panel_y + 98),
                   cv2.FONT_HERSHEY_DUPLEX, 0.38, (200, 220, 180), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # HEADER PREMIUM
        # ═══════════════════════════════════════

        cv2.rectangle(output, (10, 10), (170, 45), (50, 40, 30), -1)
        cv2.rectangle(output, (10, 10), (170, 45), (100, 180, 255), 1)
        cv2.putText(output, "PDI MODE", (20, 33),
                   cv2.FONT_HERSHEY_DUPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)

        # Info
        cv2.putText(output, "No directional info", (10, h - 15),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (120, 150, 180), 1, cv2.LINE_AA)

        return output

    def _process_blines(self, frame):
        """Lung AI - Sistema avancado de analise pulmonar estilo Clarius."""
        output = frame.copy()
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rx0, ry0, rw, rh = 0.15, 0.35, 0.7, 0.55
        x0 = int(w * rx0)
        y0 = int(h * ry0)
        x1 = int(w * (rx0 + rw))
        y1 = int(h * (ry0 + rh))
        tuned, gain_info = self._auto_gain_and_quality(gray, roi=(x0, y0, x1, y1))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(tuned)

        # ═══════════════════════════════════════
        # DETECCAO DA LINHA PLEURAL
        # ═══════════════════════════════════════

        # Linha pleural geralmente no terco superior
        pleura_region = enhanced[h//6:h//3, :]

        # Detectar linha horizontal brilhante (pleura)
        horizontal_kernel = np.array([[1, 1, 1, 1, 1],
                                      [-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]])
        pleura_edges = cv2.filter2D(pleura_region, -1, horizontal_kernel)
        _, pleura_thresh = cv2.threshold(pleura_edges, 100, 255, cv2.THRESH_BINARY)

        pleura_y = h // 4  # Posicao estimada

        # Encontrar linha pleural mais provavel
        lines_h = cv2.HoughLinesP(pleura_thresh, 1, np.pi/180, 50, minLineLength=w//3, maxLineGap=30)
        if lines_h is not None:
            for line in lines_h:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if angle < 15:  # Quase horizontal
                    pleura_y = h//6 + (y1 + y2) // 2
                    break
        pleura_y = int(np.clip(pleura_y, h // 8, h // 2))
        self.lung_pleura_history.append(pleura_y)
        if len(self.lung_pleura_history) >= 3:
            pleura_y = int(np.median(self.lung_pleura_history))

        # Desenhar linha pleural com glow
        cv2.line(output, (50, pleura_y), (w - 50, pleura_y), (80, 100, 80), 3)
        cv2.line(output, (50, pleura_y), (w - 50, pleura_y), (150, 200, 150), 2)
        cv2.line(output, (50, pleura_y), (w - 50, pleura_y), (200, 255, 200), 1)

        # Label pleura
        cv2.putText(output, "PLEURA", (55, pleura_y - 8),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (150, 200, 150), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # DETECCAO DE A-LINES (linhas horizontais)
        # ═══════════════════════════════════════

        a_line_count = 0
        below_pleura = enhanced[pleura_y:, :]

        # Filtro para detectar linhas horizontais
        h_kernel = np.array([[-1, -1, -1],
                             [2, 2, 2],
                             [-1, -1, -1]])
        a_edges = cv2.filter2D(below_pleura, -1, h_kernel)
        _, a_thresh = cv2.threshold(a_edges, 80, 255, cv2.THRESH_BINARY)

        # Encontrar A-lines
        a_lines = cv2.HoughLinesP(a_thresh, 1, np.pi/180, 40, minLineLength=w//4, maxLineGap=50)
        if a_lines is not None:
            drawn_y = []
            for line in a_lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if angle < 10:  # Horizontal
                    actual_y = pleura_y + (y1 + y2) // 2
                    # Evitar linhas muito proximas
                    if all(abs(actual_y - dy) > 30 for dy in drawn_y):
                        drawn_y.append(actual_y)
                        a_line_count += 1

                        # Desenhar A-line com transparencia
                        overlay = output.copy()
                        cv2.line(overlay, (80, actual_y), (w - 80, actual_y), (0, 180, 180), 1)
                        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)

                        # Label pequeno
                        cv2.putText(output, "A", (w - 75, actual_y + 4),
                                   cv2.FONT_HERSHEY_DUPLEX, 0.25, (0, 200, 200), 1, cv2.LINE_AA)

                        if a_line_count >= 4:
                            break

        self.lung_a_lines_detected = a_line_count >= 2

        # ═══════════════════════════════════════
        # DETECCAO DE B-LINES (linhas verticais)
        # ═══════════════════════════════════════
        b_line_count = 0
        b_line_positions = []
        b_line_density = 0.0

        if below_pleura.shape[0] > 20:
            # Realcar linhas verticais brilhantes
            v_sobel = cv2.Sobel(below_pleura, cv2.CV_32F, 1, 0, ksize=3)
            v_sobel = np.abs(v_sobel)
            v_sobel = cv2.normalize(v_sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            bright_thresh = np.percentile(below_pleura, 88)
            bright_mask = (below_pleura >= bright_thresh).astype(np.uint8) * 255
            v_resp = cv2.bitwise_and(v_sobel, bright_mask)

            # Conectar estruturas verticais
            k = max(15, int(below_pleura.shape[0] * 0.35))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
            v_resp = cv2.morphologyEx(v_resp, cv2.MORPH_CLOSE, v_kernel)

            nonzero = v_resp[v_resp > 0]
            thr = int(np.percentile(nonzero, 60)) if nonzero.size else 0
            thr = max(20, thr)
            _, v_bin = cv2.threshold(v_resp, thr, 255, cv2.THRESH_BINARY)

            col_sum = np.sum(v_bin > 0, axis=0)
            min_len = int(below_pleura.shape[0] * 0.35)
            col_mask = col_sum >= min_len

            clusters = []
            i = 0
            while i < len(col_mask):
                if col_mask[i]:
                    start = i
                    while i < len(col_mask) and col_mask[i]:
                        i += 1
                    end = i - 1
                    clusters.append([start, end])
                i += 1

            merged = []
            for start, end in clusters:
                if not merged or start - merged[-1][1] > 6:
                    merged.append([start, end])
                else:
                    merged[-1][1] = end
            clusters = merged

            total_width = sum((end - start + 1) for start, end in clusters)
            b_line_density = total_width / max(1, len(col_mask))
            b_line_count = min(10, max(len(clusters), int(round(b_line_density * 10))))

            for start, end in clusters[:10]:
                x_center = (start + end) // 2
                col = v_bin[:, x_center] > 0
                ys = np.where(col)[0]
                if ys.size == 0:
                    continue
                actual_y1 = pleura_y + int(ys[0])
                actual_y2 = pleura_y + int(ys[-1])
                if actual_y2 - actual_y1 < min_len * 0.6:
                    continue

                b_line_positions.append(x_center)

                # Glow
                cv2.line(output, (x_center, actual_y1), (x_center, actual_y2),
                        (0, 80, 120), 4)

                # Gradiente de intensidade
                for t in range(0, 100, 5):
                    ratio = t / 100.0
                    py1 = int(actual_y1 + (actual_y2 - actual_y1) * ratio)
                    py2 = int(actual_y1 + (actual_y2 - actual_y1) * (ratio + 0.05))
                    intensity = int(150 + 105 * ratio)
                    cv2.line(output, (x_center, py1), (x_center, py2),
                            (0, intensity, 255), 2)

                cv2.putText(output, "B", (x_center - 5, actual_y1 - 5),
                           cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 200, 255), 1, cv2.LINE_AA)

        # Fallback com Hough se nao detectar nada
        if b_line_count == 0:
            v_kernel = np.array([[-1, 2, -1]])
            b_edges = cv2.filter2D(below_pleura, -1, v_kernel)
            _, b_thresh = cv2.threshold(b_edges, 100, 255, cv2.THRESH_BINARY)
            b_lines = cv2.HoughLinesP(
                b_thresh,
                1,
                np.pi / 180,
                30,
                minLineLength=(h - pleura_y) // 3,
                maxLineGap=20
            )

            if b_lines is not None:
                for line in b_lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                    if 75 < angle < 105:
                        x_center = (x1 + x2) // 2
                        if all(abs(x_center - px) > 30 for px in b_line_positions):
                            b_line_positions.append(x_center)
                            b_line_count += 1

                            actual_y1 = pleura_y + min(y1, y2)
                            actual_y2 = pleura_y + max(y1, y2)
                            cv2.line(output, (x_center, actual_y1), (x_center, actual_y2),
                                    (0, 80, 120), 4)
                            cv2.line(output, (x_center, actual_y1), (x_center, actual_y2),
                                    (0, 200, 255), 2)
                            cv2.putText(output, "B", (x_center - 5, actual_y1 - 5),
                                       cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 200, 255), 1, cv2.LINE_AA)

                            if b_line_count >= 8:
                                break

        self.b_line_count = b_line_count
        self.lung_b_line_history.append(b_line_count)
        self.lung_b_line_density_history.append(b_line_density)

        # ═══════════════════════════════════════
        # ANALISE E CLASSIFICACAO
        # ═══════════════════════════════════════

        avg_b_lines = np.mean(list(self.lung_b_line_history)) if self.lung_b_line_history else 0
        avg_density = np.mean(list(self.lung_b_line_density_history)) if self.lung_b_line_density_history else 0

        # ═══════════════════════════════════════
        # GATING POR SCAN QUALITY (reduz falsos positivos)
        # ═══════════════════════════════════════
        scan_quality = gain_info.get('quality', 50.0) if gain_info else 50.0
        quality_threshold = 35.0
        low_quality_mode = scan_quality < quality_threshold

        if low_quality_mode:
            # Baixa qualidade - nao atualizar contagem, usar ultimo valor valido
            severity_score = getattr(self, '_last_lung_severity', 0)
        else:
            severity_score = max(avg_b_lines, avg_density * 10)
            self._last_lung_severity = severity_score

        display_b_lines = int(np.clip(round(severity_score), 0, 10))
        self.b_line_count = display_b_lines

        if severity_score <= 2:
            profile = "A-PROFILE"
            status = "NORMAL"
            status_color = (100, 255, 100)
            interpretation = "Normal lung pattern"
        elif severity_score <= 5:
            profile = "B-PROFILE"
            status = "MILD"
            status_color = (0, 255, 255)
            interpretation = "Interstitial syndrome"
        elif severity_score <= 8:
            profile = "B-PROFILE"
            status = "MODERATE"
            status_color = (0, 100, 255)
            interpretation = "Pulmonary edema likely"
        else:
            profile = "B-PROFILE"
            status = "SEVERE"
            status_color = (0, 0, 255)
            interpretation = "Severe pulmonary edema"

        # ═══════════════════════════════════════
        # PAINEL PRINCIPAL (usando UI Utils)
        # ═══════════════════════════════════════

        panel_w = 175
        panel_x = w - panel_w - 10
        panel_y = 10
        panel_h = 220

        # Painel com header (tema verde para lung)
        lung_border = (150, 200, 100)
        content_y = draw_panel(output, panel_x, panel_y, panel_w, panel_h,
                               title="LUNG AI",
                               border_color=lung_border,
                               title_color=(180, 230, 150))

        # Profile type no header
        cv2.putText(output, profile, (panel_x + 90, panel_y + 18),
                   FONT, FONT_SCALE_SMALL, status_color, 1, cv2.LINE_AA)

        # B-Lines
        bl_y = panel_y + 55
        cv2.putText(output, "B-LINES", (panel_x + 10, bl_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (120, 140, 100), 1, cv2.LINE_AA)
        cv2.putText(output, str(display_b_lines), (panel_x + 80, bl_y + 25),
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, status_color, 2, cv2.LINE_AA)

        # A-Lines
        al_y = panel_y + 105
        cv2.putText(output, "A-LINES", (panel_x + 10, al_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (120, 140, 100), 1, cv2.LINE_AA)
        a_status = "Present" if self.lung_a_lines_detected else "Absent"
        a_color = (100, 255, 200) if self.lung_a_lines_detected else (100, 100, 120)
        cv2.putText(output, a_status, (panel_x + 80, al_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, a_color, 1, cv2.LINE_AA)

        # Pleura sliding
        ps_y = panel_y + 130
        cv2.putText(output, "SLIDING", (panel_x + 10, ps_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (120, 140, 100), 1, cv2.LINE_AA)
        slide_status = "Normal" if self.lung_pleura_sliding else "Absent"
        slide_color = (100, 255, 100) if self.lung_pleura_sliding else (0, 100, 255)
        cv2.putText(output, slide_status, (panel_x + 80, ps_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, slide_color, 1, cv2.LINE_AA)

        # Status
        st_y = panel_y + 160
        cv2.rectangle(output, (panel_x + 5, st_y - 5), (panel_x + panel_w - 5, st_y + 20),
                     (status_color[0]//4, status_color[1]//4, status_color[2]//4), -1)
        cv2.putText(output, status, (panel_x + 10, st_y + 12),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, status_color, 1, cv2.LINE_AA)

        # Interpretation
        cv2.putText(output, interpretation, (panel_x + 10, panel_y + 200),
                   cv2.FONT_HERSHEY_DUPLEX, 0.25, (140, 160, 130), 1, cv2.LINE_AA)

        # B-line scoring bar
        bar_y = panel_y + 210
        bar_w = panel_w - 20
        cv2.rectangle(output, (panel_x + 10, bar_y), (panel_x + 10 + bar_w, bar_y + 6),
                     (40, 50, 35), -1)

        # Gradiente de severidade
        for i, (start, end, color) in enumerate([(0, 2, (80, 200, 80)), (2, 5, (0, 200, 200)), (5, 10, (0, 100, 255))]):
            if display_b_lines > start:
                x1 = panel_x + 10 + int(bar_w * start / 10)
                x2 = panel_x + 10 + int(bar_w * min(display_b_lines, end) / 10)
                cv2.rectangle(output, (x1, bar_y), (x2, bar_y + 6), color, -1)

        # ═══════════════════════════════════════
        # HEADER PREMIUM
        # ═══════════════════════════════════════

        cv2.rectangle(output, (10, 10), (150, 45), (40, 50, 30), -1)
        cv2.rectangle(output, (10, 10), (150, 45), (150, 200, 100), 1)
        cv2.putText(output, "LUS MODE", (20, 33),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (180, 230, 160), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # LEGENDA (usando UI Utils)
        # ═══════════════════════════════════════

        legend_x = 10
        legend_y = h - 70
        legend_items = [
            ("Pleura", (200, 255, 200)),
            ("A-lines", (0, 200, 200)),
            ("B-lines", (0, 180, 255)),
        ]

        # Mini painel para legenda
        draw_panel(output, legend_x, legend_y, 100, 55,
                  title="LEGEND", border_color=(100, 120, 80),
                  title_color=(100, 120, 100), header_height=18)
        draw_legend(output, legend_x + 5, legend_y + 25, legend_items)

        # Indicador de baixa qualidade
        if low_quality_mode:
            draw_warning_banner(output, "LOW QUALITY - ADJUST", y=95,
                              color=(180, 220, 180))

        # ═══════════════════════════════════════
        # PAINEL DE ZONAS (Protocolo BLUE)
        # ═══════════════════════════════════════

        # Inicializar timer da zona se necessário
        now = time.time()
        if self.lung_zone_start_time is None:
            self.lung_zone_start_time = now
            self.lung_exam_start_time = now

        # Atualizar status da zona atual
        self.lung_zones[self.lung_current_zone]['status'] = 'scanning'

        zone_panel_x = 10
        zone_panel_y = legend_y - 180
        zone_panel_w = 130
        zone_panel_h = 170

        overlay = output.copy()
        cv2.rectangle(overlay, (zone_panel_x, zone_panel_y),
                     (zone_panel_x + zone_panel_w, zone_panel_y + zone_panel_h),
                     (30, 40, 35), -1)
        cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)
        cv2.rectangle(output, (zone_panel_x, zone_panel_y),
                     (zone_panel_x + zone_panel_w, zone_panel_y + zone_panel_h),
                     (100, 150, 120), 1)

        cv2.putText(output, "LUNG ZONES", (zone_panel_x + 5, zone_panel_y + 15),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (150, 200, 150), 1, cv2.LINE_AA)

        # Mostrar zonas em 2 colunas (R e L)
        col_w = zone_panel_w // 2
        for col, side in enumerate(['R', 'L']):
            for row in range(4):
                zone_key = f"{side}{row + 1}"
                zone_data = self.lung_zones[zone_key]
                zx = zone_panel_x + 5 + col * col_w
                zy = zone_panel_y + 30 + row * 18

                # Cor baseada no status
                if zone_data['status'] == 'checked':
                    if zone_data['profile'] == 'B':
                        zone_color = (0, 180, 255)  # B-profile: laranja
                    else:
                        zone_color = (100, 200, 100)  # A-profile: verde
                    marker = "+"
                elif zone_key == self.lung_current_zone:
                    zone_color = (0, 255, 255)  # Atual: amarelo
                    marker = ">"
                else:
                    zone_color = (80, 80, 100)  # Pendente: cinza
                    marker = " "

                cv2.putText(output, f"{marker}{zone_key}", (zx, zy),
                           cv2.FONT_HERSHEY_DUPLEX, 0.28, zone_color, 1, cv2.LINE_AA)

        # Contagem de zonas checadas
        checked_count = sum(1 for z in self.lung_zones.values() if z['status'] == 'checked')
        progress_text = f"{checked_count}/8 ZONES"
        cv2.putText(output, progress_text, (zone_panel_x + 5, zone_panel_y + zone_panel_h - 25),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (140, 180, 140), 1, cv2.LINE_AA)

        # Timer da zona atual
        zone_elapsed = now - self.lung_zone_start_time if self.lung_zone_start_time else 0
        timer_text = f"{self.lung_current_zone}: {zone_elapsed:.1f}s"
        cv2.putText(output, timer_text, (zone_panel_x + 5, zone_panel_y + zone_panel_h - 10),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (0, 255, 255), 1, cv2.LINE_AA)

        # EXAM COMPLETE banner
        if self.lung_exam_complete:
            banner_y = h // 2 - 30
            overlay = output.copy()
            cv2.rectangle(overlay, (w//2 - 120, banner_y), (w//2 + 120, banner_y + 60),
                         (40, 80, 40), -1)
            cv2.addWeighted(overlay, 0.9, output, 0.1, 0, output)
            cv2.rectangle(output, (w//2 - 120, banner_y), (w//2 + 120, banner_y + 60),
                         (100, 255, 100), 2)
            cv2.putText(output, "LUNG EXAM COMPLETE", (w//2 - 100, banner_y + 25),
                       cv2.FONT_HERSHEY_DUPLEX, 0.55, (100, 255, 100), 1, cv2.LINE_AA)

            # Resumo
            b_zones = sum(1 for z in self.lung_zones.values() if z['profile'] == 'B')
            summary = f"A-profile: {8 - b_zones} | B-profile: {b_zones}"
            cv2.putText(output, summary, (w//2 - 80, banner_y + 45),
                       cv2.FONT_HERSHEY_DUPLEX, 0.35, (180, 220, 180), 1, cv2.LINE_AA)

        # Instruções de navegação
        nav_text = "< > Navigate | SPACE Confirm"
        cv2.putText(output, nav_text, (zone_panel_x + 5, zone_panel_y + zone_panel_h - 40),
                   cv2.FONT_HERSHEY_DUPLEX, 0.2, (100, 120, 100), 1, cv2.LINE_AA)

        self._draw_auto_gain_status(output, 20, 60, gain_info, color=(170, 200, 140))

        return output

    def _process_bladder(self, frame):
        """Bladder AI - Sistema avancado de volume vesical estilo Clarius."""
        output = frame.copy()
        h, w = frame.shape[:2]

        model = self.models.get('bladder')

        mask_found = False
        contour_to_draw = None
        bladder_center = None
        best_score = 0.0

        # ═══════════════════════════════════════
        # DETECCAO DA BEXIGA
        # ═══════════════════════════════════════

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ROI para bexiga (regiao central-inferior)
        x0, y0 = int(w * 0.2), int(h * 0.35)
        x1, y1 = int(w * 0.8), int(h * 0.9)
        tuned, gain_info = self._auto_gain_and_quality(gray, roi=(x0, y0, x1, y1))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(tuned)

        # ═══════════════════════════════════════
        # GATING POR SCAN QUALITY (reduz falsos positivos)
        # ═══════════════════════════════════════
        scan_quality = gain_info.get('quality', 50.0) if gain_info else 50.0
        quality_threshold = 35.0
        low_quality_mode = scan_quality < quality_threshold

        if model:
            try:
                resized = cv2.resize(gray, (224, 224))
                tensor = torch.from_numpy(resized).float() / 255.0
                tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    mask = model(tensor).cpu().numpy()[0, 0]

                mask = cv2.resize(mask, (w, h))
                binary = (mask > 0.5).astype(np.uint8)

                if binary.any():
                    # Suavizar mascara do modelo
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(largest) > 1000:
                            mask_found = True
                            contour_to_draw = largest

            except Exception as e:
                print(f"Erro Bladder AI: {e}")

        # Fallback CV
        if not mask_found:
            # Bexiga tipicamente anecoica (escura) - threshold adaptativo em ROI
            roi = enhanced[y0:y1, x0:x1]
            if roi.size == 0:
                roi = enhanced
                x0, y0, x1, y1 = 0, 0, w, h

            thresh_val = int(np.clip(np.percentile(roi, 20), 25, 80))
            _, thresh = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY_INV)

            # Focar no ROI para reduzir falsos positivos
            roi_mask = np.zeros_like(thresh)
            roi_mask[y0:y1, x0:x1] = 255
            thresh = cv2.bitwise_and(thresh, roi_mask)

            # Morfologia para limpar
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Preencher buracos
            h_mask, w_mask = thresh.shape[:2]
            flood = thresh.copy()
            cv2.floodFill(flood, np.zeros((h_mask + 2, w_mask + 2), np.uint8), (0, 0), 255)
            flood_inv = cv2.bitwise_not(flood)
            thresh = cv2.bitwise_or(thresh, flood_inv)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Encontrar contorno mais provavel (grande e eliptico)
                best_contour = None
                best_score = 0.0
                roi_mean = float(np.mean(roi))
                roi_area = max(1.0, (x1 - x0) * (y1 - y0))

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < roi_area * 0.01 or area > roi_area * 0.5:
                        continue

                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if cw < 20 or ch < 20:
                        continue

                    cx = x + cw // 2
                    cy = y + ch // 2
                    if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                        continue

                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0

                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0.0

                    mask = np.zeros_like(enhanced, dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    mean_inside = cv2.mean(enhanced, mask=mask)[0]
                    contrast = max(0.0, roi_mean - mean_inside)
                    if contrast < 10:
                        continue

                    area_score = min(area / (0.2 * roi_area), 1.0)
                    circ_score = min(max((circularity - 0.2) / 0.6, 0.0), 1.0)
                    solidity_score = min(max((solidity - 0.6) / 0.4, 0.0), 1.0)
                    contrast_score = min(contrast / 50.0, 1.0)

                    score = (
                        0.4 * contrast_score
                        + 0.25 * area_score
                        + 0.2 * solidity_score
                        + 0.15 * circ_score
                    )

                    if score > best_score:
                        best_score = score
                        best_contour = cnt

                if best_contour is not None:
                    contour_to_draw = best_contour

        # ═══════════════════════════════════════
        # CALCULO DE DIMENSOES E VOLUME
        # ═══════════════════════════════════════

        if contour_to_draw is not None:
            # Calcular qualidade mesmo com modelo
            if best_score <= 0:
                perimeter = cv2.arcLength(contour_to_draw, True)
                area = cv2.contourArea(contour_to_draw)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
                hull = cv2.convexHull(contour_to_draw)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0.0
                best_score = min(1.0, max(0.2, circularity + solidity) / 2.0)

        detection_quality_threshold = 0.45
        self.bladder_quality = float(np.clip(best_score, 0.0, 1.0))

        # Se qualidade de imagem estiver baixa, nao atualizar deteccao
        if low_quality_mode:
            contour_to_draw = None  # Ignora deteccao

        if contour_to_draw is not None and self.bladder_quality >= detection_quality_threshold:
            area = cv2.contourArea(contour_to_draw)

            # Retangulo rotacionado para dimensoes
            rect = cv2.minAreaRect(contour_to_draw)
            (cx, cy), (width, height), angle = rect
            bladder_center = (int(cx), int(cy))

            # Garantir que width > height
            if width < height:
                width, height = height, width

            px_to_mm = 0.3
            major_mm = width * px_to_mm
            minor_mm = height * px_to_mm

            aspect = width / (height + 1e-6)
            view = "transverse" if aspect < 1.35 else "sagittal"
            self.bladder_view = view
            self.bladder_view_history[view].append((major_mm, minor_mm))

            view_hist = self.bladder_view_history[view]
            majors = [m[0] for m in view_hist]
            minors = [m[1] for m in view_hist]
            major_mm = float(np.median(majors)) if majors else major_mm
            minor_mm = float(np.median(minors)) if minors else minor_mm

            self.bladder_view_data[view] = {
                'major': major_mm,
                'minor': minor_mm,
                't': now,
            }

            trans = self.bladder_view_data['transverse']
            sag = self.bladder_view_data['sagittal']
            use_dual = (now - trans['t'] < 8.0) and (now - sag['t'] < 8.0)

            if use_dual and trans['major'] and sag['major']:
                self.bladder_d1 = trans['major']  # largura LR
                self.bladder_d2 = sag['major']    # altura CC
                d3_vals = [v for v in [trans['minor'], sag['minor']] if v]
                self.bladder_d3 = float(np.mean(d3_vals)) if d3_vals else (self.bladder_d1 + self.bladder_d2) / 2
            else:
                self.bladder_d1 = major_mm
                self.bladder_d2 = minor_mm
                self.bladder_d3 = (self.bladder_d1 + self.bladder_d2) / 2

            # Formula do elipsoide: V = 0.52 x D1 x D2 x D3
            volume_ml = 0.52 * self.bladder_d1 * self.bladder_d2 * self.bladder_d3 / 1000
            volume_ml = max(0, min(1000, volume_ml))

            self.bladder_history.append(volume_ml)
            if len(self.bladder_history) >= 3:
                volume_ml = float(np.median(self.bladder_history))
            self.bladder_volume = volume_ml
            self.bladder_last_update = now

            # ═══════════════════════════════════════
            # VISUALIZACAO DA BEXIGA
            # ═══════════════════════════════════════

            # Glow effect
            cv2.drawContours(output, [contour_to_draw], -1, (100, 50, 100), 5)
            cv2.drawContours(output, [contour_to_draw], -1, (180, 100, 180), 3)
            cv2.drawContours(output, [contour_to_draw], -1, (220, 150, 255), 2)

            # Preenchimento semi-transparente
            overlay = output.copy()
            cv2.drawContours(overlay, [contour_to_draw], -1, (200, 120, 230), -1)
            cv2.addWeighted(overlay, 0.2, output, 0.8, 0, output)

            # Desenhar eixos de medicao
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Eixo maior (D1)
            mid1 = ((box[0][0] + box[1][0])//2, (box[0][1] + box[1][1])//2)
            mid2 = ((box[2][0] + box[3][0])//2, (box[2][1] + box[3][1])//2)
            cv2.line(output, mid1, mid2, (255, 200, 100), 1, cv2.LINE_AA)

            # Eixo menor (D2)
            mid3 = ((box[1][0] + box[2][0])//2, (box[1][1] + box[2][1])//2)
            mid4 = ((box[0][0] + box[3][0])//2, (box[0][1] + box[3][1])//2)
            cv2.line(output, mid3, mid4, (100, 200, 255), 1, cv2.LINE_AA)

            # Labels de dimensao
            d1_mid = ((mid1[0] + mid2[0])//2, (mid1[1] + mid2[1])//2)
            cv2.putText(output, f"D1:{self.bladder_d1:.0f}mm", (d1_mid[0] + 5, d1_mid[1] - 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 200, 100), 1, cv2.LINE_AA)

            d2_mid = ((mid3[0] + mid4[0])//2, (mid3[1] + mid4[1])//2)
            cv2.putText(output, f"D2:{self.bladder_d2:.0f}mm", (d2_mid[0] + 5, d2_mid[1] + 12),
                       cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 200, 255), 1, cv2.LINE_AA)

            # Centro
            cv2.circle(output, bladder_center, 4, (255, 255, 255), -1)
            cv2.circle(output, bladder_center, 6, (200, 150, 255), 1)
        else:
            if now - self.bladder_last_update > 2.5:
                self.bladder_volume = None
                self.bladder_d1 = None
                self.bladder_d2 = None
                self.bladder_d3 = None
                self.bladder_view = None
            self.bladder_quality = max(0.0, self.bladder_quality - 0.05)

        # ═══════════════════════════════════════
        # PAINEL PRINCIPAL
        # ═══════════════════════════════════════

        panel_w = 170
        panel_x = w - panel_w - 10
        panel_y = 10
        panel_h = 220

        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (35, 25, 40), -1)
        cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (200, 120, 255), 1)

        cv2.putText(output, "BLADDER AI", (panel_x + 10, panel_y + 22),
                   cv2.FONT_HERSHEY_DUPLEX, 0.48, (230, 180, 255), 1, cv2.LINE_AA)

        # Volume
        vol_y = panel_y + 55
        cv2.putText(output, "VOLUME", (panel_x + 10, vol_y - 10),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (140, 120, 160), 1, cv2.LINE_AA)

        if self.bladder_volume is not None:
            vol_text = f"{int(self.bladder_volume)}"
            cv2.putText(output, vol_text, (panel_x + 15, vol_y + 25),
                       cv2.FONT_HERSHEY_DUPLEX, 1.3, (230, 180, 255), 2, cv2.LINE_AA)
            cv2.putText(output, "mL", (panel_x + 100, vol_y + 25),
                       cv2.FONT_HERSHEY_DUPLEX, 0.45, (160, 130, 200), 1, cv2.LINE_AA)

            # Status
            if self.bladder_volume < 50:
                status = "EMPTY"
                status_color = (120, 120, 150)
            elif self.bladder_volume < 150:
                status = "LOW"
                status_color = (150, 200, 150)
            elif self.bladder_volume < 350:
                status = "NORMAL"
                status_color = (100, 255, 100)
            elif self.bladder_volume < 500:
                status = "FULL"
                status_color = (0, 255, 255)
            else:
                status = "DISTENDED"
                status_color = (0, 100, 255)

            cv2.putText(output, status, (panel_x + 10, vol_y + 50),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, status_color, 1, cv2.LINE_AA)

            # Barra de volume
            bar_y = vol_y + 60
            bar_w = panel_w - 20
            cv2.rectangle(output, (panel_x + 10, bar_y), (panel_x + 10 + bar_w, bar_y + 8),
                         (40, 35, 50), -1)

            fill_w = int(bar_w * min(self.bladder_volume, 600) / 600)
            cv2.rectangle(output, (panel_x + 10, bar_y), (panel_x + 10 + fill_w, bar_y + 8),
                         status_color, -1)

            # Marcadores
            for mark_ml, label in [(150, "150"), (350, "350")]:
                mx = panel_x + 10 + int(bar_w * mark_ml / 600)
                cv2.line(output, (mx, bar_y), (mx, bar_y + 8), (80, 80, 100), 1)

        else:
            cv2.putText(output, "---", (panel_x + 40, vol_y + 25),
                       cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 90, 120), 2, cv2.LINE_AA)
            cv2.putText(output, "SCANNING", (panel_x + 10, vol_y + 50),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, (100, 100, 130), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # DIMENSOES
        # ═══════════════════════════════════════

        dim_y = panel_y + 140
        cv2.putText(output, "DIMENSIONS", (panel_x + 10, dim_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (120, 110, 140), 1, cv2.LINE_AA)

        if self.bladder_d1 is not None:
            cv2.putText(output, f"D1: {self.bladder_d1:.0f}mm", (panel_x + 10, dim_y + 18),
                       cv2.FONT_HERSHEY_DUPLEX, 0.3, (200, 180, 150), 1, cv2.LINE_AA)
            cv2.putText(output, f"D2: {self.bladder_d2:.0f}mm", (panel_x + 85, dim_y + 18),
                       cv2.FONT_HERSHEY_DUPLEX, 0.3, (150, 180, 220), 1, cv2.LINE_AA)
            cv2.putText(output, f"D3: {self.bladder_d3:.0f}mm (est)", (panel_x + 10, dim_y + 35),
                       cv2.FONT_HERSHEY_DUPLEX, 0.28, (140, 140, 160), 1, cv2.LINE_AA)

        # Formula
        cv2.putText(output, "V = 0.52 x D1 x D2 x D3", (panel_x + 10, dim_y + 55),
                   cv2.FONT_HERSHEY_DUPLEX, 0.25, (100, 90, 120), 1, cv2.LINE_AA)

        # View + qualidade
        view_label = self.bladder_view.upper() if self.bladder_view else "---"
        if self.bladder_view == "transverse":
            view_label = "TRANS"
        elif self.bladder_view == "sagittal":
            view_label = "SAG"
        if self.bladder_view_data['transverse']['t'] and self.bladder_view_data['sagittal']['t']:
            if now - self.bladder_view_data['transverse']['t'] < 8.0 and now - self.bladder_view_data['sagittal']['t'] < 8.0:
                view_label = "DUAL"

        qual_pct = int(self.bladder_quality * 100)
        cv2.putText(output, f"VIEW: {view_label}", (panel_x + 10, dim_y + 70),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (130, 120, 150), 1, cv2.LINE_AA)
        cv2.putText(output, f"QUALITY: {qual_pct}%", (panel_x + 10, dim_y + 88),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (130, 120, 150), 1, cv2.LINE_AA)

        # PVR indicator
        pvr_y = panel_y + panel_h - 15
        if self.bladder_pvr_mode and self.bladder_pre_void:
            pvr = self.bladder_volume if self.bladder_volume else 0
            cv2.putText(output, f"PVR: {int(pvr)}mL", (panel_x + 10, pvr_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.32, (200, 200, 150), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # HEADER PREMIUM
        # ═══════════════════════════════════════

        cv2.rectangle(output, (10, 10), (160, 45), (50, 30, 55), -1)
        cv2.rectangle(output, (10, 10), (160, 45), (200, 120, 255), 1)
        cv2.putText(output, "BVI MODE", (20, 33),
                   cv2.FONT_HERSHEY_DUPLEX, 0.55, (230, 180, 255), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # INSTRUCOES E LEGENDA
        # ═══════════════════════════════════════

        legend_x = 10
        legend_y = h - 65
        legend_w = 140
        legend_h = 55

        overlay = output.copy()
        cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h),
                     (35, 30, 40), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
        cv2.rectangle(output, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h),
                     (120, 100, 140), 1)

        cv2.putText(output, "MEASUREMENT", (legend_x + 5, legend_y + 15),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (120, 100, 140), 1, cv2.LINE_AA)

        cv2.line(output, (legend_x + 10, legend_y + 28), (legend_x + 35, legend_y + 28), (255, 200, 100), 2)
        cv2.putText(output, "D1 (Width)", (legend_x + 40, legend_y + 32),
                   cv2.FONT_HERSHEY_DUPLEX, 0.25, (255, 200, 100), 1, cv2.LINE_AA)

        cv2.line(output, (legend_x + 10, legend_y + 42), (legend_x + 35, legend_y + 42), (100, 200, 255), 2)
        cv2.putText(output, "D2 (Height)", (legend_x + 40, legend_y + 46),
                   cv2.FONT_HERSHEY_DUPLEX, 0.25, (100, 200, 255), 1, cv2.LINE_AA)

        # Indicador de baixa qualidade
        if low_quality_mode:
            alert_y = 85
            cv2.rectangle(output, (w//2 - 90, alert_y), (w//2 + 90, alert_y + 28),
                         (50, 30, 50), -1)
            cv2.rectangle(output, (w//2 - 90, alert_y), (w//2 + 90, alert_y + 28),
                         (150, 100, 150), 2)
            cv2.putText(output, "LOW QUALITY - ADJUST", (w//2 - 78, alert_y + 20),
                       cv2.FONT_HERSHEY_DUPLEX, 0.38, (220, 180, 220), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # GUIA DE POSICIONAMENTO (Dual-View)
        # ═══════════════════════════════════════
        now = time.time()
        trans_valid = self.bladder_view_data['transverse']['t'] and (now - self.bladder_view_data['transverse']['t'] < 10.0)
        sag_valid = self.bladder_view_data['sagittal']['t'] and (now - self.bladder_view_data['sagittal']['t'] < 10.0)

        # Mostrar hint apenas se uma view foi capturada mas a outra não
        if (trans_valid and not sag_valid) or (sag_valid and not trans_valid):
            hint_y = 60
            needed_view = "SAGITTAL" if trans_valid else "TRANSVERSE"
            hint_text = f"Rotate probe for {needed_view} view"

            overlay = output.copy()
            cv2.rectangle(overlay, (w//2 - 130, hint_y), (w//2 + 130, hint_y + 35),
                         (50, 40, 60), -1)
            cv2.addWeighted(overlay, 0.9, output, 0.1, 0, output)
            cv2.rectangle(output, (w//2 - 130, hint_y), (w//2 + 130, hint_y + 35),
                         (180, 140, 200), 1)

            # Ícone de rotação
            cx = w//2 - 115
            cy = hint_y + 18
            cv2.ellipse(output, (cx, cy), (8, 8), 0, 30, 330, (0, 255, 255), 2)
            # Seta da rotação
            cv2.line(output, (cx + 6, cy - 5), (cx + 10, cy - 8), (0, 255, 255), 2)
            cv2.line(output, (cx + 6, cy - 5), (cx + 10, cy - 2), (0, 255, 255), 2)

            cv2.putText(output, hint_text, (w//2 - 95, hint_y + 23),
                       cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

        # Indicador de dual-view ativa
        elif trans_valid and sag_valid:
            dual_y = 60
            overlay = output.copy()
            cv2.rectangle(overlay, (w//2 - 80, dual_y), (w//2 + 80, dual_y + 28),
                         (40, 60, 40), -1)
            cv2.addWeighted(overlay, 0.9, output, 0.1, 0, output)
            cv2.rectangle(output, (w//2 - 80, dual_y), (w//2 + 80, dual_y + 28),
                         (100, 200, 100), 1)
            cv2.putText(output, "DUAL-VIEW ACTIVE", (w//2 - 65, dual_y + 19),
                       cv2.FONT_HERSHEY_DUPLEX, 0.38, (100, 255, 100), 1, cv2.LINE_AA)

        self._draw_auto_gain_status(output, 20, 60, gain_info, color=(200, 160, 220))

        return output

    def get_stats(self):
        """Retorna estatisticas de performance."""
        return {
            'mode': self.active_mode,
            'inference_time_ms': self.last_inference_time,
            'inference_count': self.inference_count,
            'device': str(self.device),
        }
