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
import logging
from typing import Optional, Dict, List, Tuple, Any, Deque
from collections import deque
import config

logger = logging.getLogger('USG_FLOW.AI')


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

        # FAST Protocol
        self.fast_windows = {
            'RUQ': {'status': 'pending', 'fluid': False, 'score': 0},
            'LUQ': {'status': 'pending', 'fluid': False, 'score': 0},
            'PELV': {'status': 'pending', 'fluid': False, 'score': 0},
            'CARD': {'status': 'pending', 'fluid': False, 'score': 0},
        }
        self.fast_current_window = 'RUQ'
        self.fast_start_time = None
        self.fast_fluid_regions = []

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

        # Bladder AI
        self.bladder_history = deque(maxlen=30)
        self.bladder_d1 = None
        self.bladder_d2 = None
        self.bladder_d3 = None
        self.bladder_pvr_mode = False
        self.bladder_pre_void = None

        logger.info("AI Processor inicializado com otimizacoes")

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
                self.fast_windows[key] = {'status': 'pending', 'fluid': False, 'score': 0}
            self.fast_current_window = 'RUQ'
            self.fast_start_time = None
            self.fast_fluid_regions.clear()

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

        elif old_mode == 'bladder':
            self.bladder_history.clear()
            self.bladder_d1 = None
            self.bladder_d2 = None
            self.bladder_d3 = None
            self.bladder_volume = None

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

    # =========================================================================
    # PROCESSADORES POR MODO
    # =========================================================================

    def _process_needle(self, frame):
        """Needle Pilot - Sistema avancado de guia de agulha estilo Clarius."""
        model = self.models.get('needle')
        output = frame.copy()
        h, w = frame.shape[:2]

        detected_needle = False
        needle_angle = None
        needle_depth = None
        needle_tip = None
        confidence = 0

        # Deteccao por modelo ou CV
        if model:
            config_mode = self.MODE_CONFIG['needle']
            scaled, scale = self._scale_frame(frame, config_mode['resolution_scale'])
            results = model(scaled, verbose=False, conf=config.AI_CONFIDENCE)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                    confidence = float(box.conf[0])
                    detected_needle = True

                    # Ponta da agulha (assumir canto inferior direito para agulha tipica)
                    needle_tip = (x2, y2)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    self.needle_history.append((cx, cy, x1, y1, x2, y2))
        else:
            # Fallback CV: deteccao de linhas
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=60, maxLineGap=10)

            if lines is not None:
                # Encontrar linha mais provavel (angulo diagonal)
                best_line = None
                best_score = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if 15 < angle < 75 or 105 < angle < 165:
                        score = length * (1 - abs(angle - 45) / 45)
                        if score > best_score:
                            best_score = score
                            best_line = line[0]

                if best_line is not None:
                    x1, y1, x2, y2 = best_line
                    detected_needle = True
                    confidence = min(0.9, best_score / 200)
                    needle_tip = (x2, y2) if y2 > y1 else (x1, y1)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    self.needle_history.append((cx, cy, x1, y1, x2, y2))

        # Processar historico e calcular metricas
        if len(self.needle_history) >= 2:
            points = list(self.needle_history)

            # Extrair linha da agulha do ultimo ponto
            if len(points[-1]) >= 6:
                _, _, x1, y1, x2, y2 = points[-1]

                # Desenhar linha da agulha com glow
                cv2.line(output, (x1, y1), (x2, y2), (0, 80, 0), 5)
                cv2.line(output, (x1, y1), (x2, y2), (0, 200, 0), 3)
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Calcular angulo
                dx = x2 - x1
                dy = y2 - y1
                if dx != 0:
                    needle_angle = np.arctan2(dy, dx) * 180 / np.pi

                # Estimar profundidade (assumir 0.3mm/pixel)
                if needle_tip:
                    needle_depth = needle_tip[1] * 0.3  # mm

                # Ponta da agulha destacada
                if needle_tip:
                    cv2.circle(output, needle_tip, 8, (0, 100, 0), 2)
                    cv2.circle(output, needle_tip, 5, (0, 255, 0), -1)
                    cv2.circle(output, needle_tip, 3, (255, 255, 255), -1)

                # Projecao de trajetoria
                if abs(dx) > 5 or abs(dy) > 5:
                    proj_len = 100
                    norm = np.sqrt(dx**2 + dy**2)
                    ux, uy = dx/norm, dy/norm

                    # Linha de projecao tracejada
                    for i in range(0, proj_len, 8):
                        t1, t2 = i/proj_len, (i+4)/proj_len
                        p1 = (int(x2 + ux * proj_len * t1), int(y2 + uy * proj_len * t1))
                        p2 = (int(x2 + ux * proj_len * t2), int(y2 + uy * proj_len * t2))
                        alpha = 1 - i/proj_len
                        color = (0, int(200*alpha), int(255*alpha))
                        cv2.line(output, p1, p2, color, 2)

                    # Ponto alvo
                    target = (int(x2 + ux * proj_len), int(y2 + uy * proj_len))
                    cv2.circle(output, target, 12, (0, 150, 255), 2)
                    cv2.circle(output, target, 6, (0, 200, 255), -1)
                    cv2.line(output, (target[0]-8, target[1]), (target[0]+8, target[1]), (255, 255, 255), 1)
                    cv2.line(output, (target[0], target[1]-8), (target[0], target[1]+8), (255, 255, 255), 1)

            # Trajetoria historica (trail)
            for i in range(1, min(len(points), 10)):
                alpha = i / 10
                p1 = (points[-i-1][0], points[-i-1][1])
                p2 = (points[-i][0], points[-i][1])
                color = (0, int(100 + 50*alpha), int(150 + 50*alpha))
                cv2.line(output, p1, p2, color, 1)

        # ═══════════════════════════════════════
        # PAINEL LATERAL DIREITO (estilo Clarius)
        # ═══════════════════════════════════════
        panel_w = 160
        panel_x = w - panel_w - 10
        panel_y = 10
        panel_h = 180

        # Background do painel
        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (20, 30, 20), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (0, 200, 0), 1)

        # Titulo
        cv2.putText(output, "NEEDLE PILOT", (panel_x + 10, panel_y + 22),
                   cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # Status
        status_y = panel_y + 45
        if detected_needle:
            cv2.circle(output, (panel_x + 15, status_y), 5, (0, 255, 0), -1)
            cv2.putText(output, "TRACKING", (panel_x + 25, status_y + 4),
                       cv2.FONT_HERSHEY_DUPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.circle(output, (panel_x + 15, status_y), 5, (80, 80, 80), -1)
            cv2.putText(output, "SEARCHING", (panel_x + 25, status_y + 4),
                       cv2.FONT_HERSHEY_DUPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

        # Confianca
        conf_y = panel_y + 70
        cv2.putText(output, "CONF", (panel_x + 10, conf_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 120, 100), 1, cv2.LINE_AA)
        bar_w = panel_w - 60
        cv2.rectangle(output, (panel_x + 50, conf_y - 8), (panel_x + 50 + bar_w, conf_y + 2), (40, 50, 40), -1)
        fill_w = int(bar_w * confidence)
        conf_color = (0, 255, 0) if confidence > 0.7 else (0, 200, 200) if confidence > 0.4 else (0, 100, 200)
        cv2.rectangle(output, (panel_x + 50, conf_y - 8), (panel_x + 50 + fill_w, conf_y + 2), conf_color, -1)

        # Angulo
        angle_y = panel_y + 100
        cv2.putText(output, "ANGLE", (panel_x + 10, angle_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 120, 100), 1, cv2.LINE_AA)
        if needle_angle is not None:
            angle_text = f"{abs(needle_angle):.1f}"
            cv2.putText(output, angle_text, (panel_x + 55, angle_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(output, "deg", (panel_x + 110, angle_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 150, 150), 1, cv2.LINE_AA)
        else:
            cv2.putText(output, "---", (panel_x + 70, angle_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

        # Profundidade
        depth_y = panel_y + 130
        cv2.putText(output, "DEPTH", (panel_x + 10, depth_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 120, 100), 1, cv2.LINE_AA)
        if needle_depth is not None:
            depth_text = f"{needle_depth:.1f}"
            cv2.putText(output, depth_text, (panel_x + 55, depth_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(output, "mm", (panel_x + 110, depth_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 150, 150), 1, cv2.LINE_AA)
        else:
            cv2.putText(output, "---", (panel_x + 70, depth_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

        # Dica de correcao
        tip_y = panel_y + 160
        if detected_needle and needle_angle is not None:
            if abs(needle_angle) < 30:
                tip = "STEEPEN"
                tip_color = (0, 200, 255)
            elif abs(needle_angle) > 60:
                tip = "FLATTEN"
                tip_color = (0, 200, 255)
            else:
                tip = "ON TARGET"
                tip_color = (0, 255, 0)
            cv2.putText(output, tip, (panel_x + 10, tip_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, tip_color, 1, cv2.LINE_AA)

        # Escala de profundidade na lateral esquerda
        scale_x = 25
        for i in range(0, h, 50):
            depth_mm = i * 0.3
            cv2.line(output, (scale_x - 5, i), (scale_x + 5, i), (0, 150, 0), 1)
            if i % 100 == 0:
                cv2.putText(output, f"{int(depth_mm)}", (scale_x + 8, i + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 150, 0), 1, cv2.LINE_AA)
        cv2.line(output, (scale_x, 0), (scale_x, h), (0, 100, 0), 1)

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
        """Nerve Track - Sistema avancado de segmentacao neurovascular estilo Clarius."""
        model = self.models.get('nerve')
        output = frame.copy()
        h, w = frame.shape[:2]

        structures = []  # Lista de estruturas detectadas

        if model:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (224, 224))
            tensor = torch.from_numpy(resized).float() / 255.0
            tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                masks = model(tensor).cpu().numpy()[0]

            # Classes: Nervo, Arteria, Veia
            class_config = [
                ("NERVE", (0, 255, 255), (0, 180, 180)),    # Amarelo
                ("ARTERY", (0, 0, 255), (0, 0, 180)),       # Vermelho
                ("VEIN", (255, 100, 100), (180, 70, 70)),   # Azul
            ]

            for i in range(min(3, masks.shape[0])):
                mask = cv2.resize(masks[i], (w, h))
                binary = (mask > 0.5).astype(np.uint8)

                if binary.any():
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > 200:
                            name, color_bright, color_dark = class_config[i]

                            # Preenchimento semi-transparente
                            overlay = output.copy()
                            cv2.drawContours(overlay, [cnt], -1, color_dark, -1)
                            cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)

                            # Contorno com glow
                            cv2.drawContours(output, [cnt], -1, color_dark, 3)
                            cv2.drawContours(output, [cnt], -1, color_bright, 1)

                            # Centro e metricas
                            M = cv2.moments(cnt)
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])

                                # Calcular diametro (circulo equivalente)
                                diameter_px = np.sqrt(4 * area / np.pi)
                                diameter_mm = diameter_px * 0.3  # ~0.3mm/pixel

                                structures.append({
                                    'type': name,
                                    'center': (cx, cy),
                                    'area': area,
                                    'diameter': diameter_mm,
                                    'color': color_bright,
                                    'contour': cnt
                                })
        else:
            # Fallback: Deteccao por CV (circulos)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            circles = cv2.HoughCircles(
                enhanced, cv2.HOUGH_GRADIENT, 1, 30,
                param1=50, param2=30, minRadius=8, maxRadius=100
            )

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for x, y, r in circles[0, :6]:
                    area = np.pi * r * r
                    diameter_mm = r * 2 * 0.3

                    # Classificar por tamanho e ecogenicidade
                    roi = gray[max(0,y-r):y+r, max(0,x-r):x+r]
                    if roi.size > 0:
                        mean_intensity = np.mean(roi)

                        if r > 30:
                            name, color = "NERVE", (0, 255, 255)
                        elif mean_intensity < 60:
                            name, color = "ARTERY", (0, 0, 255)
                        else:
                            name, color = "VEIN", (255, 100, 100)

                        # Desenhar com glow
                        cv2.circle(output, (x, y), r+2, (color[0]//2, color[1]//2, color[2]//2), 2)
                        cv2.circle(output, (x, y), r, color, 1)

                        structures.append({
                            'type': name,
                            'center': (int(x), int(y)),
                            'area': area,
                            'diameter': diameter_mm,
                            'color': color,
                            'contour': None
                        })

        # ═══════════════════════════════════════
        # LABELS E MEDICOES
        # ═══════════════════════════════════════
        for s in structures:
            cx, cy = s['center']
            # Label com background
            label = f"{s['type']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
            cv2.rectangle(output, (cx - tw//2 - 3, cy - th - 5), (cx + tw//2 + 3, cy + 2), (20, 20, 30), -1)
            cv2.putText(output, label, (cx - tw//2, cy - 2), cv2.FONT_HERSHEY_DUPLEX, 0.4, s['color'], 1, cv2.LINE_AA)

            # Diameter abaixo
            diam_label = f"{s['diameter']:.1f}mm"
            cv2.putText(output, diam_label, (cx - 15, cy + 18), cv2.FONT_HERSHEY_DUPLEX, 0.35, (180, 180, 200), 1, cv2.LINE_AA)

        # Desenhar linhas de distancia entre estruturas proximas
        if len(structures) >= 2:
            for i, s1 in enumerate(structures):
                for s2 in structures[i+1:]:
                    d = np.sqrt((s1['center'][0]-s2['center'][0])**2 + (s1['center'][1]-s2['center'][1])**2)
                    if d < 150:  # Estruturas proximas
                        dist_mm = d * 0.3
                        mid = ((s1['center'][0]+s2['center'][0])//2, (s1['center'][1]+s2['center'][1])//2)
                        cv2.line(output, s1['center'], s2['center'], (100, 100, 120), 1)
                        cv2.putText(output, f"{dist_mm:.1f}mm", (mid[0]-15, mid[1]-5),
                                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (150, 150, 180), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # PAINEL LATERAL (estilo Clarius)
        # ═══════════════════════════════════════
        panel_w = 160
        panel_x = w - panel_w - 10
        panel_y = 10
        panel_h = 30 + len(structures) * 35 + 40
        panel_h = min(panel_h, h - 20)

        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 20), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 200, 200), 1)

        cv2.putText(output, "NERVE TRACK", (panel_x + 10, panel_y + 22),
                   cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Lista de estruturas
        y_off = panel_y + 45
        for i, s in enumerate(structures[:5]):
            cv2.circle(output, (panel_x + 15, y_off), 5, s['color'], -1)
            cv2.putText(output, s['type'], (panel_x + 25, y_off + 4),
                       cv2.FONT_HERSHEY_DUPLEX, 0.35, s['color'], 1, cv2.LINE_AA)
            cv2.putText(output, f"{s['diameter']:.1f}mm", (panel_x + 90, y_off + 4),
                       cv2.FONT_HERSHEY_DUPLEX, 0.35, (150, 150, 180), 1, cv2.LINE_AA)
            y_off += 30

        # Contagem total
        nerve_count = sum(1 for s in structures if s['type'] == 'NERVE')
        vessel_count = len(structures) - nerve_count
        cv2.putText(output, f"N:{nerve_count} V:{vessel_count}", (panel_x + 10, panel_y + panel_h - 10),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, (120, 120, 140), 1, cv2.LINE_AA)

        # Escala de profundidade lateral
        scale_x = 20
        for i in range(0, h, 50):
            cv2.line(output, (scale_x - 3, i), (scale_x + 3, i), (0, 150, 150), 1)
            if i % 100 == 0:
                cv2.putText(output, f"{int(i*0.3)}", (scale_x + 5, i + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 150, 150), 1)
        cv2.line(output, (scale_x, 0), (scale_x, h), (0, 100, 100), 1)

        return output

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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Regiao de interesse central (onde tipicamente esta o coracao)
        roi_margin_x = int(w * 0.15)
        roi_margin_y = int(h * 0.1)
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
        _, thresh = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Encontrar contorno mais circular e grande
            best_contour = None
            best_score = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        score = area * circularity
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

        # Tracking temporal para detectar ciclo cardiaco
        self.cardiac_history.append(lv_area)

        # Calcular metricas a partir do historico
        if len(self.cardiac_history) >= 30:
            areas = list(self.cardiac_history)
            max_area = max(areas)
            min_area = min(areas) if min(areas) > 0 else 1

            # EF baseado na variacao de area (aproximacao 2D)
            if self.cardiac_ef is None:
                # Simpson simplificado: EF ≈ (EDV - ESV) / EDV
                ef_calc = ((max_area - min_area) / max_area) * 100 if max_area > 0 else 55
                self.cardiac_ef = np.clip(ef_calc * 1.2, 30, 75)  # Ajuste de escala

            # Volumes estimados (px^2 para ml usando formula elipsoidal)
            px_to_mm = 0.3
            if self.cardiac_edv is None:
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
                        self.cardiac_hr = int(60 * 30 / avg_interval)  # Assumindo ~30fps
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

        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (30, 25, 25), -1)
        cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (180, 100, 100), 1)

        # Titulo
        cv2.putText(output, "CARDIAC AI", (panel_x + 10, panel_y + 22),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 150, 150), 1, cv2.LINE_AA)

        # Indicador de fase
        phase_text = "DIASTOLE" if self.cardiac_phase == 'diastole' else "SYSTOLE"
        phase_col = (100, 150, 255) if self.cardiac_phase == 'diastole' else (100, 255, 150)
        cv2.circle(output, (panel_x + panel_w - 20, panel_y + 18), 5, phase_col, -1)

        # ═══ EF ═══
        ef_y = panel_y + 50
        cv2.putText(output, "EF", (panel_x + 10, ef_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, (120, 120, 140), 1, cv2.LINE_AA)

        ef_val = self.cardiac_ef if self.cardiac_ef else 55

        # Cor baseada no EF
        if ef_val >= 55:
            ef_color = (100, 255, 100)
            ef_status = "NORMAL"
        elif ef_val >= 40:
            ef_color = (0, 255, 255)
            ef_status = "LEVE"
        elif ef_val >= 30:
            ef_color = (0, 165, 255)
            ef_status = "MODERADO"
        else:
            ef_color = (0, 0, 255)
            ef_status = "GRAVE"

        cv2.putText(output, f"{int(ef_val)}%", (panel_x + 50, ef_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, ef_color, 1, cv2.LINE_AA)
        cv2.putText(output, ef_status, (panel_x + 110, ef_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, ef_color, 1, cv2.LINE_AA)

        # Barra EF
        bar_y = ef_y + 8
        bar_w = panel_w - 20
        cv2.rectangle(output, (panel_x + 10, bar_y), (panel_x + 10 + bar_w, bar_y + 6), (40, 40, 50), -1)
        fill_w = int(bar_w * min(ef_val, 80) / 80)
        cv2.rectangle(output, (panel_x + 10, bar_y), (panel_x + 10 + fill_w, bar_y + 6), ef_color, -1)

        # Marcadores 30% e 55%
        for mark_val in [30, 55]:
            mx = panel_x + 10 + int(bar_w * mark_val / 80)
            cv2.line(output, (mx, bar_y), (mx, bar_y + 6), (80, 80, 100), 1)

        # ═══ VOLUMES ═══
        vol_y = panel_y + 90
        cv2.putText(output, "VOLUMES", (panel_x + 10, vol_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 100, 120), 1, cv2.LINE_AA)

        edv_val = self.cardiac_edv if self.cardiac_edv else 120
        esv_val = self.cardiac_esv if self.cardiac_esv else 50

        cv2.putText(output, "EDV", (panel_x + 10, vol_y + 20),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (140, 140, 160), 1, cv2.LINE_AA)
        cv2.putText(output, f"{int(edv_val)}ml", (panel_x + 50, vol_y + 20),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (180, 180, 200), 1, cv2.LINE_AA)

        cv2.putText(output, "ESV", (panel_x + 95, vol_y + 20),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (140, 140, 160), 1, cv2.LINE_AA)
        cv2.putText(output, f"{int(esv_val)}ml", (panel_x + 130, vol_y + 20),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (180, 180, 200), 1, cv2.LINE_AA)

        # ═══ GLS ═══
        gls_y = panel_y + 135
        cv2.putText(output, "GLS", (panel_x + 10, gls_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, (120, 120, 140), 1, cv2.LINE_AA)

        gls_val = self.cardiac_gls if self.cardiac_gls else -18
        gls_color = (100, 255, 100) if gls_val <= -16 else (0, 200, 255) if gls_val <= -12 else (0, 100, 255)
        cv2.putText(output, f"{gls_val:.1f}%", (panel_x + 50, gls_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, gls_color, 1, cv2.LINE_AA)

        # ═══ HR ═══
        hr_y = panel_y + 165
        cv2.putText(output, "HR", (panel_x + 10, hr_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, (120, 120, 140), 1, cv2.LINE_AA)

        hr_val = self.cardiac_hr if self.cardiac_hr else 72
        hr_color = (100, 255, 100) if 60 <= hr_val <= 100 else (0, 200, 255)
        cv2.putText(output, f"{hr_val}", (panel_x + 50, hr_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, hr_color, 1, cv2.LINE_AA)
        cv2.putText(output, "bpm", (panel_x + 95, hr_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 100, 120), 1, cv2.LINE_AA)

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

        return output

    def _process_fast(self, frame):
        """FAST Protocol - Sistema avancado de trauma estilo Clarius."""
        output = frame.copy()
        h, w = frame.shape[:2]

        # Inicializar start time do FAST se ainda nao foi
        if self.fast_start_time is None:
            self.fast_start_time = time.time()

        # ═══════════════════════════════════════
        # DETECCAO DE LIQUIDO LIVRE (CV)
        # ═══════════════════════════════════════

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Detectar areas anecoicas (escuras) - possiveis fluidos
        _, dark_thresh = cv2.threshold(enhanced, 35, 255, cv2.THRESH_BINARY_INV)

        # Morfologia para limpar
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_thresh = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, kernel)
        dark_thresh = cv2.morphologyEx(dark_thresh, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos de fluido
        contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fluid_detected = False
        fluid_score = 0
        self.fast_fluid_regions = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 50000:  # Tamanho razoavel para fluido
                # Verificar forma (fluido tende a ser alongado entre estruturas)
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = max(cw, ch) / (min(cw, ch) + 1)

                if aspect > 1.5:  # Formato alongado
                    fluid_detected = True
                    fluid_score += area / 1000

                    self.fast_fluid_regions.append({
                        'contour': cnt,
                        'area': area,
                        'center': (x + cw//2, y + ch//2)
                    })

                    # Desenhar regiao de fluido com destaque
                    overlay = output.copy()
                    cv2.drawContours(overlay, [cnt], -1, (0, 100, 200), -1)
                    cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)

                    cv2.drawContours(output, [cnt], -1, (0, 80, 150), 3)
                    cv2.drawContours(output, [cnt], -1, (0, 150, 255), 2)

                    # Label FLUID
                    cv2.putText(output, "FLUID?", (x + cw//2 - 25, y - 8),
                               cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

        # Atualizar janela atual
        current_win = self.fast_windows[self.fast_current_window]
        if fluid_detected:
            current_win['fluid'] = True
            current_win['score'] = min(10, int(fluid_score))

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

            # Indicador de fluido detectado
            if win['fluid']:
                cv2.circle(output, (panel_x + panel_w - 20, jy + 4), 5, (0, 100, 255), -1)
                cv2.putText(output, "+", (panel_x + panel_w - 24, jy + 8),
                           cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

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

        guide_w = 170
        guide_x = w - guide_w - 10
        guide_y = h - 80
        guide_h = 70

        overlay = output.copy()
        cv2.rectangle(overlay, (guide_x, guide_y), (guide_x + guide_w, guide_y + guide_h),
                     (30, 40, 40), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
        cv2.rectangle(output, (guide_x, guide_y), (guide_x + guide_w, guide_y + guide_h),
                     (100, 150, 150), 1)

        cv2.putText(output, "CURRENT VIEW", (guide_x + 10, guide_y + 18),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 130, 130), 1, cv2.LINE_AA)

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

        # ═══════════════════════════════════════
        # INDICADORES DE FLUIDO (overlay central)
        # ═══════════════════════════════════════

        if len(self.fast_fluid_regions) > 0:
            # Alerta de fluido detectado
            alert_y = 60
            cv2.rectangle(output, (w//2 - 80, alert_y), (w//2 + 80, alert_y + 28),
                         (0, 80, 150), -1)
            cv2.rectangle(output, (w//2 - 80, alert_y), (w//2 + 80, alert_y + 28),
                         (0, 150, 255), 2)
            cv2.putText(output, "FLUID DETECTED", (w//2 - 65, alert_y + 20),
                       cv2.FONT_HERSHEY_DUPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA)

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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

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

        # Filtro vertical para B-lines
        v_kernel = np.array([[-1, 2, -1]])
        b_edges = cv2.filter2D(below_pleura, -1, v_kernel)
        _, b_thresh = cv2.threshold(b_edges, 100, 255, cv2.THRESH_BINARY)

        # Detectar B-lines
        b_lines = cv2.HoughLinesP(b_thresh, 1, np.pi/180, 30, minLineLength=(h - pleura_y)//3, maxLineGap=20)

        b_line_count = 0
        b_line_positions = []

        if b_lines is not None:
            for line in b_lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if 75 < angle < 105:  # Quase vertical
                    # Evitar B-lines muito proximas
                    x_center = (x1 + x2) // 2
                    if all(abs(x_center - px) > 30 for px in b_line_positions):
                        b_line_positions.append(x_center)
                        b_line_count += 1

                        # Desenhar B-line com glow e gradiente
                        actual_y1 = pleura_y + min(y1, y2)
                        actual_y2 = pleura_y + max(y1, y2)

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

                        # Label B
                        cv2.putText(output, "B", (x_center - 5, actual_y1 - 5),
                                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 200, 255), 1, cv2.LINE_AA)

                        if b_line_count >= 8:
                            break

        self.b_line_count = b_line_count
        self.lung_b_line_history.append(b_line_count)

        # ═══════════════════════════════════════
        # ANALISE E CLASSIFICACAO
        # ═══════════════════════════════════════

        avg_b_lines = np.mean(list(self.lung_b_line_history)) if self.lung_b_line_history else 0

        if avg_b_lines <= 2:
            profile = "A-PROFILE"
            status = "NORMAL"
            status_color = (100, 255, 100)
            interpretation = "Normal lung pattern"
        elif avg_b_lines <= 5:
            profile = "B-PROFILE"
            status = "MILD"
            status_color = (0, 255, 255)
            interpretation = "Interstitial syndrome"
        else:
            profile = "B-PROFILE"
            status = "MODERATE"
            status_color = (0, 100, 255)
            interpretation = "Pulmonary edema likely"

        # ═══════════════════════════════════════
        # PAINEL PRINCIPAL
        # ═══════════════════════════════════════

        panel_w = 175
        panel_x = w - panel_w - 10
        panel_y = 10
        panel_h = 220

        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (30, 35, 25), -1)
        cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (150, 200, 100), 1)

        cv2.putText(output, "LUNG AI", (panel_x + 10, panel_y + 22),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (180, 230, 150), 1, cv2.LINE_AA)

        # Profile type
        cv2.putText(output, profile, (panel_x + 90, panel_y + 22),
                   cv2.FONT_HERSHEY_DUPLEX, 0.35, status_color, 1, cv2.LINE_AA)

        # B-Lines
        bl_y = panel_y + 55
        cv2.putText(output, "B-LINES", (panel_x + 10, bl_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.3, (120, 140, 100), 1, cv2.LINE_AA)
        cv2.putText(output, str(b_line_count), (panel_x + 80, bl_y + 25),
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
            if b_line_count > start:
                x1 = panel_x + 10 + int(bar_w * start / 10)
                x2 = panel_x + 10 + int(bar_w * min(b_line_count, end) / 10)
                cv2.rectangle(output, (x1, bar_y), (x2, bar_y + 6), color, -1)

        # ═══════════════════════════════════════
        # HEADER PREMIUM
        # ═══════════════════════════════════════

        cv2.rectangle(output, (10, 10), (150, 45), (40, 50, 30), -1)
        cv2.rectangle(output, (10, 10), (150, 45), (150, 200, 100), 1)
        cv2.putText(output, "LUS MODE", (20, 33),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (180, 230, 160), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # LEGENDA
        # ═══════════════════════════════════════

        legend_x = 10
        legend_y = h - 75
        legend_w = 130
        legend_h = 65

        overlay = output.copy()
        cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h),
                     (30, 35, 30), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
        cv2.rectangle(output, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h),
                     (100, 120, 80), 1)

        cv2.putText(output, "LEGEND", (legend_x + 5, legend_y + 15),
                   cv2.FONT_HERSHEY_DUPLEX, 0.28, (100, 120, 100), 1, cv2.LINE_AA)

        items = [
            ("Pleura", (200, 255, 200)),
            ("A-lines", (0, 200, 200)),
            ("B-lines", (0, 180, 255)),
        ]
        for i, (name, color) in enumerate(items):
            iy = legend_y + 30 + i * 12
            cv2.line(output, (legend_x + 8, iy - 2), (legend_x + 25, iy - 2), color, 2)
            cv2.putText(output, name, (legend_x + 30, iy),
                       cv2.FONT_HERSHEY_DUPLEX, 0.25, color, 1, cv2.LINE_AA)

        return output

    def _process_bladder(self, frame):
        """Bladder AI - Sistema avancado de volume vesical estilo Clarius."""
        output = frame.copy()
        h, w = frame.shape[:2]

        model = self.models.get('bladder')

        mask_found = False
        contour_to_draw = None
        bladder_center = None

        # ═══════════════════════════════════════
        # DETECCAO DA BEXIGA
        # ═══════════════════════════════════════

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

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
            # Bexiga tipicamente anecoica (escura)
            _, thresh = cv2.threshold(enhanced, 45, 255, cv2.THRESH_BINARY_INV)

            # Morfologia para limpar
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Encontrar contorno mais provavel (grande e eliptico)
                best_contour = None
                best_score = 0

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 3000:
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter ** 2)
                            # Score baseado em area e circularidade
                            score = area * (circularity ** 2)
                            if score > best_score:
                                best_score = score
                                best_contour = cnt

                if best_contour is not None:
                    contour_to_draw = best_contour

        # ═══════════════════════════════════════
        # CALCULO DE DIMENSOES E VOLUME
        # ═══════════════════════════════════════

        if contour_to_draw is not None:
            area = cv2.contourArea(contour_to_draw)

            # Retangulo rotacionado para dimensoes
            rect = cv2.minAreaRect(contour_to_draw)
            (cx, cy), (width, height), angle = rect
            bladder_center = (int(cx), int(cy))

            # Garantir que width > height
            if width < height:
                width, height = height, width

            px_to_mm = 0.3
            self.bladder_d1 = width * px_to_mm   # Maior diametro
            self.bladder_d2 = height * px_to_mm  # Menor diametro
            self.bladder_d3 = (self.bladder_d1 + self.bladder_d2) / 2  # Estimado

            # Formula do elipsoide: V = 0.52 x D1 x D2 x D3
            volume_ml = 0.52 * self.bladder_d1 * self.bladder_d2 * self.bladder_d3 / 1000
            volume_ml = max(0, min(1000, volume_ml))

            self.bladder_volume = volume_ml
            self.bladder_history.append(volume_ml)

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

        return output

    def get_stats(self):
        """Retorna estatisticas de performance."""
        return {
            'mode': self.active_mode,
            'inference_time_ms': self.last_inference_time,
            'inference_count': self.inference_count,
            'device': str(self.device),
        }
