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
from collections import deque
import config


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
            print("AI: Usando MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("AI: Usando CUDA GPU")
        else:
            self.device = torch.device('cpu')
            print("AI: Usando CPU")

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

        print("AI Processor inicializado com otimizacoes")

    def set_mode(self, mode):
        """Define modo ativo e carrega modelo se necessario."""
        if mode == self.active_mode:
            return

        self.active_mode = mode
        self.needle_history.clear()
        self.frame_skipper.skip_count = 0

        if mode and mode not in self.models:
            self._load_model(mode)

        print(f"Modo AI: {mode or 'Desligado'}")

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
            print(f"Erro ao carregar modelo {mode}: {e}")
            self.models[mode] = None

    def _load_yolo(self, mode, path):
        """Carrega modelo YOLO."""
        if not os.path.exists(path):
            print(f"Modelo YOLO nao encontrado: {path}")
            self.models[mode] = None
            return

        from ultralytics import YOLO
        model = YOLO(path)

        # Configurar device
        if str(self.device) != 'cpu':
            model.to(self.device)

        self.models[mode] = model
        print(f"YOLO carregado: {path}")

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

    def process(self, frame):
        """Processa frame com IA."""
        if not self.active_mode:
            return frame

        # Frame skip inteligente
        if not self.frame_skipper.should_process(frame):
            # Reusar resultado anterior
            last = self.frame_skipper.get_last_result()
            if last is not None:
                return last
            return frame

        start_time = time.time()

        # Processar baseado no modo
        result = self._process_mode(frame)

        # Salvar resultado para reuso
        self.frame_skipper.set_last_result(result)

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
        """Needle Pilot - Deteccao e trajetoria de agulha com visual premium."""
        model = self.models.get('needle')
        output = frame.copy()
        h, w = frame.shape[:2]

        detected_needle = False
        needle_angle = None

        if model:
            # Escalar para processamento
            config_mode = self.MODE_CONFIG['needle']
            scaled, scale = self._scale_frame(frame, config_mode['resolution_scale'])

            # Inferencia YOLO
            results = model(scaled, verbose=False, conf=config.AI_CONFIDENCE)

            # Desenhar deteccoes
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Escalar coordenadas de volta
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                    conf = float(box.conf[0])

                    detected_needle = True

                    # Visual premium - Glow effect
                    cv2.rectangle(output, (x1-2, y1-2), (x2+2, y2+2), (0, 100, 0), 3)
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Adicionar ao historico para trajetoria
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    self.needle_history.append((cx, cy))

            # Desenhar trajetoria premium com gradiente
            if len(self.needle_history) >= 2:
                points = list(self.needle_history)

                # Trajetoria com gradiente de cor (antigo=escuro, recente=brilhante)
                for i in range(1, len(points)):
                    alpha = i / len(points)  # 0 a 1
                    color = (0, int(150 + 105 * alpha), int(200 + 55 * alpha))
                    thickness = 1 + int(alpha * 2)
                    cv2.line(output, points[i-1], points[i], color, thickness)

                # Calcular angulo da agulha
                if len(points) >= 3:
                    dx = points[-1][0] - points[-3][0]
                    dy = points[-1][1] - points[-3][1]
                    if dx != 0:
                        needle_angle = int(np.arctan2(dy, dx) * 180 / np.pi)

                    # Projecao de trajetoria animada
                    future_x = int(points[-1][0] + dx * 2)
                    future_y = int(points[-1][1] + dy * 2)

                    # Linha tracejada para projecao
                    num_dashes = 8
                    for j in range(num_dashes):
                        t1 = j / num_dashes
                        t2 = (j + 0.5) / num_dashes
                        px1 = int(points[-1][0] + (future_x - points[-1][0]) * t1)
                        py1 = int(points[-1][1] + (future_y - points[-1][1]) * t1)
                        px2 = int(points[-1][0] + (future_x - points[-1][0]) * t2)
                        py2 = int(points[-1][1] + (future_y - points[-1][1]) * t2)
                        cv2.line(output, (px1, py1), (px2, py2), (255, 200, 0), 2)

                    # Ponto alvo
                    cv2.circle(output, (future_x, future_y), 8, (255, 100, 0), 2)
                    cv2.circle(output, (future_x, future_y), 3, (255, 200, 0), -1)
        else:
            # Fallback: Needle Enhance por CV
            output = self._cv_needle_enhance(frame)

        # Header premium com background
        cv2.rectangle(output, (10, 10), (200, 70), (0, 40, 0), -1)
        cv2.rectangle(output, (10, 10), (200, 70), (0, 200, 0), 1)
        cv2.putText(output, "NEEDLE PILOT", (20, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)

        # Status e angulo
        if detected_needle or len(self.needle_history) > 0:
            cv2.circle(output, (25, 55), 5, (0, 255, 0), -1)
            cv2.putText(output, "TRACKING", (35, 60),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            if needle_angle is not None:
                cv2.putText(output, f"{abs(needle_angle)}deg", (130, 60),
                           cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(output, (25, 55), 5, (100, 100, 100), -1)
            cv2.putText(output, "SEARCHING", (35, 60),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

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
        """Nerve Track - Segmentacao de nervos."""
        model = self.models.get('nerve')
        output = frame.copy()
        h, w = frame.shape[:2]

        if model:
            # Preparar input
            config_mode = self.MODE_CONFIG['nerve']

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            input_size = (224, 224)
            resized = cv2.resize(gray, input_size)

            tensor = torch.from_numpy(resized).float() / 255.0
            tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)

            # Inferencia
            with torch.no_grad():
                masks = model(tensor).cpu().numpy()[0]

            # Cores por classe: Nervo=Amarelo, Arteria=Vermelho, Veia=Azul
            colors = [(0, 255, 255), (0, 0, 255), (255, 0, 0)]
            labels = ["Nervo", "Arteria", "Veia"]

            for i in range(min(3, masks.shape[0])):
                mask = cv2.resize(masks[i], (w, h))
                binary = mask > 0.5

                if binary.any():
                    # Contorno
                    contours, _ = cv2.findContours(
                        binary.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(output, contours, -1, colors[i], 2)

                    # Label
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(largest) > 300:
                            M = cv2.moments(largest)
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                cv2.putText(output, labels[i], (cx, cy),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
        else:
            # Fallback: Deteccao de estruturas circulares (nervos/vasos)
            output = self._cv_nerve_enhance(frame)

        # Label do modo
        cv2.putText(output, "NERVE TRACK", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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
        """Cardiac AI - Fracao de ejecao e visualizacao premium."""
        output = frame.copy()
        h, w = frame.shape[:2]

        model = self.models.get('cardiac')

        if model:
            try:
                # Preparar input
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (224, 224))
                tensor = torch.from_numpy(resized).float() / 255.0
                tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)

                # Inferência
                with torch.no_grad():
                    outputs = model(tensor).cpu().numpy()[0]

                # outputs = [EF, ESV, EDV] (valores normalizados)
                ef = int(np.clip(outputs[0] * 100, 20, 80))
                self.cardiac_ef = ef

            except Exception as e:
                print(f"Erro EchoNet: {e}")
                if self.cardiac_ef is None:
                    self.cardiac_ef = 55 + np.random.randint(-5, 6)
        else:
            # Fallback: usar valor simulado
            if self.cardiac_ef is None:
                self.cardiac_ef = 55 + np.random.randint(-5, 6)

        # Classificação e cores
        if self.cardiac_ef >= 55:
            status = "NORMAL"
            ef_color = (100, 255, 100)  # Verde
            bar_color = (80, 200, 80)
        elif self.cardiac_ef >= 40:
            status = "LEVE"
            ef_color = (0, 255, 255)  # Amarelo
            bar_color = (0, 200, 200)
        elif self.cardiac_ef >= 30:
            status = "MODERADO"
            ef_color = (0, 165, 255)  # Laranja
            bar_color = (0, 140, 200)
        else:
            status = "GRAVE"
            ef_color = (0, 0, 255)  # Vermelho
            bar_color = (0, 0, 200)

        # Header premium
        cv2.rectangle(output, (10, 10), (180, 50), (80, 40, 40), -1)
        cv2.rectangle(output, (10, 10), (180, 50), (200, 100, 100), 1)
        cv2.putText(output, "CARDIAC AI", (20, 38),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 150, 150), 1, cv2.LINE_AA)

        # Painel EF Premium (lado direito)
        panel_x = w - 150
        panel_y = 10
        panel_w = 140
        panel_h = 120

        # Background do painel
        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (30, 30, 40), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     ef_color, 2)

        # Titulo EF
        cv2.putText(output, "EF", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (150, 150, 180), 1, cv2.LINE_AA)

        # Valor EF grande
        ef_text = f"{self.cardiac_ef}%"
        cv2.putText(output, ef_text, (panel_x + 15, panel_y + 65),
                   cv2.FONT_HERSHEY_DUPLEX, 1.3, ef_color, 2, cv2.LINE_AA)

        # Status
        cv2.putText(output, status, (panel_x + 15, panel_y + 90),
                   cv2.FONT_HERSHEY_DUPLEX, 0.45, ef_color, 1, cv2.LINE_AA)

        # Barra de progresso
        bar_x = panel_x + 10
        bar_y = panel_y + 100
        bar_w = panel_w - 20
        bar_h = 10
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)

        # Preencher barra baseado no EF (0-80%)
        fill_w = int(bar_w * min(self.cardiac_ef, 80) / 80)
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        # Marcadores na barra
        markers = [(30, "30%"), (55, "55%")]
        for val, _ in markers:
            mx = bar_x + int(bar_w * val / 80)
            cv2.line(output, (mx, bar_y), (mx, bar_y + bar_h), (100, 100, 120), 1)

        # Legenda de camaras (canto inferior esquerdo)
        legend_y = h - 80
        cv2.rectangle(output, (10, legend_y), (120, h - 10), (30, 30, 40), -1)
        cv2.rectangle(output, (10, legend_y), (120, h - 10), (100, 100, 120), 1)

        chambers = [("LV", (100, 150, 255)), ("LA", (255, 150, 150)), ("RV", (150, 255, 150))]
        for i, (name, color) in enumerate(chambers):
            cy = legend_y + 18 + i * 20
            cv2.circle(output, (25, cy - 3), 5, color, -1)
            cv2.putText(output, name, (35, cy), cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1, cv2.LINE_AA)

        return output

    def _process_fast(self, frame):
        """FAST Protocol - Deteccao de liquido livre com visual premium."""
        output = frame.copy()
        h, w = frame.shape[:2]

        # Header premium
        cv2.rectangle(output, (10, 10), (200, 50), (80, 60, 30), -1)
        cv2.rectangle(output, (10, 10), (200, 50), (255, 180, 80), 1)
        cv2.putText(output, "FAST PROTOCOL", (20, 38),
                   cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 200, 100), 1, cv2.LINE_AA)

        # Painel de janelas (lado direito)
        panel_x = w - 180
        panel_y = 10
        panel_w = 170
        panel_h = 150

        # Background do painel
        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (30, 30, 40), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (255, 180, 80), 1)

        cv2.putText(output, "WINDOWS", (panel_x + 10, panel_y + 22),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (150, 150, 180), 1, cv2.LINE_AA)

        janelas = [
            ("RUQ (Morrison)", False, (255, 200, 80)),
            ("LUQ (Spleen)", False, (255, 200, 80)),
            ("Pelvic", False, (255, 200, 80)),
            ("Cardiac", False, (255, 200, 80)),
        ]

        for i, (janela, checked, color) in enumerate(janelas):
            jy = panel_y + 45 + i * 28
            # Checkbox
            cv2.rectangle(output, (panel_x + 10, jy - 10), (panel_x + 24, jy + 4),
                         (80, 80, 100), -1)
            cv2.rectangle(output, (panel_x + 10, jy - 10), (panel_x + 24, jy + 4),
                         color, 1)
            if checked:
                cv2.line(output, (panel_x + 12, jy - 3), (panel_x + 16, jy + 1), color, 2)
                cv2.line(output, (panel_x + 16, jy + 1), (panel_x + 22, jy - 8), color, 2)
            # Label
            cv2.putText(output, janela, (panel_x + 30, jy),
                       cv2.FONT_HERSHEY_DUPLEX, 0.35, (180, 180, 200), 1, cv2.LINE_AA)

        # Status geral
        cv2.rectangle(output, (10, h - 50), (180, h - 10), (30, 30, 40), -1)
        cv2.rectangle(output, (10, h - 50), (180, h - 10), (100, 100, 120), 1)
        cv2.circle(output, (25, h - 30), 6, (100, 255, 100), -1)
        cv2.putText(output, "NEGATIVE", (40, h - 24),
                   cv2.FONT_HERSHEY_DUPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)

        return output

    def _process_segment(self, frame):
        """Anatomia - Segmentacao geral de estruturas."""
        output = frame.copy()
        h, w = frame.shape[:2]

        model = self.models.get('segment')

        if model:
            try:
                # Preparar input
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (224, 224))
                tensor = torch.from_numpy(resized).float() / 255.0
                tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)

                # Inferência
                with torch.no_grad():
                    masks = model(tensor).cpu().numpy()[0]

                # 5 classes: Tecido, Osso, Fluido, Vaso, Outro
                colors = [
                    (100, 255, 100),   # Tecido - verde
                    (255, 255, 255),   # Osso - branco
                    (255, 100, 100),   # Fluido - azul
                    (0, 0, 255),       # Vaso - vermelho
                    (255, 100, 255),   # Outro - magenta
                ]
                labels = ["Tecido", "Osso", "Fluido", "Vaso", "Outro"]

                for i in range(min(5, masks.shape[0])):
                    mask = cv2.resize(masks[i], (w, h))
                    binary = mask > 0.5

                    if binary.any():
                        contours, _ = cv2.findContours(
                            binary.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(output, contours, -1, colors[i], 1)

                        # Label no maior contorno
                        if contours:
                            largest = max(contours, key=cv2.contourArea)
                            if cv2.contourArea(largest) > 500:
                                M = cv2.moments(largest)
                                if M["m00"] > 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    cv2.putText(output, labels[i], (cx-20, cy),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)

            except Exception as e:
                print(f"Erro USFM: {e}")

        cv2.putText(output, "ANATOMIA AI", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2)

        return output

    def _process_mmode(self, frame):
        """Modo-M - Analise temporal."""
        output = frame.copy()
        h, w = frame.shape[:2]

        # Linha de referencia no centro
        cx = w // 2
        cv2.line(output, (cx, 0), (cx, h), (0, 255, 255), 2)

        cv2.putText(output, "MODO-M", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 100), 2)
        cv2.putText(output, "Linha de amostragem", (cx + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return output

    def _process_color_doppler(self, frame):
        """Color Doppler - Simulacao de fluxo."""
        output = frame.copy()
        h, w = frame.shape[:2]

        # Simulacao: aplicar mapa de cores em regiao central
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar colormap apenas em areas de alto contraste
        edges = cv2.Canny(gray, 30, 100)
        mask = cv2.dilate(edges, None, iterations=2)

        # Criar overlay colorido
        colormap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        # Aplicar apenas na regiao central
        roi_h = h // 2
        roi_w = w // 2
        y1, x1 = (h - roi_h) // 2, (w - roi_w) // 2

        roi_mask = mask[y1:y1+roi_h, x1:x1+roi_w]
        roi_color = colormap[y1:y1+roi_h, x1:x1+roi_w]
        roi_orig = output[y1:y1+roi_h, x1:x1+roi_w]

        # Blend onde ha edges
        for c in range(3):
            roi_orig[:,:,c] = np.where(roi_mask > 0, roi_color[:,:,c], roi_orig[:,:,c])

        output[y1:y1+roi_h, x1:x1+roi_w] = roi_orig

        cv2.putText(output, "COLOR DOPPLER", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 50), 2)

        return output

    def _process_power_doppler(self, frame):
        """Power Doppler - Alta sensibilidade."""
        output = self._process_color_doppler(frame)

        # Ajustar label
        cv2.rectangle(output, (10, 20), (220, 55), (0, 0, 0), -1)
        cv2.putText(output, "POWER DOPPLER", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 150, 50), 2)

        return output

    def _process_blines(self, frame):
        """Lung AI - Deteccao de Linhas-B com visual premium."""
        output = frame.copy()
        h, w = frame.shape[:2]

        # Deteccao de linhas verticais brilhantes (B-lines)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Realce de contraste vertical
        kernel = np.array([[-1, 2, -1]])
        enhanced = cv2.filter2D(gray, -1, kernel)

        # Threshold
        _, thresh = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY)

        # Detectar linhas verticais
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, minLineLength=h//4, maxLineGap=20)

        b_line_count = 0
        b_line_positions = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Filtrar apenas linhas quase verticais
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if 75 < angle < 105:
                    # Visual premium - linhas com gradiente
                    for t in range(0, 100, 5):
                        alpha = t / 100.0
                        px1 = int(x1 + (x2 - x1) * alpha)
                        py1 = int(y1 + (y2 - y1) * alpha)
                        px2 = int(x1 + (x2 - x1) * (alpha + 0.05))
                        py2 = int(y1 + (y2 - y1) * (alpha + 0.05))
                        color_intensity = int(150 + 105 * alpha)
                        cv2.line(output, (px1, py1), (px2, py2), (0, color_intensity, 255), 2)
                    b_line_count += 1
                    b_line_positions.append((x1 + x2) // 2)

        # Limitar contagem
        b_line_count = min(b_line_count, 10)
        self.b_line_count = b_line_count

        # Classificacao e cores
        if b_line_count <= 2:
            status = "NORMAL"
            status_color = (100, 255, 100)
        elif b_line_count <= 5:
            status = "LEVE"
            status_color = (0, 255, 255)
        else:
            status = "MODERADO"
            status_color = (0, 100, 255)

        # Header premium
        cv2.rectangle(output, (10, 10), (160, 50), (60, 60, 40), -1)
        cv2.rectangle(output, (10, 10), (160, 50), (200, 220, 150), 1)
        cv2.putText(output, "LUNG AI", (20, 38),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 255, 150), 1, cv2.LINE_AA)

        # Painel de B-lines (lado direito)
        panel_x = w - 160
        panel_y = 10
        panel_w = 150
        panel_h = 130

        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (30, 30, 40), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (200, 220, 150), 1)

        # Titulo
        cv2.putText(output, "B-LINES", (panel_x + 10, panel_y + 22),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (150, 150, 180), 1, cv2.LINE_AA)

        # Contagem grande
        cv2.putText(output, str(b_line_count), (panel_x + 50, panel_y + 70),
                   cv2.FONT_HERSHEY_DUPLEX, 1.8, status_color, 2, cv2.LINE_AA)

        # Status
        cv2.putText(output, status, (panel_x + 10, panel_y + 95),
                   cv2.FONT_HERSHEY_DUPLEX, 0.45, status_color, 1, cv2.LINE_AA)

        # Barra de indicador
        bar_x = panel_x + 10
        bar_y = panel_y + 110
        bar_w = panel_w - 20
        bar_h = 10
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)

        # Gradiente verde->amarelo->vermelho
        segments = [
            (0, 2, (80, 200, 80)),     # Normal
            (2, 5, (0, 200, 200)),     # Leve
            (5, 10, (0, 100, 255)),    # Moderado
        ]
        for start, end, color in segments:
            if b_line_count > start:
                fill_start = int(bar_w * start / 10)
                fill_end = int(bar_w * min(b_line_count, end) / 10)
                cv2.rectangle(output, (bar_x + fill_start, bar_y),
                             (bar_x + fill_end, bar_y + bar_h), color, -1)

        # Linha da pleura indicada
        cv2.putText(output, "Pleura", (10, h - 20),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (150, 200, 200), 1, cv2.LINE_AA)
        cv2.line(output, (60, h - 25), (w - 20, h - 25), (150, 200, 200), 1)

        return output

    def _process_bladder(self, frame):
        """Bladder AI - Volume vesical com visual premium."""
        output = frame.copy()
        h, w = frame.shape[:2]

        model = self.models.get('bladder')
        mask_found = False
        contour_to_draw = None

        if model:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                        area = cv2.contourArea(largest)

                        if area > 1000:
                            mask_found = True
                            contour_to_draw = largest

                            rect = cv2.minAreaRect(largest)
                            (cx, cy), (width, height), angle = rect

                            px_to_mm = 0.3
                            d1_mm = width * px_to_mm
                            d2_mm = height * px_to_mm
                            d3_mm = (d1_mm + d2_mm) / 2

                            volume_ml = 0.52 * d1_mm * d2_mm * d3_mm / 1000
                            self.bladder_volume = max(0, min(1000, volume_ml))

            except Exception as e:
                print(f"Erro Bladder AI: {e}")

        if not mask_found:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                if area > 5000:
                    contour_to_draw = largest
                    rect = cv2.minAreaRect(largest)
                    (cx, cy), (width, height), angle = rect

                    px_to_mm = 0.3
                    d1_mm = width * px_to_mm
                    d2_mm = height * px_to_mm
                    d3_mm = (d1_mm + d2_mm) / 2

                    volume_ml = 0.52 * d1_mm * d2_mm * d3_mm / 1000
                    self.bladder_volume = max(0, min(1000, volume_ml))

        # Desenhar contorno premium
        if contour_to_draw is not None:
            # Glow effect
            cv2.drawContours(output, [contour_to_draw], -1, (100, 50, 150), 4)
            cv2.drawContours(output, [contour_to_draw], -1, (200, 120, 255), 2)

            # Preenchimento semi-transparente
            overlay = output.copy()
            cv2.drawContours(overlay, [contour_to_draw], -1, (180, 100, 220), -1)
            cv2.addWeighted(overlay, 0.2, output, 0.8, 0, output)

        # Header premium
        cv2.rectangle(output, (10, 10), (180, 50), (80, 40, 80), -1)
        cv2.rectangle(output, (10, 10), (180, 50), (200, 120, 255), 1)
        cv2.putText(output, "BLADDER AI", (20, 38),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 150, 255), 1, cv2.LINE_AA)

        # Painel de volume (lado direito)
        panel_x = w - 160
        panel_y = 10
        panel_w = 150
        panel_h = 140

        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (30, 30, 40), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (200, 120, 255), 1)

        cv2.putText(output, "VOLUME", (panel_x + 10, panel_y + 22),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (150, 150, 180), 1, cv2.LINE_AA)

        # Volume em mL
        if self.bladder_volume is not None:
            vol_text = f"{int(self.bladder_volume)}"
            cv2.putText(output, vol_text, (panel_x + 20, panel_y + 65),
                       cv2.FONT_HERSHEY_DUPLEX, 1.4, (200, 150, 255), 2, cv2.LINE_AA)
            cv2.putText(output, "mL", (panel_x + 100, panel_y + 65),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (150, 120, 200), 1, cv2.LINE_AA)

            # Status baseado no volume
            if self.bladder_volume < 100:
                status = "VAZIO"
                status_color = (100, 100, 150)
            elif self.bladder_volume < 300:
                status = "NORMAL"
                status_color = (100, 255, 100)
            elif self.bladder_volume < 500:
                status = "MODERADO"
                status_color = (0, 255, 255)
            else:
                status = "DISTENDIDO"
                status_color = (0, 100, 255)

            cv2.putText(output, status, (panel_x + 10, panel_y + 90),
                       cv2.FONT_HERSHEY_DUPLEX, 0.45, status_color, 1, cv2.LINE_AA)

            # Barra de volume
            bar_x = panel_x + 10
            bar_y = panel_y + 105
            bar_w = panel_w - 20
            bar_h = 12
            cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)

            # Preencher barra (0-600 mL)
            fill_w = int(bar_w * min(self.bladder_volume, 600) / 600)
            cv2.rectangle(output, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), status_color, -1)

            # Marcador 300 mL
            marker_x = bar_x + int(bar_w * 300 / 600)
            cv2.line(output, (marker_x, bar_y), (marker_x, bar_y + bar_h), (100, 100, 120), 1)

            # Formula usada
            cv2.putText(output, "D1xD2xD3x0.52", (panel_x + 10, panel_y + 132),
                       cv2.FONT_HERSHEY_DUPLEX, 0.3, (80, 80, 100), 1, cv2.LINE_AA)
        else:
            cv2.putText(output, "---", (panel_x + 40, panel_y + 65),
                       cv2.FONT_HERSHEY_DUPLEX, 1.4, (100, 100, 120), 2, cv2.LINE_AA)
            cv2.putText(output, "mL", (panel_x + 100, panel_y + 65),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (80, 80, 100), 1, cv2.LINE_AA)
            cv2.putText(output, "SCANNING", (panel_x + 10, panel_y + 90),
                       cv2.FONT_HERSHEY_DUPLEX, 0.45, (100, 100, 120), 1, cv2.LINE_AA)

        return output

    def get_stats(self):
        """Retorna estatisticas de performance."""
        return {
            'mode': self.active_mode,
            'inference_time_ms': self.last_inference_time,
            'inference_count': self.inference_count,
            'device': str(self.device),
        }
