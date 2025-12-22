#!/usr/bin/env python3
"""
USG FLOW - Interface Premium Completa
======================================
Baseado no projeto HTML PlataformaUSG.
Todas as funcionalidades, 100% Python.
Imagem ORIGINAL preservada (sem modificacao).
"""

import cv2
import numpy as np
import time
import threading
import os
import sys
import logging
from typing import Optional, Tuple, List, Dict, Any

# Configuracao de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('USG_FLOW')

# Inicializar NSApplication para controle de fullscreen no macOS
try:
    from AppKit import NSApplication
    NSApplication.sharedApplication()
except ImportError:
    pass  # AppKit não disponível (não é macOS ou não instalado)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.window_capture import WindowCapture
from src.clip_recorder import ClipRecorder


# =============================================================================
# CAPTURA EM THREAD SEPARADO (OTIMIZADA)
# =============================================================================
class CapturaThread(threading.Thread):
    """Thread de captura otimizada com buffer circular e lock minimo."""

    def __init__(self, buffer_size: int = 2) -> None:
        super().__init__(daemon=True)
        self._frame: Optional[np.ndarray] = None
        self._lock: threading.Lock = threading.Lock()
        self._running: bool = True
        self._connected: bool = False
        self._frame_count: int = 0
        self._last_frame_time: float = 0

    @property
    def connected(self) -> bool:
        return self._connected

    def run(self) -> None:
        cap = WindowCapture("AIRPLAY")
        consecutive_fails = 0

        while self._running:
            try:
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Lock minimo - apenas para atribuicao
                    with self._lock:
                        self._frame = frame
                        self._connected = True
                    self._frame_count += 1
                    self._last_frame_time = time.time()
                    consecutive_fails = 0
                else:
                    consecutive_fails += 1
                    self._connected = False
                    # Sleep exponencial para economizar CPU quando desconectado
                    # Começa em 10ms, dobra até max 500ms
                    sleep_time = min(0.5, 0.01 * (2 ** min(consecutive_fails, 6)))
                    time.sleep(sleep_time)
            except Exception as e:
                self._connected = False
                logger.error(f"Captura: {type(e).__name__}: {e}")
                time.sleep(0.1)

    def get_frame(self) -> Optional[np.ndarray]:
        """Retorna frame atual (copia para thread safety)."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def get_frame_if_new(self, last_count: int) -> Tuple[Optional[np.ndarray], int]:
        """Retorna frame apenas se for novo (evita copia desnecessaria)."""
        if self._frame_count == last_count:
            return None, last_count
        with self._lock:
            if self._frame is not None:
                return self._frame.copy(), self._frame_count
            return None, last_count

    def stop(self):
        self._running = False


# =============================================================================
# PROCESSAMENTO AI EM THREAD SEPARADO (EVITA TRAVAMENTO)
# =============================================================================
class AIThread(threading.Thread):
    """Thread dedicada para processamento AI - evita bloquear UI."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._running: bool = True
        self._input_frame: Optional[np.ndarray] = None
        self._output_frame: Optional[np.ndarray] = None
        self._roi: Optional[Tuple[int, int, int, int]] = None
        self._lock: threading.Lock = threading.Lock()
        self._new_frame_event: threading.Event = threading.Event()
        self._ai: Optional[Any] = None  # AIProcessor quando carregado
        self._processing: bool = False
        self._last_process_time: float = 0
        self._mode_changed: bool = False

    def set_ai(self, ai: Any) -> None:
        """Define o processador AI."""
        with self._lock:
            self._ai = ai
            self._mode_changed = True
            # Limpar resultados pendentes ao trocar AI
            self._output_frame = None
            self._input_frame = None

    def reset(self) -> None:
        """Reseta estado da thread (usar ao trocar de modo)."""
        with self._lock:
            self._output_frame = None
            self._input_frame = None
            self._mode_changed = True

    def submit_frame(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """Submete frame para processamento (non-blocking)."""
        if self._processing:
            return False  # Ainda processando anterior

        with self._lock:
            self._input_frame = frame.copy()
            self._roi = roi
        self._new_frame_event.set()
        return True

    def get_result(self) -> Optional[np.ndarray]:
        """Retorna resultado do processamento (ou None se nao pronto)."""
        with self._lock:
            if self._output_frame is not None:
                result = self._output_frame
                self._output_frame = None
                return result
        return None

    @property
    def is_processing(self) -> bool:
        return self._processing

    @property
    def last_process_time(self) -> float:
        """Tempo do ultimo processamento em ms."""
        return self._last_process_time

    def run(self) -> None:
        while self._running:
            # Esperar por novo frame (com timeout para permitir shutdown)
            if not self._new_frame_event.wait(timeout=0.1):
                continue

            self._new_frame_event.clear()
            self._processing = True

            try:
                with self._lock:
                    frame = self._input_frame
                    roi = self._roi
                    self._input_frame = None
                    mode_changed = self._mode_changed
                    self._mode_changed = False

                # Se modo mudou, ignorar frame pendente
                if mode_changed and frame is not None:
                    # Pegar frame novo apos mudanca de modo
                    pass

                if frame is not None and self._ai is not None:
                    start = time.time()

                    try:
                        if roi:
                            # Processar apenas ROI
                            rx, ry, rw, rh = roi
                            h, w = frame.shape[:2]
                            rx = max(0, min(rx, w - 1))
                            ry = max(0, min(ry, h - 1))
                            rw = max(USGApp.MIN_ROI_SIZE, min(rw, w - rx))
                            rh = max(USGApp.MIN_ROI_SIZE, min(rh, h - ry))

                            if rw > USGApp.MIN_ROI_SIZE and rh > USGApp.MIN_ROI_SIZE:
                                roi_frame = frame[ry:ry+rh, rx:rx+rw].copy()
                                roi_processed = self._ai.process(roi_frame)
                                # Garantir que dimensoes batem
                                if roi_processed.shape[:2] == (rh, rw):
                                    frame[ry:ry+rh, rx:rx+rw] = roi_processed
                                # Desenhar borda da ROI
                                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)
                        else:
                            # Processar frame inteiro
                            frame = self._ai.process(frame)

                        self._last_process_time = (time.time() - start) * 1000

                        with self._lock:
                            self._output_frame = frame

                    except Exception as e:
                        print(f"AI Process Error: {e}")
                        # Em caso de erro, retornar frame original
                        with self._lock:
                            self._output_frame = frame

            except Exception as e:
                print(f"AI Thread Error: {e}")
                import traceback
                traceback.print_exc()

            self._processing = False

    def stop(self):
        self._running = False
        self._new_frame_event.set()  # Desbloquear wait


# =============================================================================
# CLASSE BOTAO - Otimizada para Performance
# =============================================================================
class Botao:
    # Cores pré-calculadas (evita criar tuplas a cada frame)
    COR_BG_NORMAL = (22, 22, 30)
    COR_BG_HOVER = (40, 40, 52)
    COR_BORDA = (35, 35, 45)
    COR_BORDA_HOVER = (60, 60, 75)
    COR_TEXTO_NORMAL = (140, 140, 155)
    COR_TEXTO_ATIVO = (255, 255, 255)
    COR_TEXTO_HOVER = (230, 230, 240)
    COR_BADGE = (50, 50, 65)
    COR_BADGE_ATIVO = (180, 180, 200)
    COR_TECLA = (90, 90, 110)
    COR_TECLA_ATIVO = (200, 200, 220)
    COR_ACCENT = (255, 255, 255)

    def __init__(self, x, y, w, h, texto, tecla, cor_normal, cor_ativo, tooltip=""):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.texto = texto
        self.tecla = tecla
        self.cor_ativo = cor_ativo
        self.ativo = False
        self.hover = False
        # Pré-calcular posições
        self.x2 = x + w
        self.y2 = y + h
        self.cy = y + h // 2
        self.badge_x = x + 14
        self.text_x = x + 40

    def contem(self, px, py):
        return self.x <= px <= self.x2 and self.y <= py <= self.y2

    def desenhar(self, img, pulse_phase=0):
        if self.ativo:
            # Cor ativa sem efeito pulsante (visual estavel)
            cv2.rectangle(img, (self.x, self.y), (self.x2, self.y2), self.cor_ativo, -1)
            cv2.rectangle(img, (self.x, self.y + 2), (self.x + 3, self.y2 - 2), self.COR_ACCENT, -1)
            cor_texto = self.COR_TEXTO_ATIVO
            badge_cor = self.COR_BADGE_ATIVO
            tecla_cor = self.COR_TECLA_ATIVO
        elif self.hover:
            cv2.rectangle(img, (self.x, self.y), (self.x2, self.y2), self.COR_BG_HOVER, -1)
            cv2.rectangle(img, (self.x, self.y), (self.x2, self.y2), self.COR_BORDA_HOVER, 1)
            cv2.rectangle(img, (self.x, self.y + 4), (self.x + 2, self.y2 - 4), self.cor_ativo, -1)
            cor_texto = self.COR_TEXTO_HOVER
            badge_cor = self.COR_BADGE
            tecla_cor = self.COR_TECLA
        else:
            cv2.rectangle(img, (self.x, self.y), (self.x2, self.y2), self.COR_BG_NORMAL, -1)
            cv2.rectangle(img, (self.x, self.y), (self.x2, self.y2), self.COR_BORDA, 1)
            cor_texto = self.COR_TEXTO_NORMAL
            badge_cor = self.COR_BADGE
            tecla_cor = self.COR_TECLA

        if self.tecla:
            cv2.rectangle(img, (self.badge_x - 4, self.cy - 8), (self.badge_x + 12, self.cy + 8), badge_cor, -1)
            cv2.putText(img, self.tecla, (self.badge_x, self.cy + 4), cv2.FONT_HERSHEY_DUPLEX, 0.38, tecla_cor, 1, cv2.LINE_AA)
            cv2.putText(img, self.texto, (self.text_x + 2, self.cy + 4), cv2.FONT_HERSHEY_DUPLEX, 0.44, cor_texto, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, self.texto, (self.x + self.w // 2 - len(self.texto) * 5, self.cy + 4),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, cor_texto, 1, cv2.LINE_AA)


# =============================================================================
# INSTRUCOES POR MODO
# =============================================================================
MODE_INSTRUCTIONS = {
    'B-MODE': "Imagem 2D padrao para avaliacao anatomica geral em escala de cinza.",
    'AGULHA': "Realce de agulha por IA com projecao de trajetoria para puncoes guiadas.",
    'NERVO': "Segmentacao automatica de nervos perifericos e estruturas adjacentes.",
    'CARDIACO': "Calculo automatico de Fracao de Ejecao (Simpson) e visualizacao de camaras.",
    'FAST': "Deteccao rapida de liquido livre abdominal e pelvico para trauma.",
    'ANATOMIA': "Identificacao e rotulagem de estruturas anatomicas em tempo real.",
    'MODO-M': "Analise temporal de movimento (ex: pleura, valvulas cardiacas).",
    'COLOR': "Mapeamento de fluxo sanguineo direcional (Vermelho/Azul).",
    'POWER': "Alta sensibilidade para fluxos lentos/perifericos (Angio).",
    'PULMAO': "Quantificacao automatica de Linhas-B e sinal de deslizamento pleural.",
    'BEXIGA': "Calculo automatizado do volume da bexiga com correcao geometrica.",
}


# =============================================================================
# APLICATIVO PRINCIPAL
# =============================================================================
class USGApp:
    """
    Interface Premium Completa.
    Baseada no projeto HTML PlataformaUSG.
    """

    # Modos AI - Cores estilo Butterfly/Clarius (vibrantes e premium)
    MODOS = [
        ('B-MODE', '1', (140, 140, 150), "Modo Brilho - Imagem 2D padrao"),
        ('NEEDLE', '2', (80, 220, 120), "Needle Pilot - Realce de agulha"),
        ('NERVE', '3', (50, 220, 220), "Nerve Track - Segmentacao nervos"),
        ('CARDIAC', '4', (90, 130, 230), "Cardiac AI - Fracao Ejecao"),
        ('FAST', '5', (60, 180, 240), "Protocolo FAST - Trauma"),
        ('ANATOMY', '6', (200, 120, 220), "Anatomia - ID estruturas"),
        ('M-MODE', '7', (180, 180, 100), "Modo-M - Movimento temporal"),
        ('COLOR', '8', (220, 80, 80), "Color Doppler - Fluxo"),
        ('POWER', '9', (240, 160, 60), "Power Doppler - Alta sens."),
        ('LUNG', '0', (100, 180, 240), "Lung AI - Linhas-B"),
        ('BLADDER', 'V', (160, 100, 220), "Volume Vesical"),
    ]

    SIDEBAR_W = 280  # Sidebar premium (mais larga para melhor legibilidade)
    MIN_ROI_SIZE = 20  # Tamanho minimo para ROI valida (pixels)
    MIN_ROI_GRID = 100  # Tamanho minimo para mostrar grid de tercos

    # Mapeamento centralizado: plugin_idx -> nome do modo AI
    # Indices correspondem a MODOS[1:] (excluindo B-MODE que é indice 0)
    AI_MODE_MAP = {
        0: 'needle',   # NEEDLE
        1: 'nerve',    # NERVE
        2: 'cardiac',  # CARDIAC
        3: 'fast',     # FAST
        4: 'segment',  # ANATOMY
        5: 'm_mode',   # M-MODE
        6: 'color',    # COLOR
        7: 'power',    # POWER
        8: 'b_lines',  # LUNG
        9: 'bladder',  # BLADDER
    }

    def __init__(self):
        # Estado principal - GRUPO EXCLUSIVO
        # 0 = B-MODE (sem IA), 1 = IA ZONE, 2 = IA FULL
        self.source_mode = 0  # Começa em B-MODE

        # Plugin de IA selecionado (0=NEEDLE, 1=NERVE, etc.)
        self.plugin_idx = 0

        # Estados gerais
        self.pause = False
        self.recording = False
        self.show_sidebar = True
        self.show_overlays = True
        self.show_help = False
        self.show_instructions = True
        self.fullscreen = True
        self.biplane_active = False
        self.biplane_ref = None
        self.running = True
        self.fps = 0
        self.mouse_x = 0
        self.mouse_y = 0

        # Zoom/Pan
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # ROI para IA ZONE (sistema profissional)
        self.process_roi = None
        self.selecting_roi = False

        # ═══════════════════════════════════════════════════════════════════════
        # NERVE TRACK v2.0 - Seleção de Bloqueio Nervoso
        # ═══════════════════════════════════════════════════════════════════════
        self.nerve_block_idx = 0  # Índice do bloqueio selecionado
        self.nerve_block_ids = []  # Lista de IDs de bloqueios disponíveis
        self.nerve_block_names = {}  # Mapa ID -> Nome
        self.nerve_block_show_menu = False  # Mostrar menu de seleção
        self.roi_start = None
        self.roi_end = None
        self.roi_confirmed = False  # ROI confirmada com ENTER
        self.roi_drag_handle = None  # Para redimensionar: 'tl', 'tr', 'bl', 'br', 'move'
        self.roi_drag_start = None  # Posicao inicial do arraste
        self.roi_original = None    # ROI original antes do arraste
        self.roi_animation_frame = 0  # Para marching ants

        # Cache da sidebar para performance
        self._sidebar_cache = None
        self._sidebar_dirty = True  # Força redesenho quando True

        # Animacoes de transicao
        self.transition_alpha = 1.0  # 0=escuro, 1=normal
        self.transition_target = 1.0
        self.transition_speed = 0.15  # Velocidade do fade
        self.pulse_phase = 0  # Para animacao de pulse
        self.last_mode_change = 0  # Timestamp da ultima mudanca

        # Info de escala para converter coordenadas
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0

        # Cache de canvas para evitar realocacoes a cada frame
        self._canvas_cache = None
        self._canvas_size = (0, 0)

        # AI (thread separada para nao bloquear UI)
        self.ai = None
        self.ai_loading = False  # Flag para indicar carregamento
        self.ai_thread = AIThread()
        self.ai_thread.start()
        self._last_ai_frame = None  # Cache do ultimo frame processado

        # Gravacao
        self.recorder = ClipRecorder(output_dir="captures", fps=30)

        # Captura
        self.captura = CapturaThread()
        self.captura.start()
        self.frame_original = None
        self._last_frame_count = 0  # Para detectar frames novos

        # Janela
        self.janela = "USG FLOW"
        cv2.namedWindow(self.janela, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.janela, self._mouse_callback)

        # Iniciar em fullscreen real
        self.fullscreen = True
        cv2.setWindowProperty(self.janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Criar botoes
        self._criar_botoes()

        self._print_ajuda()

    def _print_ajuda(self):
        print("\n" + "=" * 60)
        print("  USG FLOW - Interface Premium Completa")
        print("=" * 60)
        print("  MODOS (11 disponiveis):")
        for nome, tecla, _, desc in self.MODOS:
            print(f"    [{tecla}] {nome}: {desc}")
        print("-" * 60)
        print("  CONTROLES:")
        print("    F       = Pause/Congelar")
        print("    A       = AI On/Off")
        print("    R       = Gravar")
        print("    S       = Screenshot")
        print("    B       = Biplane (lado a lado)")
        print("    H       = Esconder/Mostrar overlays")
        print("    I       = Esconder/Mostrar instrucoes")
        print("    T       = Esconder/Mostrar sidebar")
        print("    +/-     = Zoom In/Out")
        print("    Setas   = Pan (mover imagem)")
        print("    0       = Reset View")
        print("    ?       = Ajuda")
        print("    F11     = Fullscreen")
        print("    Q/ESC   = Sair")
        print("-" * 60)
        print("  NERVE TRACK (28 bloqueios nervosos):")
        print("    < / ,   = Bloco anterior")
        print("    > / .   = Próximo bloco")
        print("    N       = Menu de bloqueios")
        print("=" * 60 + "\n")

    def _criar_botoes(self):
        """Cria todos os botoes da sidebar - Layout Premium Moderno."""
        self.botoes_modo = []
        self.botoes_ctrl = []
        self.botoes_view = []
        self.botoes_source = []  # Grupo exclusivo: B-MODE, IA ZONE, IA FULL
        self.section_titles = []

        # Cores premium
        bg_btn = (28, 28, 35)

        # Dimensões padronizadas
        BTN_W = 220
        BTN_H = 30
        BTN_GAP = 2
        MARGIN = 30

        y = 70

        # ═══════════════════════════════════════
        # SEÇÃO 1: GRUPO EXCLUSIVO (B-MODE / IA ZONE / IA FULL)
        # ═══════════════════════════════════════
        source_btns = [
            ('B-MODE', '1', (140, 140, 160)),    # Cinza neutro
            ('IA ZONE', 'C', (255, 200, 60)),    # Amarelo dourado
            ('IA FULL', 'A', (100, 220, 140)),   # Verde suave
        ]
        for texto, tecla, cor in source_btns:
            btn = Botao(MARGIN, y, BTN_W, BTN_H, texto, tecla, bg_btn, cor, "")
            self.botoes_source.append(btn)
            y += BTN_H + BTN_GAP

        # ═══════════════════════════════════════
        # SEÇÃO 2: PLUGINS DE IA (só funcionam com IA ZONE ou IA FULL)
        # ═══════════════════════════════════════
        y += 8
        self.plugins_title_y = y  # Guardar posição do título
        y += 18  # Espaço para o título

        for i, (nome, tecla, cor, tooltip) in enumerate(self.MODOS[1:]):
            btn = Botao(MARGIN, y, BTN_W, BTN_H, nome, tecla, bg_btn, cor, tooltip)
            self.botoes_modo.append(btn)
            y += BTN_H + BTN_GAP

        # ═══════════════════════════════════════
        # SEÇÃO 3: RECORDING
        # ═══════════════════════════════════════
        y += 8
        self.recording_title_y = y
        y += 18

        controles = [
            ('FREEZE', 'F', (0, 200, 255)),
            ('REC', 'R', (100, 100, 230)),
            ('CAPTURE', 'S', (80, 180, 255)),
        ]
        for texto, tecla, cor in controles:
            btn = Botao(MARGIN, y, BTN_W, BTN_H, texto, tecla, bg_btn, cor, "")
            self.botoes_ctrl.append(btn)
            y += BTN_H + BTN_GAP

        # ═══════════════════════════════════════
        # SEÇÃO 4: SYSTEM
        # ═══════════════════════════════════════
        y += 8
        self.system_title_y = y
        y += 18

        # Zoom lado a lado
        half_w = (BTN_W - 6) // 2
        btn_zoom_plus = Botao(MARGIN, y, half_w, BTN_H, '+', None, bg_btn, (100, 200, 100), "")
        btn_zoom_minus = Botao(MARGIN + half_w + 6, y, half_w, BTN_H, '-', None, bg_btn, (200, 100, 100), "")
        self.botoes_view.append(btn_zoom_plus)
        self.botoes_view.append(btn_zoom_minus)
        y += BTN_H + BTN_GAP

        sistema_btns = [
            ('FULLSCREEN', 'X', (120, 120, 160)),
            ('HELP', '?', (120, 120, 160)),
            ('EXIT', 'Q', (160, 80, 80)),
        ]
        for texto, tecla, cor in sistema_btns:
            btn = Botao(MARGIN, y, BTN_W, BTN_H, texto, tecla, bg_btn, cor, "")
            self.botoes_ctrl.append(btn)
            y += BTN_H + BTN_GAP

    def _mouse_callback(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y

        sidebar_w = self.SIDEBAR_W if self.show_sidebar else 0
        click_in_sidebar = x < sidebar_w

        # ═══════════════════════════════════════
        # SELEÇÃO DE ROI (para IA ZONE)
        # ═══════════════════════════════════════
        if self.selecting_roi or (self.source_mode == 1 and self.process_roi):
            if click_in_sidebar and event == cv2.EVENT_LBUTTONDOWN:
                # Verificar se clicou em um botão de source (B-MODE, IA ZONE, IA FULL)
                clicked_source = False
                for i, btn in enumerate(self.botoes_source):
                    if btn.contem(x, y):
                        clicked_source = True
                        self.selecting_roi = False
                        self.roi_start = None
                        self.roi_end = None
                        self.roi_drag_handle = None
                        self._set_source_mode(i)
                        return

                # Se não clicou em source, cancelar seleção e voltar para B-MODE
                if not clicked_source:
                    self.selecting_roi = False
                    self.roi_start = None
                    self.roi_end = None
                    self.roi_drag_handle = None
                    self.source_mode = 0
                    self._invalidate_sidebar()
                    print("Seleção cancelada")
                return

            if not click_in_sidebar:
                # Converter coordenadas display -> imagem
                def display_to_image(dx, dy):
                    dx = dx - sidebar_w - self.display_offset_x
                    dy = dy - self.display_offset_y
                    if self.display_scale > 0:
                        return max(0, int(dx / self.display_scale)), max(0, int(dy / self.display_scale))
                    return dx, dy

                img_x, img_y = display_to_image(x, y)

                # Verificar se ROI já existe (para redimensionar/mover)
                if self.process_roi:
                    rx, ry, rw, rh = self.process_roi
                    handle_dist = 15  # Distância máxima para detectar handle

                    # Posições dos handles
                    handles = {
                        'tl': (rx, ry),           # Top-left
                        'tr': (rx + rw, ry),      # Top-right
                        'bl': (rx, ry + rh),      # Bottom-left
                        'br': (rx + rw, ry + rh), # Bottom-right
                    }

                    if event == cv2.EVENT_LBUTTONDOWN:
                        # Verificar clique nos handles
                        for handle_name, (hx, hy) in handles.items():
                            if abs(img_x - hx) < handle_dist and abs(img_y - hy) < handle_dist:
                                self.roi_drag_handle = handle_name
                                self.roi_drag_start = (img_x, img_y)
                                self.roi_original = self.process_roi
                                return

                        # Verificar clique dentro da ROI (para mover)
                        if rx < img_x < rx + rw and ry < img_y < ry + rh:
                            self.roi_drag_handle = 'move'
                            self.roi_drag_start = (img_x, img_y)
                            self.roi_original = self.process_roi
                            return

                        # Clique fora da ROI - iniciar nova seleção
                        self.roi_start = (img_x, img_y)
                        self.roi_end = self.roi_start
                        self.process_roi = None

                    elif event == cv2.EVENT_MOUSEMOVE:
                        if self.roi_drag_handle and self.roi_original:
                            ox, oy, ow, oh = self.roi_original
                            sx, sy = self.roi_drag_start
                            dx, dy = img_x - sx, img_y - sy

                            if self.roi_drag_handle == 'move':
                                # Mover toda a ROI
                                self.process_roi = (ox + dx, oy + dy, ow, oh)
                            elif self.roi_drag_handle == 'tl':
                                # Redimensionar pelo canto superior-esquerdo
                                new_x = ox + dx
                                new_y = oy + dy
                                new_w = ow - dx
                                new_h = oh - dy
                                if new_w > 30 and new_h > 30:
                                    self.process_roi = (new_x, new_y, new_w, new_h)
                            elif self.roi_drag_handle == 'tr':
                                # Redimensionar pelo canto superior-direito
                                new_y = oy + dy
                                new_w = ow + dx
                                new_h = oh - dy
                                if new_w > 30 and new_h > 30:
                                    self.process_roi = (ox, new_y, new_w, new_h)
                            elif self.roi_drag_handle == 'bl':
                                # Redimensionar pelo canto inferior-esquerdo
                                new_x = ox + dx
                                new_w = ow - dx
                                new_h = oh + dy
                                if new_w > 30 and new_h > 30:
                                    self.process_roi = (new_x, oy, new_w, new_h)
                            elif self.roi_drag_handle == 'br':
                                # Redimensionar pelo canto inferior-direito
                                new_w = ow + dx
                                new_h = oh + dy
                                if new_w > 30 and new_h > 30:
                                    self.process_roi = (ox, oy, new_w, new_h)
                        elif self.roi_start:
                            self.roi_end = (img_x, img_y)
                            x1, y1 = self.roi_start
                            x2, y2 = self.roi_end
                            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                                self.process_roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

                    elif event == cv2.EVENT_LBUTTONUP:
                        if self.roi_drag_handle:
                            self.roi_drag_handle = None
                            self.roi_drag_start = None
                            self.roi_original = None
                            if self.process_roi:
                                print(f"ROI ajustada: {self.process_roi[2]}x{self.process_roi[3]}")
                        elif self.roi_start:
                            self.roi_end = (img_x, img_y)
                            x1, y1 = self.roi_start
                            x2, y2 = self.roi_end
                            if abs(x2 - x1) > 30 and abs(y2 - y1) > 30:
                                self.process_roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                                print(f"ROI selecionada: {self.process_roi[2]}x{self.process_roi[3]} - Pressione ENTER para confirmar")
                            else:
                                print("Arraste para criar uma área maior (min 30x30)")
                            self.roi_start = None
                            self.roi_end = None
                else:
                    # Sem ROI - criar nova
                    if event == cv2.EVENT_LBUTTONDOWN:
                        self.roi_start = (img_x, img_y)
                        self.roi_end = self.roi_start
                    elif event == cv2.EVENT_MOUSEMOVE and self.roi_start:
                        self.roi_end = (img_x, img_y)
                        x1, y1 = self.roi_start
                        x2, y2 = self.roi_end
                        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                            self.process_roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                    elif event == cv2.EVENT_LBUTTONUP and self.roi_start:
                        self.roi_end = (img_x, img_y)
                        x1, y1 = self.roi_start
                        x2, y2 = self.roi_end
                        if abs(x2 - x1) > 30 and abs(y2 - y1) > 30:
                            self.process_roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                            print(f"ROI selecionada: {self.process_roi[2]}x{self.process_roi[3]} - Pressione ENTER para confirmar")
                        else:
                            print("Arraste para criar uma área maior (min 30x30)")
                        self.roi_start = None
                        self.roi_end = None
            return

        # ═══════════════════════════════════════
        # HOVER em todos os botões
        # ═══════════════════════════════════════
        all_btns = self.botoes_source + self.botoes_modo + self.botoes_ctrl + self.botoes_view
        for btn in all_btns:
            btn.hover = btn.contem(x, y)

        if event != cv2.EVENT_LBUTTONDOWN or not self.show_sidebar:
            return

        # ═══════════════════════════════════════
        # CLIQUE NOS BOTÕES SOURCE (B-MODE / IA ZONE / IA FULL)
        # ═══════════════════════════════════════
        for i, btn in enumerate(self.botoes_source):
            if btn.contem(x, y):
                self._set_source_mode(i)
                return

        # ═══════════════════════════════════════
        # CLIQUE NOS PLUGINS DE IA
        # ═══════════════════════════════════════
        for i, btn in enumerate(self.botoes_modo):
            if btn.contem(x, y):
                self._set_plugin(i)
                return

        # ═══════════════════════════════════════
        # CLIQUE NOS CONTROLES
        # ═══════════════════════════════════════
        for btn in self.botoes_ctrl:
            if btn.contem(x, y):
                if btn.texto == 'FREEZE':
                    self._toggle_pause()
                elif btn.texto == 'REC':
                    self._toggle_recording()
                elif btn.texto == 'CAPTURE':
                    self._screenshot()
                elif btn.texto == 'FULLSCREEN':
                    self._toggle_fullscreen()
                elif btn.texto == 'HELP':
                    self.show_help = not self.show_help
                elif btn.texto == 'EXIT':
                    self.running = False
                return

        # ═══════════════════════════════════════
        # CLIQUE NOS BOTÕES DE ZOOM
        # ═══════════════════════════════════════
        for btn in self.botoes_view:
            if btn.contem(x, y):
                if btn.texto == '+':
                    self._zoom_in()
                elif btn.texto == '-':
                    self._zoom_out()
                return

    def _invalidate_sidebar(self):
        """Marca sidebar para redesenho"""
        self._sidebar_dirty = True

    def _set_source_mode(self, mode):
        """Define o modo fonte: 0=B-MODE, 1=IA ZONE, 2=IA FULL"""
        if mode == self.source_mode:
            return

        # Iniciar transicao de fade
        self.transition_alpha = 0.3
        self.transition_target = 1.0
        self.last_mode_change = time.time()

        self.source_mode = mode
        self._invalidate_sidebar()

        if mode == 0:  # B-MODE
            self.process_roi = None
            self.selecting_roi = False
            print("B-MODE")

        elif mode == 1:  # IA ZONE
            # Manter ROI existente se houver, senao iniciar selecao
            if self.process_roi is None:
                self.selecting_roi = True
                print("IA ZONE - arraste na imagem para selecionar região")
            else:
                self.selecting_roi = False
                print(f"IA ZONE - usando ROI existente")
            # Carregar AI para que esteja pronta quando ROI for selecionada
            self._load_ai_if_needed()

        elif mode == 2:  # IA FULL
            self.process_roi = None
            self.selecting_roi = False
            self._load_ai_if_needed()
            print("IA FULL - IA aplicada em toda a imagem")

    def _set_plugin(self, idx):
        """Define o plugin de IA selecionado"""
        if idx == self.plugin_idx:
            return

        # Iniciar transicao de fade
        self.transition_alpha = 0.4
        self.transition_target = 1.0
        self.last_mode_change = time.time()

        self.plugin_idx = idx
        self._invalidate_sidebar()
        nome = self.MODOS[idx + 1][0]
        logger.debug(f"Plugin: {nome}")

        # Se estiver em modo IA (ZONE ou FULL), garantir que AI esta carregada
        if self.source_mode > 0:
            self._load_ai_if_needed()

        # Configurar modo na IA
        if self.ai:
            new_mode = self.AI_MODE_MAP.get(idx, 'needle')
            self.ai.set_mode(new_mode)
            # Resetar thread para limpar frames do modo anterior
            self.ai_thread.reset()
            logger.debug(f"AI modo: {new_mode}")

            # Carregar bloqueios nervosos quando NERVE é selecionado
            if idx == 1:  # NERVE
                self._load_nerve_blocks()
                if self.nerve_block_ids:
                    self._apply_nerve_block()

    def _load_ai_if_needed(self):
        """Carrega a IA se ainda não foi carregada"""
        if self.ai is None and not self.ai_loading:
            self.ai_loading = True
            self._invalidate_sidebar()
            try:
                from src.ai_processor import AIProcessor
                self.ai = AIProcessor()
                # Configurar na thread de AI
                self.ai_thread.set_ai(self.ai)
                # Configurar plugin atual usando mapeamento centralizado
                self.ai.set_mode(self.AI_MODE_MAP.get(self.plugin_idx, 'needle'))
                logger.info("IA carregada (thread separada)")
            except Exception as e:
                logger.error(f"Erro ao carregar IA: {e}", exc_info=True)
            finally:
                self.ai_loading = False
                self._invalidate_sidebar()

    def _toggle_pause(self):
        self.pause = not self.pause
        self._invalidate_sidebar()

    def _toggle_recording(self):
        self.recording = not self.recording
        self._invalidate_sidebar()
        if self.recording:
            self.recorder.start_recording()
        else:
            self.recorder.stop_recording()

    def _screenshot(self):
        if self.frame_original is not None:
            os.makedirs("captures", exist_ok=True)
            path = f"captures/usg_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(path, self.frame_original, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"Foto salva: {path}")

    def _toggle_biplane(self):
        if not self.biplane_active:
            if self.frame_original is not None:
                self.biplane_ref = self.frame_original.copy()
                self.biplane_active = True
                print("Biplane: ON (referencia capturada)")
        else:
            self.biplane_active = False
            self.biplane_ref = None
            print("Biplane: OFF")

    def _toggle_sidebar(self):
        self.show_sidebar = not self.show_sidebar
        self._invalidate_sidebar()

    # ═══════════════════════════════════════════════════════════════════════════
    # NERVE TRACK v2.0 - Gerenciamento de Bloqueios Nervosos
    # ═══════════════════════════════════════════════════════════════════════════

    def _load_nerve_blocks(self):
        """Carrega lista de bloqueios nervosos disponíveis."""
        if self.ai is None:
            return

        try:
            blocks = self.ai.get_available_nerve_blocks()
            if blocks:
                self.nerve_block_names = blocks
                self.nerve_block_ids = list(blocks.keys())
                logger.info(f"Carregados {len(self.nerve_block_ids)} bloqueios nervosos")
        except Exception as e:
            logger.warning(f"Erro ao carregar bloqueios: {e}")

    def _next_nerve_block(self):
        """Avança para o próximo bloqueio nervoso."""
        if not self.nerve_block_ids:
            self._load_nerve_blocks()

        if self.nerve_block_ids:
            self.nerve_block_idx = (self.nerve_block_idx + 1) % len(self.nerve_block_ids)
            self._apply_nerve_block()

    def _prev_nerve_block(self):
        """Volta para o bloqueio nervoso anterior."""
        if not self.nerve_block_ids:
            self._load_nerve_blocks()

        if self.nerve_block_ids:
            self.nerve_block_idx = (self.nerve_block_idx - 1) % len(self.nerve_block_ids)
            self._apply_nerve_block()

    def _apply_nerve_block(self):
        """Aplica o bloqueio nervoso selecionado."""
        if not self.nerve_block_ids or self.ai is None:
            return

        block_id = self.nerve_block_ids[self.nerve_block_idx]
        block_name = self.nerve_block_names.get(block_id, block_id)

        try:
            success = self.ai.set_nerve_block(block_id)
            if success:
                logger.info(f"Bloqueio: {block_name}")
                print(f"NERVE TRACK: {block_name}")
            self._invalidate_sidebar()
        except Exception as e:
            logger.error(f"Erro ao aplicar bloqueio: {e}")

    def _toggle_nerve_block_menu(self):
        """Mostra/esconde menu de seleção de bloqueio."""
        self.nerve_block_show_menu = not self.nerve_block_show_menu
        if self.nerve_block_show_menu and not self.nerve_block_ids:
            self._load_nerve_blocks()
        self._invalidate_sidebar()

    def _get_current_nerve_block_name(self) -> str:
        """Retorna o nome do bloqueio atualmente selecionado."""
        if not self.nerve_block_ids:
            return "Nenhum"
        block_id = self.nerve_block_ids[self.nerve_block_idx]
        return self.nerve_block_names.get(block_id, block_id)[:25]

    def _toggle_overlays(self):
        self.show_overlays = not self.show_overlays

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self._invalidate_sidebar()
        if self.fullscreen:
            cv2.setWindowProperty(self.janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self.janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.janela, 1400, 800)

    def _get_screen_size(self):
        """Obtem tamanho TOTAL da tela (para fullscreen real)."""
        try:
            from AppKit import NSScreen
            screen = NSScreen.mainScreen()
            frame = screen.frame()
            # OpenCV espera pontos (points), nao pixels fisicos
            return int(frame.size.width), int(frame.size.height)
        except Exception as e:
            logger.warning(f"Erro ao obter tamanho da tela: {e}")
            return 1920, 1080  # Fallback

    def _zoom_in(self):
        self.zoom_level = min(3.0, self.zoom_level + 0.1)
        print(f"Zoom: {self.zoom_level:.1f}x")

    def _zoom_out(self):
        self.zoom_level = max(0.5, self.zoom_level - 0.1)
        print(f"Zoom: {self.zoom_level:.1f}x")

    def _pan_up(self):
        self.pan_y -= 20

    def _pan_down(self):
        self.pan_y += 20

    def _pan_left(self):
        self.pan_x -= 20

    def _pan_right(self):
        self.pan_x += 20

    def _reset_view(self):
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        print("View resetada")

    def _update_transitions(self):
        """Atualiza animacoes de transicao"""
        # Atualizar fade
        if self.transition_alpha < self.transition_target:
            self.transition_alpha = min(self.transition_target,
                                       self.transition_alpha + self.transition_speed)
        elif self.transition_alpha > self.transition_target:
            self.transition_alpha = max(self.transition_target,
                                       self.transition_alpha - self.transition_speed)

        # Atualizar pulse (ciclo de 0 a 1)
        self.pulse_phase = (self.pulse_phase + 0.08) % 1.0

    def _apply_transition_effect(self, frame):
        """Aplica efeito de fade/transicao ao frame"""
        if self.transition_alpha >= 0.99:
            return frame

        # Aplicar fade
        output = frame.copy()
        alpha = self.transition_alpha
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=0)
        return output

    def _apply_roi_preset(self, preset):
        """Aplica um preset de ROI.
        preset: '50%', '75%', '100%', 'square'
        """
        if self.frame_original is None:
            return

        h, w = self.frame_original.shape[:2]

        if preset == '50%':
            # ROI centralizada com 50% da tela
            rw, rh = w // 2, h // 2
            rx, ry = w // 4, h // 4
        elif preset == '75%':
            # ROI centralizada com 75% da tela
            rw, rh = int(w * 0.75), int(h * 0.75)
            rx, ry = (w - rw) // 2, (h - rh) // 2
        elif preset == '100%':
            # ROI cobrindo toda a tela
            rx, ry = 0, 0
            rw, rh = w, h
        elif preset == 'square':
            # ROI quadrada no centro (usa menor dimensao)
            size = min(w, h) // 2
            rx = (w - size) // 2
            ry = (h - size) // 2
            rw, rh = size, size
        else:
            return

        self.process_roi = (rx, ry, rw, rh)
        print(f"Preset ROI {preset}: {rw}x{rh} px")

    def _desenhar_sidebar(self, altura, pulse_phase=0):
        sidebar = np.zeros((altura, self.SIDEBAR_W, 3), dtype=np.uint8)
        sidebar[:] = (15, 15, 20)

        # Borda direita
        cv2.line(sidebar, (self.SIDEBAR_W - 1, 0), (self.SIDEBAR_W - 1, altura), (30, 30, 40), 1)

        # HEADER (com LINE_AA para visual premium)
        cv2.rectangle(sidebar, (0, 0), (self.SIDEBAR_W, 64), (20, 20, 28), -1)
        cv2.putText(sidebar, "USG", (28, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(sidebar, "FLOW", (88, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(sidebar, "AI Ultrasound", (28, 54), cv2.FONT_HERSHEY_DUPLEX, 0.35, (70, 70, 90), 1, cv2.LINE_AA)
        cv2.line(sidebar, (0, 64), (self.SIDEBAR_W, 64), (0, 180, 220), 2)

        # ═══════════════════════════════════════
        # GRUPO EXCLUSIVO (B-MODE / IA ZONE / IA FULL)
        # ═══════════════════════════════════════
        for i, btn in enumerate(self.botoes_source):
            btn.ativo = (i == self.source_mode) or (i == 1 and self.selecting_roi)
            btn.desenhar(sidebar, pulse_phase)

        # TÍTULO "PLUGINS IA"
        ia_ativa = self.source_mode > 0
        titulo_cor = (110, 110, 130) if ia_ativa else (55, 55, 70)
        cv2.line(sidebar, (30, self.plugins_title_y), (self.SIDEBAR_W - 30, self.plugins_title_y), (35, 35, 45), 1)
        cv2.putText(sidebar, "PLUGINS IA", (30, self.plugins_title_y + 15),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, titulo_cor, 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # PLUGINS DE IA
        # ═══════════════════════════════════════
        for i, btn in enumerate(self.botoes_modo):
            btn.ativo = (i == self.plugin_idx) and ia_ativa
            if not ia_ativa:
                btn.hover = False
            btn.desenhar(sidebar, pulse_phase)

        # ═══════════════════════════════════════
        # INFO DO BLOCO NERVOSO (quando NERVE ativo)
        # ═══════════════════════════════════════
        if ia_ativa and self.plugin_idx == 1 and self.nerve_block_ids:
            # Posição abaixo do botão NERVE (índice 1)
            nerve_btn = self.botoes_modo[1]
            info_y = nerve_btn.y2 + 2

            # Background sutil
            cv2.rectangle(sidebar, (nerve_btn.x + 20, info_y),
                         (nerve_btn.x2 - 10, info_y + 16), (28, 35, 40), -1)

            # Ícone de nervo e nome do bloco
            block_name = self._get_current_nerve_block_name()
            cv2.putText(sidebar, f"→ {block_name}", (nerve_btn.x + 25, info_y + 11),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, (50, 220, 220), 1, cv2.LINE_AA)

            # Indicador de navegação (< >)
            nav_text = f"< {self.nerve_block_idx + 1}/{len(self.nerve_block_ids)} >"
            cv2.putText(sidebar, nav_text, (nerve_btn.x2 - 60, info_y + 11),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 100, 120), 1, cv2.LINE_AA)

        # TÍTULO "RECORDING"
        cv2.line(sidebar, (30, self.recording_title_y), (self.SIDEBAR_W - 30, self.recording_title_y), (35, 35, 45), 1)
        cv2.putText(sidebar, "RECORDING", (30, self.recording_title_y + 15),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (110, 110, 130), 1, cv2.LINE_AA)

        # Botões de recording (FREEZE, REC, CAPTURE)
        for btn in self.botoes_ctrl[:3]:
            if btn.texto == 'FREEZE':
                btn.ativo = self.pause
            elif btn.texto == 'REC':
                btn.ativo = self.recording
            btn.desenhar(sidebar, pulse_phase)

        # TÍTULO "SYSTEM"
        cv2.line(sidebar, (30, self.system_title_y), (self.SIDEBAR_W - 30, self.system_title_y), (35, 35, 45), 1)
        cv2.putText(sidebar, "SYSTEM", (30, self.system_title_y + 15),
                   cv2.FONT_HERSHEY_DUPLEX, 0.4, (110, 110, 130), 1, cv2.LINE_AA)

        # Zoom
        for btn in self.botoes_view:
            btn.desenhar(sidebar, pulse_phase)

        # Botões de sistema (FULLSCREEN, HELP, EXIT)
        for btn in self.botoes_ctrl[3:]:
            if btn.texto == 'FULLSCREEN':
                btn.ativo = self.fullscreen
            elif btn.texto == 'HELP':
                btn.ativo = self.show_help
            btn.desenhar(sidebar, pulse_phase)

        # FOOTER (mais compacto)
        footer_y = altura - 60
        cv2.rectangle(sidebar, (0, footer_y), (self.SIDEBAR_W, altura), (12, 12, 18), -1)
        cv2.line(sidebar, (28, footer_y + 2), (self.SIDEBAR_W - 28, footer_y + 2), (35, 35, 45), 1)

        # Status + FPS na mesma linha
        sy = footer_y + 18
        if self.captura.connected:
            cv2.circle(sidebar, (38, sy - 3), 4, (80, 220, 80), -1)
            cv2.putText(sidebar, "ONLINE", (52, sy), cv2.FONT_HERSHEY_DUPLEX, 0.35, (80, 220, 80), 1, cv2.LINE_AA)
        else:
            cv2.circle(sidebar, (38, sy - 3), 4, (80, 80, 100), -1)
            cv2.putText(sidebar, "WAITING", (52, sy), cv2.FONT_HERSHEY_DUPLEX, 0.35, (80, 80, 100), 1, cv2.LINE_AA)

        # FPS ao lado
        fps_cor = (80, 220, 80) if self.fps >= 25 else (0, 180, 220) if self.fps >= 15 else (100, 100, 180)
        cv2.putText(sidebar, f"{self.fps} FPS", (140, sy), cv2.FONT_HERSHEY_DUPLEX, 0.38, fps_cor, 1, cv2.LINE_AA)
        fps_bar_w = min(self.fps * 2, 80)
        cv2.rectangle(sidebar, (195, sy - 7), (self.SIDEBAR_W - 15, sy - 1), (30, 30, 40), -1)
        cv2.rectangle(sidebar, (195, sy - 7), (195 + fps_bar_w, sy - 1), fps_cor, -1)

        # Modo
        sy += 18
        source_names = ['B-MODE', 'IA ZONE', 'IA FULL']
        source_cors = [(140, 140, 160), (255, 200, 60), (100, 220, 140)]
        modo_nome = source_names[self.source_mode]
        modo_cor = source_cors[self.source_mode]

        # Indicador de carregamento de IA
        if self.ai_loading:
            # Animacao de loading (pontos rotativos)
            dots = '.' * (int(time.time() * 3) % 4)
            cv2.putText(sidebar, f"Loading AI{dots}", (180, sy),
                       cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 200, 60), 1, cv2.LINE_AA)
        elif self.source_mode > 0:
            modo_nome = f"{modo_nome} | {self.MODOS[self.plugin_idx + 1][0]}"
            # Tempo de processamento AI
            ai_time = self.ai_thread.last_process_time
            if ai_time > 0:
                ai_time_cor = (100, 220, 100) if ai_time < 50 else (0, 180, 220) if ai_time < 100 else (100, 100, 200)
                cv2.putText(sidebar, f"AI: {ai_time:.0f}ms", (180, sy), cv2.FONT_HERSHEY_DUPLEX, 0.3, ai_time_cor, 1, cv2.LINE_AA)
        cv2.putText(sidebar, modo_nome, (38, sy), cv2.FONT_HERSHEY_DUPLEX, 0.38, modo_cor, 1, cv2.LINE_AA)

        # REC com tempo de gravacao
        if self.recording:
            status = self.recorder.get_recording_status()
            elapsed = status.get('elapsed_seconds', 0)
            max_dur = status.get('max_duration', 30)

            # Indicador piscante
            if int(time.time() * 2) % 2:
                cv2.circle(sidebar, (38, footer_y + 42), 5, (0, 0, 255), -1)

            # Tempo formatado (MM:SS)
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"REC {mins:02d}:{secs:02d}"
            cv2.putText(sidebar, time_str, (52, footer_y + 46),
                       cv2.FONT_HERSHEY_DUPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

            # Barra de progresso do tempo restante
            progress = min(elapsed / max_dur, 1.0)
            bar_w = int(80 * progress)
            cv2.rectangle(sidebar, (140, footer_y + 39), (220, footer_y + 47), (30, 30, 40), -1)
            cv2.rectangle(sidebar, (140, footer_y + 39), (140 + bar_w, footer_y + 47), (0, 0, 255), -1)

        return sidebar

    def _desenhar_roi_selection(self, frame):
        """Desenha overlay profissional de seleção de ROI."""
        if not self.selecting_roi and self.process_roi is None:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Determinar coordenadas da ROI
        if self.roi_start and self.roi_end:
            # Seleção em andamento
            x1, y1 = self.roi_start
            x2, y2 = self.roi_end
            rx, ry = min(x1, x2), min(y1, y2)
            rw, rh = abs(x2 - x1), abs(y2 - y1)
        elif self.process_roi:
            # ROI já definida
            rx, ry, rw, rh = self.process_roi
        else:
            # Nenhuma ROI ainda - mostrar instrução
            if self.selecting_roi:
                # Fundo escurecido
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                # Texto de instrução centralizado
                texto = "ARRASTE PARA SELECIONAR A REGIAO"
                (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
                tx = (w - tw) // 2
                ty = h // 2
                cv2.putText(frame, texto, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                texto2 = "ESC para cancelar"
                (tw2, _), _ = cv2.getTextSize(texto2, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                cv2.putText(frame, texto2, ((w - tw2) // 2, ty + 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
            return frame

        if rw < self.MIN_ROI_SIZE or rh < self.MIN_ROI_SIZE:
            return frame

        # ═══════════════════════════════════════
        # OVERLAY ESCURO FORA DA SELEÇÃO
        # ═══════════════════════════════════════
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (rx, ry), (rx + rw, ry + rh), 255, -1)

        # Escurecer área fora da seleção
        dark_overlay = frame.copy()
        dark_overlay[mask == 0] = (dark_overlay[mask == 0] * 0.4).astype(np.uint8)
        frame = dark_overlay

        # ═══════════════════════════════════════
        # BORDA ANIMADA (MARCHING ANTS)
        # ═══════════════════════════════════════
        self.roi_animation_frame = (self.roi_animation_frame + 1) % 16
        offset = self.roi_animation_frame

        # Linha tracejada animada
        for i in range(0, rw + rh * 2, 8):
            # Topo
            if i < rw:
                px = rx + (i + offset) % rw
                cv2.line(frame, (px, ry), (min(px + 4, rx + rw), ry), (0, 255, 255), 2)
            # Direita
            if i < rh:
                py = ry + (i + offset) % rh
                cv2.line(frame, (rx + rw, py), (rx + rw, min(py + 4, ry + rh)), (0, 255, 255), 2)
            # Baixo
            if i < rw:
                px = rx + rw - (i + offset) % rw
                cv2.line(frame, (px, ry + rh), (max(px - 4, rx), ry + rh), (0, 255, 255), 2)
            # Esquerda
            if i < rh:
                py = ry + rh - (i + offset) % rh
                cv2.line(frame, (rx, py), (rx, max(py - 4, ry)), (0, 255, 255), 2)

        # Borda sólida fina por baixo
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 200, 200), 1)

        # ═══════════════════════════════════════
        # HANDLES NOS CANTOS (para redimensionar)
        # ═══════════════════════════════════════
        handle_size = 8
        handle_color = (255, 255, 255)
        handle_border = (0, 0, 0)

        handles = [
            (rx, ry),                    # Top-left
            (rx + rw, ry),               # Top-right
            (rx, ry + rh),               # Bottom-left
            (rx + rw, ry + rh),          # Bottom-right
        ]

        for hx, hy in handles:
            cv2.rectangle(frame, (hx - handle_size//2, hy - handle_size//2),
                         (hx + handle_size//2, hy + handle_size//2), handle_border, -1)
            cv2.rectangle(frame, (hx - handle_size//2 + 1, hy - handle_size//2 + 1),
                         (hx + handle_size//2 - 1, hy + handle_size//2 - 1), handle_color, -1)

        # ═══════════════════════════════════════
        # CROSSHAIR CENTRAL (guias de centro)
        # ═══════════════════════════════════════
        cx, cy = rx + rw // 2, ry + rh // 2
        cross_size = min(rw, rh) // 6
        cross_color = (0, 255, 255)

        # Linhas do crosshair (tracejadas)
        for i in range(0, cross_size, 6):
            # Horizontal
            cv2.line(frame, (cx - cross_size + i, cy), (cx - cross_size + i + 3, cy), cross_color, 1)
            cv2.line(frame, (cx + i, cy), (cx + i + 3, cy), cross_color, 1)
            # Vertical
            cv2.line(frame, (cx, cy - cross_size + i), (cx, cy - cross_size + i + 3), cross_color, 1)
            cv2.line(frame, (cx, cy + i), (cx, cy + i + 3), cross_color, 1)

        # Círculo central pequeno
        cv2.circle(frame, (cx, cy), 3, cross_color, 1)

        # ═══════════════════════════════════════
        # GRID DE REGRA DOS TERÇOS (linhas sutis)
        # ═══════════════════════════════════════
        if rw > self.MIN_ROI_GRID and rh > self.MIN_ROI_GRID:  # Grid de tercos
            third_color = (60, 60, 60)
            # Linhas verticais (terços)
            for i in [1, 2]:
                lx = rx + (rw * i) // 3
                cv2.line(frame, (lx, ry), (lx, ry + rh), third_color, 1)
            # Linhas horizontais (terços)
            for i in [1, 2]:
                ly = ry + (rh * i) // 3
                cv2.line(frame, (rx, ly), (rx + rw, ly), third_color, 1)

        # ═══════════════════════════════════════
        # INFORMAÇÕES DA SELEÇÃO
        # ═══════════════════════════════════════
        # Dimensões com mais detalhes
        dim_text = f"{rw} x {rh} px"
        area_text = f"Area: {(rw * rh) // 1000}K px"

        (tw, th), _ = cv2.getTextSize(dim_text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)

        # Posição do texto (acima ou abaixo da seleção)
        if ry > 50:
            text_y = ry - 8
        else:
            text_y = ry + rh + 22
        text_x = rx + (rw - tw) // 2

        # Background do texto com cantos arredondados (simulado)
        pad = 8
        cv2.rectangle(frame, (text_x - pad, text_y - th - pad), (text_x + tw + pad, text_y + pad), (20, 20, 25), -1)
        cv2.rectangle(frame, (text_x - pad, text_y - th - pad), (text_x + tw + pad, text_y + pad), (0, 200, 200), 1)
        cv2.putText(frame, dim_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # ═══════════════════════════════════════
        # BARRA DE STATUS INFERIOR (premium)
        # ═══════════════════════════════════════
        if self.selecting_roi:
            bar_h = 55
            bar_y = h - bar_h

            # Fundo gradiente simulado
            overlay_bar = frame.copy()
            cv2.rectangle(overlay_bar, (0, bar_y), (w, h), (15, 15, 20), -1)
            cv2.addWeighted(overlay_bar, 0.9, frame, 0.1, 0, frame)

            # Linha de separação
            cv2.line(frame, (0, bar_y), (w, bar_y), (0, 180, 200), 2)

            if self.process_roi:
                # ROI definida - mostrar controles e presets
                icon_y = bar_y + 22

                # ENTER
                cv2.rectangle(frame, (20, icon_y - 14), (90, icon_y + 10), (0, 150, 150), -1)
                cv2.putText(frame, "ENTER", (28, icon_y + 2), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Confirmar", (95, icon_y + 2), cv2.FONT_HERSHEY_DUPLEX, 0.35, (120, 120, 130), 1, cv2.LINE_AA)

                # ESC
                cv2.rectangle(frame, (180, icon_y - 14), (230, icon_y + 10), (100, 60, 60), -1)
                cv2.putText(frame, "ESC", (192, icon_y + 2), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Cancelar", (235, icon_y + 2), cv2.FONT_HERSHEY_DUPLEX, 0.35, (120, 120, 130), 1, cv2.LINE_AA)

                # Separador
                cv2.line(frame, (320, bar_y + 10), (320, h - 10), (50, 50, 60), 1)

                # Presets de ROI
                cv2.putText(frame, "PRESETS:", (340, icon_y + 2), cv2.FONT_HERSHEY_DUPLEX, 0.35, (80, 80, 100), 1, cv2.LINE_AA)

                preset_x = 420
                presets = [
                    ('P', '50%', (60, 140, 140)),
                    ('O', '75%', (60, 140, 140)),
                    ('L', '100%', (60, 140, 140)),
                    ('K', 'SQ', (60, 140, 140)),
                ]
                for key, label, color in presets:
                    cv2.rectangle(frame, (preset_x, icon_y - 14), (preset_x + 22, icon_y + 10), color, -1)
                    cv2.putText(frame, key, (preset_x + 6, icon_y + 2), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, label, (preset_x + 28, icon_y + 2), cv2.FONT_HERSHEY_DUPLEX, 0.35, (140, 140, 150), 1, cv2.LINE_AA)
                    preset_x += 70

                # Dicas de interação
                icon_y2 = bar_y + 42
                cv2.putText(frame, "Arraste os cantos para redimensionar | Arraste o centro para mover",
                           (20, icon_y2), cv2.FONT_HERSHEY_DUPLEX, 0.32, (80, 80, 100), 1, cv2.LINE_AA)
            else:
                # Sem ROI - instrução de arraste e presets
                icon_y = bar_y + 22

                cv2.putText(frame, "Clique e arraste para selecionar a regiao de interesse",
                           (20, icon_y), cv2.FONT_HERSHEY_DUPLEX, 0.4, (150, 200, 200), 1, cv2.LINE_AA)

                # Separador
                cv2.line(frame, (420, bar_y + 10), (420, h - 10), (50, 50, 60), 1)

                # Presets rapidos
                cv2.putText(frame, "PRESETS:", (440, icon_y), cv2.FONT_HERSHEY_DUPLEX, 0.35, (80, 80, 100), 1, cv2.LINE_AA)

                preset_x = 520
                presets = [('P', '50%'), ('O', '75%'), ('L', '100%'), ('K', 'SQ')]
                for key, label in presets:
                    cv2.rectangle(frame, (preset_x, icon_y - 14), (preset_x + 22, icon_y + 10), (60, 140, 140), -1)
                    cv2.putText(frame, key, (preset_x + 6, icon_y), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, label, (preset_x + 28, icon_y), cv2.FONT_HERSHEY_DUPLEX, 0.35, (140, 140, 150), 1, cv2.LINE_AA)
                    preset_x += 65

                # ESC
                cv2.rectangle(frame, (w - 100, icon_y - 14), (w - 50, icon_y + 10), (100, 60, 60), -1)
                cv2.putText(frame, "ESC", (w - 92, icon_y), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Sair", (w - 45, icon_y), cv2.FONT_HERSHEY_DUPLEX, 0.35, (120, 120, 130), 1, cv2.LINE_AA)

        return frame

    def _desenhar_instrucoes(self, frame):
        """Desenha instrucoes contextuais na parte inferior - OTIMIZADO."""
        if not self.show_instructions or not self.show_overlays:
            return frame

        h, w = frame.shape[:2]
        # Se IA ativa, mostrar instrução do plugin
        if self.source_mode > 0:
            modo_nome = self.MODOS[self.plugin_idx + 1][0]
        else:
            modo_nome = 'B-MODE'
        instrucao = MODE_INSTRUCTIONS.get(modo_nome, "")

        if not instrucao:
            return frame

        # Fundo semi-transparente OTIMIZADO (blend apenas na area necessaria)
        box_h = 40
        roi = frame[h - box_h:h, 0:w]
        # Escurecer apenas a ROI (mais rapido que full blend)
        roi[:] = (roi * 0.3).astype(np.uint8)

        # Texto
        cv2.putText(frame, instrucao, (20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        return frame

    def _desenhar_ajuda(self, frame):
        """Desenha modal de ajuda."""
        h, w = frame.shape[:2]

        # Lista de atalhos
        atalhos = [
            ("1", "B-MODE (sem IA)"),
            ("C", "IA ZONE (selecionar regiao)"),
            ("A", "IA FULL (IA em toda tela)"),
            ("2-9,0,V", "Selecionar plugin IA"),
            ("F", "Freeze/Pause"),
            ("R", "Iniciar/Parar gravacao"),
            ("S", "Screenshot (PNG)"),
            ("B", "Toggle Biplane"),
            ("H", "Esconder/Mostrar overlays"),
            ("+/-", "Zoom In/Out"),
            ("?", "Esta ajuda"),
            ("X", "Fullscreen"),
            ("Q/ESC", "Sair"),
        ]

        # Calcular tamanho do modal baseado no conteudo
        num_items = len(atalhos)
        line_h = min(26, max(18, (h - 150) // (num_items + 3)))
        modal_h = min(h - 80, 60 + num_items * line_h + 40)
        modal_w = min(w - 100, 450)
        modal_x = (w - modal_w) // 2
        modal_y = (h - modal_h) // 2

        # Fundo escuro
        overlay = frame.copy()
        cv2.rectangle(overlay, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)
        cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (50, 50, 60), 2)

        # Titulo
        cv2.putText(frame, "ATALHOS DO TECLADO", (modal_x + 20, modal_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2, cv2.LINE_AA)

        # Desenhar atalhos
        y = modal_y + 65
        for tecla, desc in atalhos:
            cv2.putText(frame, f"[{tecla}]", (modal_x + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, desc, (modal_x + 120, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
            y += line_h

        # Instrucao para fechar
        cv2.putText(frame, "Pressione ? para fechar", (modal_x + 20, modal_y + modal_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 90), 1, cv2.LINE_AA)

        return frame

    def _desenhar_nerve_block_menu(self, frame):
        """Desenha menu de seleção de bloqueios nervosos (28 disponíveis)."""
        if not self.nerve_block_show_menu:
            return frame

        h, w = frame.shape[:2]

        # Agrupar bloqueios por região
        regions = {
            'CERVICAL': [],
            'BRACHIAL': [],
            'THORACIC': [],
            'ABDOMEN': [],
            'LOWER LIMB': []
        }

        for block_id in self.nerve_block_ids:
            name = self.nerve_block_names.get(block_id, block_id)
            # Determinar região pelo nome
            name_lower = block_id.lower()
            if any(x in name_lower for x in ['cervical', 'interescalenico', 'supraclavicular']):
                regions['CERVICAL'].append((block_id, name))
            elif any(x in name_lower for x in ['axilar', 'infraclavicular', 'radial', 'mediano', 'ulnar', 'braquial']):
                regions['BRACHIAL'].append((block_id, name))
            elif any(x in name_lower for x in ['paravertebral', 'serratus', 'pecs', 'intercostal', 'erector', 'esp']):
                regions['THORACIC'].append((block_id, name))
            elif any(x in name_lower for x in ['tap', 'quadratus', 'ilioinguinal', 'reto', 'abdominal']):
                regions['ABDOMEN'].append((block_id, name))
            else:
                regions['LOWER LIMB'].append((block_id, name))

        # Calcular tamanho do modal
        total_items = sum(len(v) for v in regions.values()) + len([k for k, v in regions.items() if v])
        line_h = min(22, max(16, (h - 200) // (total_items + 4)))
        modal_h = min(h - 60, 80 + total_items * line_h + 60)
        modal_w = min(w - 80, 500)
        modal_x = (w - modal_w) // 2
        modal_y = (h - modal_h) // 2

        # Fundo escuro com borda cyan
        overlay = frame.copy()
        cv2.rectangle(overlay, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (15, 20, 25), -1)
        cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)
        cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (220, 220, 50), 2)

        # Título
        cv2.putText(frame, "NERVE TRACK - BLOQUEIOS", (modal_x + 20, modal_y + 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (50, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, f"28 bloqueios nervosos disponíveis", (modal_x + 20, modal_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 160), 1, cv2.LINE_AA)

        # Desenhar regiões e bloqueios
        y = modal_y + 75
        region_colors = {
            'CERVICAL': (100, 200, 255),
            'BRACHIAL': (100, 255, 150),
            'THORACIC': (255, 180, 100),
            'ABDOMEN': (255, 120, 180),
            'LOWER LIMB': (180, 140, 255)
        }

        current_block_id = self.nerve_block_ids[self.nerve_block_idx] if self.nerve_block_ids else None

        for region_name, blocks in regions.items():
            if not blocks:
                continue

            # Título da região
            region_color = region_colors.get(region_name, (200, 200, 200))
            cv2.putText(frame, f"▸ {region_name}", (modal_x + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, region_color, 1, cv2.LINE_AA)
            y += line_h

            # Bloqueios da região
            for block_id, block_name in blocks:
                is_selected = (block_id == current_block_id)

                # Background para item selecionado
                if is_selected:
                    cv2.rectangle(frame, (modal_x + 25, y - line_h + 4),
                                 (modal_x + modal_w - 20, y + 2), (40, 60, 60), -1)

                # Nome do bloqueio
                text_color = (50, 220, 220) if is_selected else (160, 160, 170)
                prefix = "● " if is_selected else "  "
                display_name = block_name[:35] + "..." if len(block_name) > 35 else block_name
                cv2.putText(frame, f"{prefix}{display_name}", (modal_x + 30, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)
                y += line_h

        # Instruções de navegação
        y = modal_y + modal_h - 35
        cv2.line(frame, (modal_x + 20, y - 10), (modal_x + modal_w - 20, y - 10), (40, 40, 50), 1)
        cv2.putText(frame, "[</>] Navegar   [N] Fechar   [ENTER] Selecionar", (modal_x + 20, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 120), 1, cv2.LINE_AA)

        return frame

    def _aplicar_zoom_pan(self, frame):
        """Aplica zoom e pan ao frame."""
        if self.zoom_level == 1.0 and self.pan_x == 0 and self.pan_y == 0:
            return frame

        h, w = frame.shape[:2]
        new_w = int(w * self.zoom_level)
        new_h = int(h * self.zoom_level)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Calcular crop com pan
        cx = (new_w - w) // 2 - self.pan_x
        cy = (new_h - h) // 2 - self.pan_y

        # Limites
        cx = max(0, min(new_w - w, cx))
        cy = max(0, min(new_h - h, cy))

        # Crop
        cropped = resized[cy:cy+h, cx:cx+w]

        return cropped

    def rodar(self):
        """Loop principal OTIMIZADO - non-blocking AI, cache de sidebar."""
        frame_count = 0
        fps_time = time.time()
        last_sidebar = None
        last_sidebar_height = 0

        # Variaveis para AI non-blocking
        ai_submitted = False
        ai_display_frame = None

        while self.running:
            loop_start = time.time()

            # Atualizar animacoes (apenas se necessario)
            if self.transition_alpha != self.transition_target:
                self._update_transitions()

            # Captura (otimizada - so copia se frame novo)
            if not self.pause:
                new_frame, new_count = self.captura.get_frame_if_new(self._last_frame_count)
                if new_frame is not None:
                    self.frame_original = new_frame
                    self._last_frame_count = new_count
                    if self.recording:
                        self.recorder.add_frame(new_frame)

            # Montar display
            if self.frame_original is not None:
                h, w = self.frame_original.shape[:2]

                # ══════════════════════════════════════════════════════════
                # AI NON-BLOCKING (thread separada)
                # ══════════════════════════════════════════════════════════
                ia_ativa = self.source_mode > 0 and self.ai is not None

                if ia_ativa:
                    # Verificar se ha resultado pronto da AI thread
                    ai_result = self.ai_thread.get_result()
                    if ai_result is not None:
                        ai_display_frame = ai_result
                        ai_submitted = False

                    # Submeter novo frame se nao esta processando
                    if not ai_submitted and not self.ai_thread.is_processing:
                        roi = self.process_roi if self.source_mode == 1 else None
                        if self.ai_thread.submit_frame(self.frame_original, roi):
                            ai_submitted = True

                    # Usar ultimo frame processado ou original
                    if ai_display_frame is not None:
                        display_img = ai_display_frame.copy()
                    else:
                        display_img = self.frame_original.copy()
                else:
                    display_img = self.frame_original.copy()
                    ai_display_frame = None  # Reset quando AI desativada

                # Sistema profissional de seleção de ROI
                if self.selecting_roi or (self.source_mode == 1 and self.process_roi):
                    display_img = self._desenhar_roi_selection(display_img)

                # Zoom/Pan
                display_img = self._aplicar_zoom_pan(display_img)

                # Biplane
                if self.biplane_active and self.biplane_ref is not None:
                    ref_resized = cv2.resize(self.biplane_ref, (w//2, h))
                    curr_resized = cv2.resize(display_img, (w//2, h))
                    display_img = np.hstack([ref_resized, curr_resized])
                    if self.show_overlays:
                        cv2.putText(display_img, "REF", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(display_img, "LIVE", (w//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Instrucoes
                if self.show_overlays:
                    display_img = self._desenhar_instrucoes(display_img)

                # Aplicar efeito de transicao (fade)
                display_img = self._apply_transition_effect(display_img)

                # Escalar imagem para preencher área disponível
                if self.fullscreen:
                    # Fullscreen: usar tamanho da tela
                    screen_w, screen_h = self._get_screen_size()
                    sidebar_w = self.SIDEBAR_W if self.show_sidebar else 0
                    target_w = screen_w - sidebar_w
                    target_h = screen_h
                else:
                    # Janela normal: 1400x800
                    sidebar_w = self.SIDEBAR_W if self.show_sidebar else 0
                    target_w = 1400 - sidebar_w
                    target_h = 800

                img_h, img_w = display_img.shape[:2]

                # Proteger contra dimensões inválidas
                if img_w < 1 or img_h < 1:
                    continue

                # Calcular escala mantendo proporção
                scale = min(target_w / img_w, target_h / img_h)
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)

                # Guardar info de escala para conversão de coordenadas
                self.display_scale = scale
                self.display_offset_x = (target_w - new_w) // 2
                self.display_offset_y = (target_h - new_h) // 2

                # Redimensionar (usar INTER_AREA para downscale, INTER_LINEAR para upscale)
                if new_w < img_w or new_h < img_h:
                    display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # Reutilizar canvas se tamanho nao mudou (evita alocacao)
                canvas_needed_size = (target_h, target_w)
                if self._canvas_cache is None or self._canvas_size != canvas_needed_size:
                    # Liberar canvas antigo explicitamente para evitar memory leak
                    if self._canvas_cache is not None:
                        del self._canvas_cache
                    self._canvas_cache = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    self._canvas_size = canvas_needed_size
                    # Preencher uma vez com cor de fundo
                    self._canvas_cache[:] = (12, 12, 15)
                else:
                    # Limpar apenas area onde sera desenhada a imagem (mais rapido)
                    self._canvas_cache[self.display_offset_y:self.display_offset_y+new_h,
                                       self.display_offset_x:self.display_offset_x+new_w] = (12, 12, 15)

                # Centralizar imagem no canvas
                self._canvas_cache[self.display_offset_y:self.display_offset_y+new_h,
                                   self.display_offset_x:self.display_offset_x+new_w] = display_img
                display_img = self._canvas_cache

                # ══════════════════════════════════════════════════════════
                # SIDEBAR COM CACHE (evita redesenho desnecessario)
                # ══════════════════════════════════════════════════════════
                if self.show_sidebar:
                    # Redesenhar apenas se necessario
                    if self._sidebar_dirty or last_sidebar is None or last_sidebar_height != target_h:
                        sidebar = self._desenhar_sidebar(target_h, self.pulse_phase)
                        last_sidebar = sidebar.copy()
                        last_sidebar_height = target_h
                        self._sidebar_dirty = False
                    else:
                        sidebar = last_sidebar

                    display = np.hstack([sidebar, display_img])
                else:
                    display = display_img

                # Ajuda
                if self.show_help:
                    display = self._desenhar_ajuda(display)

                # Menu de bloqueios nervosos
                if self.nerve_block_show_menu:
                    display = self._desenhar_nerve_block_menu(display)

            else:
                # Tela de espera
                if self.fullscreen:
                    screen_w, screen_h = self._get_screen_size()
                    h = screen_h
                    w = screen_w - (self.SIDEBAR_W if self.show_sidebar else 0)
                else:
                    h = 800
                    w = 1200

                espera = np.zeros((h, w, 3), dtype=np.uint8)
                espera[:] = (12, 12, 15)
                cx, cy = w // 2, h // 2
                cv2.putText(espera, "AGUARDANDO SINAL", (cx - 180, cy - 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, (80, 80, 90), 2, cv2.LINE_AA)
                cv2.putText(espera, "Conecte iPhone via QuickTime", (cx - 190, cy + 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 165, 0), 1, cv2.LINE_AA)
                cv2.putText(espera, "Menu > Arquivo > Nova Gravacao de Filme > iPhone", (cx - 250, cy + 60),
                            cv2.FONT_HERSHEY_DUPLEX, 0.4, (60, 60, 70), 1, cv2.LINE_AA)

                if self.show_sidebar:
                    if self._sidebar_dirty or last_sidebar is None or last_sidebar_height != h:
                        sidebar = self._desenhar_sidebar(h, self.pulse_phase)
                        last_sidebar = sidebar.copy()
                        last_sidebar_height = h
                        self._sidebar_dirty = False
                    else:
                        sidebar = last_sidebar

                    display = np.hstack([sidebar, espera])
                else:
                    display = espera

                # Ajuda
                if self.show_help:
                    display = self._desenhar_ajuda(display)

                # Menu de bloqueios nervosos
                if self.nerve_block_show_menu:
                    display = self._desenhar_nerve_block_menu(display)

            cv2.imshow(self.janela, display)

            # FPS
            frame_count += 1
            if time.time() - fps_time >= 1:
                self.fps = frame_count
                frame_count = 0
                fps_time = time.time()

            # Teclas
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                if self.selecting_roi:
                    self.selecting_roi = False
                    self.roi_start = None
                    self.roi_end = None
                    self.process_roi = None
                    self.source_mode = 0  # Voltar para B-MODE
                    self._invalidate_sidebar()
                    print("Seleção cancelada")
                else:
                    break
            elif k == 13 or k == 10:  # ENTER - confirmar ROI
                if self.selecting_roi and self.process_roi:
                    self.selecting_roi = False
                    self.roi_confirmed = True
                    self._load_ai_if_needed()
                    print(f"ROI confirmada: {self.process_roi}")
            # Presets de ROI (P=50%, O=75%, L=100%, K=Square)
            elif k == ord('p') and self.selecting_roi:
                self._apply_roi_preset('50%')
            elif k == ord('o') and self.selecting_roi:
                self._apply_roi_preset('75%')
            elif k == ord('l') and self.selecting_roi:
                self._apply_roi_preset('100%')
            elif k == ord('k') and self.selecting_roi:
                self._apply_roi_preset('square')
            elif k == ord('q'):
                break
            elif k == ord('f'):
                self._toggle_pause()
            elif k == ord('1'):
                # B-MODE
                self._set_source_mode(0)
            elif k == ord('c'):
                # IA ZONE
                self._set_source_mode(1)
            elif k == ord('a'):
                # IA FULL
                self._set_source_mode(2)
            elif k == ord('r'):
                self._toggle_recording()
            elif k == ord('s'):
                self._screenshot()
            elif k == ord('b'):
                self._toggle_biplane()
            elif k == ord('h'):
                self._toggle_overlays()
            elif k == ord('i'):
                self.show_instructions = not self.show_instructions
            elif k == ord('+') or k == ord('='):
                self._zoom_in()
            elif k == ord('-'):
                self._zoom_out()
            elif k == 82 or k == 0:  # Seta cima
                self._pan_up()
            elif k == 84 or k == 1:  # Seta baixo
                self._pan_down()
            elif k == 81 or k == 2:  # Seta esquerda
                self._pan_left()
            elif k == 83 or k == 3:  # Seta direita
                self._pan_right()
            elif k == ord('/') or k == ord('?'):
                self.show_help = not self.show_help
            elif k == ord('x') or k == 122 or k == 201 or k == 144:
                self._toggle_fullscreen()
            # Plugins por numero (2-9, 0, V)
            elif ord('2') <= k <= ord('9'):
                self._set_plugin(k - ord('2'))  # 2=NEEDLE(0), 3=NERVE(1), etc
            elif k == ord('0'):
                if self.zoom_level != 1.0 or self.pan_y != 0:
                    self._reset_view()
                else:
                    self._set_plugin(8)  # LUNG
            elif k == ord('v'):
                self._set_plugin(9)  # BLADDER
            # ═══════════════════════════════════════════════════════════════════════
            # CALIBRAÇÃO - Ajuste de profundidade (FASE 3 NEEDLE PILOT)
            # ═══════════════════════════════════════════════════════════════════════
            elif k == ord('['):
                # Diminuir profundidade
                current = self.ai_processor.calibration['depth_mm']
                new_depth = max(20, current - 10)
                self.ai_processor.set_ultrasound_depth(new_depth)
                print(f"📐 Profundidade: {new_depth}mm")
            elif k == ord(']'):
                # Aumentar profundidade
                current = self.ai_processor.calibration['depth_mm']
                new_depth = min(200, current + 10)
                self.ai_processor.set_ultrasound_depth(new_depth)
                print(f"Profundidade: {new_depth}mm")

            # ═══════════════════════════════════════════════════════════════════════
            # NERVE TRACK v2.0 - Navegação de Bloqueios (<, >, n)
            # ═══════════════════════════════════════════════════════════════════════
            elif k == ord(',') or k == ord('<'):
                # Bloqueio anterior (quando NERVE ativo)
                if self.plugin_idx == 1:  # NERVE
                    self._prev_nerve_block()
            elif k == ord('.') or k == ord('>'):
                # Próximo bloqueio (quando NERVE ativo)
                if self.plugin_idx == 1:  # NERVE
                    self._next_nerve_block()
            elif k == ord('n'):
                # Mostrar/esconder menu de bloqueios (quando NERVE ativo)
                if self.plugin_idx == 1:  # NERVE
                    self._toggle_nerve_block_menu()
            elif k == 13 or k == 10:  # ENTER
                # Selecionar bloco atual e fechar menu
                if self.nerve_block_show_menu:
                    self._apply_nerve_block()
                    self.nerve_block_show_menu = False
                    self._invalidate_sidebar()

        # Cleanup - parar todas as threads
        if self.recording:
            self.recorder.stop_recording()
        self.ai_thread.stop()
        self.captura.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Validar configuracao antes de iniciar
    if not config.print_config_status():
        print("ERRO CRITICO: Configuracao invalida. Corrija os erros acima.")
        import sys
        sys.exit(1)

    USGApp().rodar()
