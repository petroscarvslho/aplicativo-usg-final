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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.window_capture import WindowCapture
from src.clip_recorder import ClipRecorder


# =============================================================================
# CAPTURA EM THREAD SEPARADO
# =============================================================================
class CapturaThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.connected = False

    def run(self):
        cap = WindowCapture("AIRPLAY")
        while self.running:
            ret, frame = cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.connected = True
            else:
                self.connected = False
                time.sleep(0.02)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False


# =============================================================================
# CLASSE BOTAO
# =============================================================================
class Botao:
    def __init__(self, x, y, w, h, texto, tecla, cor_normal, cor_ativo, tooltip=""):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.texto = texto
        self.tecla = tecla
        self.cor_normal = cor_normal
        self.cor_ativo = cor_ativo
        self.tooltip = tooltip
        self.ativo = False
        self.hover = False

    def contem(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def desenhar(self, img):
        if self.ativo:
            cor_bg = self.cor_ativo
            cor_texto = (0, 0, 0)
        elif self.hover:
            cor_bg = tuple(min(c + 25, 255) for c in self.cor_normal)
            cor_texto = (220, 220, 220)
        else:
            cor_bg = self.cor_normal
            cor_texto = (180, 180, 180)

        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), cor_bg, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (50, 50, 60), 1)

        label = f"[{self.tecla}] {self.texto}" if self.tecla else self.texto
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, cor_texto, 1, cv2.LINE_AA)


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

    # Todos os modos disponiveis (igual ao HTML)
    MODOS = [
        ('B-MODE', '1', (180, 180, 180), "Modo Brilho - Imagem 2D padrao"),
        ('AGULHA', '2', (100, 255, 100), "Needle Pilot - Realce de agulha"),
        ('NERVO', '3', (0, 255, 255), "Nerve Track - Segmentacao nervos"),
        ('CARDIACO', '4', (100, 100, 255), "Cardiac AI - Fracao Ejecao"),
        ('FAST', '5', (0, 165, 255), "Protocolo FAST - Trauma"),
        ('ANATOMIA', '6', (255, 100, 255), "Anatomia - ID estruturas"),
        ('MODO-M', '7', (200, 200, 100), "Modo-M - Movimento temporal"),
        ('COLOR', '8', (255, 50, 50), "Color Doppler - Fluxo"),
        ('POWER', '9', (255, 150, 50), "Power Doppler - Alta sens."),
        ('PULMAO', '0', (100, 200, 255), "Lung AI - Linhas-B"),
        ('BEXIGA', 'V', (150, 100, 255), "Volume Vesical"),
    ]

    SIDEBAR_W = 200

    def __init__(self):
        # Estado
        self.modo_idx = 0
        self.pause = False
        self.ai_on = False
        self.recording = False
        self.show_sidebar = True
        self.show_overlays = True
        self.show_help = False
        self.show_instructions = True
        self.fullscreen = False
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

        # AI
        self.ai = None

        # Gravacao
        self.recorder = ClipRecorder(output_dir="captures", fps=30)

        # Captura
        self.captura = CapturaThread()
        self.captura.start()
        self.frame_original = None

        # Janela
        self.janela = "USG FLOW"
        cv2.namedWindow(self.janela, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.janela, 1700, 950)
        cv2.setMouseCallback(self.janela, self._mouse_callback)

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
        print("=" * 60 + "\n")

    def _criar_botoes(self):
        """Cria todos os botoes da sidebar."""
        self.botoes_modo = []
        self.botoes_ctrl = []
        self.botoes_view = []

        bg_btn = (40, 40, 48)

        # Botoes de MODO
        y = 55
        for i, (nome, tecla, cor, tooltip) in enumerate(self.MODOS):
            btn = Botao(10, y, 180, 26, nome, tecla, bg_btn, cor, tooltip)
            self.botoes_modo.append(btn)
            y += 30

        # Separador - Controles
        y += 15

        # Botoes de CONTROLE
        controles = [
            ('Freeze', 'F', (0, 200, 255), "Congelar imagem"),
            ('AI', 'A', (100, 255, 100), "Ativar processamento IA"),
            ('Gravar', 'R', (80, 80, 255), "Iniciar/parar gravacao"),
            ('Foto', 'S', (255, 165, 0), "Salvar screenshot"),
            ('Biplane', 'B', (200, 100, 200), "Modo lado a lado"),
        ]
        for texto, tecla, cor, tooltip in controles:
            btn = Botao(10, y, 180, 26, texto, tecla, bg_btn, cor, tooltip)
            self.botoes_ctrl.append(btn)
            y += 30

        # Separador - View
        y += 15

        # Botoes de VIEW
        view_btns = [
            ('Zoom +', '+', (100, 150, 200), "Aumentar zoom"),
            ('Zoom -', '-', (100, 150, 200), "Diminuir zoom"),
            ('Reset', '0', (150, 150, 150), "Resetar visualizacao"),
            ('Overlays', 'H', (120, 120, 140), "Mostrar/esconder overlays"),
            ('Sidebar', 'T', (120, 120, 140), "Mostrar/esconder sidebar"),
            ('Ajuda', '?', (100, 150, 255), "Mostrar atalhos"),
        ]
        for texto, tecla, cor, tooltip in view_btns:
            btn = Botao(10, y, 88 if texto.startswith('Zoom') else 180, 24, texto, tecla, bg_btn, cor, tooltip)
            if texto == 'Zoom +':
                btn.w = 88
            elif texto == 'Zoom -':
                btn.x = 102
                btn.w = 88
                y -= 28  # Mesma linha
            self.botoes_view.append(btn)
            y += 28

    def _mouse_callback(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y

        # Atualizar hover
        for btn in self.botoes_modo + self.botoes_ctrl + self.botoes_view:
            btn.hover = btn.contem(x, y)

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not self.show_sidebar:
            return

        # Cliques nos botoes de modo
        for i, btn in enumerate(self.botoes_modo):
            if btn.contem(x, y):
                self._set_modo(i)
                return

        # Cliques nos botoes de controle
        for i, btn in enumerate(self.botoes_ctrl):
            if btn.contem(x, y):
                if i == 0:
                    self._toggle_pause()
                elif i == 1:
                    self._toggle_ai()
                elif i == 2:
                    self._toggle_recording()
                elif i == 3:
                    self._screenshot()
                elif i == 4:
                    self._toggle_biplane()
                return

        # Cliques nos botoes de view
        for i, btn in enumerate(self.botoes_view):
            if btn.contem(x, y):
                if btn.texto == 'Zoom +':
                    self._zoom_in()
                elif btn.texto == 'Zoom -':
                    self._zoom_out()
                elif btn.texto == 'Reset':
                    self._reset_view()
                elif btn.texto == 'Overlays':
                    self._toggle_overlays()
                elif btn.texto == 'Sidebar':
                    self._toggle_sidebar()
                elif btn.texto == 'Ajuda':
                    self.show_help = not self.show_help
                return

    def _set_modo(self, idx):
        if 0 <= idx < len(self.MODOS):
            self.modo_idx = idx
            nome = self.MODOS[idx][0]
            print(f"Modo: {nome}")
            if self.ai:
                mode_map = {
                    0: None, 1: 'needle', 2: 'nerve', 3: 'cardiac',
                    4: 'fast', 5: 'segment', 6: 'm_mode', 7: 'color',
                    8: 'power', 9: 'b_lines', 10: 'bladder'
                }
                self.ai.set_mode(mode_map.get(idx))

    def _toggle_pause(self):
        self.pause = not self.pause
        print(f"Pause: {'ON' if self.pause else 'OFF'}")

    def _toggle_ai(self):
        self.ai_on = not self.ai_on
        if self.ai_on and self.ai is None:
            try:
                from src.ai_processor import AIProcessor
                self.ai = AIProcessor()
                # Configurar modo atual
                self._set_modo(self.modo_idx)
            except Exception as e:
                print(f"Erro AI: {e}")
                self.ai_on = False
        print(f"AI: {'ON' if self.ai_on else 'OFF'}")

    def _toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            self.recorder.start_recording()
        else:
            self.recorder.stop_recording()
        print(f"Gravacao: {'ON' if self.recording else 'OFF'}")

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

    def _toggle_overlays(self):
        self.show_overlays = not self.show_overlays
        print(f"Overlays: {'ON' if self.show_overlays else 'OFF'}")

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        prop = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.janela, cv2.WND_PROP_FULLSCREEN, prop)
        if not self.fullscreen:
            cv2.resizeWindow(self.janela, 1700, 950)

    def _get_screen_size(self):
        """Obtem tamanho da tela para fullscreen."""
        try:
            import Quartz
            main_monitor = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
            return int(main_monitor.size.width), int(main_monitor.size.height)
        except:
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

    def _desenhar_sidebar(self, altura):
        sidebar = np.zeros((altura, self.SIDEBAR_W, 3), dtype=np.uint8)
        sidebar[:] = (20, 20, 25)

        # Logo
        cv2.putText(sidebar, "USG", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 165, 0), 2, cv2.LINE_AA)
        cv2.putText(sidebar, "FLOW", (62, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(sidebar, (10, 42), (190, 42), (40, 40, 50), 1)

        # Label MODOS
        cv2.putText(sidebar, "MODOS", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 110), 1, cv2.LINE_AA)

        # Botoes de modo
        for i, btn in enumerate(self.botoes_modo):
            btn.ativo = (i == self.modo_idx)
            btn.desenhar(sidebar)

        # Separador
        y_sep = self.botoes_modo[-1].y + self.botoes_modo[-1].h + 8
        cv2.line(sidebar, (10, y_sep), (190, y_sep), (40, 40, 50), 1)
        cv2.putText(sidebar, "CONTROLES", (10, y_sep + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 110), 1, cv2.LINE_AA)

        # Botoes de controle
        self.botoes_ctrl[0].ativo = self.pause
        self.botoes_ctrl[1].ativo = self.ai_on
        self.botoes_ctrl[2].ativo = self.recording
        self.botoes_ctrl[4].ativo = self.biplane_active
        for btn in self.botoes_ctrl:
            btn.desenhar(sidebar)

        # Separador
        y_sep2 = self.botoes_ctrl[-1].y + self.botoes_ctrl[-1].h + 8
        cv2.line(sidebar, (10, y_sep2), (190, y_sep2), (40, 40, 50), 1)
        cv2.putText(sidebar, "VISUALIZACAO", (10, y_sep2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 110), 1, cv2.LINE_AA)

        # Botoes de view
        for btn in self.botoes_view:
            if btn.texto == 'Overlays':
                btn.ativo = self.show_overlays
            elif btn.texto == 'Ajuda':
                btn.ativo = self.show_help
            btn.desenhar(sidebar)

        # Footer - Status
        footer_y = altura - 85
        cv2.line(sidebar, (10, footer_y), (190, footer_y), (40, 40, 50), 1)

        # Conexao
        sy = footer_y + 18
        if self.captura.connected:
            cv2.circle(sidebar, (20, sy - 4), 4, (100, 255, 100), -1)
            cv2.putText(sidebar, "Conectado", (30, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1, cv2.LINE_AA)
        else:
            cv2.circle(sidebar, (20, sy - 4), 4, (80, 80, 80), -1)
            cv2.putText(sidebar, "Aguardando...", (30, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)

        # FPS
        sy += 18
        fps_cor = (100, 255, 100) if self.fps >= 25 else (0, 200, 255) if self.fps >= 15 else (0, 100, 255)
        cv2.putText(sidebar, f"FPS: {self.fps}", (15, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fps_cor, 1, cv2.LINE_AA)

        # REC
        if self.recording:
            if int(time.time() * 2) % 2:
                cv2.circle(sidebar, (160, sy - 5), 6, (0, 0, 255), -1)
            cv2.putText(sidebar, "REC", (130, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

        # Modo e Zoom
        sy += 18
        modo_nome = self.MODOS[self.modo_idx][0]
        modo_cor = self.MODOS[self.modo_idx][2]
        cv2.putText(sidebar, f"{modo_nome}", (15, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, modo_cor, 1, cv2.LINE_AA)
        cv2.putText(sidebar, f"Zoom: {self.zoom_level:.1f}x", (110, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 110), 1, cv2.LINE_AA)

        # Resolucao
        if self.frame_original is not None:
            h, w = self.frame_original.shape[:2]
            sy += 16
            cv2.putText(sidebar, f"{w}x{h}", (15, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 90), 1, cv2.LINE_AA)

        return sidebar

    def _desenhar_instrucoes(self, frame):
        """Desenha instrucoes contextuais na parte inferior."""
        if not self.show_instructions or not self.show_overlays:
            return frame

        h, w = frame.shape[:2]
        modo_nome = self.MODOS[self.modo_idx][0]
        instrucao = MODE_INSTRUCTIONS.get(modo_nome, "")

        if not instrucao:
            return frame

        # Fundo semi-transparente
        overlay = frame.copy()
        box_h = 50
        cv2.rectangle(overlay, (0, h - box_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Texto
        cv2.putText(frame, instrucao, (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        return frame

    def _desenhar_ajuda(self, frame):
        """Desenha modal de ajuda."""
        h, w = frame.shape[:2]

        # Fundo escuro
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//5, h//8), (4*w//5, 7*h//8), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)
        cv2.rectangle(frame, (w//5, h//8), (4*w//5, 7*h//8), (50, 50, 60), 2)

        # Titulo
        cv2.putText(frame, "ATALHOS DO TECLADO", (w//5 + 30, h//8 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2, cv2.LINE_AA)

        # Lista de atalhos
        atalhos = [
            ("1-9, 0, V", "Selecionar modo"),
            ("F", "Freeze/Pause"),
            ("A", "Ativar/Desativar AI"),
            ("R", "Iniciar/Parar gravacao"),
            ("S", "Screenshot (PNG)"),
            ("B", "Toggle Biplane"),
            ("H", "Esconder/Mostrar overlays"),
            ("I", "Esconder/Mostrar instrucoes"),
            ("T", "Esconder/Mostrar sidebar"),
            ("+/-", "Zoom In/Out"),
            ("Setas", "Pan (mover imagem)"),
            ("0 (view)", "Reset visualizacao"),
            ("?", "Esta ajuda"),
            ("F11", "Fullscreen"),
            ("Q/ESC", "Sair"),
        ]

        y = h//8 + 80
        for tecla, desc in atalhos:
            cv2.putText(frame, f"[{tecla}]", (w//5 + 30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, desc, (w//5 + 150, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
            y += 28

        # Instrucao para fechar
        cv2.putText(frame, "Pressione ? para fechar", (w//5 + 30, 7*h//8 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 90), 1, cv2.LINE_AA)

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
        """Loop principal."""
        frame_count = 0
        fps_time = time.time()

        while self.running:
            # Captura
            if not self.pause:
                f = self.captura.get_frame()
                if f is not None:
                    self.frame_original = f
                    if self.recording:
                        self.recorder.add_frame(f)

            # Montar display
            if self.frame_original is not None:
                h, w = self.frame_original.shape[:2]

                # Imagem para display (COPIA do original)
                display_img = self.frame_original.copy()

                # AI (se ligada)
                if self.ai_on and self.ai:
                    try:
                        display_img = self.ai.process(display_img)
                    except:
                        pass

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

                # Sidebar
                if self.show_sidebar:
                    sidebar = self._desenhar_sidebar(display_img.shape[0])
                    display = np.hstack([sidebar, display_img])
                else:
                    display = display_img

                # Ajuda
                if self.show_help:
                    display = self._desenhar_ajuda(display)

            else:
                # Tela de espera
                h = 720
                espera = np.zeros((h, 1280, 3), dtype=np.uint8)
                espera[:] = (12, 12, 15)
                cv2.putText(espera, "AGUARDANDO SINAL", (450, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 2)
                cv2.putText(espera, "Conecte iPhone via QuickTime", (420, 370),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 1)
                cv2.putText(espera, "Menu > Arquivo > Nova Gravacao de Filme > iPhone", (350, 410),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 70), 1)

                if self.show_sidebar:
                    sidebar = self._desenhar_sidebar(h)
                    display = np.hstack([sidebar, espera])
                else:
                    display = espera

            # Fullscreen: redimensionar para preencher a tela toda
            if self.fullscreen:
                screen_w, screen_h = self._get_screen_size()
                if display.shape[1] != screen_w or display.shape[0] != screen_h:
                    display = cv2.resize(display, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)

            cv2.imshow(self.janela, display)

            # FPS
            frame_count += 1
            if time.time() - fps_time >= 1:
                self.fps = frame_count
                frame_count = 0
                fps_time = time.time()

            # Teclas
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break
            elif k == ord('f'):
                self._toggle_pause()
            elif k == ord('a'):
                self._toggle_ai()
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
            elif k == ord('t'):
                self._toggle_sidebar()
            elif k == ord('+') or k == ord('='):
                self._zoom_in()
            elif k == ord('-'):
                self._zoom_out()
            # Setas (codigos variam por sistema - testando multiplos)
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
            # F11 (varios codigos possiveis)
            elif k == 122 or k == 201 or k == 144:  # F11
                self._toggle_fullscreen()
            # Modos por numero
            elif ord('1') <= k <= ord('9'):
                self._set_modo(k - ord('1'))
            elif k == ord('0'):
                if self.zoom_level != 1.0 or self.pan_y != 0:
                    self._reset_view()
                else:
                    self._set_modo(9)  # PULMAO
            elif k == ord('v'):
                self._set_modo(10)  # BEXIGA

        # Cleanup
        if self.recording:
            self.recorder.stop_recording()
        self.captura.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    USGApp().rodar()
