#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        APLICATIVO USG FINAL                                   ║
║                                                                                ║
║  100% Python/OpenCV - ZERO conversao de imagem                                ║
║  Interface Premium inspirada em Apple + Butterfly iQ3                         ║
║                                                                                ║
║  Pixels perfeitos - A imagem exibida e IDENTICA ao que vem do iPhone         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import time
import os
import sys
import math

# Path setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.capture import VideoCapture
from src.design_system import get_design_system, Color


class PremiumUI:
    """
    Interface Premium para o aplicativo USG
    Design inspirado em Apple Human Interface Guidelines + Butterfly iQ3
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.ds = get_design_system('dark')

        # Layout dimensions
        self.header_height = 56
        self.sidebar_width = 300
        self.footer_height = 48
        self.padding = 16
        self.border_radius = 12

        # Animation states
        self.hover_states = {}
        self.pulse_phase = 0

        # Precompute colors (BGR for OpenCV)
        self.c = {
            'bg': (20, 15, 15),           # Fundo principal
            'surface': (35, 32, 28),      # Superficie elevada
            'surface2': (48, 45, 38),     # Superficie mais elevada
            'border': (75, 70, 60),       # Bordas sutis
            'text': (255, 255, 255),      # Texto principal
            'text_dim': (165, 160, 150),  # Texto secundario
            'text_muted': (115, 110, 100),# Texto terciario
            'accent': (255, 165, 0),      # Laranja accent (BGR)
            'accent2': (255, 200, 50),    # Amarelo accent
            'success': (100, 255, 50),    # Verde
            'danger': (80, 80, 255),      # Vermelho (BGR)
            'warning': (0, 200, 255),     # Amarelo warning
            'info': (255, 150, 50),       # Azul info
            'medical_needle': (100, 255, 100),  # Verde agulha
            'medical_nerve': (0, 255, 255),     # Amarelo nervo
            'medical_artery': (80, 80, 255),    # Vermelho arteria
            'medical_vein': (255, 100, 100),    # Azul veia
        }

    def draw_rounded_rect(self, img, pt1, pt2, color, radius=12, thickness=-1, border_color=None):
        """Desenha retangulo com bordas arredondadas."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Limitar radius ao tamanho do retangulo
        radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

        if thickness == -1:
            # Preenchido
            # Retangulos internos
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)

            # Cantos arredondados
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
        else:
            # Apenas borda
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

        if border_color:
            self.draw_rounded_rect(img, pt1, pt2, border_color, radius, 1)

    def draw_gradient_rect(self, img, pt1, pt2, color1, color2, vertical=True):
        """Desenha retangulo com gradiente."""
        x1, y1 = pt1
        x2, y2 = pt2

        if vertical:
            for y in range(y1, y2):
                t = (y - y1) / max(1, (y2 - y1))
                color = tuple(int(c1 * (1 - t) + c2 * t) for c1, c2 in zip(color1, color2))
                cv2.line(img, (x1, y), (x2, y), color, 1)
        else:
            for x in range(x1, x2):
                t = (x - x1) / max(1, (x2 - x1))
                color = tuple(int(c1 * (1 - t) + c2 * t) for c1, c2 in zip(color1, color2))
                cv2.line(img, (x, y1), (x, y2), color, 1)

    def draw_header(self, img, app):
        """Desenha header premium com gradiente e blur."""
        h = self.header_height

        # Gradiente de fundo
        self.draw_gradient_rect(img, (0, 0), (self.width, h),
                                (35, 30, 25), (25, 22, 18), vertical=True)

        # Linha de accent inferior
        cv2.line(img, (0, h - 2), (self.width, h - 2), self.c['accent'], 2)

        # Logo com glow
        logo_text = "USG FLOW VISION"
        cv2.putText(img, logo_text, (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.c['accent'], 2, cv2.LINE_AA)

        # Versao
        cv2.putText(img, "v2.0", (230, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.c['text_muted'], 1, cv2.LINE_AA)

        # Modo atual - Pill badge
        mode_name = app.current_mode[0]
        mode_color = app.current_mode[2]

        pill_x = 300
        pill_w = len(mode_name) * 12 + 30
        self.draw_rounded_rect(img, (pill_x, 15), (pill_x + pill_w, 42),
                               mode_color, radius=14)
        cv2.putText(img, mode_name, (pill_x + 15, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # Status indicators no centro
        center_x = self.width // 2 - 150

        # Freeze indicator
        if app.freeze:
            self.draw_status_badge(img, center_x, 15, "CONGELADO", self.c['warning'], pulse=True)
            center_x += 130

        # Recording indicator
        if app.recording:
            elapsed = int(time.time() - app.record_start) if app.record_start else 0
            mins, secs = divmod(elapsed, 60)
            rec_text = f"REC {mins:02d}:{secs:02d}"
            self.draw_status_badge(img, center_x, 15, rec_text, self.c['danger'], pulse=True, dot=True)
            center_x += 140

        # AI indicator
        if app.ai_enabled:
            self.draw_status_badge(img, center_x, 15, "IA ATIVA", self.c['success'])

        # FPS e metricas no canto direito
        fps_x = self.width - 200

        # FPS com indicador de cor
        fps_color = self.c['success'] if app.fps >= 25 else self.c['warning'] if app.fps >= 15 else self.c['danger']
        cv2.putText(img, f"{app.fps}", (fps_x, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2, cv2.LINE_AA)
        cv2.putText(img, "FPS", (fps_x + 35, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.c['text_muted'], 1, cv2.LINE_AA)

        # Resolucao
        if app.frame is not None:
            h_frame, w_frame = app.frame.shape[:2]
            res_text = f"{w_frame}x{h_frame}"
            cv2.putText(img, res_text, (fps_x + 80, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.c['text_dim'], 1, cv2.LINE_AA)

    def draw_status_badge(self, img, x, y, text, color, pulse=False, dot=False):
        """Desenha badge de status premium."""
        w = len(text) * 10 + 25
        h = 27

        # Pulse effect
        alpha = 1.0
        if pulse:
            self.pulse_phase += 0.1
            alpha = 0.7 + 0.3 * math.sin(self.pulse_phase)

        # Background com alpha
        bg_color = tuple(int(c * 0.3) for c in color)
        self.draw_rounded_rect(img, (x, y), (x + w, y + h), bg_color, radius=14)

        # Borda
        border_color = tuple(int(c * alpha) for c in color)
        self.draw_rounded_rect(img, (x, y), (x + w, y + h), border_color, radius=14, thickness=1)

        # Dot pulsante
        text_x = x + 10
        if dot:
            if int(time.time() * 3) % 2:
                cv2.circle(img, (x + 12, y + h // 2), 5, color, -1)
            text_x = x + 22

        # Texto
        cv2.putText(img, text, (text_x, y + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def draw_sidebar(self, img, app):
        """Desenha sidebar premium com sombra e blur."""
        x = self.width - self.sidebar_width
        y = self.header_height

        # Sombra
        for i in range(10):
            alpha = 0.02 * (10 - i)
            shadow_color = tuple(int(c * alpha) for c in (0, 0, 0))
            cv2.line(img, (x - i, y), (x - i, self.height), shadow_color, 1)

        # Background
        self.draw_rounded_rect(img, (x, y), (self.width, self.height),
                               self.c['surface'], radius=0)

        # Borda esquerda com gradiente
        for i in range(2):
            cv2.line(img, (x + i, y), (x + i, self.height),
                     tuple(int(c * (1 - i * 0.5)) for c in self.c['accent']), 1)

        # === SECAO MODOS ===
        self.draw_section_title(img, x + 20, y + 30, "MODOS DE IMAGEM")

        btn_y = y + 55
        for i, (name, key, color) in enumerate(app.modes):
            is_selected = (i == app.mode_idx)
            self.draw_mode_button(img, x + 15, btn_y, self.sidebar_width - 30, 40,
                                  name, f"[{i+1}]", color, is_selected)
            btn_y += 48

        # === SECAO CONTROLES ===
        btn_y += 15
        self.draw_section_title(img, x + 20, btn_y, "CONTROLES")
        btn_y += 25

        controls = [
            ("Congelar", "F", app.freeze, self.c['warning']),
            ("Gravar", "R", app.recording, self.c['danger']),
            ("IA", "A", app.ai_enabled, self.c['success']),
            ("Screenshot", "S", False, self.c['info']),
        ]

        for label, key, active, color in controls:
            self.draw_control_item(img, x + 15, btn_y, self.sidebar_width - 30,
                                   label, key, active, color)
            btn_y += 38

        # === SECAO ATALHOS ===
        btn_y += 15
        self.draw_section_title(img, x + 20, btn_y, "ATALHOS")
        btn_y += 25

        shortcuts = [
            ("Ajuda", "H"),
            ("Camera", "C"),
            ("Sidebar", "T"),
            ("Sair", "Q"),
        ]

        for label, key in shortcuts:
            cv2.putText(img, f"[{key}]", (x + 20, btn_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.c['accent'], 1, cv2.LINE_AA)
            cv2.putText(img, label, (x + 55, btn_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.c['text_dim'], 1, cv2.LINE_AA)
            btn_y += 28

        # === RODAPE ===
        footer_y = self.height - 80
        cv2.line(img, (x + 20, footer_y), (self.width - 20, footer_y),
                 self.c['border'], 1)

        # Logo pequeno
        cv2.putText(img, "USG FLOW VISION", (x + 20, footer_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.c['text_muted'], 1, cv2.LINE_AA)
        cv2.putText(img, "Zero Conversao", (x + 20, footer_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.c['accent'], 1, cv2.LINE_AA)

    def draw_section_title(self, img, x, y, text):
        """Desenha titulo de secao."""
        cv2.putText(img, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.c['text_muted'], 1, cv2.LINE_AA)
        # Linha decorativa
        text_w = len(text) * 8
        cv2.line(img, (x + text_w + 10, y - 5), (x + self.sidebar_width - 50, y - 5),
                 self.c['border'], 1)

    def draw_mode_button(self, img, x, y, w, h, text, shortcut, color, selected):
        """Desenha botao de modo premium."""
        if selected:
            # Fundo colorido
            self.draw_rounded_rect(img, (x, y), (x + w, y + h), color, radius=10)
            # Glow effect
            for i in range(3):
                glow_color = tuple(int(c * 0.3 / (i + 1)) for c in color)
                self.draw_rounded_rect(img, (x - i, y - i), (x + w + i, y + h + i),
                                       glow_color, radius=10 + i, thickness=1)
            text_color = (0, 0, 0)
        else:
            # Fundo transparente com borda
            self.draw_rounded_rect(img, (x, y), (x + w, y + h),
                                   self.c['surface2'], radius=10)
            self.draw_rounded_rect(img, (x, y), (x + w, y + h),
                                   self.c['border'], radius=10, thickness=1)
            text_color = self.c['text_dim']

        # Texto
        cv2.putText(img, text, (x + 15, y + h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        # Shortcut
        cv2.putText(img, shortcut, (x + w - 35, y + h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)

    def draw_control_item(self, img, x, y, w, label, key, active, color):
        """Desenha item de controle."""
        h = 32

        # Toggle switch
        switch_x = x + w - 50
        switch_w = 44
        switch_h = 22

        if active:
            # Switch ativo
            self.draw_rounded_rect(img, (switch_x, y + 5), (switch_x + switch_w, y + 5 + switch_h),
                                   color, radius=11)
            # Knob
            cv2.circle(img, (switch_x + switch_w - 11, y + 5 + switch_h // 2), 8,
                       (255, 255, 255), -1)
        else:
            # Switch inativo
            self.draw_rounded_rect(img, (switch_x, y + 5), (switch_x + switch_w, y + 5 + switch_h),
                                   self.c['surface2'], radius=11)
            self.draw_rounded_rect(img, (switch_x, y + 5), (switch_x + switch_w, y + 5 + switch_h),
                                   self.c['border'], radius=11, thickness=1)
            # Knob
            cv2.circle(img, (switch_x + 11, y + 5 + switch_h // 2), 8,
                       self.c['text_muted'], -1)

        # Key badge
        cv2.putText(img, f"[{key}]", (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.c['accent'], 1, cv2.LINE_AA)

        # Label
        label_color = color if active else self.c['text_dim']
        cv2.putText(img, label, (x + 35, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)

    def draw_image_area(self, img, frame, app):
        """Desenha area principal da imagem."""
        if app.show_sidebar:
            area_w = self.width - self.sidebar_width - self.padding
        else:
            area_w = self.width - self.padding

        area_h = self.height - self.header_height - self.footer_height - self.padding
        area_x = self.padding // 2
        area_y = self.header_height + self.padding // 2

        if frame is None:
            # Tela de espera
            self.draw_waiting_screen(img, area_x, area_y, area_w, area_h)
        else:
            # Processar e exibir imagem
            processed = frame.copy()

            # Aplicar IA se ativada
            if app.ai_enabled and app.ai:
                try:
                    processed = app.ai.process(processed)
                except:
                    pass

            # Calcular escala mantendo aspect ratio
            fh, fw = processed.shape[:2]
            scale = min(area_w / fw, area_h / fh)
            new_w = int(fw * scale)
            new_h = int(fh * scale)

            # Resize com melhor qualidade
            if scale < 1:
                processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
            elif scale > 1:
                processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Centralizar
            x_offset = area_x + (area_w - new_w) // 2
            y_offset = area_y + (area_h - new_h) // 2

            # Borda da imagem
            border = 2
            cv2.rectangle(img, (x_offset - border, y_offset - border),
                          (x_offset + new_w + border, y_offset + new_h + border),
                          self.c['border'], border)

            # Colocar imagem
            img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = processed

            # Overlay info no canto da imagem
            self.draw_image_overlay(img, x_offset, y_offset, new_w, new_h, fw, fh, app)

    def draw_waiting_screen(self, img, x, y, w, h):
        """Desenha tela de espera animada."""
        # Fundo com grid sutil
        for i in range(0, w, 50):
            cv2.line(img, (x + i, y), (x + i, y + h), self.c['surface'], 1)
        for i in range(0, h, 50):
            cv2.line(img, (x, y + i), (x + w, y + i), self.c['surface'], 1)

        # Circulo central animado
        center_x = x + w // 2
        center_y = y + h // 2

        # Arcos animados
        t = time.time()
        for i in range(3):
            angle = int((t * 60 + i * 120) % 360)
            color = tuple(int(c * (1 - i * 0.2)) for c in self.c['accent'])
            cv2.ellipse(img, (center_x, center_y), (60 + i * 20, 60 + i * 20),
                        angle, 0, 60, color, 3)

        # Texto central
        msg = "AGUARDANDO SINAL"
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        tx = center_x - text_size[0] // 2
        ty = center_y + 100
        cv2.putText(img, msg, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.c['text_dim'], 2, cv2.LINE_AA)

        # Instrucoes
        instructions = [
            "Para conectar o iPhone:",
            "1. Abra QuickTime Player",
            "2. Arquivo > Nova Gravacao de Filme",
            "3. Selecione seu iPhone como fonte",
            "",
            "Ou use Espelhamento do macOS Sequoia"
        ]

        ty += 50
        for line in instructions:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            tx = center_x - text_size[0] // 2
            color = self.c['accent'] if "1." in line or "2." in line or "3." in line else self.c['text_muted']
            cv2.putText(img, line, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            ty += 25

    def draw_image_overlay(self, img, x, y, w, h, orig_w, orig_h, app):
        """Desenha overlay de informacoes sobre a imagem."""
        # Badge de resolucao no canto superior esquerdo
        res_text = f"{orig_w}x{orig_h}"
        badge_w = len(res_text) * 8 + 16
        badge_h = 22

        # Fundo semi-transparente
        overlay = img.copy()
        cv2.rectangle(overlay, (x + 8, y + 8), (x + 8 + badge_w, y + 8 + badge_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        cv2.putText(img, res_text, (x + 16, y + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.c['text'], 1, cv2.LINE_AA)

        # Timestamp no canto inferior direito
        timestamp = time.strftime("%H:%M:%S")
        ts_w = len(timestamp) * 10 + 16

        overlay = img.copy()
        cv2.rectangle(overlay, (x + w - ts_w - 8, y + h - 30),
                      (x + w - 8, y + h - 8), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        cv2.putText(img, timestamp, (x + w - ts_w, y + h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.c['text'], 1, cv2.LINE_AA)

    def draw_footer(self, img, app):
        """Desenha footer com informacoes."""
        y = self.height - self.footer_height

        # Background
        cv2.rectangle(img, (0, y), (self.width, self.height), self.c['surface'], -1)

        # Linha superior
        cv2.line(img, (0, y), (self.width, y), self.c['border'], 1)

        # Status da conexao
        if app.frame is not None:
            status = "CONECTADO"
            status_color = self.c['success']
        else:
            status = "DESCONECTADO"
            status_color = self.c['danger']

        cv2.circle(img, (20, y + self.footer_height // 2), 5, status_color, -1)
        cv2.putText(img, status, (35, y + self.footer_height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1, cv2.LINE_AA)

        # Fonte de video
        source_text = f"Fonte: {config.VIDEO_SOURCE}"
        cv2.putText(img, source_text, (160, y + self.footer_height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.c['text_muted'], 1, cv2.LINE_AA)

        # Hora
        current_time = time.strftime("%H:%M:%S")
        cv2.putText(img, current_time, (self.width - 100, y + self.footer_height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.c['text_dim'], 1, cv2.LINE_AA)

    def draw_help_modal(self, img):
        """Desenha modal de ajuda premium."""
        # Overlay escuro
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        # Modal
        modal_w = 600
        modal_h = 500
        mx = (self.width - modal_w) // 2
        my = (self.height - modal_h) // 2

        # Sombra do modal
        for i in range(20):
            alpha = 0.03 * (20 - i)
            shadow_color = tuple(int(30 * alpha) for _ in range(3))
            self.draw_rounded_rect(img, (mx - i, my - i), (mx + modal_w + i, my + modal_h + i),
                                   shadow_color, radius=20 + i)

        # Background do modal
        self.draw_rounded_rect(img, (mx, my), (mx + modal_w, my + modal_h),
                               self.c['surface'], radius=20)
        self.draw_rounded_rect(img, (mx, my), (mx + modal_w, my + modal_h),
                               self.c['accent'], radius=20, thickness=2)

        # Titulo
        title = "AJUDA - CONTROLES"
        cv2.putText(img, title, (mx + 30, my + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.c['accent'], 2, cv2.LINE_AA)

        # Linha decorativa
        cv2.line(img, (mx + 30, my + 60), (mx + modal_w - 30, my + 60),
                 self.c['border'], 1)

        # Comandos
        commands = [
            ("1-6", "Selecionar modo diretamente"),
            ("M", "Proximo modo"),
            ("F", "Congelar / Descongelar imagem"),
            ("R", "Iniciar / Parar gravacao"),
            ("S", "Salvar screenshot (PNG lossless)"),
            ("A", "Ligar / Desligar IA"),
            ("C", "Trocar camera / fonte"),
            ("T", "Mostrar / Esconder sidebar"),
            ("H", "Mostrar / Esconder esta ajuda"),
            ("Q / ESC", "Sair do aplicativo"),
        ]

        cy = my + 100
        for key, desc in commands:
            # Key badge
            key_w = len(key) * 12 + 20
            self.draw_rounded_rect(img, (mx + 40, cy - 15), (mx + 40 + key_w, cy + 10),
                                   self.c['surface2'], radius=6)
            cv2.putText(img, key, (mx + 50, cy + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.c['accent'], 1, cv2.LINE_AA)

            # Descricao
            cv2.putText(img, desc, (mx + 130, cy + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.c['text'], 1, cv2.LINE_AA)

            cy += 38

        # Rodape
        footer = "Pressione H para fechar"
        text_size = cv2.getTextSize(footer, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.putText(img, footer, (mx + (modal_w - text_size[0]) // 2, my + modal_h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.c['text_muted'], 1, cv2.LINE_AA)

    def render(self, app):
        """Renderiza interface completa."""
        # Canvas base
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.c['bg']

        # Componentes
        self.draw_header(img, app)
        self.draw_image_area(img, app.frame, app)

        if app.show_sidebar:
            self.draw_sidebar(img, app)

        self.draw_footer(img, app)

        if app.show_help:
            self.draw_help_modal(img)

        return img


class USGFinal:
    """Aplicativo principal."""

    def __init__(self):
        self.print_banner()

        self.running = True
        self.freeze = False
        self.recording = False
        self.show_help = False
        self.ai_enabled = False
        self.show_sidebar = True

        # Modos (nome, key_ia, cor_bgr)
        self.modes = [
            ('B-MODE', None, (180, 180, 180)),
            ('AGULHA', 'needle', (100, 255, 100)),
            ('NERVO', 'nerve', (0, 255, 255)),
            ('CARDIACO', 'cardiac', (100, 100, 255)),
            ('FAST', 'fast', (0, 165, 255)),
            ('PULMAO', 'lung', (0, 200, 255)),
        ]
        self.mode_idx = 0
        self.current_mode = self.modes[0]

        # Captura
        print("\n[1/3] Iniciando captura...")
        self.cap = None
        self.init_capture()

        # IA
        print("[2/3] Carregando IA...")
        try:
            from src.ai_processor import AIProcessor
            self.ai = AIProcessor()
        except Exception as e:
            print(f"      IA nao disponivel: {e}")
            self.ai = None

        # Gravacao
        print("[3/3] Inicializando...")
        self.video_writer = None
        self.record_path = None
        self.record_start = None

        # Frame
        self.frame = None
        self.fps = 0
        self.frame_count = 0
        self.fps_time = time.time()

        # UI
        self.window_width = 1500
        self.window_height = 900
        self.ui = PremiumUI(self.window_width, self.window_height)

        # Janela
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_NAME, self.window_width, self.window_height)

        print("\n" + "=" * 60)
        print("  PRONTO! Use H para ajuda")
        print("=" * 60 + "\n")

    def print_banner(self):
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██╗   ██╗███████╗ ██████╗     ███████╗██╗      ██████╗     ║
║   ██║   ██║██╔════╝██╔════╝     ██╔════╝██║     ██╔═══██╗    ║
║   ██║   ██║███████╗██║  ███╗    █████╗  ██║     ██║   ██║    ║
║   ██║   ██║╚════██║██║   ██║    ██╔══╝  ██║     ██║   ██║    ║
║   ╚██████╔╝███████║╚██████╔╝    ██║     ███████╗╚██████╔╝    ║
║    ╚═════╝ ╚══════╝ ╚═════╝     ╚═╝     ╚══════╝ ╚═════╝     ║
║                                                              ║
║              APLICATIVO USG FINAL                            ║
║              100% Python - Zero Conversao                    ║
║              Pixels Perfeitos                                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def init_capture(self):
        try:
            self.cap = VideoCapture(config.VIDEO_SOURCE)
            ret, frame = self.cap.read()
            if ret and frame is not None:
                print(f"      Fonte: {config.VIDEO_SOURCE} - OK")
                return True
        except Exception as e:
            print(f"      Erro: {e}")

        if config.VIDEO_SOURCE == "AIRPLAY":
            print("\n" + "!" * 60)
            print("  AIRPLAY NAO ENCONTRADO")
            print("  ")
            print("  Abra o QuickTime e conecte o iPhone")
            print("!" * 60 + "\n")

        return False

    def screenshot(self):
        if self.frame is None:
            return
        if not os.path.exists("captures"):
            os.makedirs("captures")
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"captures/screenshot_{ts}.png"
        cv2.imwrite(path, self.frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"Screenshot: {path}")

    def start_recording(self):
        if self.frame is None:
            return
        if not os.path.exists("captures"):
            os.makedirs("captures")
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.record_path = f"captures/video_{ts}.avi"
        h, w = self.frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(self.record_path, fourcc, 30, (w, h))
        self.record_start = time.time()
        self.recording = True
        print(f"Gravando: {self.record_path}")

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            elapsed = int(time.time() - self.record_start) if self.record_start else 0
            print(f"Gravacao salva ({elapsed}s): {self.record_path}")
        self.recording = False
        self.record_start = None

    def set_mode(self, idx):
        if 0 <= idx < len(self.modes):
            self.mode_idx = idx
            self.current_mode = self.modes[idx]
            name, key, _ = self.current_mode
            if self.ai and key:
                self.ai.set_mode(key)
            elif self.ai:
                self.ai.set_mode(None)
            print(f"Modo: {name}")

    def next_mode(self):
        self.set_mode((self.mode_idx + 1) % len(self.modes))

    def next_camera(self):
        if self.cap:
            self.cap.release()
        sources = ["AIRPLAY", 0, 1, 2]
        try:
            idx = sources.index(config.VIDEO_SOURCE)
        except:
            idx = 0
        config.VIDEO_SOURCE = sources[(idx + 1) % len(sources)]
        print(f"Camera: {config.VIDEO_SOURCE}")
        self.cap = VideoCapture(config.VIDEO_SOURCE)

    def run(self):
        while self.running:
            # Captura
            if not self.freeze and self.cap:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.frame = frame

            # Render UI
            display = self.ui.render(self)

            # Gravar frame original
            if self.recording and self.video_writer and self.frame is not None:
                self.video_writer.write(self.frame)

            # Mostrar
            cv2.imshow(config.WINDOW_NAME, display)

            # FPS
            self.frame_count += 1
            if time.time() - self.fps_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.fps_time = time.time()

            # Teclas
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                self.running = False
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('f'):
                self.freeze = not self.freeze
            elif key == ord('r'):
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording()
            elif key == ord('s'):
                self.screenshot()
            elif key == ord('m'):
                self.next_mode()
            elif key == ord('a'):
                self.ai_enabled = not self.ai_enabled
                print(f"IA: {'ON' if self.ai_enabled else 'OFF'}")
            elif key == ord('c'):
                self.next_camera()
            elif key == ord('t'):
                self.show_sidebar = not self.show_sidebar
            elif ord('1') <= key <= ord('6'):
                self.set_mode(key - ord('1'))

        # Cleanup
        if self.recording:
            self.stop_recording()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\nAte mais!")


if __name__ == '__main__':
    app = USGFinal()
    app.run()
