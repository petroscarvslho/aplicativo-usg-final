#!/usr/bin/env python3
"""
APLICATIVO USG FINAL
====================
100% Python/OpenCV
ZERO conversao de imagem - Pixels perfeitos
"""

import cv2
import numpy as np
import time
import os
import config
from src.capture import VideoCapture
from src.ai_processor import AIProcessor

class USGFinal:
    def __init__(self):
        print("=" * 60)
        print("  APLICATIVO USG FINAL")
        print("  100% Python - ZERO conversao de imagem")
        print("=" * 60)

        self.running = True
        self.freeze = False
        self.recording = False
        self.show_help = True  # Mostrar ajuda inicial
        self.ai_enabled = False

        # Modos
        self.modes = ['B-Mode', 'AGULHA', 'NERVO']
        self.mode_idx = 0
        self.current_mode = self.modes[0]

        # Captura
        print("\n[1/2] Iniciando captura...")
        self.cap = VideoCapture(config.VIDEO_SOURCE)

        # IA
        print("[2/2] Carregando IA...")
        self.ai = AIProcessor()

        # Gravacao
        self.video_writer = None
        self.record_path = None

        # Frame
        self.frame = None
        self.fps = 0
        self.frame_count = 0
        self.fps_time = time.time()

        # Janela
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_NAME, 1280, 720)

        print("\n[PRONTO] Pressione H para ajuda\n")

    def draw_ui(self, frame):
        """Desenha interface minima sobre a imagem."""
        h, w = frame.shape[:2]

        # Barra superior
        cv2.rectangle(frame, (0, 0), (w, 35), (20, 20, 20), -1)

        # Modo
        cv2.putText(frame, f"MODO: {self.current_mode}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # FPS
        cv2.putText(frame, f"{self.fps} FPS", (w - 80, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Status
        x = 200
        if self.freeze:
            cv2.putText(frame, "CONGELADO", (x, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            x += 120

        if self.recording:
            if int(time.time() * 2) % 2:
                cv2.circle(frame, (x + 8, 18), 6, (0, 0, 255), -1)
            cv2.putText(frame, "GRAVANDO", (x + 20, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            x += 120

        if self.ai_enabled:
            cv2.putText(frame, "IA ON", (x, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Ajuda
        if self.show_help:
            self.draw_help(frame)

        return frame

    def draw_help(self, frame):
        """Painel de ajuda."""
        h, w = frame.shape[:2]

        # Fundo
        overlay = frame.copy()
        x1, y1 = w // 4, h // 4
        x2, y2 = 3 * w // 4, 3 * h // 4
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Borda
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 2)

        # Titulo
        cv2.putText(frame, "CONTROLES", (x1 + 20, y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Comandos
        comandos = [
            ("H", "Mostrar/esconder ajuda"),
            ("Q / ESC", "Sair"),
            ("F", "Congelar imagem"),
            ("R", "Iniciar/parar gravacao"),
            ("S", "Screenshot (PNG)"),
            ("M", "Trocar modo"),
            ("A", "Ligar/desligar IA"),
            ("C", "Trocar camera"),
            ("+ / -", "Zoom"),
        ]

        y = y1 + 70
        for tecla, desc in comandos:
            cv2.putText(frame, tecla, (x1 + 30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, desc, (x1 + 120, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 28

        cv2.putText(frame, "Pressione H para fechar", (x1 + 30, y2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    def screenshot(self):
        """Salva screenshot em PNG (lossless)."""
        if self.frame is None:
            return

        if not os.path.exists("captures"):
            os.makedirs("captures")

        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"captures/screenshot_{ts}.png"

        # PNG sem compressao = qualidade maxima
        cv2.imwrite(path, self.frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"Screenshot: {path}")

    def start_recording(self):
        """Inicia gravacao de video."""
        if self.frame is None:
            return

        if not os.path.exists("captures"):
            os.makedirs("captures")

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.record_path = f"captures/video_{ts}.avi"

        h, w = self.frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(self.record_path, fourcc, 30, (w, h))

        self.recording = True
        print(f"Gravando: {self.record_path}")

    def stop_recording(self):
        """Para gravacao."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print(f"Gravacao salva: {self.record_path}")

        self.recording = False

    def next_mode(self):
        """Proximo modo."""
        self.mode_idx = (self.mode_idx + 1) % len(self.modes)
        self.current_mode = self.modes[self.mode_idx]

        # Configurar IA
        if self.current_mode == 'AGULHA':
            self.ai.set_mode('needle')
        elif self.current_mode == 'NERVO':
            self.ai.set_mode('nerve')
        else:
            self.ai.set_mode(None)

        print(f"Modo: {self.current_mode}")

    def next_camera(self):
        """Proxima camera."""
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
        """Loop principal."""
        while self.running:
            # Captura
            if not self.freeze:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.frame = frame

            if self.frame is None:
                time.sleep(0.01)
                continue

            # Processar
            display = self.frame.copy()

            # IA
            if self.ai_enabled and self.current_mode in ['AGULHA', 'NERVO']:
                display = self.ai.process(display)

            # UI
            display = self.draw_ui(display)

            # Gravar
            if self.recording and self.video_writer:
                self.video_writer.write(display)

            # Mostrar - IMAGEM DIRETA, SEM CONVERSAO
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

        # Cleanup
        if self.recording:
            self.stop_recording()
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nAte mais!")


if __name__ == '__main__':
    app = USGFinal()
    app.run()
