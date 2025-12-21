import cv2
import numpy as np
import struct
import time
import config
from src.window_capture import WindowCapture

# Tentar importar memória compartilhada
try:
    from multiprocessing import shared_memory
    SHM_AVAILABLE = True
except ImportError:
    SHM_AVAILABLE = False


class SharedMemoryCapture:
    """Captura frames da memória compartilhada (do capture_server.py)."""

    SHARED_MEMORY_NAME = 'usgapp_frame'
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FRAME_CHANNELS = 3
    FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS
    METADATA_SIZE = 32

    def __init__(self):
        self.shm = None
        self.connected = False
        self.last_frame_count = -1
        self.last_frame = None

    def connect(self) -> bool:
        """Conecta à memória compartilhada."""
        if not SHM_AVAILABLE:
            print("❌ shared_memory não disponível")
            return False

        try:
            self.shm = shared_memory.SharedMemory(name=self.SHARED_MEMORY_NAME)
            self.connected = True
            print(f"✅ Conectado à memória compartilhada: {self.SHARED_MEMORY_NAME}")
            return True
        except FileNotFoundError:
            print("⚠️ Servidor de captura não está rodando.")
            print("   Execute primeiro: python3 capture_server.py")
            return False
        except Exception as e:
            print(f"❌ Erro ao conectar: {e}")
            return False

    def read(self):
        """Lê frame da memória compartilhada."""
        if not self.connected or not self.shm:
            if not self.connect():
                return False, None

        try:
            # Ler metadata
            metadata = bytes(self.shm.buf[:self.METADATA_SIZE])
            timestamp, frame_count, status = struct.unpack('dII', metadata[:16])

            # Verificar se frame é válido
            if status != 1:
                return False, None

            # Verificar se é frame novo
            if frame_count == self.last_frame_count:
                # Retornar último frame
                if self.last_frame is not None:
                    return True, self.last_frame
                return False, None

            # Ler frame
            frame_bytes = bytes(self.shm.buf[self.METADATA_SIZE:self.METADATA_SIZE + self.FRAME_SIZE])
            frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = frame.reshape((self.FRAME_HEIGHT, self.FRAME_WIDTH, self.FRAME_CHANNELS))

            self.last_frame_count = frame_count
            self.last_frame = frame.copy()

            return True, frame

        except Exception as e:
            print(f"⚠️ Erro ao ler frame: {e}")
            self.connected = False
            return False, None

    def release(self):
        """Libera recursos."""
        if self.shm:
            self.shm.close()
            self.shm = None
        self.connected = False


class VideoCapture:
    def __init__(self, source=config.VIDEO_SOURCE):
        self.capture = None
        self.cap = None
        self.shm_capture = None

        # Verificar tipo de fonte
        if source == 'SHM' or source == 'shared_memory':
            # Captura via memória compartilhada (do capture_server.py)
            print("🔗 Iniciando captura via memória compartilhada...")
            self.shm_capture = SharedMemoryCapture()
            if not self.shm_capture.connect():
                print("⚠️ Fallback: tentando captura de janela direta")
                self.shm_capture = None
                self.capture = WindowCapture("AIRPLAY")

        elif isinstance(source, str) and not source.endswith(('.mp4', '.avi', '.mov')):
            # Assume que é nome de janela se for string e não for arquivo de vídeo
            print(f"🖥️ Iniciando captura de janela: {source}")
            self.capture = WindowCapture(source)

            # Aplicar ROI se configurado
            if hasattr(config, 'USE_ROI') and config.USE_ROI:
                preset = getattr(config, 'ROI_PRESET', None)
                if preset == "butterfly":
                    self.capture.set_roi_butterfly()
                elif preset == "clarius":
                    self.capture.set_roi_clarius()
                elif preset == "custom":
                    roi = getattr(config, 'ROI_CUSTOM', (0.1, 0.1, 0.8, 0.8))
                    self.capture.set_roi_percent(*roi)
                else:
                    # Usar ROI_CUSTOM como padrão
                    roi = getattr(config, 'ROI_CUSTOM', None)
                    if roi:
                        self.capture.set_roi_percent(*roi)
        else:
            # Arquivo ou Câmera
            print(f"📹 Iniciando captura de vídeo/câmera: {source}")
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                print(f"❌ Erro: Não foi possível abrir a fonte de vídeo: {source}")

        # Configura resolução e buffer (apenas se for câmera/webcam)
        if isinstance(source, int) and self.cap:
            # Latência: Buffer size 1 para pegar sempre o frame mais recente
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
        
    def read(self):
        # Prioridade: memória compartilhada > captura de janela > OpenCV
        if self.shm_capture:
            return self.shm_capture.read()
        elif self.capture:
            return self.capture.read()
        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            # Loop para arquivos de vídeo
            if not ret and isinstance(config.VIDEO_SOURCE, str) and config.VIDEO_SOURCE.endswith(('.mp4', '.avi', '.mov')):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return self.cap.read()
            return ret, frame
        return False, None

    def release(self):
        if self.shm_capture:
            self.shm_capture.release()
        if self.capture:
            self.capture.release()
        if self.cap and self.cap.isOpened():
            self.cap.release()
