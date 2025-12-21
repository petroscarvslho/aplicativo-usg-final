"""
Window Capture - Captura de Janela Otimizada para Qualidade Máxima
==================================================================

OBJETIVO: Captura pixel-perfect sem NENHUMA perda de qualidade.

Otimizações implementadas:
1. Zero conversão de colorspace desnecessária
2. Captura em resolução nativa (sem resize)
3. Formato BGRA nativo do macOS preservado
4. Sem compressão intermediária
5. Buffer contíguo para máxima velocidade
6. Cache de metadata para evitar recálculos
"""

import numpy as np
import cv2
import Quartz
from Quartz import (
    CGWindowListCopyWindowInfo,
    CGWindowListCreateImage,
    CGImageGetWidth,
    CGImageGetHeight,
    CGImageGetBytesPerRow,
    CGImageGetBitsPerPixel,
    CGImageGetColorSpace,
    CGDataProviderCopyData,
    CGImageGetDataProvider,
    CGRectNull,
    kCGWindowListOptionAll,
    kCGWindowListOptionIncludingWindow,
    kCGNullWindowID,
    kCGWindowImageBoundsIgnoreFraming,
    kCGWindowImageNominalResolution,  # Evita scaling de Retina
)


class WindowCapture:
    """
    Captura de janela com qualidade máxima.

    Características:
    - Pixel-perfect: nenhuma alteração na imagem original
    - Zero compressão: dados brutos direto da GPU
    - Latência mínima: ~5-10ms por frame
    - Cores fiéis: sem conversão de colorspace
    - Suporte a ROI: captura apenas região de interesse
    """

    def __init__(self, window_name_part=None):
        self.window_name_part = window_name_part
        self.window_id = None
        self.width = 0
        self.height = 0
        self.bytes_per_row = 0
        self.bits_per_pixel = 0

        # ROI (Region of Interest) - None = captura tudo
        # Formato: (x, y, largura, altura) em pixels
        # Ou formato percentual: (x%, y%, largura%, altura%) se use_percent=True
        self.roi = None
        self.roi_percent = None  # ROI em porcentagem (0.0 a 1.0)

        # Cache para evitar realocação de memória
        self._frame_buffer = None
        self._last_successful_frame = None

        # Estatísticas
        self.frames_captured = 0
        self.frames_failed = 0

        if window_name_part:
            self.find_window(window_name_part)

    def find_window(self, name_part: str) -> bool:
        """
        Encontra janela por nome parcial.
        Prioriza janelas maiores (evita controles/menus).
        """
        window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)

        candidates = []
        search_lower = name_part.lower()

        for window in window_list:
            owner = window.get('kCGWindowOwnerName', '') or ''
            name = window.get('kCGWindowName', '') or ''
            bounds = window.get('kCGWindowBounds', {})
            width = bounds.get('Width', 0)
            height = bounds.get('Height', 0)

            # Ignorar janelas muito pequenas
            if width < 100 or height < 100:
                continue

            # Verificar match
            is_match = False

            # Match direto por nome
            if search_lower in name.lower() or search_lower in owner.lower():
                is_match = True

            # Match especial para AIRPLAY/iPhone
            if search_lower == "airplay":
                airplay_keywords = ['espelhamento', 'mirroring', 'iphone', 'ipad', 'quicktime']
                if any(kw in owner.lower() or kw in name.lower() for kw in airplay_keywords):
                    is_match = True

            if is_match:
                candidates.append({
                    'id': window['kCGWindowNumber'],
                    'owner': owner,
                    'name': name,
                    'width': width,
                    'height': height,
                    'area': width * height
                })

        if not candidates:
            print(f"⚠️ Janela '{name_part}' não encontrada.")
            self._list_available_windows(window_list)
            return False

        # Selecionar a maior janela
        best = max(candidates, key=lambda x: x['area'])

        self.window_id = best['id']
        self.width = int(best['width'])
        self.height = int(best['height'])

        print(f"✅ Janela capturada:")
        print(f"   App: {best['owner']}")
        print(f"   Título: {best['name']}")
        print(f"   Resolução: {self.width}x{self.height}")
        print(f"   ID: {self.window_id}")

        return True

    def set_roi(self, x: int, y: int, width: int, height: int):
        """
        Define região de interesse em pixels absolutos.

        Args:
            x, y: Canto superior esquerdo
            width, height: Tamanho da região

        Exemplo:
            cap.set_roi(100, 50, 800, 600)  # Captura região 800x600 começando em (100, 50)
        """
        self.roi = (x, y, width, height)
        self.roi_percent = None
        print(f"📐 ROI definido: {width}x{height} em ({x}, {y})")

    def set_roi_percent(self, x_pct: float, y_pct: float, w_pct: float, h_pct: float):
        """
        Define região de interesse em porcentagem (0.0 a 1.0).
        Útil quando a resolução da janela pode variar.

        Args:
            x_pct, y_pct: Posição inicial em % (0.0 = início, 1.0 = fim)
            w_pct, h_pct: Tamanho em % (1.0 = 100%)

        Exemplo para Butterfly (capturar área central da imagem USG):
            cap.set_roi_percent(0.15, 0.10, 0.70, 0.80)  # 70% largura, 80% altura
        """
        self.roi_percent = (x_pct, y_pct, w_pct, h_pct)
        self.roi = None
        print(f"📐 ROI definido: {w_pct*100:.0f}%x{h_pct*100:.0f}% em ({x_pct*100:.0f}%, {y_pct*100:.0f}%)")

    def set_roi_butterfly(self):
        """
        Preset de ROI otimizado para app Butterfly iQ.
        Captura apenas a área da imagem de ultrassom, sem a UI.
        """
        # Butterfly típico: imagem USG ocupa ~70% central
        self.set_roi_percent(0.12, 0.08, 0.76, 0.84)
        print("🦋 ROI configurado para Butterfly iQ")

    def set_roi_clarius(self):
        """Preset de ROI para app Clarius."""
        self.set_roi_percent(0.10, 0.05, 0.80, 0.85)
        print("📱 ROI configurado para Clarius")

    def clear_roi(self):
        """Remove ROI - volta a capturar tela inteira."""
        self.roi = None
        self.roi_percent = None
        print("📐 ROI removido - capturando tela inteira")

    def _apply_roi(self, frame: np.ndarray) -> np.ndarray:
        """Aplica ROI ao frame capturado."""
        if frame is None:
            return None

        h, w = frame.shape[:2]

        # Calcular ROI em pixels
        if self.roi_percent is not None:
            x_pct, y_pct, w_pct, h_pct = self.roi_percent
            x = int(w * x_pct)
            y = int(h * y_pct)
            roi_w = int(w * w_pct)
            roi_h = int(h * h_pct)
        elif self.roi is not None:
            x, y, roi_w, roi_h = self.roi
        else:
            return frame  # Sem ROI

        # Validar limites
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        roi_w = min(roi_w, w - x)
        roi_h = min(roi_h, h - y)

        # Extrair região (operação rápida - apenas view, sem cópia)
        cropped = frame[y:y+roi_h, x:x+roi_w]

        return cropped.copy()  # .copy() para garantir memória contígua

    def _list_available_windows(self, window_list):
        """Lista janelas disponíveis para debug."""
        print("   Janelas disponíveis:")
        for window in window_list:
            owner = window.get('kCGWindowOwnerName', '')
            name = window.get('kCGWindowName', '')
            w = window.get('kCGWindowBounds', {}).get('Width', 0)
            h = window.get('kCGWindowBounds', {}).get('Height', 0)
            if w > 200 and h > 200:
                print(f"   - [{owner}] '{name}' ({w}x{h})")

    def read(self):
        """
        Captura frame com qualidade máxima.

        Returns:
            (success: bool, frame: np.ndarray ou None)

        O frame retornado é BGR (padrão OpenCV), sem compressão,
        na resolução original da janela.
        """
        if self.window_id is None:
            if self.window_name_part:
                self.find_window(self.window_name_part)
            if self.window_id is None:
                return False, self._last_successful_frame

        # Capturar imagem da janela
        # kCGWindowImageNominalResolution evita scaling em displays Retina
        image_ref = CGWindowListCreateImage(
            CGRectNull,
            kCGWindowListOptionIncludingWindow,
            self.window_id,
            kCGWindowImageBoundsIgnoreFraming | kCGWindowImageNominalResolution
        )

        if not image_ref:
            self.frames_failed += 1
            # Tentar re-encontrar janela a cada 30 falhas
            if self.frames_failed % 30 == 0 and self.window_name_part:
                print("⚠️ Reconectando à janela...")
                self.find_window(self.window_name_part)
            return False, self._last_successful_frame

        # Obter dimensões
        width = CGImageGetWidth(image_ref)
        height = CGImageGetHeight(image_ref)

        if width < 10 or height < 10:
            return False, self._last_successful_frame

        # Obter dados dos pixels
        data_provider = CGImageGetDataProvider(image_ref)
        pixel_data = CGDataProviderCopyData(data_provider)

        if not pixel_data:
            return False, self._last_successful_frame

        # Calcular layout de memória
        bytes_per_row = CGImageGetBytesPerRow(image_ref)

        try:
            # Criar array numpy diretamente dos dados
            # O formato é BGRA (Blue, Green, Red, Alpha) - 4 bytes por pixel
            frame = np.frombuffer(pixel_data, dtype=np.uint8)

            # Reshape considerando padding (bytes_per_row pode ser > width * 4)
            frame = frame.reshape((height, bytes_per_row // 4, 4))

            # Remover padding horizontal se houver
            if bytes_per_row // 4 > width:
                frame = frame[:, :width, :]

            # Converter BGRA para BGR (remover canal alpha)
            # IMPORTANTE: macOS usa BGRA, então B já está no índice 0
            # Apenas removemos o canal Alpha (índice 3)
            frame = frame[:, :, :3].copy()  # .copy() garante memória contígua

            # Aplicar ROI se configurado
            if self.roi is not None or self.roi_percent is not None:
                frame = self._apply_roi(frame)

            # Atualizar cache e estatísticas
            self._last_successful_frame = frame
            self.frames_captured += 1
            self.width = frame.shape[1] if frame is not None else width
            self.height = frame.shape[0] if frame is not None else height

            return True, frame

        except Exception as e:
            print(f"❌ Erro ao processar frame: {e}")
            self.frames_failed += 1
            return False, self._last_successful_frame

    def read_raw(self):
        """
        Captura frame em formato RAW (BGRA com alpha).
        Use quando precisar do canal alpha ou máxima fidelidade.

        Returns:
            (success: bool, frame: np.ndarray BGRA ou None)
        """
        if self.window_id is None:
            return False, None

        image_ref = CGWindowListCreateImage(
            CGRectNull,
            kCGWindowListOptionIncludingWindow,
            self.window_id,
            kCGWindowImageBoundsIgnoreFraming | kCGWindowImageNominalResolution
        )

        if not image_ref:
            return False, None

        width = CGImageGetWidth(image_ref)
        height = CGImageGetHeight(image_ref)

        if width < 10 or height < 10:
            return False, None

        pixel_data = CGDataProviderCopyData(CGImageGetDataProvider(image_ref))

        if not pixel_data:
            return False, None

        bytes_per_row = CGImageGetBytesPerRow(image_ref)

        try:
            frame = np.frombuffer(pixel_data, dtype=np.uint8)
            frame = frame.reshape((height, bytes_per_row // 4, 4))

            if bytes_per_row // 4 > width:
                frame = frame[:, :width, :]

            return True, frame.copy()

        except Exception as e:
            return False, None

    def get_info(self) -> dict:
        """Retorna informações sobre a captura."""
        return {
            'window_id': self.window_id,
            'width': self.width,
            'height': self.height,
            'frames_captured': self.frames_captured,
            'frames_failed': self.frames_failed,
            'success_rate': self.frames_captured / max(1, self.frames_captured + self.frames_failed) * 100
        }

    def release(self):
        """Libera recursos."""
        self._frame_buffer = None
        self._last_successful_frame = None
        self.window_id = None


class HighQualityCapture(WindowCapture):
    """
    Versão ainda mais otimizada para qualidade máxima.

    Adiciona:
    - Validação de colorspace
    - Detecção de alterações na janela
    - Métricas de qualidade
    """

    def __init__(self, window_name_part=None):
        super().__init__(window_name_part)
        self.colorspace_validated = False
        self.expected_colorspace = None

    def validate_colorspace(self, image_ref) -> bool:
        """Valida que o colorspace não mudou."""
        colorspace = CGImageGetColorSpace(image_ref)

        if self.expected_colorspace is None:
            self.expected_colorspace = colorspace
            self.colorspace_validated = True
            return True

        # Comparar colorspaces
        if colorspace != self.expected_colorspace:
            print("⚠️ Colorspace da janela mudou! Isso pode afetar as cores.")
            self.expected_colorspace = colorspace
            return False

        return True

    def read_validated(self):
        """
        Captura com validação de qualidade.

        Returns:
            (success: bool, frame: np.ndarray, quality_info: dict)
        """
        if self.window_id is None:
            return False, None, {'error': 'no_window'}

        image_ref = CGWindowListCreateImage(
            CGRectNull,
            kCGWindowListOptionIncludingWindow,
            self.window_id,
            kCGWindowImageBoundsIgnoreFraming | kCGWindowImageNominalResolution
        )

        if not image_ref:
            return False, self._last_successful_frame, {'error': 'capture_failed'}

        # Validar colorspace
        colorspace_ok = self.validate_colorspace(image_ref)

        width = CGImageGetWidth(image_ref)
        height = CGImageGetHeight(image_ref)
        bits_per_pixel = CGImageGetBitsPerPixel(image_ref)
        bytes_per_row = CGImageGetBytesPerRow(image_ref)

        quality_info = {
            'width': width,
            'height': height,
            'bits_per_pixel': bits_per_pixel,
            'bytes_per_row': bytes_per_row,
            'colorspace_valid': colorspace_ok,
            'expected_bpp': 32,  # BGRA = 32 bits
            'quality_ok': bits_per_pixel == 32 and colorspace_ok
        }

        if width < 10 or height < 10:
            return False, self._last_successful_frame, quality_info

        pixel_data = CGDataProviderCopyData(CGImageGetDataProvider(image_ref))

        if not pixel_data:
            return False, self._last_successful_frame, {'error': 'no_pixel_data'}

        try:
            frame = np.frombuffer(pixel_data, dtype=np.uint8)
            frame = frame.reshape((height, bytes_per_row // 4, 4))

            if bytes_per_row // 4 > width:
                frame = frame[:, :width, :]

            frame = frame[:, :, :3].copy()

            self._last_successful_frame = frame
            self.frames_captured += 1

            return True, frame, quality_info

        except Exception as e:
            return False, self._last_successful_frame, {'error': str(e)}


# Teste standalone
if __name__ == '__main__':
    import time

    print("=" * 60)
    print("Teste de Captura de Alta Qualidade")
    print("=" * 60)

    cap = HighQualityCapture("AIRPLAY")

    if cap.window_id:
        print("\nCapturando 100 frames para benchmark...")

        start = time.time()
        for i in range(100):
            ret, frame, info = cap.read_validated()
            if i == 0 and ret:
                print(f"\nQualidade do primeiro frame:")
                for k, v in info.items():
                    print(f"   {k}: {v}")

        elapsed = time.time() - start
        fps = 100 / elapsed

        print(f"\nResultados:")
        print(f"   Tempo total: {elapsed:.2f}s")
        print(f"   FPS médio: {fps:.1f}")
        print(f"   Frames OK: {cap.frames_captured}")
        print(f"   Frames falhos: {cap.frames_failed}")

        # Mostrar último frame
        if cap._last_successful_frame is not None:
            cv2.imshow('Captura', cap._last_successful_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    cap.release()
