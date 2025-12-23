"""
UI Utilities - Sistema de interface visual unificado para o aplicativo USG.

Funcoes reutilizaveis para desenhar paineis, barras, indicadores e labels
com estilo consistente em todos os modos.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


# ═══════════════════════════════════════════════════════════════════════════
# CORES DO TEMA
# ═══════════════════════════════════════════════════════════════════════════

class Theme:
    """Cores padronizadas para toda a UI."""

    # Backgrounds
    PANEL_BG = (20, 25, 30)           # Fundo dos paineis
    PANEL_BG_LIGHT = (35, 40, 45)     # Fundo secundario
    HEADER_BG = (40, 60, 70)          # Header do painel

    # Bordas
    BORDER_PRIMARY = (0, 180, 180)    # Cyan principal
    BORDER_SECONDARY = (80, 100, 110) # Cinza medio
    BORDER_ACCENT = (0, 200, 255)     # Laranja/amarelo

    # Textos
    TEXT_PRIMARY = (220, 220, 220)    # Branco principal
    TEXT_SECONDARY = (150, 150, 160)  # Cinza claro
    TEXT_ACCENT = (0, 255, 255)       # Cyan
    TEXT_MUTED = (100, 100, 110)      # Cinza escuro

    # Status
    STATUS_OK = (0, 255, 100)         # Verde
    STATUS_WARNING = (0, 200, 255)    # Laranja
    STATUS_ERROR = (0, 0, 255)        # Vermelho
    STATUS_INFO = (255, 200, 100)     # Azul claro
    STATUS_INACTIVE = (80, 80, 80)    # Cinza

    # Estruturas medicas
    NERVE = (0, 255, 255)             # Amarelo
    ARTERY = (0, 0, 255)              # Vermelho
    VEIN = (255, 100, 100)            # Azul
    FASCIA = (200, 200, 200)          # Cinza claro
    MUSCLE = (0, 180, 255)            # Laranja
    BONE = (255, 255, 255)            # Branco

    # Barras de progresso
    BAR_BG = (40, 40, 50)             # Fundo da barra
    BAR_FILL_OK = (0, 200, 100)       # Preenchimento OK
    BAR_FILL_LOW = (0, 100, 200)      # Preenchimento baixo
    BAR_FILL_HIGH = (0, 0, 200)       # Preenchimento alto


# ═══════════════════════════════════════════════════════════════════════════
# FONTE PADRAO
# ═══════════════════════════════════════════════════════════════════════════

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE_TITLE = 0.45
FONT_SCALE_LABEL = 0.35
FONT_SCALE_VALUE = 0.4
FONT_SCALE_SMALL = 0.3


# ═══════════════════════════════════════════════════════════════════════════
# FUNCOES DE DESENHO DE PAINEIS
# ═══════════════════════════════════════════════════════════════════════════

def draw_panel(
    frame: np.ndarray,
    x: int, y: int,
    width: int, height: int,
    title: Optional[str] = None,
    title_color: Tuple[int, int, int] = Theme.TEXT_ACCENT,
    border_color: Tuple[int, int, int] = Theme.BORDER_PRIMARY,
    bg_color: Tuple[int, int, int] = Theme.PANEL_BG,
    alpha: float = 0.9,
    header_height: int = 25,
    version: Optional[str] = None
) -> int:
    """
    Desenha um painel padronizado com header opcional.

    Returns:
        int: Y inicial para conteudo (abaixo do header)
    """
    # Background com transparencia
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Borda externa
    cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, 1)

    content_y = y + 5

    # Header se tiver titulo
    if title:
        # Fundo do header
        header_bg = tuple(max(0, c - 20) for c in border_color)
        cv2.rectangle(frame, (x + 1, y + 1), (x + width - 1, y + header_height), header_bg, -1)

        # Titulo
        cv2.putText(frame, title, (x + 8, y + header_height - 7),
                   FONT, FONT_SCALE_TITLE, title_color, 1, cv2.LINE_AA)

        # Versao
        if version:
            (vw, _), _ = cv2.getTextSize(version, FONT, FONT_SCALE_SMALL, 1)
            cv2.putText(frame, version, (x + width - vw - 8, y + header_height - 7),
                       FONT, FONT_SCALE_SMALL, Theme.TEXT_MUTED, 1, cv2.LINE_AA)

        content_y = y + header_height + 8

    return content_y


def draw_status_indicator(
    frame: np.ndarray,
    x: int, y: int,
    status: str,
    active: bool = True,
    color: Optional[Tuple[int, int, int]] = None
) -> None:
    """Desenha um indicador de status (circulo + texto)."""
    if color is None:
        if active:
            color = Theme.STATUS_OK
        else:
            color = Theme.STATUS_INACTIVE

    # Circulo externo (glow)
    cv2.circle(frame, (x + 6, y), 7, tuple(c // 2 for c in color), 1)
    # Circulo interno
    cv2.circle(frame, (x + 6, y), 5, color, -1)
    # Texto
    cv2.putText(frame, status, (x + 18, y + 4),
               FONT, FONT_SCALE_LABEL, color, 1, cv2.LINE_AA)


def draw_progress_bar(
    frame: np.ndarray,
    x: int, y: int,
    width: int, height: int,
    value: float,
    max_value: float = 100.0,
    color: Optional[Tuple[int, int, int]] = None,
    bg_color: Tuple[int, int, int] = Theme.BAR_BG,
    show_markers: Optional[List[float]] = None,
    gradient: bool = False
) -> None:
    """
    Desenha uma barra de progresso.

    Args:
        value: Valor atual
        max_value: Valor maximo
        color: Cor do preenchimento (auto se None)
        show_markers: Lista de valores para mostrar marcadores
        gradient: Se True, usa gradiente de cor baseado no valor
    """
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)

    # Calcular preenchimento
    fill_ratio = min(1.0, max(0.0, value / max_value))
    fill_w = int(width * fill_ratio)

    # Cor automatica baseada no valor
    if color is None:
        if gradient:
            if fill_ratio < 0.35:
                color = Theme.STATUS_ERROR
            elif fill_ratio < 0.65:
                color = Theme.STATUS_WARNING
            else:
                color = Theme.STATUS_OK
        else:
            color = Theme.BAR_FILL_OK

    # Preenchimento
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), color, -1)

    # Marcadores
    if show_markers:
        for marker_val in show_markers:
            if 0 < marker_val < max_value:
                mx = x + int(width * marker_val / max_value)
                cv2.line(frame, (mx, y), (mx, y + height), Theme.TEXT_MUTED, 1)


def draw_labeled_value(
    frame: np.ndarray,
    x: int, y: int,
    label: str,
    value: str,
    unit: str = "",
    label_color: Tuple[int, int, int] = Theme.TEXT_SECONDARY,
    value_color: Tuple[int, int, int] = Theme.TEXT_PRIMARY,
    label_width: int = 50
) -> int:
    """
    Desenha um label com valor alinhado.

    Returns:
        int: Largura total usada
    """
    # Label
    cv2.putText(frame, label, (x, y),
               FONT, FONT_SCALE_LABEL, label_color, 1, cv2.LINE_AA)

    # Valor
    cv2.putText(frame, value, (x + label_width, y),
               FONT, FONT_SCALE_VALUE, value_color, 1, cv2.LINE_AA)

    # Unidade
    if unit:
        (vw, _), _ = cv2.getTextSize(value, FONT, FONT_SCALE_VALUE, 1)
        cv2.putText(frame, unit, (x + label_width + vw + 3, y),
                   FONT, FONT_SCALE_SMALL, Theme.TEXT_MUTED, 1, cv2.LINE_AA)

    return label_width + 50


def draw_mini_chart(
    frame: np.ndarray,
    x: int, y: int,
    width: int, height: int,
    values: List[float],
    color: Tuple[int, int, int] = Theme.STATUS_OK,
    bg_color: Tuple[int, int, int] = Theme.PANEL_BG_LIGHT,
    filled: bool = False
) -> None:
    """Desenha um mini grafico de linha."""
    if not values or len(values) < 2:
        return

    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), Theme.BORDER_SECONDARY, 1)

    # Normalizar valores
    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val != min_val else 1

    # Pontos
    points = []
    for i, v in enumerate(values):
        px = x + int((i / (len(values) - 1)) * width)
        py = y + height - int(((v - min_val) / val_range) * (height - 4)) - 2
        points.append((px, py))

    # Desenhar
    pts = np.array(points, dtype=np.int32)

    if filled:
        # Area preenchida
        fill_pts = np.vstack([pts, [[x + width, y + height], [x, y + height]]])
        overlay = frame.copy()
        cv2.fillPoly(overlay, [fill_pts], color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Linha
    cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)


def draw_structure_list(
    frame: np.ndarray,
    x: int, y: int,
    structures: List[Dict[str, Any]],
    max_items: int = 5,
    item_height: int = 20
) -> int:
    """
    Desenha uma lista de estruturas detectadas.

    Returns:
        int: Altura total usada
    """
    for i, struct in enumerate(structures[:max_items]):
        sy = y + i * item_height

        # Cor da estrutura
        color = struct.get('color', Theme.TEXT_PRIMARY)

        # Indicador
        cv2.circle(frame, (x + 5, sy + 5), 4, color, -1)

        # Nome
        name = struct.get('type', 'UNKNOWN')[:8]
        cv2.putText(frame, name, (x + 15, sy + 9),
                   FONT, FONT_SCALE_SMALL, color, 1, cv2.LINE_AA)

        # Valor extra (CSA, confianca, etc)
        if 'csa_mm2' in struct:
            cv2.putText(frame, f"{struct['csa_mm2']:.1f}mm²", (x + 70, sy + 9),
                       FONT, FONT_SCALE_SMALL, Theme.TEXT_MUTED, 1, cv2.LINE_AA)
        elif 'confidence' in struct:
            conf_pct = int(struct['confidence'] * 100)
            cv2.putText(frame, f"{conf_pct}%", (x + 70, sy + 9),
                       FONT, FONT_SCALE_SMALL, Theme.TEXT_MUTED, 1, cv2.LINE_AA)

    return min(len(structures), max_items) * item_height


def draw_scan_quality(
    frame: np.ndarray,
    x: int, y: int,
    quality: float,
    label: str = "SCAN Q"
) -> None:
    """Desenha indicador de qualidade de scan."""
    q_pct = int(quality * 100) if quality <= 1.0 else int(quality)

    # Cor baseada na qualidade
    if q_pct >= 60:
        color = Theme.STATUS_OK
    elif q_pct >= 35:
        color = Theme.STATUS_WARNING
    else:
        color = Theme.STATUS_ERROR

    # Label
    cv2.putText(frame, f"{label}: {q_pct}%", (x, y),
               FONT, FONT_SCALE_SMALL, color, 1, cv2.LINE_AA)


def draw_warning_banner(
    frame: np.ndarray,
    text: str,
    y: Optional[int] = None,
    color: Tuple[int, int, int] = Theme.STATUS_ERROR
) -> None:
    """Desenha um banner de aviso centralizado."""
    h, w = frame.shape[:2]
    if y is None:
        y = 30

    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    tx = (w - tw) // 2

    # Background
    cv2.rectangle(frame, (tx - 10, y - th - 5), (tx + tw + 10, y + 5), (0, 0, 0), -1)
    cv2.rectangle(frame, (tx - 10, y - th - 5), (tx + tw + 10, y + 5), color, 1)

    # Texto
    cv2.putText(frame, text, (tx, y), FONT, 0.5, color, 1, cv2.LINE_AA)


def draw_legend(
    frame: np.ndarray,
    x: int, y: int,
    items: List[Tuple[str, Tuple[int, int, int]]],
    horizontal: bool = False,
    item_width: int = 80
) -> None:
    """
    Desenha uma legenda de cores.

    Args:
        items: Lista de tuplas (nome, cor)
        horizontal: Se True, desenha horizontalmente
    """
    for i, (name, color) in enumerate(items):
        if horizontal:
            ix = x + i * item_width
            iy = y
        else:
            ix = x
            iy = y + i * 18

        cv2.circle(frame, (ix + 5, iy), 4, color, -1)
        cv2.putText(frame, name, (ix + 12, iy + 4),
                   FONT, FONT_SCALE_SMALL, Theme.TEXT_SECONDARY, 1, cv2.LINE_AA)


def draw_keyboard_hint(
    frame: np.ndarray,
    x: int, y: int,
    key: str,
    action: str
) -> int:
    """
    Desenha uma dica de atalho de teclado.

    Returns:
        int: Largura usada
    """
    # Caixa da tecla
    (kw, kh), _ = cv2.getTextSize(key, FONT, FONT_SCALE_SMALL, 1)
    box_w = max(kw + 8, 20)

    cv2.rectangle(frame, (x, y - kh - 4), (x + box_w, y + 4), Theme.BORDER_SECONDARY, 1)
    cv2.putText(frame, key, (x + 4, y),
               FONT, FONT_SCALE_SMALL, Theme.TEXT_PRIMARY, 1, cv2.LINE_AA)

    # Acao
    cv2.putText(frame, action, (x + box_w + 5, y),
               FONT, FONT_SCALE_SMALL, Theme.TEXT_MUTED, 1, cv2.LINE_AA)

    (aw, _), _ = cv2.getTextSize(action, FONT, FONT_SCALE_SMALL, 1)
    return box_w + aw + 10


# ═══════════════════════════════════════════════════════════════════════════
# EFEITOS VISUAIS
# ═══════════════════════════════════════════════════════════════════════════

def draw_glow_circle(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    intensity: float = 0.5
) -> None:
    """Desenha um circulo com efeito de glow."""
    for i in range(3, 0, -1):
        alpha = intensity * (i / 3)
        glow_color = tuple(int(c * alpha) for c in color)
        cv2.circle(frame, center, radius + i * 2, glow_color, 1, cv2.LINE_AA)
    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)


def draw_pulsing_text(
    frame: np.ndarray,
    text: str,
    x: int, y: int,
    color: Tuple[int, int, int],
    phase: float,
    base_scale: float = 0.5
) -> None:
    """Desenha texto com efeito de pulsacao."""
    import math
    pulse = 0.15 * math.sin(phase * 2 * math.pi)
    scale = base_scale + pulse

    # Sombra
    cv2.putText(frame, text, (x + 1, y + 1),
               FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    # Texto
    cv2.putText(frame, text, (x, y),
               FONT, scale, color, 1, cv2.LINE_AA)


def apply_vignette(
    frame: np.ndarray,
    strength: float = 0.3
) -> np.ndarray:
    """Aplica efeito de vinheta (escurecimento nas bordas)."""
    h, w = frame.shape[:2]

    # Criar mascara gaussiana
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)

    dist = np.sqrt(X**2 + Y**2)
    mask = 1 - np.clip(dist * strength, 0, 1)

    # Aplicar
    mask = mask[:, :, np.newaxis]
    result = (frame * mask).astype(np.uint8)

    return result
