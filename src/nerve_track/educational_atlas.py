#!/usr/bin/env python3
"""
NERVE TRACK v2.0 - Educational Atlas
=====================================
Fallback visual educacional que funciona SEM modelo treinado.

Mostra:
- Atlas anatômico do bloqueio selecionado
- Estruturas esperadas em posições típicas
- Instruções de inserção de agulha
- Referências de profundidade e tamanho

Baseado em:
- ScanNav Anatomy PNB
- Nerveblox Atlas
- NYSORA (NY School of Regional Anesthesia)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

# Importar database de bloqueios
try:
    from .block_database import (
        get_block_config, get_all_block_ids,
        StructureType, NerveBlock, Structure
    )
except ImportError:
    from block_database import (
        get_block_config, get_all_block_ids,
        StructureType, NerveBlock, Structure
    )


# =============================================================================
# CONFIGURAÇÃO DE CORES
# =============================================================================

COLORS = {
    # Estruturas anatômicas (BGR)
    "nerve": (0, 255, 255),      # Amarelo/Cyan
    "artery": (0, 0, 255),       # Vermelho
    "vein": (255, 0, 0),         # Azul
    "muscle": (100, 80, 60),     # Marrom
    "fascia": (200, 200, 200),   # Cinza claro
    "bone": (255, 255, 255),     # Branco
    "pleura": (180, 100, 255),   # Rosa
    "other": (150, 150, 150),    # Cinza

    # UI
    "text": (255, 255, 255),
    "text_shadow": (0, 0, 0),
    "panel_bg": (20, 20, 30),
    "panel_border": (60, 60, 80),
    "needle_path": (0, 255, 0),  # Verde
    "danger_zone": (0, 0, 200),  # Vermelho escuro
    "target_zone": (0, 200, 0),  # Verde escuro
}

# Mapeamento de tipo para cor
TYPE_COLORS = {
    StructureType.NERVE: COLORS["nerve"],
    StructureType.ARTERY: COLORS["artery"],
    StructureType.VEIN: COLORS["vein"],
    StructureType.MUSCLE: COLORS["muscle"],
    StructureType.FASCIA: COLORS["fascia"],
    StructureType.BONE: COLORS["bone"],
    StructureType.PLEURA: COLORS["pleura"],
    StructureType.OTHER: COLORS["other"],
}


# =============================================================================
# ATLAS EDUCACIONAL
# =============================================================================

@dataclass
class AtlasStructure:
    """Estrutura para renderização no atlas"""
    name: str
    type: StructureType
    x: float  # Posição X normalizada (0-1)
    y: float  # Posição Y normalizada (0-1)
    width: float  # Largura normalizada
    height: float  # Altura normalizada
    shape: str  # "ellipse", "circle", "rect", "line"
    is_target: bool = False
    is_danger: bool = False
    label: str = ""
    depth_mm: float = 0


class EducationalAtlas:
    """
    Atlas educacional para visualização de bloqueios nervosos.
    Funciona completamente sem modelo de IA treinado.
    """

    def __init__(self, mm_per_pixel: float = 0.1):
        self.mm_per_pixel = mm_per_pixel
        self.current_block_id: Optional[str] = None
        self.current_block: Optional[NerveBlock] = None

        # Cache de layouts
        self._layout_cache: Dict[str, List[AtlasStructure]] = {}

        # Configurações de visualização
        self.show_labels = True
        self.show_depth_ruler = True
        self.show_needle_guide = True
        self.show_instructions = True
        self.animation_phase = 0

    def set_block(self, block_id: str):
        """Define o bloqueio atual"""
        self.current_block_id = block_id
        self.current_block = get_block_config(block_id)

    def set_scale(self, mm_per_pixel: float):
        """Atualiza escala"""
        self.mm_per_pixel = mm_per_pixel

    def render_atlas(
        self,
        frame: np.ndarray,
        overlay_mode: str = "side"  # "side", "overlay", "full"
    ) -> np.ndarray:
        """
        Renderiza o atlas educacional.

        Args:
            frame: Frame de ultrassom original
            overlay_mode: Modo de exibição
                - "side": Painel lateral com atlas
                - "overlay": Sobreposição semi-transparente
                - "full": Atlas em tela cheia

        Returns:
            Frame com atlas renderizado
        """
        if self.current_block is None:
            return self._render_no_block_selected(frame)

        h, w = frame.shape[:2]

        if overlay_mode == "side":
            return self._render_side_panel(frame)
        elif overlay_mode == "overlay":
            return self._render_overlay(frame)
        else:  # full
            return self._render_full_atlas(frame)

    def _render_no_block_selected(self, frame: np.ndarray) -> np.ndarray:
        """Renderiza mensagem quando nenhum bloqueio está selecionado"""
        h, w = frame.shape[:2]

        # Painel de instrução
        panel_h = 100
        panel_y = h - panel_h - 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (20, panel_y), (w - 20, panel_y + panel_h),
                     COLORS["panel_bg"], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (20, panel_y), (w - 20, panel_y + panel_h),
                     COLORS["panel_border"], 2)

        # Texto
        cv2.putText(frame, "NERVE TRACK - Atlas Educacional",
                   (40, panel_y + 30), cv2.FONT_HERSHEY_DUPLEX,
                   0.7, COLORS["nerve"], 2, cv2.LINE_AA)
        cv2.putText(frame, "Pressione [N] para selecionar um bloqueio nervoso",
                   (40, panel_y + 55), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, COLORS["text"], 1, cv2.LINE_AA)
        cv2.putText(frame, "Use [<] [>] para navegar entre os 28 bloqueios disponiveis",
                   (40, panel_y + 80), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (150, 150, 150), 1, cv2.LINE_AA)

        return frame

    def _render_side_panel(self, frame: np.ndarray) -> np.ndarray:
        """Renderiza painel lateral com atlas"""
        h, w = frame.shape[:2]
        panel_w = min(350, w // 3)

        # Criar painel
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = COLORS["panel_bg"]

        # Header
        cv2.rectangle(panel, (0, 0), (panel_w, 50), (30, 30, 45), -1)
        cv2.putText(panel, "ATLAS ANATOMICO",
                   (15, 32), cv2.FONT_HERSHEY_DUPLEX,
                   0.6, COLORS["nerve"], 2, cv2.LINE_AA)
        cv2.line(panel, (0, 50), (panel_w, 50), COLORS["nerve"], 2)

        # Nome do bloqueio
        if self.current_block:
            cv2.putText(panel, self.current_block.name[:30],
                       (15, 75), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, COLORS["text"], 1, cv2.LINE_AA)
            cv2.putText(panel, f"Regiao: {self.current_block.region.value}",
                       (15, 95), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # Área do diagrama
        diagram_y = 110
        diagram_h = min(250, h - 300)
        diagram_area = panel[diagram_y:diagram_y + diagram_h, 10:panel_w - 10]

        # Desenhar diagrama anatômico
        self._draw_anatomy_diagram(diagram_area)

        # Legenda
        legend_y = diagram_y + diagram_h + 20
        self._draw_legend(panel, 15, legend_y)

        # Instruções de agulha
        if self.current_block and self.show_instructions:
            instr_y = legend_y + 120
            self._draw_needle_instructions(panel, 15, instr_y, panel_w - 30)

        # Régua de profundidade
        if self.show_depth_ruler:
            self._draw_depth_ruler(panel, panel_w - 30, diagram_y, diagram_h)

        # Combinar com frame original
        result = np.hstack([panel, frame])

        return result

    def _render_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Renderiza atlas como sobreposição transparente"""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        if self.current_block is None:
            return frame

        # Área de sobreposição
        margin = 50
        atlas_x = margin
        atlas_y = margin
        atlas_w = w // 2 - margin * 2
        atlas_h = h - margin * 2

        # Fundo semi-transparente
        cv2.rectangle(overlay, (atlas_x, atlas_y),
                     (atlas_x + atlas_w, atlas_y + atlas_h),
                     COLORS["panel_bg"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Borda
        cv2.rectangle(frame, (atlas_x, atlas_y),
                     (atlas_x + atlas_w, atlas_y + atlas_h),
                     COLORS["nerve"], 2)

        # Título
        cv2.putText(frame, self.current_block.name[:35],
                   (atlas_x + 15, atlas_y + 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, COLORS["nerve"], 2, cv2.LINE_AA)

        # Diagrama
        diagram_area = frame[atlas_y + 50:atlas_y + atlas_h - 100,
                            atlas_x + 10:atlas_x + atlas_w - 10]
        self._draw_anatomy_diagram(diagram_area)

        # Legenda compacta
        self._draw_legend_compact(frame, atlas_x + 15, atlas_y + atlas_h - 90)

        return frame

    def _render_full_atlas(self, frame: np.ndarray) -> np.ndarray:
        """Renderiza atlas em tela cheia"""
        h, w = frame.shape[:2]

        # Criar canvas
        atlas = np.zeros((h, w, 3), dtype=np.uint8)
        atlas[:] = (15, 15, 20)

        if self.current_block is None:
            return self._render_no_block_selected(atlas)

        # Layout: diagrama à esquerda, info à direita
        diagram_w = w // 2
        info_w = w - diagram_w

        # Header
        cv2.rectangle(atlas, (0, 0), (w, 60), (25, 25, 35), -1)
        cv2.putText(atlas, f"NERVE TRACK ATLAS - {self.current_block.name}",
                   (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                   0.8, COLORS["nerve"], 2, cv2.LINE_AA)
        cv2.line(atlas, (0, 60), (w, 60), COLORS["nerve"], 2)

        # Diagrama anatômico (lado esquerdo)
        diagram_area = atlas[80:h - 100, 20:diagram_w - 20]
        self._draw_anatomy_diagram(diagram_area, detailed=True)

        # Informações (lado direito)
        info_x = diagram_w + 20
        info_y = 80

        # Região e descrição
        cv2.putText(atlas, f"Regiao: {self.current_block.region.value}",
                   (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # Estruturas
        info_y += 40
        cv2.putText(atlas, "ESTRUTURAS:", (info_x, info_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS["text"], 1, cv2.LINE_AA)

        info_y += 25
        for struct in self.current_block.structures[:10]:
            color = TYPE_COLORS.get(struct.type, COLORS["other"])

            # Indicador de cor
            cv2.circle(atlas, (info_x + 8, info_y - 4), 5, color, -1)

            # Nome e tipo
            label = f"{struct.name}"
            if struct.is_target:
                label += " [ALVO]"
            if struct.is_danger:
                label += " [!]"

            cv2.putText(atlas, label, (info_x + 20, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            info_y += 22

        # Instruções de agulha
        info_y += 20
        self._draw_needle_instructions(atlas, info_x, info_y, info_w - 40)

        # Legenda
        self._draw_legend(atlas, 20, h - 90)

        # Mini preview do frame real (canto inferior direito)
        preview_h = 120
        preview_w = int(preview_h * w / h)
        preview = cv2.resize(frame, (preview_w, preview_h))
        atlas[h - preview_h - 10:h - 10, w - preview_w - 10:w - 10] = preview
        cv2.rectangle(atlas, (w - preview_w - 10, h - preview_h - 10),
                     (w - 10, h - 10), COLORS["panel_border"], 1)
        cv2.putText(atlas, "LIVE", (w - preview_w - 5, h - preview_h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        return atlas

    def _draw_anatomy_diagram(
        self,
        area: np.ndarray,
        detailed: bool = False
    ):
        """Desenha diagrama anatômico esquemático"""
        h, w = area.shape[:2]

        if self.current_block is None:
            return

        # Fundo de ultrassom simulado
        self._draw_ultrasound_background(area)

        # Desenhar estruturas
        structures = self._get_structure_layout()

        for struct in structures:
            self._draw_structure(area, struct, detailed)

        # Guia de agulha (se habilitado)
        if self.show_needle_guide and self.current_block.needle_approach:
            self._draw_needle_guide(area)

    def _draw_ultrasound_background(self, area: np.ndarray):
        """Desenha fundo simulando textura de ultrassom"""
        h, w = area.shape[:2]

        # Gradiente de profundidade
        for y in range(h):
            brightness = int(60 - (y / h) * 30)
            area[y, :] = (brightness, brightness, brightness + 5)

        # Ruído speckle simulado
        noise = np.random.randint(0, 20, (h, w), dtype=np.uint8)
        area[:, :, 0] = np.clip(area[:, :, 0].astype(int) + noise, 0, 255).astype(np.uint8)
        area[:, :, 1] = np.clip(area[:, :, 1].astype(int) + noise, 0, 255).astype(np.uint8)
        area[:, :, 2] = np.clip(area[:, :, 2].astype(int) + noise, 0, 255).astype(np.uint8)

        # Suavizar
        area[:] = cv2.GaussianBlur(area, (5, 5), 1)

    def _get_structure_layout(self) -> List[AtlasStructure]:
        """Gera layout das estruturas para o bloqueio atual"""
        if self.current_block_id in self._layout_cache:
            return self._layout_cache[self.current_block_id]

        structures = []

        if self.current_block is None:
            return structures

        # Converter estruturas do bloqueio para layout visual
        for i, struct in enumerate(self.current_block.structures):
            # Posição baseada na posição típica ou calculada
            if struct.typical_position:
                x, y = struct.typical_position
            else:
                # Distribuir automaticamente
                x = 0.3 + (i % 3) * 0.2
                y = 0.2 + (i // 3) * 0.15

            # Tamanho baseado no tipo
            if struct.type == StructureType.NERVE:
                width, height = 0.12, 0.08
                shape = "ellipse"
            elif struct.type == StructureType.ARTERY:
                width, height = 0.08, 0.08
                shape = "circle"
            elif struct.type == StructureType.VEIN:
                width, height = 0.10, 0.06
                shape = "ellipse"
            elif struct.type == StructureType.MUSCLE:
                width, height = 0.25, 0.12
                shape = "rect"
            elif struct.type == StructureType.FASCIA:
                width, height = 0.8, 0.02
                shape = "line"
            elif struct.type == StructureType.BONE:
                width, height = 0.3, 0.05
                shape = "rect"
            else:
                width, height = 0.1, 0.08
                shape = "ellipse"

            atlas_struct = AtlasStructure(
                name=struct.name,
                type=struct.type,
                x=x,
                y=y,
                width=width,
                height=height,
                shape=shape,
                is_target=struct.is_target,
                is_danger=struct.is_danger,
                label=struct.abbreviation or struct.name[:3].upper(),
                depth_mm=struct.typical_depth_mm or 0
            )
            structures.append(atlas_struct)

        self._layout_cache[self.current_block_id] = structures
        return structures

    def _draw_structure(
        self,
        area: np.ndarray,
        struct: AtlasStructure,
        detailed: bool = False
    ):
        """Desenha uma estrutura anatômica"""
        h, w = area.shape[:2]

        # Converter coordenadas normalizadas para pixels
        cx = int(struct.x * w)
        cy = int(struct.y * h)
        sw = int(struct.width * w)
        sh = int(struct.height * h)

        color = TYPE_COLORS.get(struct.type, COLORS["other"])

        # Animação para alvos
        if struct.is_target:
            pulse = abs(math.sin(self.animation_phase * 0.1)) * 0.3 + 0.7
            color = tuple(int(c * pulse) for c in color)

        # Desenhar forma
        if struct.shape == "circle":
            radius = max(sw, sh) // 2
            # Preenchimento escuro (lúmen)
            cv2.circle(area, (cx, cy), radius, (20, 20, 30), -1)
            # Borda
            cv2.circle(area, (cx, cy), radius, color, 2)

        elif struct.shape == "ellipse":
            # Para nervos: padrão honeycomb
            if struct.type == StructureType.NERVE:
                cv2.ellipse(area, (cx, cy), (sw//2, sh//2), 0, 0, 360,
                           (180, 180, 150), -1)
                # Fascículos
                for _ in range(5):
                    fx = cx + np.random.randint(-sw//3, sw//3)
                    fy = cy + np.random.randint(-sh//3, sh//3)
                    cv2.circle(area, (fx, fy), 3, (80, 80, 60), -1)
                cv2.ellipse(area, (cx, cy), (sw//2, sh//2), 0, 0, 360, color, 2)
            else:
                cv2.ellipse(area, (cx, cy), (sw//2, sh//2), 0, 0, 360,
                           (20, 20, 30), -1)
                cv2.ellipse(area, (cx, cy), (sw//2, sh//2), 0, 0, 360, color, 2)

        elif struct.shape == "rect":
            x1, y1 = cx - sw//2, cy - sh//2
            x2, y2 = cx + sw//2, cy + sh//2

            if struct.type == StructureType.MUSCLE:
                # Textura muscular
                cv2.rectangle(area, (x1, y1), (x2, y2), (60, 50, 40), -1)
                for i in range(y1, y2, 4):
                    cv2.line(area, (x1, i), (x2, i), (80, 70, 60), 1)
            else:
                cv2.rectangle(area, (x1, y1), (x2, y2), color, 2)

        elif struct.shape == "line":
            y_line = cy
            cv2.line(area, (int(w * 0.1), y_line), (int(w * 0.9), y_line),
                    color, 2)

        # Zona de perigo
        if struct.is_danger:
            cv2.circle(area, (cx, cy), max(sw, sh)//2 + 10,
                      COLORS["danger_zone"], 1)

        # Label
        if self.show_labels:
            label_y = cy - sh//2 - 8 if struct.shape != "line" else cy - 15

            # Sombra do texto
            cv2.putText(area, struct.label, (cx - 15, label_y + 1),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text_shadow"], 2)
            cv2.putText(area, struct.label, (cx - 15, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # Marcador de alvo
        if struct.is_target:
            # Crosshair
            cv2.line(area, (cx - 15, cy), (cx - 5, cy), COLORS["target_zone"], 2)
            cv2.line(area, (cx + 5, cy), (cx + 15, cy), COLORS["target_zone"], 2)
            cv2.line(area, (cx, cy - 15), (cx, cy - 5), COLORS["target_zone"], 2)
            cv2.line(area, (cx, cy + 5), (cx, cy + 15), COLORS["target_zone"], 2)

    def _draw_needle_guide(self, area: np.ndarray):
        """Desenha guia de trajetória da agulha"""
        h, w = area.shape[:2]

        if not self.current_block or not self.current_block.needle_approach:
            return

        approach = self.current_block.needle_approach

        # Ponto de entrada
        entry_x = int(w * 0.15)
        entry_y = int(h * 0.1)

        # Encontrar alvo
        target_x, target_y = w // 2, h // 2
        for struct in self._get_structure_layout():
            if struct.is_target:
                target_x = int(struct.x * w)
                target_y = int(struct.y * h)
                break

        # Linha tracejada da trajetória
        self._draw_dashed_line(area, (entry_x, entry_y), (target_x, target_y),
                              COLORS["needle_path"], 2, 10)

        # Ponta de seta no alvo
        angle = math.atan2(target_y - entry_y, target_x - entry_x)
        arrow_len = 15
        pt1 = (int(target_x - arrow_len * math.cos(angle - 0.3)),
               int(target_y - arrow_len * math.sin(angle - 0.3)))
        pt2 = (int(target_x - arrow_len * math.cos(angle + 0.3)),
               int(target_y - arrow_len * math.sin(angle + 0.3)))
        cv2.line(area, (target_x, target_y), pt1, COLORS["needle_path"], 2)
        cv2.line(area, (target_x, target_y), pt2, COLORS["needle_path"], 2)

        # Texto de abordagem
        cv2.putText(area, approach[:20], (entry_x, entry_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["needle_path"], 1)

    def _draw_dashed_line(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 1,
        dash_length: int = 10
    ):
        """Desenha linha tracejada"""
        dist = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        dashes = int(dist / dash_length)

        for i in range(0, dashes, 2):
            start_ratio = i / dashes
            end_ratio = min((i + 1) / dashes, 1.0)

            start = (int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio),
                    int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio))
            end = (int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio),
                  int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio))

            cv2.line(img, start, end, color, thickness)

    def _draw_depth_ruler(
        self,
        panel: np.ndarray,
        x: int,
        y_start: int,
        height: int
    ):
        """Desenha régua de profundidade"""
        # Linha vertical
        cv2.line(panel, (x, y_start), (x, y_start + height), (100, 100, 100), 1)

        # Marcações a cada 1cm (10mm)
        max_depth_mm = int(height * self.mm_per_pixel)

        for depth in range(0, max_depth_mm + 1, 10):
            y = y_start + int(depth / self.mm_per_pixel)
            if y > y_start + height:
                break

            # Marcação
            cv2.line(panel, (x - 5, y), (x, y), (100, 100, 100), 1)

            # Label
            cv2.putText(panel, f"{depth}mm", (x - 35, y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100, 100, 100), 1)

    def _draw_legend(self, panel: np.ndarray, x: int, y: int):
        """Desenha legenda de cores"""
        cv2.putText(panel, "LEGENDA:", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text"], 1)

        items = [
            ("Nervo", COLORS["nerve"]),
            ("Arteria", COLORS["artery"]),
            ("Veia", COLORS["vein"]),
            ("Musculo", COLORS["muscle"]),
        ]

        y += 20
        for name, color in items:
            cv2.circle(panel, (x + 8, y), 5, color, -1)
            cv2.putText(panel, name, (x + 20, y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y += 18

        # Símbolos especiais
        y += 5
        cv2.putText(panel, "+ ALVO", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["target_zone"], 1)
        cv2.putText(panel, "! PERIGO", (x + 60, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["danger_zone"], 1)

    def _draw_legend_compact(self, panel: np.ndarray, x: int, y: int):
        """Legenda compacta em linha"""
        items = [
            ("N", COLORS["nerve"]),
            ("A", COLORS["artery"]),
            ("V", COLORS["vein"]),
            ("M", COLORS["muscle"]),
        ]

        for i, (letter, color) in enumerate(items):
            cx = x + i * 50
            cv2.circle(panel, (cx, y), 8, color, -1)
            cv2.putText(panel, letter, (cx - 4, y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    def _draw_needle_instructions(
        self,
        panel: np.ndarray,
        x: int,
        y: int,
        max_width: int
    ):
        """Desenha instruções de inserção de agulha"""
        if self.current_block is None:
            return

        cv2.putText(panel, "TECNICA:", (x, y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.45, COLORS["needle_path"], 1)

        y += 22

        # Abordagem
        if self.current_block.needle_approach:
            approach = self.current_block.needle_approach
            # Quebrar texto longo
            words = approach.split()
            line = ""
            for word in words:
                test_line = line + " " + word if line else word
                if len(test_line) * 7 > max_width:
                    cv2.putText(panel, line, (x, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text"], 1)
                    y += 16
                    line = word
                else:
                    line = test_line
            if line:
                cv2.putText(panel, line, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text"], 1)
                y += 16

        # Dicas adicionais
        y += 10
        tips = [
            "• Manter visualizacao da ponta",
            "• Aspirar antes de injetar",
            "• Observar spread do AL",
        ]

        for tip in tips:
            cv2.putText(panel, tip, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 140), 1)
            y += 14

    def update_animation(self):
        """Atualiza fase da animação"""
        self.animation_phase += 1


# =============================================================================
# FACTORY
# =============================================================================

def create_educational_atlas(mm_per_pixel: float = 0.1) -> EducationalAtlas:
    """Cria instância do atlas educacional"""
    return EducationalAtlas(mm_per_pixel=mm_per_pixel)


# =============================================================================
# TESTE
# =============================================================================

if __name__ == "__main__":
    # Teste do atlas
    atlas = create_educational_atlas()

    # Criar frame de teste
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 50)

    # Sem bloqueio selecionado
    result = atlas.render_atlas(frame.copy(), "side")
    cv2.imshow("Atlas - No Block", result)

    # Com bloqueio selecionado
    atlas.set_block("interscalene")
    result = atlas.render_atlas(frame.copy(), "side")
    cv2.imshow("Atlas - Interscalene (Side)", result)

    result = atlas.render_atlas(frame.copy(), "overlay")
    cv2.imshow("Atlas - Interscalene (Overlay)", result)

    result = atlas.render_atlas(frame.copy(), "full")
    cv2.imshow("Atlas - Interscalene (Full)", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
