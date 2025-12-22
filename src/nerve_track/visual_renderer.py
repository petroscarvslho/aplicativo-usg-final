"""
NERVE TRACK v2.0 - Visual Renderer Premium
==========================================
Sistema de visualização premium para bloqueios nervosos.

Características:
- Visualização específica por tipo de bloqueio
- Cores diferenciadas por estrutura anatômica
- Halos de incerteza e zonas de perigo
- Trajetórias suavizadas
- Labels anatômicos detalhados
- Indicadores de CSA e profundidade
- Guias de inserção de agulha

Baseado em:
- Clarius Smart Visualization
- GE Healthcare cNerve UI
- ScanNav Visual Design
- Nerveblox Premium
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import IntEnum
import math

from .block_database import (
    get_block_config,
    NerveBlock,
    StructureType,
)
from .nerve_identifier import IdentifiedStructure
from .nerve_tracker import TrackedStructure


# =============================================================================
# Configuração Visual
# =============================================================================

@dataclass
class VisualConfig:
    """Configuração visual para renderização."""

    # Cores por tipo de estrutura (BGR)
    colors: Dict[StructureType, Tuple[int, int, int]] = field(default_factory=lambda: {
        StructureType.NERVE: (0, 255, 255),        # Amarelo
        StructureType.ARTERY: (0, 0, 255),         # Vermelho
        StructureType.VEIN: (255, 0, 0),           # Azul
        StructureType.FASCIA: (180, 180, 180),     # Cinza claro
        StructureType.MUSCLE: (128, 0, 128),       # Roxo
        StructureType.BONE: (255, 255, 255),       # Branco
        StructureType.PLEURA: (0, 165, 255),       # Laranja
        StructureType.OTHER: (128, 128, 128),      # Cinza
    })

    # Cores especiais
    target_color: Tuple[int, int, int] = (0, 255, 0)        # Verde (alvo)
    danger_color: Tuple[int, int, int] = (0, 0, 255)        # Vermelho (perigo)
    uncertainty_color: Tuple[int, int, int] = (255, 255, 0)  # Ciano (incerteza)
    trajectory_color: Tuple[int, int, int] = (255, 200, 0)   # Azul claro (trajetória)

    # Opções de visualização
    show_contours: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    show_csa: bool = True
    show_depth: bool = True
    show_trajectory: bool = True
    show_uncertainty_halo: bool = True
    show_danger_zones: bool = True
    show_target_highlight: bool = True
    show_block_info: bool = True
    show_scale_bar: bool = True

    # Parâmetros de renderização
    contour_thickness: int = 2
    label_font_scale: float = 0.5
    overlay_alpha: float = 0.3
    trajectory_length: int = 30
    halo_blur_size: int = 21

    # Animação
    pulse_enabled: bool = True
    pulse_speed: float = 2.0


# =============================================================================
# Premium Visual Renderer
# =============================================================================

class PremiumVisualRenderer:
    """
    Renderizador visual premium para NERVE TRACK.

    Cria visualizações de alta qualidade com:
    - Overlay colorido por estrutura
    - Halos de incerteza
    - Alertas de zonas de perigo
    - Informações contextuais do bloqueio
    """

    def __init__(
        self,
        config: Optional[VisualConfig] = None,
        mm_per_pixel: float = 0.1
    ):
        """
        Args:
            config: Configuração visual (usa padrão se None)
            mm_per_pixel: Escala de conversão
        """
        self.config = config or VisualConfig()
        self.mm_per_pixel = mm_per_pixel
        self.current_block: Optional[NerveBlock] = None
        self.frame_count = 0

    def set_scale(self, mm_per_pixel: float):
        """Atualiza escala de conversão."""
        self.mm_per_pixel = mm_per_pixel

    def set_block(self, block_id: str):
        """Define bloqueio atual para visualização contextual."""
        self.current_block = get_block_config(block_id)

    def render(
        self,
        frame: np.ndarray,
        identified: List[IdentifiedStructure],
        tracks: Optional[List[TrackedStructure]] = None
    ) -> np.ndarray:
        """
        Renderiza visualização premium.

        Args:
            frame: Frame de ultrassom (BGR)
            identified: Lista de estruturas identificadas
            tracks: Lista de tracks (opcional, para trajetórias)

        Returns:
            Frame com visualização
        """
        self.frame_count += 1
        vis = frame.copy()

        # 1. Renderizar overlay de segmentação
        if self.config.show_contours:
            vis = self._render_overlay(vis, identified)

        # 2. Renderizar contornos
        vis = self._render_contours(vis, identified)

        # 3. Renderizar halos de incerteza
        if self.config.show_uncertainty_halo:
            vis = self._render_uncertainty_halos(vis, identified)

        # 4. Renderizar zonas de perigo
        if self.config.show_danger_zones:
            vis = self._render_danger_zones(vis, identified)

        # 5. Renderizar destaque do alvo
        if self.config.show_target_highlight:
            vis = self._render_target_highlight(vis, identified)

        # 6. Renderizar trajetórias
        if self.config.show_trajectory and tracks:
            vis = self._render_trajectories(vis, tracks)

        # 7. Renderizar labels
        if self.config.show_labels:
            vis = self._render_labels(vis, identified)

        # 8. Renderizar informações do bloqueio
        if self.config.show_block_info and self.current_block:
            vis = self._render_block_info(vis)

        # 9. Renderizar barra de escala
        if self.config.show_scale_bar:
            vis = self._render_scale_bar(vis)

        return vis

    def _render_overlay(
        self,
        vis: np.ndarray,
        identified: List[IdentifiedStructure]
    ) -> np.ndarray:
        """Renderiza overlay colorido por estrutura."""
        overlay = np.zeros_like(vis)

        for struct in identified:
            if struct.contour is None or len(struct.contour) == 0:
                continue

            color = self._get_color(struct)
            cv2.fillPoly(overlay, [struct.contour], color)

        # Blend com alpha
        return cv2.addWeighted(vis, 1.0, overlay, self.config.overlay_alpha, 0)

    def _render_contours(
        self,
        vis: np.ndarray,
        identified: List[IdentifiedStructure]
    ) -> np.ndarray:
        """Renderiza contornos das estruturas."""
        for struct in identified:
            if struct.contour is None or len(struct.contour) == 0:
                continue

            color = self._get_color(struct)
            thickness = self.config.contour_thickness

            # Alvo tem contorno mais grosso
            if struct.is_target:
                thickness = self.config.contour_thickness + 2

            cv2.drawContours(vis, [struct.contour], -1, color, thickness)

        return vis

    def _render_uncertainty_halos(
        self,
        vis: np.ndarray,
        identified: List[IdentifiedStructure]
    ) -> np.ndarray:
        """Renderiza halos de incerteza baseado na confiança."""
        for struct in identified:
            if struct.confidence >= 0.9:  # Alta confiança, sem halo
                continue

            if struct.contour is None or len(struct.contour) == 0:
                continue

            # Criar máscara do halo
            halo_mask = np.zeros(vis.shape[:2], dtype=np.uint8)

            # Dilatar contorno baseado na incerteza
            uncertainty = 1.0 - struct.confidence
            dilation_size = int(uncertainty * 20) + 5

            cv2.drawContours(halo_mask, [struct.contour], -1, 255, dilation_size)

            # Blur para suavizar
            halo_mask = cv2.GaussianBlur(halo_mask, (self.config.halo_blur_size, self.config.halo_blur_size), 0)

            # Aplicar cor do halo
            halo_color = self.config.uncertainty_color
            halo_overlay = np.zeros_like(vis)
            halo_overlay[halo_mask > 0] = halo_color

            # Blend com intensidade baseada na incerteza
            alpha = uncertainty * 0.3
            vis = cv2.addWeighted(vis, 1.0, halo_overlay, alpha, 0)

        return vis

    def _render_danger_zones(
        self,
        vis: np.ndarray,
        identified: List[IdentifiedStructure]
    ) -> np.ndarray:
        """Renderiza alertas visuais para zonas de perigo."""
        for struct in identified:
            if not struct.is_danger_zone:
                continue

            if struct.contour is None or len(struct.contour) == 0:
                continue

            # Efeito pulsante para zonas de perigo
            if self.config.pulse_enabled:
                pulse = (math.sin(self.frame_count * 0.2) + 1) / 2
                thickness = int(2 + pulse * 3)
            else:
                thickness = 3

            # Contorno vermelho pulsante
            cv2.drawContours(
                vis, [struct.contour], -1,
                self.config.danger_color, thickness
            )

            # Ícone de alerta no centroide
            cx, cy = int(struct.centroid[0]), int(struct.centroid[1])
            self._draw_warning_icon(vis, (cx, cy - 30))

        return vis

    def _render_target_highlight(
        self,
        vis: np.ndarray,
        identified: List[IdentifiedStructure]
    ) -> np.ndarray:
        """Renderiza destaque especial para estruturas alvo."""
        for struct in identified:
            if not struct.is_target:
                continue

            if struct.contour is None or len(struct.contour) == 0:
                continue

            # Efeito de brilho pulsante
            if self.config.pulse_enabled:
                pulse = (math.sin(self.frame_count * 0.15) + 1) / 2
                glow_size = int(5 + pulse * 10)
            else:
                glow_size = 10

            # Criar glow effect
            glow_mask = np.zeros(vis.shape[:2], dtype=np.uint8)
            cv2.drawContours(glow_mask, [struct.contour], -1, 255, glow_size)
            glow_mask = cv2.GaussianBlur(glow_mask, (21, 21), 0)

            # Aplicar glow verde
            glow_overlay = np.zeros_like(vis)
            glow_overlay[glow_mask > 0] = self.config.target_color

            vis = cv2.addWeighted(vis, 1.0, glow_overlay, 0.3, 0)

            # Desenhar crosshair no centroide
            cx, cy = int(struct.centroid[0]), int(struct.centroid[1])
            self._draw_crosshair(vis, (cx, cy), self.config.target_color)

        return vis

    def _render_trajectories(
        self,
        vis: np.ndarray,
        tracks: List[TrackedStructure]
    ) -> np.ndarray:
        """Renderiza trajetórias dos tracks."""
        for track in tracks:
            if len(track.trajectory) < 2:
                continue

            points = list(track.trajectory)[-self.config.trajectory_length:]

            # Desenhar trajetória com fade
            for i in range(1, len(points)):
                alpha = i / len(points)
                color = tuple(int(c * alpha) for c in self.config.trajectory_color)

                pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))

                cv2.line(vis, pt1, pt2, color, 2)

        return vis

    def _render_labels(
        self,
        vis: np.ndarray,
        identified: List[IdentifiedStructure]
    ) -> np.ndarray:
        """Renderiza labels das estruturas."""
        for struct in identified:
            cx, cy = int(struct.centroid[0]), int(struct.centroid[1])
            color = self._get_color(struct)

            # Construir texto do label
            label_parts = [struct.abbreviation or struct.name[:10]]

            if self.config.show_confidence:
                label_parts.append(f"{struct.confidence:.0%}")

            if self.config.show_csa and struct.area_mm2:
                label_parts.append(f"{struct.area_mm2:.1f}mm²")

            if self.config.show_depth and struct.depth_mm:
                label_parts.append(f"{struct.depth_mm:.1f}mm")

            label = " | ".join(label_parts)

            # Posicionar label acima do centroide
            label_y = cy - 20

            # Background do label
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,
                self.config.label_font_scale, 1
            )

            label_x = cx - text_w // 2

            # Garantir que label está dentro da imagem
            label_x = max(5, min(label_x, vis.shape[1] - text_w - 5))
            label_y = max(text_h + 5, label_y)

            # Desenhar background
            cv2.rectangle(
                vis,
                (label_x - 3, label_y - text_h - 3),
                (label_x + text_w + 3, label_y + 3),
                (0, 0, 0),
                -1
            )

            # Borda colorida
            cv2.rectangle(
                vis,
                (label_x - 3, label_y - text_h - 3),
                (label_x + text_w + 3, label_y + 3),
                color,
                1
            )

            # Texto
            cv2.putText(
                vis, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.label_font_scale,
                color, 1, cv2.LINE_AA
            )

            # Nome completo se for alvo
            if struct.is_target and struct.name != struct.abbreviation:
                name_label = f"ALVO: {struct.name}"
                cv2.putText(
                    vis, name_label, (label_x, label_y + text_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.label_font_scale * 0.8,
                    self.config.target_color, 1, cv2.LINE_AA
                )

        return vis

    def _render_block_info(self, vis: np.ndarray) -> np.ndarray:
        """Renderiza informações do bloqueio no canto superior."""
        if not self.current_block:
            return vis

        # Background semi-transparente
        info_height = 80
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (vis.shape[1], info_height), (0, 0, 0), -1)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        # Nome do bloqueio
        cv2.putText(
            vis, f"BLOQUEIO: {self.current_block.name}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 1, cv2.LINE_AA
        )

        # Região
        cv2.putText(
            vis, f"Regiao: {self.current_block.region.name}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (180, 180, 180), 1, cv2.LINE_AA
        )

        # Alvos
        target_names = [t.abbreviation for t in self.current_block.targets[:3]]
        cv2.putText(
            vis, f"Alvos: {', '.join(target_names)}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            self.config.target_color, 1, cv2.LINE_AA
        )

        # Legenda de cores no canto direito
        legend_x = vis.shape[1] - 150
        legend_items = [
            ("NERVO", self.config.colors[StructureType.NERVE]),
            ("ARTERIA", self.config.colors[StructureType.ARTERY]),
            ("VEIA", self.config.colors[StructureType.VEIN]),
        ]

        for i, (name, color) in enumerate(legend_items):
            y = 20 + i * 20
            cv2.circle(vis, (legend_x, y), 5, color, -1)
            cv2.putText(
                vis, name, (legend_x + 12, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                color, 1, cv2.LINE_AA
            )

        return vis

    def _render_scale_bar(self, vis: np.ndarray) -> np.ndarray:
        """Renderiza barra de escala."""
        # Calcular tamanho da barra para 10mm
        scale_mm = 10
        scale_pixels = int(scale_mm / self.mm_per_pixel)

        # Limitar tamanho máximo
        scale_pixels = min(scale_pixels, vis.shape[1] // 4)

        # Posição no canto inferior direito
        x_end = vis.shape[1] - 20
        x_start = x_end - scale_pixels
        y = vis.shape[0] - 30

        # Desenhar barra
        cv2.line(vis, (x_start, y), (x_end, y), (255, 255, 255), 2)
        cv2.line(vis, (x_start, y - 5), (x_start, y + 5), (255, 255, 255), 2)
        cv2.line(vis, (x_end, y - 5), (x_end, y + 5), (255, 255, 255), 2)

        # Label
        label = f"{scale_mm}mm"
        (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            vis, label,
            (x_start + (scale_pixels - text_w) // 2, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA
        )

        return vis

    def _get_color(self, struct: IdentifiedStructure) -> Tuple[int, int, int]:
        """Retorna cor para uma estrutura."""
        if struct.is_target:
            return self.config.target_color
        if struct.is_danger_zone:
            return self.config.danger_color

        return self.config.colors.get(struct.structure_type, (128, 128, 128))

    def _draw_warning_icon(self, vis: np.ndarray, center: Tuple[int, int]):
        """Desenha ícone de aviso (triângulo com !)."""
        x, y = center
        size = 15

        # Triângulo
        pts = np.array([
            [x, y - size],
            [x - size, y + size // 2],
            [x + size, y + size // 2]
        ], dtype=np.int32)

        cv2.fillPoly(vis, [pts], self.config.danger_color)
        cv2.polylines(vis, [pts], True, (255, 255, 255), 1)

        # Exclamação
        cv2.putText(
            vis, "!", (x - 3, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 2, cv2.LINE_AA
        )

    def _draw_crosshair(
        self,
        vis: np.ndarray,
        center: Tuple[int, int],
        color: Tuple[int, int, int]
    ):
        """Desenha crosshair no ponto alvo."""
        x, y = center
        size = 15
        gap = 5

        # Linhas horizontais
        cv2.line(vis, (x - size, y), (x - gap, y), color, 2)
        cv2.line(vis, (x + gap, y), (x + size, y), color, 2)

        # Linhas verticais
        cv2.line(vis, (x, y - size), (x, y - gap), color, 2)
        cv2.line(vis, (x, y + gap), (x, y + size), color, 2)

        # Círculo central
        cv2.circle(vis, (x, y), gap, color, 1)


# =============================================================================
# Needle Guide Renderer
# =============================================================================

class NeedleGuideRenderer:
    """
    Renderizador de guia de inserção de agulha.

    Mostra:
    - Trajetória sugerida até o alvo
    - Ângulo de inserção
    - Profundidade estimada
    - Zonas a evitar
    """

    def __init__(self, mm_per_pixel: float = 0.1):
        self.mm_per_pixel = mm_per_pixel

    def render_guide(
        self,
        vis: np.ndarray,
        target: IdentifiedStructure,
        entry_point: Tuple[int, int],
        danger_zones: List[IdentifiedStructure]
    ) -> np.ndarray:
        """
        Renderiza guia de inserção.

        Args:
            vis: Frame de visualização
            target: Estrutura alvo
            entry_point: Ponto de entrada da agulha (x, y)
            danger_zones: Zonas de perigo a evitar

        Returns:
            Frame com guia
        """
        target_point = (int(target.centroid[0]), int(target.centroid[1]))

        # Verificar se trajetória cruza zonas de perigo
        trajectory_safe = self._check_trajectory_safe(
            entry_point, target_point, danger_zones
        )

        # Cor da trajetória
        if trajectory_safe:
            line_color = (0, 255, 0)  # Verde
        else:
            line_color = (0, 165, 255)  # Laranja (aviso)

        # Desenhar trajetória
        cv2.line(vis, entry_point, target_point, line_color, 2, cv2.LINE_AA)

        # Desenhar ponto de entrada
        cv2.circle(vis, entry_point, 8, line_color, 2)
        cv2.circle(vis, entry_point, 3, line_color, -1)

        # Calcular ângulo e distância
        dx = target_point[0] - entry_point[0]
        dy = target_point[1] - entry_point[1]
        angle = math.degrees(math.atan2(dy, dx))
        distance_pixels = math.sqrt(dx**2 + dy**2)
        distance_mm = distance_pixels * self.mm_per_pixel

        # Informações da trajetória
        info_x = entry_point[0] + 15
        info_y = entry_point[1] - 10

        cv2.putText(
            vis, f"Ang: {angle:.1f}",
            (info_x, info_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            line_color, 1, cv2.LINE_AA
        )

        cv2.putText(
            vis, f"Dist: {distance_mm:.1f}mm",
            (info_x, info_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            line_color, 1, cv2.LINE_AA
        )

        # Aviso se trajetória insegura
        if not trajectory_safe:
            cv2.putText(
                vis, "ATENCAO: Cruzando zona de perigo!",
                (vis.shape[1] // 2 - 150, vis.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2, cv2.LINE_AA
            )

        return vis

    def _check_trajectory_safe(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        danger_zones: List[IdentifiedStructure]
    ) -> bool:
        """Verifica se trajetória não cruza zonas de perigo."""
        for zone in danger_zones:
            if zone.contour is None:
                continue

            # Verificar interseção com contorno
            # Simplificação: verificar pontos ao longo da linha
            num_points = 20
            for i in range(num_points):
                t = i / num_points
                x = int(start[0] + t * (end[0] - start[0]))
                y = int(start[1] + t * (end[1] - start[1]))

                # Verificar se ponto está dentro do contorno
                result = cv2.pointPolygonTest(zone.contour, (x, y), False)
                if result >= 0:  # Dentro ou na borda
                    return False

        return True


# =============================================================================
# Factory Function
# =============================================================================

def create_visual_renderer(
    mm_per_pixel: float = 0.1,
    config: Optional[VisualConfig] = None
) -> PremiumVisualRenderer:
    """
    Factory function para criar renderer visual.

    Args:
        mm_per_pixel: Escala de conversão
        config: Configuração visual (opcional)

    Returns:
        Instância de PremiumVisualRenderer
    """
    return PremiumVisualRenderer(config=config, mm_per_pixel=mm_per_pixel)


# =============================================================================
# Teste
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NERVE TRACK v2.0 - Visual Renderer Premium")
    print("=" * 60)

    # Criar renderer
    renderer = create_visual_renderer(mm_per_pixel=0.1)
    renderer.set_block("femoral")

    print("\n[OK] Visual Renderer criado!")
    print(f"  - Bloqueio: {renderer.current_block.name if renderer.current_block else 'Nenhum'}")
    print(f"  - Configuração:")
    print(f"    - Show contours: {renderer.config.show_contours}")
    print(f"    - Show labels: {renderer.config.show_labels}")
    print(f"    - Show CSA: {renderer.config.show_csa}")
    print(f"    - Show depth: {renderer.config.show_depth}")
    print(f"    - Pulse enabled: {renderer.config.pulse_enabled}")

    # Criar imagem de teste
    print("\n[INFO] Criando visualização de teste...")
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (50, 50, 50)  # Fundo cinza escuro

    # Simular estruturas identificadas
    from .nerve_identifier import IdentifiedStructure
    from .block_database import StructureType

    # Criar contorno de teste
    contour = np.array([
        [[200, 200]], [[250, 180]], [[280, 200]],
        [[280, 250]], [[250, 270]], [[200, 250]]
    ], dtype=np.int32)

    test_struct = IdentifiedStructure(
        structure_id=1,
        structure_type=StructureType.NERVE,
        name="Nervo Femoral",
        abbreviation="FN",
        contour=contour,
        centroid=(240, 225),
        bbox=(200, 180, 80, 90),
        area_pixels=3600,
        area_mm2=36.0,
        confidence=0.92,
        match_score=0.85,
        is_target=True,
        is_danger_zone=False,
        depth_mm=25.5
    )

    # Renderizar
    vis = renderer.render(test_frame, [test_struct])

    print(f"\n[OK] Visualização renderizada!")
    print(f"  - Shape: {vis.shape}")

    print("\n" + "=" * 60)
    print("[OK] Visual Renderer funcionando!")
    print("=" * 60)
