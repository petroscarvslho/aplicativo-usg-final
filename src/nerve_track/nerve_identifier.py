"""
NERVE TRACK v2.0 - Identificador de Nervos Específicos
======================================================
Sistema de identificação de nervos baseado no tipo de bloqueio selecionado.

Funcionalidades:
1. Identificação contextual baseada no bloqueio selecionado
2. Matching de estruturas com anatomia esperada
3. Medição automática de CSA (Cross-Sectional Area)
4. Cálculo de distâncias entre estruturas
5. Validação anatômica

Baseado em:
- ScanNav Anatomy PNB
- Nerveblox Atlas
- Ultrasound-Guided Regional Anesthesia (UGRA)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import IntEnum
import cv2
import math

from .block_database import (
    ALL_NERVE_BLOCKS,
    get_block_config,
    get_structures_to_detect,
    NerveBlock,
    Structure,
    StructureType
)


# =============================================================================
# Estruturas de Dados
# =============================================================================

@dataclass
class IdentifiedStructure:
    """
    Estrutura identificada com nome anatômico.
    """
    # Identificação
    structure_id: int
    structure_type: StructureType
    name: str                        # Nome anatômico (ex: "Nervo Mediano")
    abbreviation: str                # Abreviação (ex: "MN")

    # Geometria
    contour: np.ndarray
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area_pixels: float
    area_mm2: Optional[float]        # CSA em mm²

    # Confiança
    confidence: float
    match_score: float               # Score de matching com anatomia esperada

    # Metadados
    is_target: bool                  # É o alvo principal do bloqueio?
    is_danger_zone: bool             # Zona de perigo?
    distance_to_target: Optional[float] = None  # Distância ao alvo em mm

    # Medições
    depth_mm: Optional[float] = None  # Profundidade em mm
    width_mm: Optional[float] = None  # Largura em mm
    height_mm: Optional[float] = None  # Altura em mm


@dataclass
class CSAMeasurement:
    """
    Medição de Cross-Sectional Area (CSA).
    """
    structure_name: str
    area_pixels: float
    area_mm2: float
    perimeter_mm: float
    major_axis_mm: float
    minor_axis_mm: float
    circularity: float
    is_normal: bool                  # Dentro dos valores de referência
    reference_range: Optional[Tuple[float, float]] = None  # Valores normais


@dataclass
class AnatomyContext:
    """
    Contexto anatômico para um bloqueio específico.
    """
    block_id: str
    block_name: str
    expected_structures: List[Structure]
    target_nerves: List[str]
    danger_zones: List[str]
    landmarks: List[str]
    depth_range_mm: Tuple[float, float]
    typical_probe_depth_mm: float


# =============================================================================
# CSA Calculator
# =============================================================================

class CSACalculator:
    """
    Calculador de Cross-Sectional Area (CSA) para nervos.

    A CSA é uma medida importante para:
    - Diagnóstico de neuropatias (aumento de CSA)
    - Síndrome do túnel do carpo
    - Neuropatia ulnar
    - Avaliação pré-bloqueio
    """

    # Valores de referência de CSA por nervo (em mm²)
    # Baseado em literatura médica
    REFERENCE_CSA = {
        # Membro Superior
        "median_nerve_wrist": (7.0, 12.0),       # Túnel do carpo
        "median_nerve_forearm": (6.0, 10.0),
        "median_nerve_arm": (8.0, 14.0),
        "ulnar_nerve_wrist": (4.0, 8.0),
        "ulnar_nerve_elbow": (6.0, 10.0),        # Túnel cubital
        "radial_nerve": (3.0, 7.0),
        "musculocutaneous_nerve": (4.0, 8.0),

        # Plexo Braquial
        "brachial_plexus_roots": (8.0, 15.0),
        "brachial_plexus_trunks": (15.0, 30.0),
        "brachial_plexus_cords": (10.0, 20.0),

        # Membro Inferior
        "femoral_nerve": (15.0, 40.0),
        "sciatic_nerve": (40.0, 80.0),
        "tibial_nerve": (15.0, 30.0),
        "common_peroneal_nerve": (8.0, 15.0),
        "saphenous_nerve": (2.0, 5.0),
        "lateral_femoral_cutaneous": (3.0, 8.0),
        "obturator_nerve": (4.0, 10.0),

        # Default para nervos não especificados
        "default": (5.0, 20.0)
    }

    def __init__(self, mm_per_pixel: float = 0.1):
        """
        Args:
            mm_per_pixel: Escala de conversão pixel -> mm
        """
        self.mm_per_pixel = mm_per_pixel

    def set_scale(self, mm_per_pixel: float):
        """Atualiza escala de conversão."""
        self.mm_per_pixel = mm_per_pixel

    def calculate_from_contour(
        self,
        contour: np.ndarray,
        nerve_type: str = "default"
    ) -> CSAMeasurement:
        """
        Calcula CSA a partir de contorno.

        Args:
            contour: Contorno OpenCV
            nerve_type: Tipo de nervo para referência

        Returns:
            CSAMeasurement com todas as medições
        """
        # Área em pixels
        area_pixels = cv2.contourArea(contour)

        # Área em mm²
        area_mm2 = area_pixels * (self.mm_per_pixel ** 2)

        # Perímetro em mm
        perimeter_pixels = cv2.arcLength(contour, True)
        perimeter_mm = perimeter_pixels * self.mm_per_pixel

        # Elipse ajustada (para eixos maior/menor)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (major, minor), angle = ellipse
            major_axis_mm = major * self.mm_per_pixel
            minor_axis_mm = minor * self.mm_per_pixel
        else:
            # Fallback para bbox
            x, y, w, h = cv2.boundingRect(contour)
            major_axis_mm = max(w, h) * self.mm_per_pixel
            minor_axis_mm = min(w, h) * self.mm_per_pixel

        # Circularidade
        if perimeter_pixels > 0:
            circularity = 4 * np.pi * area_pixels / (perimeter_pixels ** 2)
        else:
            circularity = 0

        # Verificar se está dentro do range de referência
        ref_range = self.REFERENCE_CSA.get(nerve_type, self.REFERENCE_CSA["default"])
        is_normal = ref_range[0] <= area_mm2 <= ref_range[1]

        return CSAMeasurement(
            structure_name=nerve_type,
            area_pixels=area_pixels,
            area_mm2=area_mm2,
            perimeter_mm=perimeter_mm,
            major_axis_mm=major_axis_mm,
            minor_axis_mm=minor_axis_mm,
            circularity=circularity,
            is_normal=is_normal,
            reference_range=ref_range
        )

    def calculate_from_mask(
        self,
        mask: np.ndarray,
        nerve_type: str = "default"
    ) -> CSAMeasurement:
        """
        Calcula CSA a partir de máscara binária.

        Args:
            mask: Máscara binária
            nerve_type: Tipo de nervo

        Returns:
            CSAMeasurement
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return CSAMeasurement(
                structure_name=nerve_type,
                area_pixels=0,
                area_mm2=0,
                perimeter_mm=0,
                major_axis_mm=0,
                minor_axis_mm=0,
                circularity=0,
                is_normal=False,
                reference_range=self.REFERENCE_CSA.get(nerve_type, self.REFERENCE_CSA["default"])
            )

        # Usar maior contorno
        largest_contour = max(contours, key=cv2.contourArea)
        return self.calculate_from_contour(largest_contour, nerve_type)


# =============================================================================
# Nerve Identifier
# =============================================================================

class NerveIdentifier:
    """
    Identificador de nervos baseado no contexto do bloqueio.

    Funcionalidades:
    - Identifica estruturas baseado no bloqueio selecionado
    - Faz matching com anatomia esperada
    - Calcula CSA
    - Identifica zonas de perigo
    """

    def __init__(
        self,
        mm_per_pixel: float = 0.1,
        min_confidence: float = 0.5
    ):
        """
        Args:
            mm_per_pixel: Escala de conversão
            min_confidence: Confiança mínima para identificação
        """
        self.csa_calculator = CSACalculator(mm_per_pixel)
        self.min_confidence = min_confidence
        self.current_block: Optional[NerveBlock] = None
        self.context: Optional[AnatomyContext] = None

    def set_scale(self, mm_per_pixel: float):
        """Atualiza escala de conversão."""
        self.csa_calculator.set_scale(mm_per_pixel)

    def set_block(self, block_id: str):
        """
        Define o tipo de bloqueio para identificação contextual.

        Args:
            block_id: ID do bloqueio (ex: "interscalene", "femoral")
        """
        block = get_block_config(block_id)
        if block is None:
            print(f"[AVISO] Bloqueio '{block_id}' não encontrado")
            return

        self.current_block = block

        # Criar contexto anatômico
        structures = get_structures_to_detect(block_id)

        # Identificar alvos e zonas de perigo
        targets = [s.name for s in block.targets]
        danger_zones = [dz.name for dz in block.danger_zones]
        landmarks = [lm.name for lm in block.landmarks]

        # Calcular range de profundidade típico
        depths = [s.typical_depth_mm for s in structures if s.typical_depth_mm]
        if depths:
            depth_range = (min(depths) * 0.5, max(depths) * 1.5)
            typical_depth = sum(depths) / len(depths)
        else:
            depth_range = (5, 50)
            typical_depth = 20

        self.context = AnatomyContext(
            block_id=block_id,
            block_name=block.name,
            expected_structures=structures,
            target_nerves=targets,
            danger_zones=danger_zones,
            landmarks=landmarks,
            depth_range_mm=depth_range,
            typical_probe_depth_mm=typical_depth
        )

    def identify_structures(
        self,
        detections: List[Dict[str, Any]],
        image_height: int
    ) -> List[IdentifiedStructure]:
        """
        Identifica estruturas detectadas baseado no contexto do bloqueio.

        Args:
            detections: Lista de detecções do classificador
                Cada detecção deve ter: contour, class_id, confidence, centroid
            image_height: Altura da imagem (para calcular profundidade)

        Returns:
            Lista de IdentifiedStructure
        """
        if self.context is None:
            # Sem contexto, apenas classificação básica
            return self._identify_basic(detections)

        identified = []

        # Ordenar estruturas esperadas por prioridade
        expected = sorted(
            self.context.expected_structures,
            key=lambda s: s.priority,
            reverse=True
        )

        # Criar cópias das detecções para não modificar original
        available_detections = list(detections)

        # Para cada estrutura esperada, tentar encontrar match
        for structure in expected:
            best_match = None
            best_score = 0

            for det in available_detections:
                score = self._calculate_match_score(det, structure, image_height)

                if score > best_score and score > self.min_confidence:
                    best_score = score
                    best_match = det

            if best_match is not None:
                # Criar IdentifiedStructure
                identified_struct = self._create_identified_structure(
                    best_match,
                    structure,
                    best_score,
                    image_height
                )
                identified.append(identified_struct)
                available_detections.remove(best_match)

        # Estruturas não identificadas
        for det in available_detections:
            # Criar identificação genérica
            identified_struct = self._create_generic_identified(det, image_height)
            identified.append(identified_struct)

        return identified

    def _calculate_match_score(
        self,
        detection: Dict[str, Any],
        expected: Structure,
        image_height: int
    ) -> float:
        """
        Calcula score de matching entre detecção e estrutura esperada.

        Args:
            detection: Detecção do classificador
            expected: Estrutura esperada do banco de dados
            image_height: Altura da imagem

        Returns:
            Score de 0 a 1
        """
        score = 0.0
        weights_total = 0.0

        # 1. Tipo de estrutura (peso alto)
        type_weight = 3.0
        det_type = detection.get('class_id', 0)
        if self._type_matches(det_type, expected.structure_type):
            score += type_weight
        weights_total += type_weight

        # 2. Profundidade esperada (se disponível)
        if expected.typical_depth_mm:
            depth_weight = 2.0
            centroid = detection.get('centroid', (0, 0))
            depth_pixels = centroid[1]  # Y = profundidade
            depth_mm = depth_pixels * self.csa_calculator.mm_per_pixel

            # Tolerância de 50%
            expected_depth = expected.typical_depth_mm
            tolerance = expected_depth * 0.5

            if abs(depth_mm - expected_depth) < tolerance:
                depth_score = 1.0 - abs(depth_mm - expected_depth) / tolerance
                score += depth_weight * depth_score

            weights_total += depth_weight

        # 3. Tamanho esperado (se disponível)
        if expected.typical_size_mm:
            size_weight = 1.5
            contour = detection.get('contour')

            if contour is not None:
                csa = self.csa_calculator.calculate_from_contour(contour)
                detected_size = max(csa.major_axis_mm, csa.minor_axis_mm)
                expected_size = expected.typical_size_mm

                # Tolerância de 100%
                tolerance = expected_size

                if abs(detected_size - expected_size) < tolerance:
                    size_score = 1.0 - abs(detected_size - expected_size) / tolerance
                    score += size_weight * size_score

            weights_total += size_weight

        # 4. Confiança da detecção
        conf_weight = 1.0
        confidence = detection.get('confidence', 0.5)
        score += conf_weight * confidence
        weights_total += conf_weight

        # Normalizar
        if weights_total > 0:
            return score / weights_total
        return 0.0

    def _type_matches(self, detected_type: int, expected_type: StructureType) -> bool:
        """Verifica se tipo detectado corresponde ao esperado."""
        # Mapeamento de class_id do classificador para StructureType
        type_map = {
            1: StructureType.NERVE,
            2: StructureType.ARTERY,
            3: StructureType.VEIN,
            4: StructureType.FASCIA,
            5: StructureType.MUSCLE,
            6: StructureType.BONE,
            7: StructureType.PLEURA
        }
        detected = type_map.get(detected_type, StructureType.OTHER)
        return detected == expected_type

    def _create_identified_structure(
        self,
        detection: Dict[str, Any],
        expected: Structure,
        match_score: float,
        image_height: int
    ) -> IdentifiedStructure:
        """Cria IdentifiedStructure a partir de match."""
        contour = detection.get('contour', np.array([]))
        centroid = detection.get('centroid', (0, 0))

        # Bounding box
        if len(contour) > 0:
            bbox = cv2.boundingRect(contour)
        else:
            bbox = (0, 0, 0, 0)

        # Área
        area_pixels = cv2.contourArea(contour) if len(contour) > 0 else 0
        area_mm2 = area_pixels * (self.csa_calculator.mm_per_pixel ** 2)

        # Profundidade
        depth_mm = centroid[1] * self.csa_calculator.mm_per_pixel

        # Verificar se é alvo ou zona de perigo
        is_target = expected.name in self.context.target_nerves
        is_danger = expected.name in self.context.danger_zones

        return IdentifiedStructure(
            structure_id=detection.get('id', 0),
            structure_type=expected.structure_type,
            name=expected.name,
            abbreviation=expected.abbreviation,
            contour=contour,
            centroid=centroid,
            bbox=bbox,
            area_pixels=area_pixels,
            area_mm2=area_mm2,
            confidence=detection.get('confidence', 0.5),
            match_score=match_score,
            is_target=is_target,
            is_danger_zone=is_danger,
            depth_mm=depth_mm
        )

    def _create_generic_identified(
        self,
        detection: Dict[str, Any],
        image_height: int
    ) -> IdentifiedStructure:
        """Cria IdentifiedStructure genérica para detecção não identificada."""
        contour = detection.get('contour', np.array([]))
        centroid = detection.get('centroid', (0, 0))
        class_id = detection.get('class_id', 0)

        # Mapear tipo
        type_names = {
            1: ("Nervo", "N", StructureType.NERVE),
            2: ("Artéria", "A", StructureType.ARTERY),
            3: ("Veia", "V", StructureType.VEIN),
            4: ("Fascia", "F", StructureType.FASCIA),
            5: ("Músculo", "M", StructureType.MUSCLE),
        }
        name, abbrev, stype = type_names.get(class_id, ("Estrutura", "?", StructureType.OTHER))

        # Bounding box
        if len(contour) > 0:
            bbox = cv2.boundingRect(contour)
        else:
            bbox = (0, 0, 0, 0)

        # Área
        area_pixels = cv2.contourArea(contour) if len(contour) > 0 else 0
        area_mm2 = area_pixels * (self.csa_calculator.mm_per_pixel ** 2)

        # Profundidade
        depth_mm = centroid[1] * self.csa_calculator.mm_per_pixel

        return IdentifiedStructure(
            structure_id=detection.get('id', 0),
            structure_type=stype,
            name=name,
            abbreviation=abbrev,
            contour=contour,
            centroid=centroid,
            bbox=bbox,
            area_pixels=area_pixels,
            area_mm2=area_mm2,
            confidence=detection.get('confidence', 0.5),
            match_score=0.0,
            is_target=False,
            is_danger_zone=False,
            depth_mm=depth_mm
        )

    def _identify_basic(self, detections: List[Dict[str, Any]]) -> List[IdentifiedStructure]:
        """Identificação básica sem contexto de bloqueio."""
        return [self._create_generic_identified(det, 500) for det in detections]

    def calculate_distances(
        self,
        structures: List[IdentifiedStructure]
    ) -> Dict[Tuple[int, int], float]:
        """
        Calcula distâncias entre estruturas.

        Args:
            structures: Lista de estruturas identificadas

        Returns:
            Dict de (id1, id2) -> distância em mm
        """
        distances = {}

        for i, s1 in enumerate(structures):
            for j, s2 in enumerate(structures):
                if i >= j:
                    continue

                # Distância euclidiana entre centroides
                dx = s1.centroid[0] - s2.centroid[0]
                dy = s1.centroid[1] - s2.centroid[1]
                dist_pixels = math.sqrt(dx**2 + dy**2)
                dist_mm = dist_pixels * self.csa_calculator.mm_per_pixel

                distances[(s1.structure_id, s2.structure_id)] = dist_mm

        return distances

    def measure_csa(
        self,
        structure: IdentifiedStructure
    ) -> CSAMeasurement:
        """
        Mede CSA de uma estrutura identificada.

        Args:
            structure: Estrutura identificada

        Returns:
            CSAMeasurement
        """
        # Determinar tipo para referência
        nerve_type_map = {
            "Nervo Mediano": "median_nerve_wrist",
            "Nervo Ulnar": "ulnar_nerve_wrist",
            "Nervo Radial": "radial_nerve",
            "Nervo Femoral": "femoral_nerve",
            "Nervo Ciático": "sciatic_nerve",
            "Plexo Braquial": "brachial_plexus_trunks",
        }

        nerve_type = "default"
        for key, value in nerve_type_map.items():
            if key in structure.name:
                nerve_type = value
                break

        return self.csa_calculator.calculate_from_contour(
            structure.contour,
            nerve_type
        )

    def get_expected_structures(self) -> List[Structure]:
        """Retorna estruturas esperadas para o bloqueio atual."""
        if self.context is None:
            return []
        return self.context.expected_structures

    def get_target_names(self) -> List[str]:
        """Retorna nomes dos alvos do bloqueio atual."""
        if self.context is None:
            return []
        return self.context.target_nerves

    def get_danger_zone_names(self) -> List[str]:
        """Retorna nomes das zonas de perigo."""
        if self.context is None:
            return []
        return self.context.danger_zones


# =============================================================================
# Factory Function
# =============================================================================

def create_nerve_identifier(
    mm_per_pixel: float = 0.1,
    block_id: Optional[str] = None
) -> NerveIdentifier:
    """
    Factory function para criar identificador de nervos.

    Args:
        mm_per_pixel: Escala de conversão
        block_id: ID do bloqueio inicial (opcional)

    Returns:
        Instância de NerveIdentifier
    """
    identifier = NerveIdentifier(mm_per_pixel=mm_per_pixel)

    if block_id:
        identifier.set_block(block_id)

    return identifier


# =============================================================================
# Teste
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NERVE TRACK v2.0 - Identificador de Nervos")
    print("=" * 60)

    # Criar identificador
    identifier = create_nerve_identifier(mm_per_pixel=0.1)

    # Testar CSA Calculator
    print("\n[1] Testando CSA Calculator...")
    csa_calc = CSACalculator(mm_per_pixel=0.1)

    # Criar contorno de teste (círculo)
    center = (100, 100)
    radius = 30
    angles = np.linspace(0, 2*np.pi, 100)
    contour = np.array([
        [[int(center[0] + radius * np.cos(a)),
          int(center[1] + radius * np.sin(a))]]
        for a in angles
    ], dtype=np.int32)

    csa = csa_calc.calculate_from_contour(contour, "median_nerve_wrist")
    print(f"  Área: {csa.area_mm2:.2f} mm² (referência: {csa.reference_range})")
    print(f"  Perímetro: {csa.perimeter_mm:.2f} mm")
    print(f"  Circularidade: {csa.circularity:.2f}")
    print(f"  Normal: {csa.is_normal}")

    # Testar identificador com bloqueio
    print("\n[2] Testando identificação contextual...")
    identifier.set_block("interscalene")

    if identifier.context:
        print(f"  Bloqueio: {identifier.context.block_name}")
        print(f"  Alvos: {identifier.context.target_nerves}")
        print(f"  Zonas de perigo: {identifier.context.danger_zones}")
        print(f"  Estruturas esperadas: {len(identifier.context.expected_structures)}")

        for struct in identifier.context.expected_structures[:5]:
            print(f"    - {struct.name} ({struct.abbreviation})")

    # Testar com detecções simuladas
    print("\n[3] Testando identificação de estruturas...")

    fake_detections = [
        {
            'id': 1,
            'class_id': 1,  # Nervo
            'confidence': 0.9,
            'centroid': (100, 150),
            'contour': contour
        },
        {
            'id': 2,
            'class_id': 2,  # Artéria
            'confidence': 0.85,
            'centroid': (200, 180),
            'contour': contour
        }
    ]

    identified = identifier.identify_structures(fake_detections, image_height=400)

    for struct in identified:
        print(f"  {struct.name}:")
        print(f"    - Tipo: {struct.structure_type.name}")
        print(f"    - Confiança: {struct.confidence:.0%}")
        print(f"    - Match score: {struct.match_score:.2f}")
        print(f"    - Alvo: {struct.is_target}")
        print(f"    - Zona de perigo: {struct.is_danger_zone}")
        print(f"    - CSA: {struct.area_mm2:.2f} mm²")
        print(f"    - Profundidade: {struct.depth_mm:.1f} mm")

    # Testar distâncias
    if len(identified) >= 2:
        print("\n[4] Testando cálculo de distâncias...")
        distances = identifier.calculate_distances(identified)
        for (id1, id2), dist in distances.items():
            print(f"  Estrutura {id1} <-> Estrutura {id2}: {dist:.1f} mm")

    print("\n" + "=" * 60)
    print("[OK] Identificador funcionando!")
    print("=" * 60)
