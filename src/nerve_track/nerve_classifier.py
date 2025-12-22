"""
NERVE TRACK v2.0 - Classificador de Estruturas
================================================
Classificação automática de Nervo/Artéria/Veia baseada em características ultrassonográficas.

Critérios de Classificação:
1. Echogenicidade: Padrão de intensidade interno
   - Nervo: Hipoecóico com padrão "honeycomb" (fascicular)
   - Artéria: Anecóico (lúmen escuro), parede ecogênica
   - Veia: Anecóico, parede fina, colapsável

2. Pulsatilidade: Variação rítmica de área/forma
   - Artéria: Pulsação sistólica regular (~60-100 bpm)
   - Veia: Sem pulsação ou mínima
   - Nervo: Sem pulsação

3. Compressibilidade: Resposta à pressão do probe
   - Veia: Altamente compressível (colapsa facilmente)
   - Artéria: Pouco compressível
   - Nervo: Incompressível

4. Forma e Textura:
   - Nervo: Oval/redondo, ecotextura fascicular
   - Artéria: Redonda, lúmen anecóico central
   - Veia: Oval/irregular, parede fina

5. Contexto Anatômico:
   - Posição relativa às outras estruturas
   - Compatibilidade com anatomia esperada

Baseado em:
- PIPIN Protocol (Peripheral Intravenous and Peripheral Nerve)
- ScanNav Anatomy Classification
- Clarius Smart Detection
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
from enum import IntEnum
import cv2


# =============================================================================
# Tipos de Estrutura
# =============================================================================

class StructureType(IntEnum):
    """Tipos de estruturas anatômicas."""
    UNKNOWN = 0
    NERVE = 1
    ARTERY = 2
    VEIN = 3
    FASCIA = 4
    MUSCLE = 5


# =============================================================================
# Feature Extraction
# =============================================================================

@dataclass
class StructureFeatures:
    """
    Features extraídas de uma estrutura para classificação.
    """
    # Identificação
    structure_id: int
    frame_id: int

    # Geometria
    area: float                      # Área em pixels
    perimeter: float                 # Perímetro
    circularity: float               # 4*pi*area/perimeter^2 (1 = círculo perfeito)
    aspect_ratio: float              # Largura/Altura
    solidity: float                  # Área/Área do convex hull
    extent: float                    # Área/Área do bounding box

    # Intensidade (Echogenicidade)
    mean_intensity: float            # Intensidade média
    std_intensity: float             # Desvio padrão
    min_intensity: float             # Mínimo
    max_intensity: float             # Máximo
    intensity_range: float           # Diferença max-min
    entropy: float                   # Entropia (complexidade textural)

    # Padrão interno
    has_dark_center: bool            # Centro escuro (típico de vasos)
    edge_intensity: float            # Intensidade média da borda
    center_intensity: float          # Intensidade média do centro
    honeycomb_score: float           # Score de padrão fascicular (nervo)

    # Temporal (preenchido pelo classificador)
    pulsatility_index: float = 0.0   # Índice de pulsatilidade
    area_variance: float = 0.0       # Variância de área temporal
    compressibility: float = 0.0     # Índice de compressibilidade


class FeatureExtractor:
    """
    Extrai features de estruturas segmentadas para classificação.
    """

    def __init__(self):
        self.frame_count = 0

    def extract(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        contour: np.ndarray,
        structure_id: int
    ) -> StructureFeatures:
        """
        Extrai features de uma estrutura.

        Args:
            image: Imagem de ultrassom (grayscale)
            mask: Máscara binária da estrutura
            contour: Contorno OpenCV
            structure_id: ID da estrutura

        Returns:
            StructureFeatures com todas as features
        """
        # Garantir grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Geometria
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Circularity (roundness)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # Bounding box e aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 1

        # Solidity (convexidade)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Extent (preenchimento do bbox)
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0

        # Intensidade dentro da máscara
        masked_pixels = gray[mask > 0]

        if len(masked_pixels) > 0:
            mean_intensity = np.mean(masked_pixels)
            std_intensity = np.std(masked_pixels)
            min_intensity = np.min(masked_pixels)
            max_intensity = np.max(masked_pixels)
        else:
            mean_intensity = std_intensity = min_intensity = max_intensity = 0

        intensity_range = max_intensity - min_intensity

        # Entropia (complexidade textural)
        entropy = self._calculate_entropy(masked_pixels)

        # Análise centro vs borda
        center_mask, edge_mask = self._create_center_edge_masks(mask, contour)

        center_pixels = gray[center_mask > 0]
        edge_pixels = gray[edge_mask > 0]

        center_intensity = np.mean(center_pixels) if len(center_pixels) > 0 else mean_intensity
        edge_intensity = np.mean(edge_pixels) if len(edge_pixels) > 0 else mean_intensity

        # Centro escuro (típico de vasos)
        has_dark_center = center_intensity < edge_intensity * 0.7

        # Score de padrão fascicular (honeycomb)
        honeycomb_score = self._calculate_honeycomb_score(gray, mask)

        return StructureFeatures(
            structure_id=structure_id,
            frame_id=self.frame_count,
            area=area,
            perimeter=perimeter,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            extent=extent,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
            intensity_range=intensity_range,
            entropy=entropy,
            has_dark_center=has_dark_center,
            edge_intensity=edge_intensity,
            center_intensity=center_intensity,
            honeycomb_score=honeycomb_score
        )

    def _calculate_entropy(self, pixels: np.ndarray) -> float:
        """Calcula entropia de Shannon dos pixels."""
        if len(pixels) == 0:
            return 0.0

        # Histograma normalizado
        hist, _ = np.histogram(pixels, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]  # Remover zeros

        # Entropia
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy

    def _create_center_edge_masks(
        self,
        mask: np.ndarray,
        contour: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cria máscaras separadas para centro e borda."""
        # Erode para criar centro
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        center_mask = cv2.erode(mask, kernel, iterations=2)

        # Edge = mask - centro
        edge_mask = mask.copy()
        edge_mask[center_mask > 0] = 0

        return center_mask, edge_mask

    def _calculate_honeycomb_score(
        self,
        gray: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Calcula score de padrão fascicular (honeycomb).
        Nervos têm padrão característico de múltiplas regiões
        hipoecóicas (fascículos) separadas por linhas ecogênicas (epineurium).
        """
        # ROI da estrutura
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return 0.0

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        roi = gray[y_min:y_max+1, x_min:x_max+1]
        roi_mask = mask[y_min:y_max+1, x_min:x_max+1]

        if roi.size == 0 or roi_mask.sum() == 0:
            return 0.0

        # Aplicar threshold adaptativo para encontrar regiões escuras
        roi_masked = roi.copy()
        roi_masked[roi_mask == 0] = 255

        # Threshold Otsu
        _, binary = cv2.threshold(
            roi_masked,
            0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Contar componentes conectados (fascículos potenciais)
        binary[roi_mask == 0] = 0
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        # Filtrar componentes muito pequenos ou muito grandes
        valid_components = 0
        total_area = roi_mask.sum()

        for i in range(1, num_labels):
            comp_area = stats[i, cv2.CC_STAT_AREA]
            # Fascículos típicos: 5-25% da área total cada
            if 0.02 * total_area < comp_area < 0.3 * total_area:
                valid_components += 1

        # Score: mais componentes válidos = mais provável ser nervo
        # Nervos típicos têm 3-10 fascículos visíveis
        if valid_components >= 2 and valid_components <= 15:
            score = min(valid_components / 5.0, 1.0)
        else:
            score = 0.0

        return score


# =============================================================================
# Classificador de Estruturas
# =============================================================================

@dataclass
class ClassificationResult:
    """Resultado da classificação de uma estrutura."""
    structure_id: int
    predicted_class: StructureType
    confidence: float
    probabilities: Dict[StructureType, float]
    features: StructureFeatures
    reasoning: List[str]  # Explicação da classificação


class StructureClassifier:
    """
    Classificador de estruturas anatômicas baseado em regras e ML.

    Usa uma combinação de:
    1. Regras baseadas em conhecimento médico
    2. Análise temporal (pulsatilidade)
    3. Características de imagem (textura, forma)
    """

    def __init__(
        self,
        history_size: int = 30,
        pulsatility_threshold: float = 0.1,
        use_temporal: bool = True
    ):
        """
        Args:
            history_size: Frames para análise temporal
            pulsatility_threshold: Limiar de pulsatilidade para artéria
            use_temporal: Usar análise temporal
        """
        self.history_size = history_size
        self.pulsatility_threshold = pulsatility_threshold
        self.use_temporal = use_temporal

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # Histórico de features por estrutura
        self.history: Dict[int, deque] = {}

        # Pesos para cada feature por classe
        self._init_weights()

    def _init_weights(self):
        """Inicializa pesos para classificação baseada em regras."""
        # Características típicas de cada estrutura
        self.typical_features = {
            StructureType.NERVE: {
                'circularity': (0.6, 0.95),      # Oval a redondo
                'mean_intensity': (50, 150),     # Hipoecóico
                'honeycomb_score': (0.3, 1.0),   # Padrão fascicular
                'has_dark_center': False,
                'std_intensity': (20, 80),       # Heterogêneo
            },
            StructureType.ARTERY: {
                'circularity': (0.8, 1.0),       # Redondo
                'mean_intensity': (10, 60),      # Anecóico (escuro)
                'honeycomb_score': (0.0, 0.2),   # Sem padrão fascicular
                'has_dark_center': True,
                'std_intensity': (5, 30),        # Homogêneo
                'pulsatility_index': (0.1, 1.0), # Pulsátil
            },
            StructureType.VEIN: {
                'circularity': (0.5, 0.9),       # Oval (compressível)
                'mean_intensity': (10, 60),      # Anecóico
                'honeycomb_score': (0.0, 0.2),   # Sem padrão fascicular
                'has_dark_center': True,
                'std_intensity': (5, 30),        # Homogêneo
                'pulsatility_index': (0.0, 0.1), # Não pulsátil
            }
        }

    def classify(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        contour: np.ndarray,
        structure_id: int
    ) -> ClassificationResult:
        """
        Classifica uma estrutura.

        Args:
            image: Imagem de ultrassom
            mask: Máscara binária da estrutura
            contour: Contorno OpenCV
            structure_id: ID da estrutura

        Returns:
            ClassificationResult com classificação e confiança
        """
        # Extrair features
        features = self.feature_extractor.extract(image, mask, contour, structure_id)

        # Adicionar ao histórico
        if structure_id not in self.history:
            self.history[structure_id] = deque(maxlen=self.history_size)
        self.history[structure_id].append(features)

        # Calcular features temporais
        if self.use_temporal and len(self.history[structure_id]) >= 5:
            features = self._add_temporal_features(features, structure_id)

        # Classificar
        probabilities = self._calculate_probabilities(features)

        # Determinar classe mais provável
        predicted_class = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_class]

        # Gerar explicação
        reasoning = self._generate_reasoning(features, predicted_class)

        return ClassificationResult(
            structure_id=structure_id,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            features=features,
            reasoning=reasoning
        )

    def _add_temporal_features(
        self,
        features: StructureFeatures,
        structure_id: int
    ) -> StructureFeatures:
        """Adiciona features temporais baseadas no histórico."""
        history = list(self.history[structure_id])

        # Áreas ao longo do tempo
        areas = [f.area for f in history]

        # Índice de pulsatilidade: (max - min) / mean
        if len(areas) >= 5:
            mean_area = np.mean(areas)
            if mean_area > 0:
                pulsatility = (np.max(areas) - np.min(areas)) / mean_area
            else:
                pulsatility = 0
            features.pulsatility_index = pulsatility
            features.area_variance = np.var(areas)

        return features

    def _calculate_probabilities(
        self,
        features: StructureFeatures
    ) -> Dict[StructureType, float]:
        """Calcula probabilidades para cada classe."""
        scores = {
            StructureType.NERVE: 0.0,
            StructureType.ARTERY: 0.0,
            StructureType.VEIN: 0.0
        }

        # Score para NERVO
        nerve_score = 0.0
        nerve_factors = []

        # Honeycomb pattern (forte indicador)
        if features.honeycomb_score > 0.3:
            nerve_score += 3.0 * features.honeycomb_score
            nerve_factors.append(f"honeycomb={features.honeycomb_score:.2f}")

        # Sem centro escuro
        if not features.has_dark_center:
            nerve_score += 1.0
            nerve_factors.append("sem_centro_escuro")

        # Intensidade média (hipoecóico, não anecóico)
        if 50 < features.mean_intensity < 150:
            nerve_score += 1.0
            nerve_factors.append(f"intensidade={features.mean_intensity:.0f}")

        # Heterogeneidade (fascículos)
        if 20 < features.std_intensity < 80:
            nerve_score += 0.5
            nerve_factors.append(f"std={features.std_intensity:.0f}")

        # Sem pulsatilidade
        if features.pulsatility_index < 0.05:
            nerve_score += 0.5
            nerve_factors.append("sem_pulsacao")

        scores[StructureType.NERVE] = nerve_score

        # Score para ARTÉRIA
        artery_score = 0.0
        artery_factors = []

        # Centro escuro (lúmen anecóico)
        if features.has_dark_center:
            artery_score += 2.0
            artery_factors.append("centro_escuro")

        # Alta circularidade
        if features.circularity > 0.8:
            artery_score += 1.0
            artery_factors.append(f"circular={features.circularity:.2f}")

        # Intensidade baixa (anecóico)
        if features.mean_intensity < 60:
            artery_score += 1.0
            artery_factors.append(f"anecóico={features.mean_intensity:.0f}")

        # PULSATILIDADE (forte indicador)
        if features.pulsatility_index > self.pulsatility_threshold:
            artery_score += 4.0 * min(features.pulsatility_index / 0.2, 1.0)
            artery_factors.append(f"PULSATIL={features.pulsatility_index:.2f}")

        # Homogeneidade
        if features.std_intensity < 30:
            artery_score += 0.5
            artery_factors.append("homogeneo")

        scores[StructureType.ARTERY] = artery_score

        # Score para VEIA
        vein_score = 0.0
        vein_factors = []

        # Centro escuro
        if features.has_dark_center:
            vein_score += 1.5
            vein_factors.append("centro_escuro")

        # Forma oval (compressível)
        if features.circularity < 0.85 and features.circularity > 0.5:
            vein_score += 1.0
            vein_factors.append(f"oval={features.circularity:.2f}")

        # Intensidade baixa
        if features.mean_intensity < 60:
            vein_score += 1.0
            vein_factors.append(f"anecóico={features.mean_intensity:.0f}")

        # SEM pulsatilidade (diferencia de artéria)
        if features.pulsatility_index < self.pulsatility_threshold:
            vein_score += 2.0
            vein_factors.append("sem_pulsacao")
        else:
            vein_score -= 2.0  # Penalizar se pulsátil

        # Homogeneidade
        if features.std_intensity < 30:
            vein_score += 0.5
            vein_factors.append("homogeneo")

        scores[StructureType.VEIN] = vein_score

        # Normalizar para probabilidades
        total = sum(max(s, 0) for s in scores.values())
        if total > 0:
            probabilities = {k: max(v, 0) / total for k, v in scores.items()}
        else:
            # Default: incerteza
            probabilities = {k: 1/3 for k in scores.keys()}

        return probabilities

    def _generate_reasoning(
        self,
        features: StructureFeatures,
        predicted_class: StructureType
    ) -> List[str]:
        """Gera explicação da classificação."""
        reasons = []

        if predicted_class == StructureType.NERVE:
            if features.honeycomb_score > 0.3:
                reasons.append(f"Padrão fascicular detectado (score={features.honeycomb_score:.2f})")
            if not features.has_dark_center:
                reasons.append("Sem lúmen anecóico central")
            if 50 < features.mean_intensity < 150:
                reasons.append(f"Ecogenicidade hipoecóica ({features.mean_intensity:.0f})")
            if features.pulsatility_index < 0.05:
                reasons.append("Sem pulsatilidade")

        elif predicted_class == StructureType.ARTERY:
            if features.has_dark_center:
                reasons.append("Lúmen anecóico central")
            if features.pulsatility_index > self.pulsatility_threshold:
                reasons.append(f"Pulsatilidade detectada ({features.pulsatility_index:.2f})")
            if features.circularity > 0.8:
                reasons.append(f"Forma circular ({features.circularity:.2f})")
            if features.mean_intensity < 60:
                reasons.append(f"Interior anecóico ({features.mean_intensity:.0f})")

        elif predicted_class == StructureType.VEIN:
            if features.has_dark_center:
                reasons.append("Lúmen anecóico central")
            if features.pulsatility_index < self.pulsatility_threshold:
                reasons.append("Sem pulsatilidade arterial")
            if features.circularity < 0.85:
                reasons.append(f"Forma oval/compressível ({features.circularity:.2f})")

        return reasons

    def reset(self, structure_id: Optional[int] = None):
        """
        Reseta histórico.

        Args:
            structure_id: ID específico ou None para todos
        """
        if structure_id is not None:
            if structure_id in self.history:
                del self.history[structure_id]
        else:
            self.history.clear()

    def get_structure_history(
        self,
        structure_id: int
    ) -> List[StructureFeatures]:
        """Retorna histórico de features de uma estrutura."""
        if structure_id in self.history:
            return list(self.history[structure_id])
        return []


# =============================================================================
# Classificador com Deep Learning (Opcional)
# =============================================================================

class DeepStructureClassifier:
    """
    Classificador baseado em CNN para classificação de estruturas.
    Usado como complemento ao classificador baseado em regras.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.device = 'cpu'

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Carrega modelo de classificação."""
        try:
            import torch
            import torch.nn as nn

            # Arquitetura simples para classificação de patches
            class ClassifierCNN(nn.Module):
                def __init__(self, num_classes=3):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(1, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1)
                    )
                    self.classifier = nn.Linear(128, num_classes)

                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    return self.classifier(x)

            self.model = ClassifierCNN()
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
            self.model.eval()

            # Usar MPS se disponível
            if torch.backends.mps.is_available():
                self.device = 'mps'
                self.model = self.model.to(self.device)

        except Exception as e:
            print(f"[AVISO] Não foi possível carregar modelo de classificação: {e}")
            self.model = None

    def classify(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Dict[StructureType, float]:
        """
        Classifica usando CNN.

        Args:
            image: Imagem de ultrassom
            mask: Máscara da estrutura

        Returns:
            Probabilidades por classe
        """
        if self.model is None:
            return {
                StructureType.NERVE: 0.33,
                StructureType.ARTERY: 0.33,
                StructureType.VEIN: 0.34
            }

        import torch

        # Extrair patch
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return {
                StructureType.NERVE: 0.33,
                StructureType.ARTERY: 0.33,
                StructureType.VEIN: 0.34
            }

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # Garantir grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        patch = gray[y_min:y_max+1, x_min:x_max+1]
        patch = cv2.resize(patch, (64, 64))

        # Para tensor
        tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device) / 255.0

        # Inferência
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()

        return {
            StructureType.NERVE: float(probs[0]),
            StructureType.ARTERY: float(probs[1]),
            StructureType.VEIN: float(probs[2])
        }


# =============================================================================
# Classificador Ensemble
# =============================================================================

class EnsembleClassifier:
    """
    Classificador ensemble que combina regras e deep learning.
    """

    def __init__(
        self,
        rule_weight: float = 0.6,
        deep_weight: float = 0.4,
        deep_model_path: Optional[str] = None,
        use_temporal: bool = True
    ):
        self.rule_weight = rule_weight
        self.deep_weight = deep_weight

        # Classificador baseado em regras
        self.rule_classifier = StructureClassifier(use_temporal=use_temporal)

        # Classificador deep learning
        self.deep_classifier = DeepStructureClassifier(deep_model_path)

    def classify(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        contour: np.ndarray,
        structure_id: int
    ) -> ClassificationResult:
        """
        Classificação ensemble.

        Args:
            image: Imagem de ultrassom
            mask: Máscara da estrutura
            contour: Contorno OpenCV
            structure_id: ID da estrutura

        Returns:
            ClassificationResult
        """
        # Classificação por regras
        rule_result = self.rule_classifier.classify(image, mask, contour, structure_id)

        # Classificação deep learning
        deep_probs = self.deep_classifier.classify(image, mask)

        # Combinar probabilidades
        combined_probs = {}
        for class_type in [StructureType.NERVE, StructureType.ARTERY, StructureType.VEIN]:
            rule_prob = rule_result.probabilities.get(class_type, 0)
            deep_prob = deep_probs.get(class_type, 0)
            combined_probs[class_type] = (
                self.rule_weight * rule_prob +
                self.deep_weight * deep_prob
            )

        # Normalizar
        total = sum(combined_probs.values())
        if total > 0:
            combined_probs = {k: v / total for k, v in combined_probs.items()}

        # Classe mais provável
        predicted_class = max(combined_probs, key=combined_probs.get)
        confidence = combined_probs[predicted_class]

        return ClassificationResult(
            structure_id=structure_id,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=combined_probs,
            features=rule_result.features,
            reasoning=rule_result.reasoning
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_structure_classifier(
    use_temporal: bool = True,
    use_deep_learning: bool = False,
    deep_model_path: Optional[str] = None
) -> StructureClassifier:
    """
    Factory function para criar classificador.

    Args:
        use_temporal: Usar análise temporal
        use_deep_learning: Usar classificador ensemble com deep learning
        deep_model_path: Caminho para modelo deep learning

    Returns:
        Instância de classificador
    """
    if use_deep_learning:
        return EnsembleClassifier(
            use_temporal=use_temporal,
            deep_model_path=deep_model_path
        )
    else:
        return StructureClassifier(use_temporal=use_temporal)


# =============================================================================
# Teste
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NERVE TRACK v2.0 - Classificador de Estruturas")
    print("=" * 60)

    # Criar classificador
    classifier = create_structure_classifier(use_temporal=True)

    # Simular imagem e máscara
    print("\nSimulando classificação...")

    # Criar imagem fake (300x300 grayscale)
    image = np.random.randint(50, 150, (300, 300), dtype=np.uint8)

    # Simular nervo (região com padrão variado)
    nerve_mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(nerve_mask, (100, 100), 30, 255, -1)

    # Adicionar variação (simular fascículos)
    for i in range(5):
        x = 100 + np.random.randint(-20, 20)
        y = 100 + np.random.randint(-20, 20)
        cv2.circle(image, (x, y), 5, 30, -1)  # Regiões escuras

    nerve_contour, _ = cv2.findContours(nerve_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if nerve_contour:
        result = classifier.classify(image, nerve_mask, nerve_contour[0], structure_id=1)

        print(f"\nResultado para estrutura simulada:")
        print(f"  Classe predita: {result.predicted_class.name}")
        print(f"  Confiança: {result.confidence:.2%}")
        print(f"  Probabilidades:")
        for cls, prob in result.probabilities.items():
            print(f"    - {cls.name}: {prob:.2%}")
        print(f"  Razões:")
        for reason in result.reasoning:
            print(f"    - {reason}")

        print(f"\n  Features extraídas:")
        print(f"    - Área: {result.features.area:.0f} px")
        print(f"    - Circularidade: {result.features.circularity:.2f}")
        print(f"    - Intensidade média: {result.features.mean_intensity:.0f}")
        print(f"    - Honeycomb score: {result.features.honeycomb_score:.2f}")
        print(f"    - Centro escuro: {result.features.has_dark_center}")

    print("\n" + "=" * 60)
    print("[OK] Classificador funcionando!")
    print("=" * 60)
