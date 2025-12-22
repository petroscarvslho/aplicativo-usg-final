"""
NERVE TRACK v2.0 PREMIUM
========================
Sistema avançado de detecção e tracking de nervos para ultrassom.

Módulos:
- block_database: Banco de dados de 28 bloqueios nervosos
- nerve_model: Modelo de segmentação (U-Net++ + CBAM + ConvLSTM)
- nerve_tracker: Sistema de tracking temporal com Kalman
- nerve_classifier: Classificação Nervo/Artéria/Veia
- nerve_identifier: Identificação de nervos específicos + CSA
- visual_renderer: Renderização visual premium

Baseado em:
- Clarius Median Nerve AI
- GE Healthcare cNerve
- ScanNav Anatomy PNB
- DeepNerve (MICCAI)
- Nerveblox Atlas

Características:
- U-Net++ com EfficientNet-B4 encoder
- CBAM (Convolutional Block Attention Module)
- ConvLSTM para consistência temporal
- Kalman Filter para tracking suave
- Classificação baseada em echogenicidade e pulsatilidade
- Medição automática de CSA (Cross-Sectional Area)
- Suporte a 28 tipos de bloqueios nervosos
- Visual premium por tipo de bloqueio
"""

__version__ = "2.0.0"
__author__ = "USG App Team"

# =============================================================================
# Banco de Dados de Bloqueios
# =============================================================================
from .block_database import (
    # Classes principais
    Structure,
    NerveBlock,
    BlockRegion,
    StructureType,

    # Dados
    ALL_NERVE_BLOCKS,
    BLOCK_NAMES,

    # Funções utilitárias
    get_block_config,
    get_blocks_by_region,
    get_structures_to_detect,
    get_all_block_ids,
    search_blocks,
)

# =============================================================================
# Modelo de Segmentação
# =============================================================================
try:
    from .nerve_model import (
        # Classes principais
        NerveSegmentationModel,
        NerveSegmentationInference,
        NervePostProcessor,

        # Módulos de atenção
        CBAM,
        ChannelAttention,
        SpatialAttention,

        # Módulo temporal
        ConvLSTMCell,
        TemporalConsistencyModule,

        # Loss functions
        DiceLoss,
        FocalLoss,
        UnifiedFocalLoss,

        # Factory
        create_nerve_model,
    )
    HAS_MODEL = True
except ImportError as e:
    HAS_MODEL = False
    print(f"[AVISO] Modelo de segmentação não disponível: {e}")

# =============================================================================
# Sistema de Tracking
# =============================================================================
from .nerve_tracker import (
    # Classes principais
    NerveTracker,
    KalmanBoxTracker,
    Detection,
    TrackedStructure,

    # Visualização
    TrackingVisualizer,

    # Factory
    create_nerve_tracker,
)

# =============================================================================
# Classificador de Estruturas
# =============================================================================
from .nerve_classifier import (
    # Classes principais
    StructureClassifier,
    EnsembleClassifier,
    DeepStructureClassifier,
    FeatureExtractor,

    # Estruturas de dados
    StructureFeatures,
    ClassificationResult,

    # Tipos
    StructureType as ClassifierStructureType,

    # Factory
    create_structure_classifier,
)

# =============================================================================
# Identificador de Nervos
# =============================================================================
from .nerve_identifier import (
    # Classes principais
    NerveIdentifier,
    CSACalculator,

    # Estruturas de dados
    IdentifiedStructure,
    CSAMeasurement,
    AnatomyContext,

    # Factory
    create_nerve_identifier,
)

# =============================================================================
# Visual Premium (será importado quando criado)
# =============================================================================
try:
    from .visual_renderer import (
        PremiumVisualRenderer,
        create_visual_renderer,
    )
    HAS_VISUAL = True
except ImportError:
    HAS_VISUAL = False

# =============================================================================
# Atlas Educacional (funciona SEM modelo treinado)
# =============================================================================
try:
    from .educational_atlas import (
        EducationalAtlas,
        create_educational_atlas,
    )
    HAS_ATLAS = True
except ImportError:
    HAS_ATLAS = False


# =============================================================================
# Factory Principal
# =============================================================================

class NerveTrackSystem:
    """
    Sistema integrado de NERVE TRACK.
    Combina todos os módulos em uma interface unificada.
    """

    def __init__(
        self,
        block_id: str = None,
        mm_per_pixel: float = 0.1,
        use_model: bool = True,
        use_temporal: bool = True,
        device: str = 'auto'
    ):
        """
        Args:
            block_id: ID do bloqueio inicial (opcional)
            mm_per_pixel: Escala de conversão pixel -> mm
            use_model: Usar modelo de segmentação (requer PyTorch)
            use_temporal: Usar processamento temporal
            device: Dispositivo para inferência ('auto', 'cpu', 'cuda', 'mps')
        """
        self.mm_per_pixel = mm_per_pixel
        self.current_block_id = block_id

        # Inicializar componentes
        self.tracker = create_nerve_tracker(
            max_age=10,
            min_hits=3,
            process_noise=0.01,
            measurement_noise=0.1
        )

        self.classifier = create_structure_classifier(use_temporal=use_temporal)

        self.identifier = create_nerve_identifier(
            mm_per_pixel=mm_per_pixel,
            block_id=block_id
        )

        # Modelo de segmentação (opcional)
        self.model = None
        if use_model and HAS_MODEL:
            try:
                self.model = create_nerve_model(
                    device=device,
                    use_temporal=use_temporal
                )
            except Exception as e:
                print(f"[AVISO] Não foi possível carregar modelo: {e}")

        # Visual renderer
        self.renderer = None
        if HAS_VISUAL:
            try:
                self.renderer = create_visual_renderer(mm_per_pixel=mm_per_pixel)
            except Exception as e:
                print(f"[AVISO] Visual renderer não disponível: {e}")

    def set_block(self, block_id: str):
        """Define o tipo de bloqueio."""
        self.current_block_id = block_id
        self.identifier.set_block(block_id)

        if self.renderer:
            self.renderer.set_block(block_id)

    def set_scale(self, mm_per_pixel: float):
        """Atualiza escala de conversão."""
        self.mm_per_pixel = mm_per_pixel
        self.identifier.set_scale(mm_per_pixel)

        if self.renderer:
            self.renderer.set_scale(mm_per_pixel)

    def process_frame(self, frame, reset_temporal: bool = False):
        """
        Processa um frame de ultrassom.

        Args:
            frame: Frame de ultrassom (BGR)
            reset_temporal: Se True, reseta estados temporais

        Returns:
            Dict com resultados do processamento
        """
        import cv2

        results = {
            'structures': [],
            'tracks': [],
            'identified': [],
            'visualization': None
        }

        if reset_temporal:
            self.tracker.reset()
            if self.model:
                self.model.model.reset_temporal_state()

        # 1. Segmentação (se modelo disponível)
        if self.model:
            seg_result = self.model.predict(frame, reset_temporal=reset_temporal)
            mask = seg_result['mask']
        else:
            # Fallback: usar classificador baseado em contornos
            mask = None

        # 2. Extrair detecções
        detections = []

        if mask is not None:
            # Extrair contornos de cada classe
            for class_id in range(1, 6):
                class_mask = (mask == class_id).astype('uint8')
                contours, _ = cv2.findContours(
                    class_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    if cv2.contourArea(contour) < 100:
                        continue

                    det = Detection.from_contour(
                        contour=contour,
                        class_id=class_id,
                        confidence=0.8,
                        mask=class_mask
                    )
                    detections.append(det)

        # 3. Tracking
        tracks = self.tracker.update(detections)
        results['tracks'] = tracks

        # 4. Classificação e Identificação
        if tracks:
            classified = []
            for track in tracks:
                if track.contour is not None:
                    # Criar máscara do contorno
                    track_mask = cv2.drawContours(
                        np.zeros(frame.shape[:2], dtype='uint8'),
                        [track.contour], -1, 255, -1
                    )

                    # Classificar
                    class_result = self.classifier.classify(
                        frame, track_mask, track.contour, track.track_id
                    )

                    classified.append({
                        'id': track.track_id,
                        'class_id': class_result.predicted_class.value,
                        'confidence': class_result.confidence,
                        'contour': track.contour,
                        'centroid': track.centroid
                    })

            results['structures'] = classified

            # Identificar
            identified = self.identifier.identify_structures(
                classified,
                frame.shape[0]
            )
            results['identified'] = identified

        # 5. Visualização
        if self.renderer:
            results['visualization'] = self.renderer.render(
                frame,
                results['identified'],
                results['tracks']
            )

        return results

    def get_block_info(self):
        """Retorna informações do bloqueio atual."""
        if self.current_block_id:
            return get_block_config(self.current_block_id)
        return None

    def get_available_blocks(self):
        """Retorna lista de bloqueios disponíveis."""
        return get_all_block_ids()


def create_nerve_track_system(
    block_id: str = None,
    mm_per_pixel: float = 0.1,
    use_model: bool = True,
    use_temporal: bool = True,
    device: str = 'auto'
) -> NerveTrackSystem:
    """
    Factory function para criar sistema NERVE TRACK completo.

    Args:
        block_id: ID do bloqueio inicial
        mm_per_pixel: Escala de conversão
        use_model: Usar modelo de segmentação
        use_temporal: Usar processamento temporal
        device: Dispositivo para inferência

    Returns:
        Instância de NerveTrackSystem
    """
    return NerveTrackSystem(
        block_id=block_id,
        mm_per_pixel=mm_per_pixel,
        use_model=use_model,
        use_temporal=use_temporal,
        device=device
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Versão
    '__version__',
    '__author__',

    # Database
    'Structure',
    'NerveBlock',
    'BlockRegion',
    'StructureType',
    'ALL_NERVE_BLOCKS',
    'BLOCK_NAMES',
    'get_block_config',
    'get_blocks_by_region',
    'get_structures_to_detect',
    'get_all_block_ids',
    'search_blocks',

    # Tracker
    'NerveTracker',
    'KalmanBoxTracker',
    'Detection',
    'TrackedStructure',
    'TrackingVisualizer',
    'create_nerve_tracker',

    # Classifier
    'StructureClassifier',
    'EnsembleClassifier',
    'FeatureExtractor',
    'StructureFeatures',
    'ClassificationResult',
    'create_structure_classifier',

    # Identifier
    'NerveIdentifier',
    'CSACalculator',
    'IdentifiedStructure',
    'CSAMeasurement',
    'AnatomyContext',
    'create_nerve_identifier',

    # Sistema integrado
    'NerveTrackSystem',
    'create_nerve_track_system',

    # Flags
    'HAS_MODEL',
    'HAS_VISUAL',
]

# Adicionar exports do modelo se disponível
if HAS_MODEL:
    __all__.extend([
        'NerveSegmentationModel',
        'NerveSegmentationInference',
        'NervePostProcessor',
        'CBAM',
        'ConvLSTMCell',
        'TemporalConsistencyModule',
        'DiceLoss',
        'FocalLoss',
        'UnifiedFocalLoss',
        'create_nerve_model',
    ])

if HAS_VISUAL:
    __all__.extend([
        'PremiumVisualRenderer',
        'create_visual_renderer',
    ])

if HAS_ATLAS:
    __all__.extend([
        'EducationalAtlas',
        'create_educational_atlas',
        'HAS_ATLAS',
    ])
