"""
NERVE TRACK v2.0 - Banco de Dados de Bloqueios Nervosos

Contém configurações completas para 28 tipos de bloqueios nervosos
guiados por ultrassom utilizados em anestesia regional.

Cada bloqueio inclui:
- Estruturas alvo (nervos a detectar)
- Landmarks (estruturas de referência)
- Zonas de perigo (estruturas a evitar)
- Configurações de probe e profundidade
- Dermátomos cobertos
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


class BlockRegion(Enum):
    """Regiões anatômicas dos bloqueios"""
    UPPER_LIMB = "upper_limb"
    LOWER_LIMB = "lower_limb"
    TRUNK = "trunk"
    HEAD_NECK = "head_neck"
    PELVIS = "pelvis"


class StructureType(Enum):
    """Tipos de estruturas detectáveis"""
    NERVE = "nerve"
    ARTERY = "artery"
    VEIN = "vein"
    MUSCLE = "muscle"
    BONE = "bone"
    FASCIA = "fascia"
    PLEURA = "pleura"
    LIGAMENT = "ligament"
    OTHER = "other"


@dataclass
class Structure:
    """Estrutura anatômica detectável"""
    name: str
    display_name: str
    structure_type: StructureType
    appearance: str  # hypoechoic, hyperechoic, anechoic, etc.
    color: Tuple[int, int, int]  # BGR
    position: Optional[str] = None
    is_target: bool = False  # Se é alvo do bloqueio
    is_warning: bool = False  # Se é zona de perigo
    key_landmark: bool = False  # Se é landmark principal
    typical_csa_mm2: Optional[Tuple[float, float]] = None
    notes: Optional[str] = None

    # Campos adicionais para identificação
    abbreviation: Optional[str] = None  # Abreviação (ex: "MN" para Median Nerve)
    typical_depth_mm: Optional[float] = None  # Profundidade típica em mm
    typical_size_mm: Optional[float] = None  # Tamanho típico em mm
    priority: int = 5  # Prioridade de detecção (1-10, maior = mais importante)

    def __post_init__(self):
        """Gera abreviação se não fornecida"""
        if self.abbreviation is None:
            # Gerar abreviação a partir do nome
            words = self.name.replace("-", " ").split()
            if len(words) >= 2:
                self.abbreviation = "".join(w[0].upper() for w in words[:3])
            else:
                self.abbreviation = self.name[:3].upper()


@dataclass
class NerveBlock:
    """Configuração completa de um bloqueio nervoso"""
    id: str
    name: str
    name_en: str
    region: BlockRegion

    # Configurações de probe
    probe_type: str  # linear, curvilinear
    frequency_mhz: Tuple[int, int]
    depth_cm: Tuple[float, float]

    # Estruturas
    targets: List[Structure]
    landmarks: List[Structure]
    danger_zones: List[Structure] = field(default_factory=list)

    # Informações clínicas
    indications: List[str] = field(default_factory=list)
    dermatomes: List[str] = field(default_factory=list)
    motor_coverage: List[str] = field(default_factory=list)

    # Posicionamento
    patient_position: str = "supine"

    # Técnica
    technique_notes: Optional[str] = None
    skill_level: str = "intermediate"  # basic, intermediate, advanced

    # Volume típico
    volume_ml: Optional[Tuple[int, int]] = None


# ══════════════════════════════════════════════════════════════════════════════
# CORES PADRÃO
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'NERVE': (0, 255, 255),       # Amarelo
    'NERVE_ALT1': (0, 255, 200),  # Amarelo-verde
    'NERVE_ALT2': (0, 200, 255),  # Amarelo-laranja
    'NERVE_ALT3': (100, 255, 200),
    'ARTERY': (0, 0, 255),        # Vermelho
    'VEIN': (255, 100, 100),      # Azul
    'MUSCLE': (150, 100, 100),    # Marrom
    'BONE': (255, 255, 255),      # Branco
    'FASCIA': (200, 200, 200),    # Cinza claro
    'PLEURA': (100, 100, 255),    # Rosa
    'WARNING': (0, 0, 255),       # Vermelho (perigo)
}


# ══════════════════════════════════════════════════════════════════════════════
# MEMBRO SUPERIOR
# ══════════════════════════════════════════════════════════════════════════════

INTERSCALENE_BLOCK = NerveBlock(
    id='INTERSCALENE',
    name='Bloqueio Interescalênico',
    name_en='Interscalene Block',
    region=BlockRegion.UPPER_LIMB,

    probe_type='linear',
    frequency_mhz=(8, 14),
    depth_cm=(2, 4),

    targets=[
        Structure(
            name='BRACHIAL_PLEXUS_ROOTS',
            display_name='Raízes C5-C6-C7',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic_circles_stacked',
            color=COLORS['NERVE'],
            position='between_scalenes',
            is_target=True,
            notes='Aparecem como círculos hipoecóicos empilhados'
        ),
    ],

    landmarks=[
        Structure(
            name='ANTERIOR_SCALENE',
            display_name='M. Escaleno Anterior',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='medial',
            key_landmark=True,
        ),
        Structure(
            name='MIDDLE_SCALENE',
            display_name='M. Escaleno Médio',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='lateral',
            key_landmark=True,
        ),
        Structure(
            name='STERNOCLEIDOMASTOID',
            display_name='M. Esternocleidomastóideo',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='superficial',
        ),
    ],

    danger_zones=[
        Structure(
            name='CAROTID_ARTERY',
            display_name='A. Carótida',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['ARTERY'],
            position='medial',
            is_warning=True,
        ),
        Structure(
            name='INTERNAL_JUGULAR',
            display_name='V. Jugular Interna',
            structure_type=StructureType.VEIN,
            appearance='anechoic_compressible',
            color=COLORS['VEIN'],
            position='medial',
            is_warning=True,
        ),
        Structure(
            name='PHRENIC_NERVE',
            display_name='N. Frênico',
            structure_type=StructureType.NERVE,
            appearance='small_hypoechoic',
            color=COLORS['WARNING'],
            position='anterior_to_anterior_scalene',
            is_warning=True,
            notes='Paralisia diafragmática se bloqueado'
        ),
        Structure(
            name='VERTEBRAL_ARTERY',
            display_name='A. Vertebral',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['WARNING'],
            position='transverse_foramen',
            is_warning=True,
        ),
    ],

    indications=['Cirurgia de ombro', 'Clavícula lateral', 'Úmero proximal'],
    dermatomes=['C5', 'C6', 'C7'],
    motor_coverage=['deltoid', 'biceps', 'supraspinatus', 'infraspinatus'],
    patient_position='supine_head_rotated_contralateral',
    technique_notes='Nível do cricóide, entre escalenos',
    volume_ml=(15, 20),
)


SUPRACLAVICULAR_BLOCK = NerveBlock(
    id='SUPRACLAVICULAR',
    name='Bloqueio Supraclavicular',
    name_en='Supraclavicular Block',
    region=BlockRegion.UPPER_LIMB,

    probe_type='linear',
    frequency_mhz=(8, 14),
    depth_cm=(2, 5),

    targets=[
        Structure(
            name='BRACHIAL_PLEXUS_TRUNKS',
            display_name='Troncos do Plexo',
            structure_type=StructureType.NERVE,
            appearance='cluster_hypoechoic',
            color=COLORS['NERVE'],
            position='lateral_to_subclavian_artery',
            is_target=True,
            notes='Aparência de "cacho de uvas" lateral à artéria'
        ),
    ],

    landmarks=[
        Structure(
            name='SUBCLAVIAN_ARTERY',
            display_name='A. Subclávia',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['ARTERY'],
            position='medial_to_plexus',
            key_landmark=True,
        ),
        Structure(
            name='FIRST_RIB',
            display_name='Primeira Costela',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_shadow',
            color=COLORS['BONE'],
            position='deep',
            key_landmark=True,
        ),
    ],

    danger_zones=[
        Structure(
            name='PLEURA',
            display_name='Pleura',
            structure_type=StructureType.PLEURA,
            appearance='hyperechoic_sliding',
            color=COLORS['PLEURA'],
            position='deep_medial',
            is_warning=True,
            notes='Risco de pneumotórax'
        ),
    ],

    indications=['Cotovelo', 'Antebraço', 'Mão', 'Úmero distal'],
    dermatomes=['C5', 'C6', 'C7', 'C8', 'T1'],
    motor_coverage=['complete_upper_limb'],
    patient_position='supine_head_rotated',
    technique_notes='"Corner pocket" - lateral à artéria, sobre a costela',
    skill_level='intermediate',
    volume_ml=(20, 30),
)


INFRACLAVICULAR_BLOCK = NerveBlock(
    id='INFRACLAVICULAR',
    name='Bloqueio Infraclavicular',
    name_en='Infraclavicular Block',
    region=BlockRegion.UPPER_LIMB,

    probe_type='linear',
    frequency_mhz=(6, 12),
    depth_cm=(3, 6),

    targets=[
        Structure(
            name='LATERAL_CORD',
            display_name='Cordão Lateral',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic',
            color=COLORS['NERVE'],
            position='9_oclock_around_artery',
            is_target=True,
        ),
        Structure(
            name='POSTERIOR_CORD',
            display_name='Cordão Posterior',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic',
            color=COLORS['NERVE_ALT1'],
            position='6_oclock_around_artery',
            is_target=True,
        ),
        Structure(
            name='MEDIAL_CORD',
            display_name='Cordão Medial',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic',
            color=COLORS['NERVE_ALT2'],
            position='3_oclock_around_artery',
            is_target=True,
        ),
    ],

    landmarks=[
        Structure(
            name='AXILLARY_ARTERY',
            display_name='A. Axilar',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['ARTERY'],
            position='central_reference',
            key_landmark=True,
        ),
        Structure(
            name='AXILLARY_VEIN',
            display_name='V. Axilar',
            structure_type=StructureType.VEIN,
            appearance='anechoic_compressible',
            color=COLORS['VEIN'],
            position='medial_inferior',
        ),
        Structure(
            name='PECTORALIS_MAJOR',
            display_name='M. Peitoral Maior',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='superficial',
        ),
        Structure(
            name='PECTORALIS_MINOR',
            display_name='M. Peitoral Menor',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=(130, 100, 100),
            position='deep_to_major',
        ),
    ],

    indications=['Cotovelo', 'Antebraço', 'Mão', 'Ideal para cateter'],
    dermatomes=['C5', 'C6', 'C7', 'C8', 'T1'],
    patient_position='supine_arm_abducted',
    technique_notes='Injeção em U ao redor da artéria',
    volume_ml=(25, 35),
)


AXILLARY_BLOCK = NerveBlock(
    id='AXILLARY',
    name='Bloqueio Axilar',
    name_en='Axillary Block',
    region=BlockRegion.UPPER_LIMB,

    probe_type='linear',
    frequency_mhz=(10, 15),
    depth_cm=(1, 3),

    targets=[
        Structure(
            name='MEDIAN_NERVE',
            display_name='N. Mediano',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic_honeycomb',
            color=COLORS['NERVE'],
            position='10_to_11_oclock',
            is_target=True,
        ),
        Structure(
            name='ULNAR_NERVE',
            display_name='N. Ulnar',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic_honeycomb',
            color=COLORS['NERVE_ALT1'],
            position='1_to_2_oclock',
            is_target=True,
        ),
        Structure(
            name='RADIAL_NERVE',
            display_name='N. Radial',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic_honeycomb',
            color=COLORS['NERVE_ALT2'],
            position='4_to_6_oclock',
            is_target=True,
        ),
        Structure(
            name='MUSCULOCUTANEOUS_NERVE',
            display_name='N. Musculocutâneo',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic_within_muscle',
            color=COLORS['NERVE_ALT3'],
            position='within_coracobrachialis',
            is_target=True,
            notes='Dentro do músculo coracobraquial'
        ),
    ],

    landmarks=[
        Structure(
            name='BRACHIAL_ARTERY',
            display_name='A. Braquial',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['ARTERY'],
            position='central',
            key_landmark=True,
        ),
        Structure(
            name='CORACOBRACHIALIS',
            display_name='M. Coracobraquial',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='lateral',
            notes='Contém o N. Musculocutâneo'
        ),
        Structure(
            name='BICEPS',
            display_name='M. Bíceps',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='superficial',
        ),
    ],

    indications=['Antebraço', 'Mão', 'Mais superficial e seguro'],
    dermatomes=['C5', 'C6', 'C7', 'C8', 'T1'],
    patient_position='supine_arm_abducted_90deg',
    technique_notes='Mnemônico MARU (sentido horário): Mediano, Artéria, Radial, Ulnar',
    skill_level='basic',
    volume_ml=(20, 30),
)


WRIST_BLOCK = NerveBlock(
    id='WRIST',
    name='Bloqueio do Pulso',
    name_en='Wrist Block',
    region=BlockRegion.UPPER_LIMB,

    probe_type='linear',
    frequency_mhz=(12, 18),
    depth_cm=(0.5, 2),

    targets=[
        Structure(
            name='MEDIAN_NERVE_WRIST',
            display_name='N. Mediano',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic_honeycomb',
            color=COLORS['NERVE'],
            position='between_fds_fdp',
            is_target=True,
            typical_csa_mm2=(6, 10),
            notes='CSA > 10mm² sugere Síndrome do Túnel do Carpo'
        ),
        Structure(
            name='ULNAR_NERVE_WRIST',
            display_name='N. Ulnar',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic_honeycomb',
            color=COLORS['NERVE_ALT1'],
            position='medial_to_ulnar_artery',
            is_target=True,
        ),
        Structure(
            name='RADIAL_NERVE_SUPERFICIAL',
            display_name='N. Radial Superficial',
            structure_type=StructureType.NERVE,
            appearance='small_hypoechoic',
            color=COLORS['NERVE_ALT2'],
            position='lateral_subcutaneous',
            is_target=True,
        ),
    ],

    landmarks=[
        Structure(
            name='RADIAL_ARTERY',
            display_name='A. Radial',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['ARTERY'],
            position='lateral',
        ),
        Structure(
            name='ULNAR_ARTERY',
            display_name='A. Ulnar',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['ARTERY'],
            position='medial',
        ),
        Structure(
            name='FLEXOR_TENDONS',
            display_name='Tendões Flexores',
            structure_type=StructureType.MUSCLE,
            appearance='hyperechoic_fibrillar',
            color=(200, 200, 200),
            position='deep',
        ),
    ],

    indications=['Mão', 'Dedos', 'Suplementação de bloqueio proximal'],
    dermatomes=['C6', 'C7', 'C8'],
    patient_position='supine_arm_supinated',
    skill_level='basic',
    volume_ml=(3, 5),
)


# ══════════════════════════════════════════════════════════════════════════════
# MEMBRO INFERIOR
# ══════════════════════════════════════════════════════════════════════════════

FEMORAL_NERVE_BLOCK = NerveBlock(
    id='FEMORAL',
    name='Bloqueio do Nervo Femoral',
    name_en='Femoral Nerve Block',
    region=BlockRegion.LOWER_LIMB,

    probe_type='linear',
    frequency_mhz=(8, 14),
    depth_cm=(2, 5),

    targets=[
        Structure(
            name='FEMORAL_NERVE',
            display_name='N. Femoral',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic_triangular',
            color=COLORS['NERVE'],
            position='lateral_to_artery',
            is_target=True,
            notes='Forma triangular, lateral à artéria, sob fascia ilíaca'
        ),
    ],

    landmarks=[
        Structure(
            name='FEMORAL_ARTERY',
            display_name='A. Femoral',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['ARTERY'],
            position='medial_to_nerve',
            key_landmark=True,
        ),
        Structure(
            name='FEMORAL_VEIN',
            display_name='V. Femoral',
            structure_type=StructureType.VEIN,
            appearance='anechoic_compressible',
            color=COLORS['VEIN'],
            position='medial_to_artery',
        ),
        Structure(
            name='FASCIA_ILIACA',
            display_name='Fascia Ilíaca',
            structure_type=StructureType.FASCIA,
            appearance='hyperechoic_line',
            color=COLORS['FASCIA'],
            position='superficial_to_nerve',
            key_landmark=True,
        ),
        Structure(
            name='ILIOPSOAS',
            display_name='M. Iliopsoas',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='deep',
        ),
    ],

    indications=['Quadril', 'Fêmur', 'Joelho', 'Analgesia pós-operatória'],
    dermatomes=['L2', 'L3', 'L4'],
    motor_coverage=['quadriceps'],
    patient_position='supine_leg_slightly_abducted',
    technique_notes='Mnemônico NAVEL (lateral → medial): Nervo-Artéria-Veia-Espaço-Linfáticos',
    volume_ml=(15, 25),
)


FASCIA_ILIACA_BLOCK = NerveBlock(
    id='FASCIA_ILIACA',
    name='Bloqueio da Fascia Ilíaca',
    name_en='Fascia Iliaca Compartment Block',
    region=BlockRegion.LOWER_LIMB,

    probe_type='linear',
    frequency_mhz=(8, 14),
    depth_cm=(2, 6),

    targets=[
        Structure(
            name='FASCIA_ILIACA_PLANE',
            display_name='Plano da Fascia Ilíaca',
            structure_type=StructureType.FASCIA,
            appearance='fascial_plane',
            color=COLORS['FASCIA'],
            position='between_fascias',
            is_target=True,
            notes='Injeção entre fascia lata e fascia ilíaca'
        ),
    ],

    landmarks=[
        Structure(
            name='FASCIA_LATA',
            display_name='Fascia Lata',
            structure_type=StructureType.FASCIA,
            appearance='hyperechoic_superficial',
            color=COLORS['FASCIA'],
            position='superficial',
        ),
        Structure(
            name='FASCIA_ILIACA',
            display_name='Fascia Ilíaca',
            structure_type=StructureType.FASCIA,
            appearance='hyperechoic_deep',
            color=COLORS['FASCIA'],
            position='deep',
            key_landmark=True,
        ),
        Structure(
            name='ILIACUS_MUSCLE',
            display_name='M. Ilíaco',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='deep_to_fascia_iliaca',
        ),
        Structure(
            name='SARTORIUS',
            display_name='M. Sartório',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='lateral_reference',
            key_landmark=True,
        ),
    ],

    indications=['Fratura de quadril', 'Cirurgias de quadril e fêmur'],
    dermatomes=['L1', 'L2', 'L3', 'L4'],
    patient_position='supine',
    technique_notes='Variantes: infrainguinal ou suprainguinal (melhor cobertura)',
    volume_ml=(30, 40),
)


PENG_BLOCK = NerveBlock(
    id='PENG',
    name='Bloqueio PENG',
    name_en='Pericapsular Nerve Group Block',
    region=BlockRegion.LOWER_LIMB,

    probe_type='curvilinear',
    frequency_mhz=(4, 8),
    depth_cm=(4, 8),

    targets=[
        Structure(
            name='PENG_PLANE',
            display_name='Plano PENG',
            structure_type=StructureType.FASCIA,
            appearance='fascial_plane',
            color=COLORS['NERVE'],
            position='between_psoas_tendon_and_bone',
            is_target=True,
            notes='Entre tendão do iliopsoas e osso púbico'
        ),
    ],

    landmarks=[
        Structure(
            name='AIIS',
            display_name='EIAI',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_bony',
            color=COLORS['BONE'],
            position='lateral',
            key_landmark=True,
            notes='Espinha Ilíaca Ântero-Inferior'
        ),
        Structure(
            name='ILIOPUBIC_EMINENCE',
            display_name='Eminência Iliopúbica',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_bony',
            color=COLORS['BONE'],
            position='medial',
            key_landmark=True,
        ),
        Structure(
            name='ILIOPSOAS',
            display_name='M. Iliopsoas',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='superficial',
        ),
        Structure(
            name='FEMORAL_HEAD',
            display_name='Cabeça Femoral',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_curved',
            color=COLORS['BONE'],
            position='deep',
        ),
    ],

    danger_zones=[
        Structure(
            name='LFCN',
            display_name='N. Cutâneo Femoral Lateral',
            structure_type=StructureType.NERVE,
            appearance='small_hypoechoic',
            color=COLORS['WARNING'],
            position='near_injection_site',
            is_warning=True,
            notes='Verificar parestesia antes de injetar'
        ),
    ],

    indications=['Fratura de quadril', 'Artroplastia de quadril'],
    dermatomes=['L2', 'L3'],
    patient_position='supine',
    technique_notes='Preserva força do quadríceps - alternativa ao femoral',
    skill_level='intermediate',
    volume_ml=(20, 25),
)


SCIATIC_SUBGLUTEAL_BLOCK = NerveBlock(
    id='SCIATIC_SUBGLUTEAL',
    name='Bloqueio Ciático Subglúteo',
    name_en='Subgluteal Sciatic Block',
    region=BlockRegion.LOWER_LIMB,

    probe_type='curvilinear',
    frequency_mhz=(4, 8),
    depth_cm=(4, 10),

    targets=[
        Structure(
            name='SCIATIC_NERVE',
            display_name='N. Ciático',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic_large_oval',
            color=COLORS['NERVE'],
            position='between_gt_and_it',
            is_target=True,
            typical_csa_mm2=(40, 80),
            notes='Maior nervo do corpo - entre trocanter e tuberosidade'
        ),
    ],

    landmarks=[
        Structure(
            name='GREATER_TROCHANTER',
            display_name='Trocanter Maior',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_bony',
            color=COLORS['BONE'],
            position='lateral',
            key_landmark=True,
        ),
        Structure(
            name='ISCHIAL_TUBEROSITY',
            display_name='Tuberosidade Isquiática',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_bony',
            color=COLORS['BONE'],
            position='medial',
            key_landmark=True,
        ),
        Structure(
            name='GLUTEUS_MAXIMUS',
            display_name='M. Glúteo Máximo',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='superficial',
        ),
        Structure(
            name='QUADRATUS_FEMORIS',
            display_name='M. Quadrado Femoral',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='deep_to_sciatic',
        ),
    ],

    indications=['Perna', 'Tornozelo', 'Pé', 'Amputações'],
    dermatomes=['L4', 'L5', 'S1', 'S2', 'S3'],
    patient_position='lateral_or_prone',
    volume_ml=(20, 30),
)


SCIATIC_POPLITEAL_BLOCK = NerveBlock(
    id='SCIATIC_POPLITEAL',
    name='Bloqueio Ciático Poplíteo',
    name_en='Popliteal Sciatic Block',
    region=BlockRegion.LOWER_LIMB,

    probe_type='linear',
    frequency_mhz=(8, 14),
    depth_cm=(2, 5),

    targets=[
        Structure(
            name='SCIATIC_NERVE',
            display_name='N. Ciático',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic_honeycomb',
            color=COLORS['NERVE'],
            position='superficial_to_vessels',
            is_target=True,
            notes='Proximal à bifurcação (5-7cm acima do joelho)'
        ),
        Structure(
            name='TIBIAL_NERVE',
            display_name='N. Tibial',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic',
            color=COLORS['NERVE_ALT1'],
            position='medial_after_bifurcation',
            is_target=True,
            notes='Maior, mais medial após bifurcação'
        ),
        Structure(
            name='COMMON_PERONEAL_NERVE',
            display_name='N. Peroneal Comum',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic_smaller',
            color=COLORS['NERVE_ALT2'],
            position='lateral_superficial',
            is_target=True,
            notes='Menor, mais lateral e superficial'
        ),
    ],

    landmarks=[
        Structure(
            name='POPLITEAL_ARTERY',
            display_name='A. Poplítea',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['ARTERY'],
            position='deep_to_nerve',
            key_landmark=True,
        ),
        Structure(
            name='POPLITEAL_VEIN',
            display_name='V. Poplítea',
            structure_type=StructureType.VEIN,
            appearance='anechoic_compressible',
            color=COLORS['VEIN'],
            position='between_nerve_and_artery',
        ),
        Structure(
            name='BICEPS_FEMORIS',
            display_name='M. Bíceps Femoral',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='lateral',
        ),
        Structure(
            name='SEMIMEMBRANOSUS',
            display_name='M. Semimembranoso',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='medial',
        ),
    ],

    indications=['Tornozelo', 'Pé'],
    dermatomes=['L4', 'L5', 'S1', 'S2'],
    patient_position='prone_or_lateral',
    technique_notes='Bloquear proximal à bifurcação (5-7cm acima do joelho)',
    volume_ml=(20, 30),
)


ADDUCTOR_CANAL_BLOCK = NerveBlock(
    id='ADDUCTOR_CANAL',
    name='Bloqueio do Canal Adutor',
    name_en='Adductor Canal Block',
    region=BlockRegion.LOWER_LIMB,

    probe_type='linear',
    frequency_mhz=(10, 15),
    depth_cm=(2, 4),

    targets=[
        Structure(
            name='SAPHENOUS_NERVE',
            display_name='N. Safeno',
            structure_type=StructureType.NERVE,
            appearance='small_hypoechoic',
            color=COLORS['NERVE'],
            position='anterolateral_to_artery',
            is_target=True,
        ),
    ],

    landmarks=[
        Structure(
            name='FEMORAL_ARTERY_CANAL',
            display_name='A. Femoral (no canal)',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['ARTERY'],
            position='within_canal',
            key_landmark=True,
        ),
        Structure(
            name='SARTORIUS',
            display_name='M. Sartório',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_triangular',
            color=COLORS['MUSCLE'],
            position='roof_of_canal',
            key_landmark=True,
            notes='Forma o "teto" do canal'
        ),
        Structure(
            name='VASTUS_MEDIALIS',
            display_name='M. Vasto Medial',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='lateral_wall',
        ),
        Structure(
            name='ADDUCTOR_LONGUS',
            display_name='M. Adutor Longo',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='medial_wall',
        ),
    ],

    indications=['Joelho', 'Artroplastia total de joelho'],
    dermatomes=['L3', 'L4'],
    patient_position='supine_leg_externally_rotated',
    technique_notes='Preserva força do quadríceps vs. bloqueio femoral',
    volume_ml=(15, 20),
)


OBTURATOR_NERVE_BLOCK = NerveBlock(
    id='OBTURATOR',
    name='Bloqueio do Nervo Obturador',
    name_en='Obturator Nerve Block',
    region=BlockRegion.LOWER_LIMB,

    probe_type='linear',
    frequency_mhz=(8, 14),
    depth_cm=(2, 5),

    targets=[
        Structure(
            name='ANTERIOR_DIVISION',
            display_name='Divisão Anterior',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic',
            color=COLORS['NERVE'],
            position='between_AL_and_AB',
            is_target=True,
            notes='Entre adutor longo e adutor curto'
        ),
        Structure(
            name='POSTERIOR_DIVISION',
            display_name='Divisão Posterior',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic',
            color=COLORS['NERVE_ALT1'],
            position='between_AB_and_AM',
            is_target=True,
            notes='Entre adutor curto e adutor magno'
        ),
    ],

    landmarks=[
        Structure(
            name='ADDUCTOR_LONGUS',
            display_name='M. Adutor Longo',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='superficial',
            key_landmark=True,
        ),
        Structure(
            name='ADDUCTOR_BREVIS',
            display_name='M. Adutor Curto',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='middle',
            key_landmark=True,
        ),
        Structure(
            name='ADDUCTOR_MAGNUS',
            display_name='M. Adutor Magno',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='deep',
            key_landmark=True,
        ),
        Structure(
            name='PECTINEUS',
            display_name='M. Pectíneo',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='lateral',
        ),
    ],

    indications=['Prevenção de reflexo adutor em RTU', 'Suplementação joelho'],
    dermatomes=['L2', 'L3', 'L4'],
    patient_position='supine_leg_abducted',
    technique_notes='Mnemônico ALABAma: AL, AB, AMagnus (anterior → posterior)',
    volume_ml=(5, 10),
)


ANKLE_BLOCK = NerveBlock(
    id='ANKLE',
    name='Bloqueio do Tornozelo',
    name_en='Ankle Block',
    region=BlockRegion.LOWER_LIMB,

    probe_type='linear',
    frequency_mhz=(10, 15),
    depth_cm=(0.5, 3),

    targets=[
        # Nervos profundos
        Structure(
            name='TIBIAL_NERVE',
            display_name='N. Tibial',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic_honeycomb',
            color=COLORS['NERVE'],
            position='posterior_to_medial_malleolus',
            is_target=True,
            notes='Posterior à A. tibial posterior'
        ),
        Structure(
            name='DEEP_PERONEAL_NERVE',
            display_name='N. Peroneal Profundo',
            structure_type=StructureType.NERVE,
            appearance='small_hypoechoic',
            color=COLORS['NERVE_ALT1'],
            position='lateral_to_anterior_tibial_artery',
            is_target=True,
        ),
        # Nervos superficiais
        Structure(
            name='SUPERFICIAL_PERONEAL_NERVE',
            display_name='N. Peroneal Superficial',
            structure_type=StructureType.NERVE,
            appearance='small_hypoechoic',
            color=COLORS['NERVE_ALT2'],
            position='anterolateral_subcutaneous',
            is_target=True,
        ),
        Structure(
            name='SURAL_NERVE',
            display_name='N. Sural',
            structure_type=StructureType.NERVE,
            appearance='small_near_vein',
            color=COLORS['NERVE_ALT3'],
            position='near_small_saphenous_vein',
            is_target=True,
            notes='Próximo à V. safena parva'
        ),
        Structure(
            name='SAPHENOUS_NERVE',
            display_name='N. Safeno',
            structure_type=StructureType.NERVE,
            appearance='small_subcutaneous',
            color=(200, 255, 100),
            position='anteromedial',
            is_target=True,
            notes='Pode ser omitido para antepé'
        ),
    ],

    landmarks=[
        Structure(
            name='POSTERIOR_TIBIAL_ARTERY',
            display_name='A. Tibial Posterior',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['ARTERY'],
            position='posterior_medial',
            key_landmark=True,
        ),
        Structure(
            name='ANTERIOR_TIBIAL_ARTERY',
            display_name='A. Tibial Anterior',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['ARTERY'],
            position='anterior',
        ),
        Structure(
            name='SMALL_SAPHENOUS_VEIN',
            display_name='V. Safena Parva',
            structure_type=StructureType.VEIN,
            appearance='anechoic_compressible',
            color=COLORS['VEIN'],
            position='posterolateral',
        ),
        Structure(
            name='MEDIAL_MALLEOLUS',
            display_name='Maléolo Medial',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_bony',
            color=COLORS['BONE'],
            position='medial',
        ),
        Structure(
            name='LATERAL_MALLEOLUS',
            display_name='Maléolo Lateral',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_bony',
            color=COLORS['BONE'],
            position='lateral',
        ),
    ],

    indications=['Cirurgias de pé e dedos'],
    dermatomes=['L4', 'L5', 'S1', 'S2'],
    patient_position='supine',
    technique_notes='5 nervos. Superficiais podem ser bloqueados com wheal subcutâneo',
    skill_level='intermediate',
    volume_ml=(3, 5),
)


LATERAL_FEMORAL_CUTANEOUS_BLOCK = NerveBlock(
    id='LFCN',
    name='Bloqueio do N. Cutâneo Femoral Lateral',
    name_en='Lateral Femoral Cutaneous Nerve Block',
    region=BlockRegion.LOWER_LIMB,

    probe_type='linear',
    frequency_mhz=(10, 15),
    depth_cm=(1, 3),

    targets=[
        Structure(
            name='LFCN',
            display_name='N. Cutâneo Femoral Lateral',
            structure_type=StructureType.NERVE,
            appearance='small_hypoechoic',
            color=COLORS['NERVE'],
            position='medial_to_ASIS_under_fascia',
            is_target=True,
        ),
    ],

    landmarks=[
        Structure(
            name='ASIS',
            display_name='EIAS',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_bony',
            color=COLORS['BONE'],
            position='lateral',
            key_landmark=True,
            notes='Espinha Ilíaca Ântero-Superior'
        ),
        Structure(
            name='INGUINAL_LIGAMENT',
            display_name='Lig. Inguinal',
            structure_type=StructureType.LIGAMENT,
            appearance='hyperechoic_linear',
            color=COLORS['FASCIA'],
            position='reference',
        ),
        Structure(
            name='SARTORIUS',
            display_name='M. Sartório',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='deep',
        ),
    ],

    indications=['Meralgia parestésica', 'Suplementação PENG/FICB'],
    dermatomes=['L2', 'L3'],
    patient_position='supine',
    volume_ml=(5, 10),
)


# ══════════════════════════════════════════════════════════════════════════════
# TRONCO
# ══════════════════════════════════════════════════════════════════════════════

TAP_BLOCK = NerveBlock(
    id='TAP',
    name='Bloqueio TAP',
    name_en='Transversus Abdominis Plane Block',
    region=BlockRegion.TRUNK,

    probe_type='linear',
    frequency_mhz=(8, 14),
    depth_cm=(2, 5),

    targets=[
        Structure(
            name='TAP_PLANE',
            display_name='Plano TAP',
            structure_type=StructureType.FASCIA,
            appearance='fascial_plane',
            color=COLORS['NERVE'],
            position='between_IO_and_TA',
            is_target=True,
            notes='Entre oblíquo interno e transverso abdominal'
        ),
    ],

    landmarks=[
        Structure(
            name='EXTERNAL_OBLIQUE',
            display_name='M. Oblíquo Externo',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=(180, 100, 100),
            position='superficial',
            key_landmark=True,
        ),
        Structure(
            name='INTERNAL_OBLIQUE',
            display_name='M. Oblíquo Interno',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=(160, 100, 100),
            position='middle',
            key_landmark=True,
            notes='Geralmente o mais espesso'
        ),
        Structure(
            name='TRANSVERSUS_ABDOMINIS',
            display_name='M. Transverso Abdominal',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=(140, 100, 100),
            position='deep',
            key_landmark=True,
            notes='Frequentemente o mais fino'
        ),
    ],

    danger_zones=[
        Structure(
            name='PERITONEUM',
            display_name='Peritônio',
            structure_type=StructureType.FASCIA,
            appearance='hyperechoic_line',
            color=COLORS['WARNING'],
            position='deep_to_TA',
            is_warning=True,
            notes='Risco de perfuração visceral'
        ),
    ],

    indications=['Cirurgias abdominais', 'Cesárea', 'Laparotomia'],
    dermatomes=['T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1'],
    patient_position='supine',
    technique_notes='Variantes: Lateral (abaixo umbigo), Subcostal (acima umbigo)',
    volume_ml=(15, 25),
)


QUADRATUS_LUMBORUM_BLOCK = NerveBlock(
    id='QL',
    name='Bloqueio do Quadrado Lombar',
    name_en='Quadratus Lumborum Block',
    region=BlockRegion.TRUNK,

    probe_type='curvilinear',
    frequency_mhz=(4, 8),
    depth_cm=(4, 10),

    targets=[
        Structure(
            name='QL_PLANE',
            display_name='Plano QL',
            structure_type=StructureType.FASCIA,
            appearance='fascial_plane',
            color=COLORS['NERVE'],
            position='varies_by_approach',
            is_target=True,
            notes='QL1 (lateral), QL2 (posterior), QL3 (anterior)'
        ),
    ],

    landmarks=[
        Structure(
            name='QUADRATUS_LUMBORUM',
            display_name='M. Quadrado Lombar',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='central',
            key_landmark=True,
        ),
        Structure(
            name='ERECTOR_SPINAE',
            display_name='M. Eretor da Espinha',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=(130, 100, 100),
            position='posterior_to_QL',
            key_landmark=True,
        ),
        Structure(
            name='PSOAS_MAJOR',
            display_name='M. Psoas Maior',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=(170, 100, 100),
            position='anterior_to_QL',
            key_landmark=True,
        ),
        Structure(
            name='L4_TRANSVERSE_PROCESS',
            display_name='Processo Transverso L4',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_shadow',
            color=COLORS['BONE'],
            position='deep',
            key_landmark=True,
            notes='Centro do "sinal do trevo" (shamrock)'
        ),
    ],

    indications=['Cirurgias abdominais extensas', 'Nefrectomia', 'Cesárea'],
    dermatomes=['T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2'],
    patient_position='lateral',
    technique_notes='Sinal do trevo (shamrock): PT=tronco, QL/ES/Psoas=folhas',
    skill_level='advanced',
    volume_ml=(20, 30),
)


ERECTOR_SPINAE_BLOCK = NerveBlock(
    id='ESP',
    name='Bloqueio Erector Spinae',
    name_en='Erector Spinae Plane Block',
    region=BlockRegion.TRUNK,

    probe_type='linear',
    frequency_mhz=(6, 12),
    depth_cm=(2, 6),

    targets=[
        Structure(
            name='ESP_PLANE',
            display_name='Plano ESP',
            structure_type=StructureType.FASCIA,
            appearance='fascial_plane',
            color=COLORS['NERVE'],
            position='deep_to_erector_spinae',
            is_target=True,
            notes='Profundo ao eretor da espinha, sobre processo transverso'
        ),
    ],

    landmarks=[
        Structure(
            name='TRANSVERSE_PROCESS',
            display_name='Processo Transverso',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_flat_shadow',
            color=COLORS['BONE'],
            position='deep',
            key_landmark=True,
        ),
        Structure(
            name='ERECTOR_SPINAE',
            display_name='M. Eretor da Espinha',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='superficial_to_TP',
            key_landmark=True,
        ),
        Structure(
            name='TRAPEZIUS',
            display_name='M. Trapézio',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='superficial',
            notes='Apenas em níveis torácicos'
        ),
        Structure(
            name='RHOMBOID',
            display_name='M. Romboide',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='superficial',
            notes='Apenas em níveis torácicos'
        ),
    ],

    indications=['Toracotomia', 'Fraturas costais', 'Mastectomia', 'Cirurgias lombares'],
    dermatomes=['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'],
    patient_position='sitting_or_lateral',
    technique_notes='Mais seguro que paravertebral (longe da pleura)',
    volume_ml=(20, 30),
)


SERRATUS_ANTERIOR_BLOCK = NerveBlock(
    id='SAPB',
    name='Bloqueio do Serrátil Anterior',
    name_en='Serratus Anterior Plane Block',
    region=BlockRegion.TRUNK,

    probe_type='linear',
    frequency_mhz=(8, 14),
    depth_cm=(1, 4),

    targets=[
        Structure(
            name='SERRATUS_PLANE',
            display_name='Plano do Serrátil',
            structure_type=StructureType.FASCIA,
            appearance='fascial_plane',
            color=COLORS['NERVE'],
            position='superficial_or_deep_to_serratus',
            is_target=True,
            notes='Superficial (entre latíssimo e serrátil) ou profundo'
        ),
    ],

    landmarks=[
        Structure(
            name='SERRATUS_ANTERIOR',
            display_name='M. Serrátil Anterior',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='central',
            key_landmark=True,
        ),
        Structure(
            name='LATISSIMUS_DORSI',
            display_name='M. Grande Dorsal',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='superficial',
        ),
        Structure(
            name='RIBS',
            display_name='Costelas',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_shadow',
            color=COLORS['BONE'],
            position='deep',
        ),
    ],

    indications=['Fraturas costais', 'Toracotomia lateral', 'Cirurgia de mama'],
    dermatomes=['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9'],
    patient_position='lateral_or_supine',
    technique_notes='Nível da linha axilar média, 4ª-5ª costela',
    volume_ml=(20, 40),
)


PECS_BLOCK = NerveBlock(
    id='PECS',
    name='Bloqueio PECS',
    name_en='Pectoral Nerve Block',
    region=BlockRegion.TRUNK,

    probe_type='linear',
    frequency_mhz=(8, 14),
    depth_cm=(1, 4),

    targets=[
        Structure(
            name='PECS_I_PLANE',
            display_name='Plano PECS I',
            structure_type=StructureType.FASCIA,
            appearance='fascial_plane',
            color=COLORS['NERVE'],
            position='between_pec_major_and_minor',
            is_target=True,
            notes='Entre peitoral maior e menor'
        ),
        Structure(
            name='PECS_II_PLANE',
            display_name='Plano PECS II',
            structure_type=StructureType.FASCIA,
            appearance='fascial_plane',
            color=COLORS['NERVE_ALT1'],
            position='between_pec_minor_and_serratus',
            is_target=True,
            notes='Entre peitoral menor e serrátil'
        ),
    ],

    landmarks=[
        Structure(
            name='PECTORALIS_MAJOR',
            display_name='M. Peitoral Maior',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='superficial',
            key_landmark=True,
        ),
        Structure(
            name='PECTORALIS_MINOR',
            display_name='M. Peitoral Menor',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='middle',
            key_landmark=True,
        ),
        Structure(
            name='SERRATUS_ANTERIOR',
            display_name='M. Serrátil Anterior',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='deep',
        ),
        Structure(
            name='THORACOACROMIAL_ARTERY',
            display_name='A. Toracoacromial',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['ARTERY'],
            position='reference',
        ),
    ],

    indications=['Mastectomia', 'Cirurgia de mama', 'Implante de dispositivos'],
    dermatomes=['T2', 'T3', 'T4', 'T5', 'T6'],
    patient_position='supine_arm_abducted',
    technique_notes='PECS I: marca-passo. PECS II: mama',
    volume_ml=(10, 20),
)


PARAVERTEBRAL_BLOCK = NerveBlock(
    id='PVB',
    name='Bloqueio Paravertebral Torácico',
    name_en='Thoracic Paravertebral Block',
    region=BlockRegion.TRUNK,

    probe_type='linear',
    frequency_mhz=(6, 12),
    depth_cm=(2, 5),

    targets=[
        Structure(
            name='PARAVERTEBRAL_SPACE',
            display_name='Espaço Paravertebral',
            structure_type=StructureType.FASCIA,
            appearance='fascial_space',
            color=COLORS['NERVE'],
            position='between_TP_and_pleura',
            is_target=True,
        ),
    ],

    landmarks=[
        Structure(
            name='TRANSVERSE_PROCESS',
            display_name='Processo Transverso',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_shadow',
            color=COLORS['BONE'],
            position='reference',
            key_landmark=True,
        ),
        Structure(
            name='COSTOTRANSVERSE_LIGAMENT',
            display_name='Lig. Costotransverso',
            structure_type=StructureType.LIGAMENT,
            appearance='hyperechoic_linear',
            color=COLORS['FASCIA'],
            position='superficial_to_space',
        ),
    ],

    danger_zones=[
        Structure(
            name='PLEURA',
            display_name='Pleura',
            structure_type=StructureType.PLEURA,
            appearance='hyperechoic_sliding',
            color=COLORS['WARNING'],
            position='deep',
            is_warning=True,
            notes='Risco de pneumotórax'
        ),
    ],

    indications=['Toracotomia', 'Mastectomia', 'Herpes zoster'],
    dermatomes=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'],
    patient_position='sitting_or_lateral',
    technique_notes='Visualização essencial da agulha - risco de pneumotórax',
    skill_level='advanced',
    volume_ml=(15, 20),
)


INTERCOSTAL_BLOCK = NerveBlock(
    id='INTERCOSTAL',
    name='Bloqueio Intercostal',
    name_en='Intercostal Nerve Block',
    region=BlockRegion.TRUNK,

    probe_type='linear',
    frequency_mhz=(10, 15),
    depth_cm=(1, 3),

    targets=[
        Structure(
            name='INTERCOSTAL_NERVE',
            display_name='N. Intercostal',
            structure_type=StructureType.NERVE,
            appearance='small_hypoechoic',
            color=COLORS['NERVE'],
            position='inferior_rib_groove',
            is_target=True,
            notes='No sulco costal inferior'
        ),
    ],

    landmarks=[
        Structure(
            name='RIB',
            display_name='Costela',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_shadow',
            color=COLORS['BONE'],
            position='reference',
            key_landmark=True,
        ),
        Structure(
            name='INTERCOSTAL_SPACE',
            display_name='Espaço Intercostal',
            structure_type=StructureType.FASCIA,
            appearance='between_ribs',
            color=COLORS['FASCIA'],
            position='between_ribs',
        ),
    ],

    danger_zones=[
        Structure(
            name='PLEURA',
            display_name='Pleura',
            structure_type=StructureType.PLEURA,
            appearance='hyperechoic_sliding',
            color=COLORS['WARNING'],
            position='deep',
            is_warning=True,
        ),
        Structure(
            name='INTERCOSTAL_VESSELS',
            display_name='Vasos Intercostais',
            structure_type=StructureType.ARTERY,
            appearance='anechoic',
            color=COLORS['WARNING'],
            position='with_nerve',
            is_warning=True,
        ),
    ],

    indications=['Fraturas costais', 'Dor pós-toracotomia'],
    dermatomes=['T1-T11'],
    patient_position='sitting_or_lateral',
    technique_notes='Mnemônico VAN: Veia-Artéria-Nervo no sulco costal',
    volume_ml=(3, 5),
)


RECTUS_SHEATH_BLOCK = NerveBlock(
    id='RECTUS_SHEATH',
    name='Bloqueio da Bainha do Reto',
    name_en='Rectus Sheath Block',
    region=BlockRegion.TRUNK,

    probe_type='linear',
    frequency_mhz=(10, 15),
    depth_cm=(1, 4),

    targets=[
        Structure(
            name='POSTERIOR_RECTUS_SHEATH',
            display_name='Bainha Posterior do Reto',
            structure_type=StructureType.FASCIA,
            appearance='fascial_plane',
            color=COLORS['NERVE'],
            position='between_rectus_and_sheath',
            is_target=True,
        ),
    ],

    landmarks=[
        Structure(
            name='RECTUS_ABDOMINIS',
            display_name='M. Reto Abdominal',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='superficial',
            key_landmark=True,
        ),
        Structure(
            name='LINEA_ALBA',
            display_name='Linha Alba',
            structure_type=StructureType.FASCIA,
            appearance='hyperechoic_midline',
            color=COLORS['FASCIA'],
            position='midline',
        ),
        Structure(
            name='POSTERIOR_SHEATH',
            display_name='Bainha Posterior',
            structure_type=StructureType.FASCIA,
            appearance='hyperechoic_line',
            color=COLORS['FASCIA'],
            position='deep_to_rectus',
        ),
    ],

    indications=['Incisões medianas', 'Laparotomia', 'Umbilical'],
    dermatomes=['T9', 'T10', 'T11'],
    patient_position='supine',
    technique_notes='Bilateral para cobertura completa',
    volume_ml=(10, 15),
)


# ══════════════════════════════════════════════════════════════════════════════
# CABEÇA E PESCOÇO
# ══════════════════════════════════════════════════════════════════════════════

SUPERFICIAL_CERVICAL_PLEXUS_BLOCK = NerveBlock(
    id='SCP',
    name='Bloqueio do Plexo Cervical Superficial',
    name_en='Superficial Cervical Plexus Block',
    region=BlockRegion.HEAD_NECK,

    probe_type='linear',
    frequency_mhz=(10, 15),
    depth_cm=(1, 2),

    targets=[
        Structure(
            name='SCP',
            display_name='Plexo Cervical Superficial',
            structure_type=StructureType.NERVE,
            appearance='small_hypoechoic_cluster',
            color=COLORS['NERVE'],
            position='posterior_border_SCM',
            is_target=True,
        ),
    ],

    landmarks=[
        Structure(
            name='STERNOCLEIDOMASTOID',
            display_name='M. Esternocleidomastóideo',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='reference',
            key_landmark=True,
        ),
        Structure(
            name='EXTERNAL_JUGULAR',
            display_name='V. Jugular Externa',
            structure_type=StructureType.VEIN,
            appearance='anechoic_compressible',
            color=COLORS['VEIN'],
            position='superficial',
        ),
    ],

    indications=['Endarterectomia carotídea', 'Tireoide', 'Traqueostomia'],
    dermatomes=['C2', 'C3', 'C4'],
    patient_position='supine_head_rotated',
    technique_notes='Borda posterior do ECM, nível do cricóide',
    skill_level='basic',
    volume_ml=(10, 15),
)


DEEP_CERVICAL_PLEXUS_BLOCK = NerveBlock(
    id='DCP',
    name='Bloqueio do Plexo Cervical Profundo',
    name_en='Deep Cervical Plexus Block',
    region=BlockRegion.HEAD_NECK,

    probe_type='linear',
    frequency_mhz=(8, 14),
    depth_cm=(2, 4),

    targets=[
        Structure(
            name='C2_C4_ROOTS',
            display_name='Raízes C2-C4',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic',
            color=COLORS['NERVE'],
            position='deep_to_prevertebral_fascia',
            is_target=True,
        ),
    ],

    landmarks=[
        Structure(
            name='SCM',
            display_name='M. Esternocleidomastóideo',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='superficial',
        ),
        Structure(
            name='LEVATOR_SCAPULAE',
            display_name='M. Elevador da Escápula',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='deep',
        ),
    ],

    danger_zones=[
        Structure(
            name='CAROTID_ARTERY',
            display_name='A. Carótida',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['WARNING'],
            position='medial',
            is_warning=True,
        ),
        Structure(
            name='INTERNAL_JUGULAR',
            display_name='V. Jugular Interna',
            structure_type=StructureType.VEIN,
            appearance='anechoic_compressible',
            color=COLORS['WARNING'],
            position='lateral',
            is_warning=True,
        ),
        Structure(
            name='VERTEBRAL_ARTERY',
            display_name='A. Vertebral',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['WARNING'],
            position='deep',
            is_warning=True,
        ),
    ],

    indications=['Endarterectomia carotídea', 'Procedimentos cervicais'],
    dermatomes=['C2', 'C3', 'C4'],
    patient_position='supine_head_rotated',
    technique_notes='CUIDADO: Riscos de injeção intratecal, vertebral, frênico',
    skill_level='advanced',
    volume_ml=(10, 15),
)


STELLATE_GANGLION_BLOCK = NerveBlock(
    id='SGB',
    name='Bloqueio do Gânglio Estrelado',
    name_en='Stellate Ganglion Block',
    region=BlockRegion.HEAD_NECK,

    probe_type='linear',
    frequency_mhz=(8, 14),
    depth_cm=(2, 4),

    targets=[
        Structure(
            name='STELLATE_GANGLION',
            display_name='Gânglio Estrelado',
            structure_type=StructureType.NERVE,
            appearance='small_hypoechoic',
            color=COLORS['NERVE'],
            position='anterolateral_to_longus_colli',
            is_target=True,
            notes='Nível C6-C7'
        ),
    ],

    landmarks=[
        Structure(
            name='CHASSAIGNAC_TUBERCLE',
            display_name='Tubérculo de Chassaignac',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_bony',
            color=COLORS['BONE'],
            position='C6_anterior_tubercle',
            key_landmark=True,
        ),
        Structure(
            name='LONGUS_COLLI',
            display_name='M. Longo do Pescoço',
            structure_type=StructureType.MUSCLE,
            appearance='muscle_texture',
            color=COLORS['MUSCLE'],
            position='deep',
            key_landmark=True,
        ),
        Structure(
            name='THYROID',
            display_name='Tireoide',
            structure_type=StructureType.MUSCLE,
            appearance='homogeneous',
            color=(150, 150, 100),
            position='medial',
        ),
    ],

    danger_zones=[
        Structure(
            name='CAROTID_ARTERY',
            display_name='A. Carótida',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['WARNING'],
            position='lateral',
            is_warning=True,
        ),
        Structure(
            name='VERTEBRAL_ARTERY',
            display_name='A. Vertebral',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['WARNING'],
            position='posterior',
            is_warning=True,
        ),
        Structure(
            name='ESOPHAGUS',
            display_name='Esôfago',
            structure_type=StructureType.MUSCLE,
            appearance='target_sign',
            color=COLORS['WARNING'],
            position='medial',
            is_warning=True,
        ),
    ],

    indications=['SDRC', 'Hiperidrose', 'Doença vascular periférica'],
    patient_position='supine_neck_extended',
    technique_notes='Sucesso = Síndrome de Horner (ptose, miose, anidrose)',
    skill_level='advanced',
    volume_ml=(5, 10),
)


# ══════════════════════════════════════════════════════════════════════════════
# PELVE
# ══════════════════════════════════════════════════════════════════════════════

PUDENDAL_NERVE_BLOCK = NerveBlock(
    id='PUDENDAL',
    name='Bloqueio do Nervo Pudendo',
    name_en='Pudendal Nerve Block',
    region=BlockRegion.PELVIS,

    probe_type='curvilinear',
    frequency_mhz=(2, 5),
    depth_cm=(5, 10),

    targets=[
        Structure(
            name='PUDENDAL_NERVE',
            display_name='N. Pudendo',
            structure_type=StructureType.NERVE,
            appearance='hypoechoic',
            color=COLORS['NERVE'],
            position='between_SSL_and_STL',
            is_target=True,
            notes='Entre ligamentos sacroespinhoso e sacrotuberal'
        ),
    ],

    landmarks=[
        Structure(
            name='ISCHIAL_SPINE',
            display_name='Espinha Isquiática',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_bony',
            color=COLORS['BONE'],
            position='reference',
            key_landmark=True,
        ),
        Structure(
            name='SACROSPINOUS_LIGAMENT',
            display_name='Lig. Sacroespinhoso',
            structure_type=StructureType.LIGAMENT,
            appearance='hyperechoic_linear',
            color=COLORS['FASCIA'],
            position='superficial_to_nerve',
        ),
        Structure(
            name='SACROTUBEROUS_LIGAMENT',
            display_name='Lig. Sacrotuberal',
            structure_type=StructureType.LIGAMENT,
            appearance='hyperechoic_linear',
            color=COLORS['FASCIA'],
            position='deep_to_nerve',
        ),
        Structure(
            name='INTERNAL_PUDENDAL_ARTERY',
            display_name='A. Pudenda Interna',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['ARTERY'],
            position='with_nerve',
        ),
    ],

    danger_zones=[
        Structure(
            name='SCIATIC_NERVE',
            display_name='N. Ciático',
            structure_type=StructureType.NERVE,
            appearance='large_hypoechoic',
            color=COLORS['WARNING'],
            position='lateral',
            is_warning=True,
            notes='Evitar spread lateral excessivo'
        ),
    ],

    indications=['Neuralgia pudenda', 'Cirurgia perineal', 'Hemorroidectomia'],
    patient_position='prone',
    technique_notes='Abordagem transglútea',
    skill_level='advanced',
    volume_ml=(5, 10),
)


IPACK_BLOCK = NerveBlock(
    id='IPACK',
    name='Bloqueio IPACK',
    name_en='Infiltration between Popliteal Artery and Capsule of Knee',
    region=BlockRegion.PELVIS,  # Classificado aqui para joelho posterior

    probe_type='linear',
    frequency_mhz=(6, 12),
    depth_cm=(3, 6),

    targets=[
        Structure(
            name='IPACK_SPACE',
            display_name='Espaço IPACK',
            structure_type=StructureType.FASCIA,
            appearance='fascial_space',
            color=COLORS['NERVE'],
            position='between_popliteal_artery_and_femur',
            is_target=True,
        ),
    ],

    landmarks=[
        Structure(
            name='POPLITEAL_ARTERY',
            display_name='A. Poplítea',
            structure_type=StructureType.ARTERY,
            appearance='anechoic_pulsatile',
            color=COLORS['ARTERY'],
            position='superficial',
            key_landmark=True,
        ),
        Structure(
            name='FEMORAL_CONDYLES',
            display_name='Côndilos Femorais',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_curved',
            color=COLORS['BONE'],
            position='deep',
            key_landmark=True,
        ),
        Structure(
            name='POSTERIOR_CAPSULE',
            display_name='Cápsula Posterior',
            structure_type=StructureType.FASCIA,
            appearance='hyperechoic_linear',
            color=COLORS['FASCIA'],
            position='between_artery_and_bone',
        ),
    ],

    indications=['Artroplastia total de joelho', 'LCA'],
    patient_position='supine_knee_flexed',
    technique_notes='Analgesia posterior sem bloqueio motor',
    volume_ml=(15, 20),
)


GENICULAR_NERVE_BLOCK = NerveBlock(
    id='GENICULAR',
    name='Bloqueio dos Nervos Geniculares',
    name_en='Genicular Nerve Block',
    region=BlockRegion.PELVIS,

    probe_type='linear',
    frequency_mhz=(10, 15),
    depth_cm=(1, 3),

    targets=[
        Structure(
            name='SUPEROMEDIAL_GENICULAR',
            display_name='N. Genicular Superomedial',
            structure_type=StructureType.NERVE,
            appearance='small_hypoechoic',
            color=COLORS['NERVE'],
            position='junction_shaft_condyle_medial',
            is_target=True,
        ),
        Structure(
            name='SUPEROLATERAL_GENICULAR',
            display_name='N. Genicular Superolateral',
            structure_type=StructureType.NERVE,
            appearance='small_hypoechoic',
            color=COLORS['NERVE_ALT1'],
            position='junction_shaft_condyle_lateral',
            is_target=True,
        ),
        Structure(
            name='INFEROMEDIAL_GENICULAR',
            display_name='N. Genicular Inferomedial',
            structure_type=StructureType.NERVE,
            appearance='small_hypoechoic',
            color=COLORS['NERVE_ALT2'],
            position='below_tibial_plateau_medial',
            is_target=True,
        ),
    ],

    landmarks=[
        Structure(
            name='ADDUCTOR_TUBERCLE',
            display_name='Tubérculo Adutor',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_bony',
            color=COLORS['BONE'],
            position='medial_femur',
            key_landmark=True,
        ),
        Structure(
            name='FEMORAL_CONDYLES',
            display_name='Côndilos Femorais',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_curved',
            color=COLORS['BONE'],
            position='reference',
        ),
        Structure(
            name='TIBIAL_PLATEAU',
            display_name='Platô Tibial',
            structure_type=StructureType.BONE,
            appearance='hyperechoic_flat',
            color=COLORS['BONE'],
            position='distal',
        ),
    ],

    indications=['Dor crônica de joelho', 'Osteoartrite'],
    patient_position='supine',
    technique_notes='Pode usar ablação por radiofrequência',
    volume_ml=(2, 4),
)


# ══════════════════════════════════════════════════════════════════════════════
# DICIONÁRIO PRINCIPAL DE BLOQUEIOS
# ══════════════════════════════════════════════════════════════════════════════

ALL_NERVE_BLOCKS: Dict[str, NerveBlock] = {
    # Membro Superior
    'INTERSCALENE': INTERSCALENE_BLOCK,
    'SUPRACLAVICULAR': SUPRACLAVICULAR_BLOCK,
    'INFRACLAVICULAR': INFRACLAVICULAR_BLOCK,
    'AXILLARY': AXILLARY_BLOCK,
    'WRIST': WRIST_BLOCK,

    # Membro Inferior
    'FEMORAL': FEMORAL_NERVE_BLOCK,
    'FASCIA_ILIACA': FASCIA_ILIACA_BLOCK,
    'PENG': PENG_BLOCK,
    'SCIATIC_SUBGLUTEAL': SCIATIC_SUBGLUTEAL_BLOCK,
    'SCIATIC_POPLITEAL': SCIATIC_POPLITEAL_BLOCK,
    'ADDUCTOR_CANAL': ADDUCTOR_CANAL_BLOCK,
    'OBTURATOR': OBTURATOR_NERVE_BLOCK,
    'ANKLE': ANKLE_BLOCK,
    'LFCN': LATERAL_FEMORAL_CUTANEOUS_BLOCK,

    # Tronco
    'TAP': TAP_BLOCK,
    'QL': QUADRATUS_LUMBORUM_BLOCK,
    'RECTUS_SHEATH': RECTUS_SHEATH_BLOCK,
    'ESP': ERECTOR_SPINAE_BLOCK,
    'SAPB': SERRATUS_ANTERIOR_BLOCK,
    'PECS': PECS_BLOCK,
    'PVB': PARAVERTEBRAL_BLOCK,
    'INTERCOSTAL': INTERCOSTAL_BLOCK,

    # Cabeça e Pescoço
    'SCP': SUPERFICIAL_CERVICAL_PLEXUS_BLOCK,
    'DCP': DEEP_CERVICAL_PLEXUS_BLOCK,
    'SGB': STELLATE_GANGLION_BLOCK,

    # Pelve
    'PUDENDAL': PUDENDAL_NERVE_BLOCK,
    'IPACK': IPACK_BLOCK,
    'GENICULAR': GENICULAR_NERVE_BLOCK,
}


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE ACESSO
# ══════════════════════════════════════════════════════════════════════════════

def get_block_config(block_id: str) -> Optional[NerveBlock]:
    """Retorna configuração de um bloqueio específico"""
    return ALL_NERVE_BLOCKS.get(block_id.upper())


def get_blocks_by_region(region: BlockRegion) -> List[NerveBlock]:
    """Retorna todos os bloqueios de uma região"""
    return [block for block in ALL_NERVE_BLOCKS.values() if block.region == region]


def get_all_block_ids() -> List[str]:
    """Retorna lista de todos os IDs de bloqueios"""
    return list(ALL_NERVE_BLOCKS.keys())


def get_structures_to_detect(block_id: str) -> List[Structure]:
    """Retorna todas as estruturas que a IA deve detectar para um bloqueio"""
    block = get_block_config(block_id)
    if not block:
        return []

    structures = []
    structures.extend(block.targets)
    structures.extend(block.landmarks)
    structures.extend(block.danger_zones)

    return structures


def get_target_structures(block_id: str) -> List[Structure]:
    """Retorna apenas as estruturas alvo (nervos a bloquear)"""
    block = get_block_config(block_id)
    if not block:
        return []
    return block.targets


def get_danger_zones(block_id: str) -> List[Structure]:
    """Retorna estruturas de perigo/alerta"""
    block = get_block_config(block_id)
    if not block:
        return []
    return block.danger_zones


def get_blocks_by_skill_level(skill_level: str) -> List[NerveBlock]:
    """Retorna bloqueios por nível de habilidade"""
    return [
        block for block in ALL_NERVE_BLOCKS.values()
        if block.skill_level == skill_level.lower()
    ]


def search_blocks_by_indication(indication: str) -> List[NerveBlock]:
    """Busca bloqueios por indicação clínica"""
    indication_lower = indication.lower()
    return [
        block for block in ALL_NERVE_BLOCKS.values()
        if any(indication_lower in ind.lower() for ind in block.indications)
    ]


def search_blocks(query: str) -> List[NerveBlock]:
    """Busca bloqueios por nome, ID ou indicação"""
    query_lower = query.lower()
    results = []

    for block in ALL_NERVE_BLOCKS.values():
        # Buscar por nome
        if query_lower in block.name.lower():
            results.append(block)
            continue

        # Buscar por ID
        if query_lower in block.id.lower():
            results.append(block)
            continue

        # Buscar por nome em inglês
        if query_lower in block.name_en.lower():
            results.append(block)
            continue

        # Buscar por indicações
        if any(query_lower in ind.lower() for ind in block.indications):
            results.append(block)
            continue

    return results


# ══════════════════════════════════════════════════════════════════════════════
# NOMES DOS BLOQUEIOS (para UI)
# ══════════════════════════════════════════════════════════════════════════════

BLOCK_NAMES: Dict[str, str] = {
    block_id: block.name for block_id, block in ALL_NERVE_BLOCKS.items()
}


# ══════════════════════════════════════════════════════════════════════════════
# ESTATÍSTICAS
# ══════════════════════════════════════════════════════════════════════════════

def get_database_stats() -> dict:
    """Retorna estatísticas do banco de dados"""
    stats = {
        'total_blocks': len(ALL_NERVE_BLOCKS),
        'by_region': {},
        'by_skill_level': {},
        'total_structures': 0,
    }

    for region in BlockRegion:
        count = len(get_blocks_by_region(region))
        stats['by_region'][region.value] = count

    for level in ['basic', 'intermediate', 'advanced']:
        count = len(get_blocks_by_skill_level(level))
        stats['by_skill_level'][level] = count

    for block in ALL_NERVE_BLOCKS.values():
        stats['total_structures'] += len(block.targets)
        stats['total_structures'] += len(block.landmarks)
        stats['total_structures'] += len(block.danger_zones)

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# TESTE DO MÓDULO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("NERVE TRACK v2.0 - Banco de Dados de Bloqueios")
    print("=" * 60)

    stats = get_database_stats()
    print(f"\nTotal de bloqueios: {stats['total_blocks']}")
    print(f"Total de estruturas: {stats['total_structures']}")

    print("\nPor região:")
    for region, count in stats['by_region'].items():
        print(f"  - {region}: {count}")

    print("\nPor nível:")
    for level, count in stats['by_skill_level'].items():
        print(f"  - {level}: {count}")

    print("\n" + "=" * 60)
    print("Lista de bloqueios:")
    print("=" * 60)

    for block_id, block in ALL_NERVE_BLOCKS.items():
        targets = len(block.targets)
        landmarks = len(block.landmarks)
        dangers = len(block.danger_zones)
        print(f"  [{block_id}] {block.name}")
        print(f"       Alvos: {targets} | Landmarks: {landmarks} | Perigos: {dangers}")
