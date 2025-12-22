# 🏥 ATLAS COMPLETO DE BLOQUEIOS NERVOSOS GUIADOS POR ULTRASSOM

## Para implementação no NERVE TRACK v2.0 PREMIUM

Este documento contém TODOS os bloqueios nervosos utilizados em anestesia regional,
com anatomia detalhada, estruturas a detectar, e configurações para IA.

---

# 📑 ÍNDICE GERAL

## MEMBRO SUPERIOR (Upper Extremity)
1. [Plexo Braquial - Interescalênico](#1-bloqueio-interescalênico)
2. [Plexo Braquial - Supraclavicular](#2-bloqueio-supraclavicular)
3. [Plexo Braquial - Infraclavicular](#3-bloqueio-infraclavicular)
4. [Plexo Braquial - Axilar](#4-bloqueio-axilar)
5. [Bloqueio do Pulso (Wrist Block)](#5-bloqueio-do-pulso)

## MEMBRO INFERIOR (Lower Extremity)
6. [Nervo Femoral](#6-bloqueio-do-nervo-femoral)
7. [Fascia Ilíaca (FICB)](#7-bloqueio-da-fascia-ilíaca)
8. [PENG Block](#8-peng-block)
9. [Nervo Ciático - Subglúteo](#9-bloqueio-ciático-subglúteo)
10. [Nervo Ciático - Poplíteo](#10-bloqueio-ciático-poplíteo)
11. [Canal Adutor (Safeno)](#11-bloqueio-do-canal-adutor)
12. [Nervo Obturador](#12-bloqueio-do-nervo-obturador)
13. [Bloqueio do Tornozelo](#13-bloqueio-do-tornozelo)
14. [Nervo Cutâneo Femoral Lateral](#14-nervo-cutâneo-femoral-lateral)

## TRONCO E PAREDE ABDOMINAL
15. [TAP Block (Transversus Abdominis)](#15-tap-block)
16. [Quadratus Lumborum (QL Block)](#16-quadratus-lumborum-block)
17. [Bainha do Reto (Rectus Sheath)](#17-bloqueio-da-bainha-do-reto)
18. [Erector Spinae (ESP Block)](#18-erector-spinae-block)
19. [Serratus Anterior (SAPB)](#19-serratus-anterior-block)
20. [PECS I e II](#20-pecs-block)
21. [Paravertebral Torácico](#21-bloqueio-paravertebral)
22. [Intercostal](#22-bloqueio-intercostal)

## CABEÇA E PESCOÇO
23. [Plexo Cervical Superficial](#23-plexo-cervical-superficial)
24. [Plexo Cervical Profundo](#24-plexo-cervical-profundo)
25. [Gânglio Estrelado](#25-gânglio-estrelado)

## PELVE E PERÍNEO
26. [Nervo Pudendo](#26-bloqueio-do-nervo-pudendo)
27. [IPACK Block](#27-ipack-block)
28. [Bloqueio Genicular](#28-bloqueio-genicular)

---

# MEMBRO SUPERIOR

---

## 1. BLOQUEIO INTERESCALÊNICO

### Indicações
- Cirurgias de ombro (artroscopia, artroplastia)
- Clavícula (2/3 laterais)
- Úmero proximal

### Anatomia Ultrassonográfica

```
ESTRUTURAS A DETECTAR:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    ┌─────────────────┐                                      │
│    │ M. Escaleno     │                                      │
│    │ Anterior        │                                      │
│    └────────┬────────┘                                      │
│             │                                               │
│    ┌────────┴────────┐                                      │
│    │   RAÍZES C5-C6  │  ← Alvo principal                   │
│    │   (hypoechoic   │                                      │
│    │    circles)     │                                      │
│    └────────┬────────┘                                      │
│             │                                               │
│    ┌────────┴────────┐                                      │
│    │ M. Escaleno     │                                      │
│    │ Médio           │                                      │
│    └─────────────────┘                                      │
│                                                             │
│    Profundo: Processo transverso cervical                   │
│    Lateral: A. Vertebral                                    │
│    Medial: A. Carótida, V. Jugular Interna                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Configuração para IA

```python
INTERSCALENE_BLOCK = {
    'id': 'INTERSCALENE',
    'name': 'Bloqueio Interescalênico',
    'name_en': 'Interscalene Block',

    # Probe e profundidade
    'probe': 'linear',
    'frequency_mhz': (8, 14),
    'depth_cm': (2, 4),

    # Estruturas alvo (a detectar)
    'targets': [
        {
            'name': 'BRACHIAL_PLEXUS_ROOTS',
            'display': 'Raízes C5-C6-C7',
            'appearance': 'hypoechoic_circles',
            'pattern': 'stacked',  # empilhadas verticalmente
            'typical_count': (2, 3),
            'color': (0, 255, 255),  # Amarelo
        },
    ],

    # Estruturas adjacentes (landmarks)
    'landmarks': [
        {
            'name': 'ANTERIOR_SCALENE',
            'display': 'M. Escaleno Anterior',
            'appearance': 'muscle_texture',
            'position': 'medial',
            'color': (180, 100, 100),
        },
        {
            'name': 'MIDDLE_SCALENE',
            'display': 'M. Escaleno Médio',
            'appearance': 'muscle_texture',
            'position': 'lateral',
            'color': (180, 100, 100),
        },
        {
            'name': 'CAROTID_ARTERY',
            'display': 'A. Carótida',
            'appearance': 'anechoic_pulsatile',
            'position': 'medial',
            'color': (0, 0, 255),  # Vermelho
            'warning': True,  # Estrutura a evitar
        },
        {
            'name': 'INTERNAL_JUGULAR',
            'display': 'V. Jugular Interna',
            'appearance': 'anechoic_compressible',
            'position': 'medial',
            'color': (255, 100, 100),  # Azul
            'warning': True,
        },
    ],

    # Estruturas de alerta
    'danger_zones': [
        {
            'name': 'PHRENIC_NERVE',
            'display': 'N. Frênico',
            'description': 'Anterior ao escaleno anterior',
            'complication': 'Paralisia diafragmática',
        },
        {
            'name': 'VERTEBRAL_ARTERY',
            'display': 'A. Vertebral',
            'description': 'Nos forames transversos',
            'complication': 'Injeção intravascular',
        },
    ],

    # Posição do paciente
    'patient_position': 'supine_head_rotated_contralateral',

    # Referências
    'dermatomes': ['C5', 'C6', 'C7'],
    'motor_coverage': ['deltoid', 'biceps', 'supraspinatus', 'infraspinatus'],
}
```

---

## 2. BLOQUEIO SUPRACLAVICULAR

### Indicações
- Cirurgias de cotovelo, antebraço, mão
- Úmero distal
- "Anestesia espinhal do membro superior"

### Anatomia Ultrassonográfica

```
ESTRUTURAS A DETECTAR:
┌─────────────────────────────────────────────────────────────┐
│                     Pele                                    │
│    ┌──────────────────────────────────────────────────┐     │
│    │            Subcutâneo                            │     │
│    └──────────────────────────────────────────────────┘     │
│                                                             │
│    ┌────────────────────┐    ┌────────────────────┐         │
│    │                    │    │  PLEXO BRAQUIAL    │         │
│    │   A. Subclávia     │    │  (cluster of       │         │
│    │   (pulsátil)       │    │   grapes)          │         │
│    │                    │    │                    │         │
│    └────────────────────┘    └────────────────────┘         │
│                                                             │
│    ════════════════════════════════════════════════         │
│    ░░░░░░░░░░░░ PRIMEIRA COSTELA ░░░░░░░░░░░░░░░░░          │
│    ════════════════════════════════════════════════         │
│                                                             │
│    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ PLEURA ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

VISTA: Transversal supraclavicular
POSIÇÃO: "Corner pocket" - lateral à artéria, sobre a costela
```

### Configuração para IA

```python
SUPRACLAVICULAR_BLOCK = {
    'id': 'SUPRACLAVICULAR',
    'name': 'Bloqueio Supraclavicular',
    'name_en': 'Supraclavicular Block',

    'probe': 'linear',
    'frequency_mhz': (8, 14),
    'depth_cm': (2, 5),

    'targets': [
        {
            'name': 'BRACHIAL_PLEXUS_TRUNKS',
            'display': 'Troncos do Plexo',
            'appearance': 'cluster_hypoechoic',  # "cacho de uvas"
            'pattern': 'grape_cluster',
            'position': 'lateral_to_artery',
            'color': (0, 255, 255),
        },
    ],

    'landmarks': [
        {
            'name': 'SUBCLAVIAN_ARTERY',
            'display': 'A. Subclávia',
            'appearance': 'anechoic_pulsatile',
            'position': 'medial_to_plexus',
            'color': (0, 0, 255),
            'key_landmark': True,  # Usar como referência principal
        },
        {
            'name': 'FIRST_RIB',
            'display': 'Primeira Costela',
            'appearance': 'hyperechoic_shadow',
            'position': 'deep',
            'color': (255, 255, 255),
        },
        {
            'name': 'PLEURA',
            'display': 'Pleura',
            'appearance': 'hyperechoic_sliding',
            'position': 'deep_medial',
            'color': (100, 100, 255),
            'warning': True,
        },
    ],

    'danger_zones': [
        {
            'name': 'PLEURA',
            'display': 'Pleura',
            'complication': 'Pneumotórax',
            'prevention': 'Manter agulha superficial à costela',
        },
        {
            'name': 'SUBCLAVIAN_ARTERY',
            'display': 'A. Subclávia',
            'complication': 'Hematoma',
        },
    ],

    'patient_position': 'supine_head_rotated',
    'dermatomes': ['C5', 'C6', 'C7', 'C8', 'T1'],
    'motor_coverage': ['complete_upper_limb'],
}
```

---

## 3. BLOQUEIO INFRACLAVICULAR

### Indicações
- Cirurgias de cotovelo, antebraço, mão
- Ideal para cateter contínuo (posição estável)

### Anatomia Ultrassonográfica

```
ESTRUTURAS A DETECTAR:
┌─────────────────────────────────────────────────────────────┐
│                         Pele                                │
│    ┌──────────────────────────────────────────────────┐     │
│    │         M. Peitoral Maior                        │     │
│    └──────────────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────┐     │
│    │         M. Peitoral Menor                        │     │
│    └──────────────────────────────────────────────────┘     │
│                                                             │
│              ┌─────────┐                                    │
│    LATERAL   │         │                                    │
│    CORD ●────┤  A.     │                                    │
│              │ AXILAR  │                                    │
│    POST  ●───┤         │                                    │
│    CORD      │  (●)    │────● MEDIAL CORD                   │
│              └─────────┘                                    │
│                  │                                          │
│              V. Axilar (medial)                             │
│                                                             │
│    POSIÇÃO DOS CORDÕES AO REDOR DA ARTÉRIA:                 │
│    - Lateral: 9 horas                                       │
│    - Posterior: 6 horas                                     │
│    - Medial: 3 horas                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Configuração para IA

```python
INFRACLAVICULAR_BLOCK = {
    'id': 'INFRACLAVICULAR',
    'name': 'Bloqueio Infraclavicular',
    'name_en': 'Infraclavicular Block',

    'probe': 'linear_or_curvilinear',
    'frequency_mhz': (6, 12),
    'depth_cm': (3, 6),

    'targets': [
        {
            'name': 'LATERAL_CORD',
            'display': 'Cordão Lateral',
            'appearance': 'hypoechoic',
            'clock_position': '9_oclock',
            'color': (0, 255, 255),
        },
        {
            'name': 'POSTERIOR_CORD',
            'display': 'Cordão Posterior',
            'appearance': 'hypoechoic',
            'clock_position': '6_oclock',
            'color': (0, 255, 200),
        },
        {
            'name': 'MEDIAL_CORD',
            'display': 'Cordão Medial',
            'appearance': 'hypoechoic',
            'clock_position': '3_oclock',
            'color': (0, 200, 255),
        },
    ],

    'landmarks': [
        {
            'name': 'AXILLARY_ARTERY',
            'display': 'A. Axilar',
            'appearance': 'anechoic_pulsatile',
            'position': 'central_reference',
            'color': (0, 0, 255),
            'key_landmark': True,
        },
        {
            'name': 'AXILLARY_VEIN',
            'display': 'V. Axilar',
            'appearance': 'anechoic_compressible',
            'position': 'medial_inferior',
            'color': (255, 100, 100),
        },
        {
            'name': 'PECTORALIS_MAJOR',
            'display': 'M. Peitoral Maior',
            'appearance': 'muscle_texture',
            'position': 'superficial',
            'color': (150, 100, 100),
        },
        {
            'name': 'PECTORALIS_MINOR',
            'display': 'M. Peitoral Menor',
            'appearance': 'muscle_texture',
            'position': 'deep_to_major',
            'color': (130, 100, 100),
        },
    ],

    'injection_target': 'U_shape_around_artery',
    'patient_position': 'supine_arm_abducted',
    'dermatomes': ['C5', 'C6', 'C7', 'C8', 'T1'],
}
```

---

## 4. BLOQUEIO AXILAR

### Indicações
- Cirurgias de antebraço e mão
- Ideal para pacientes acordados (superficial, seguro)

### Anatomia Ultrassonográfica

```
ESTRUTURAS A DETECTAR:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    Posição: Axila, braço abduzido 90°                       │
│                                                             │
│                    ┌─────────────┐                          │
│                    │ BICEPS      │                          │
│                    └─────────────┘                          │
│                                                             │
│    N. Musculocutâneo ●                                      │
│    (dentro do coracobraquial)                               │
│                                                             │
│    ┌──────────────────────────────────┐                     │
│    │                                  │                     │
│    │   N. Mediano ●     ● N. Ulnar    │                     │
│    │        ↖           ↗             │                     │
│    │         ┌───────┐                │                     │
│    │         │   A.  │                │                     │
│    │         │BRAQ.  │                │                     │
│    │         └───────┘                │                     │
│    │              ↓                   │                     │
│    │         ● N. Radial              │                     │
│    │                                  │                     │
│    └──────────────────────────────────┘                     │
│                                                             │
│    MNEMÔNICO (sentido horário a partir das 12h):            │
│    M.A.R.U = Musculocutâneo, A. braquial,                   │
│              Radial (profundo), Ulnar                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Configuração para IA

```python
AXILLARY_BLOCK = {
    'id': 'AXILLARY',
    'name': 'Bloqueio Axilar',
    'name_en': 'Axillary Block',

    'probe': 'linear',
    'frequency_mhz': (10, 15),
    'depth_cm': (1, 3),

    'targets': [
        {
            'name': 'MEDIAN_NERVE',
            'display': 'N. Mediano',
            'appearance': 'hypoechoic_honeycomb',
            'position': 'superficial_lateral_to_artery',
            'clock_position': '10_to_11_oclock',
            'color': (0, 255, 255),
        },
        {
            'name': 'ULNAR_NERVE',
            'display': 'N. Ulnar',
            'appearance': 'hypoechoic_honeycomb',
            'position': 'superficial_medial_to_artery',
            'clock_position': '1_to_2_oclock',
            'color': (0, 255, 200),
        },
        {
            'name': 'RADIAL_NERVE',
            'display': 'N. Radial',
            'appearance': 'hypoechoic_honeycomb',
            'position': 'deep_to_artery',
            'clock_position': '4_to_6_oclock',
            'color': (0, 200, 255),
        },
        {
            'name': 'MUSCULOCUTANEOUS_NERVE',
            'display': 'N. Musculocutâneo',
            'appearance': 'hypoechoic_within_muscle',
            'position': 'within_coracobrachialis',
            'color': (100, 255, 200),
        },
    ],

    'landmarks': [
        {
            'name': 'BRACHIAL_ARTERY',
            'display': 'A. Braquial',
            'appearance': 'anechoic_pulsatile',
            'position': 'central',
            'color': (0, 0, 255),
            'key_landmark': True,
        },
        {
            'name': 'CORACOBRACHIALIS',
            'display': 'M. Coracobraquial',
            'appearance': 'muscle_texture',
            'contains': 'MUSCULOCUTANEOUS_NERVE',
            'color': (150, 100, 100),
        },
    ],

    'patient_position': 'supine_arm_abducted_90deg',
    'dermatomes': ['C5', 'C6', 'C7', 'C8', 'T1'],
    'notes': 'Bloqueio mais superficial e seguro do plexo braquial',
}
```

---

## 5. BLOQUEIO DO PULSO

### Indicações
- Cirurgias de mão e dedos
- Suplementação de bloqueio proximal incompleto

### Configuração para IA

```python
WRIST_BLOCK = {
    'id': 'WRIST',
    'name': 'Bloqueio do Pulso',
    'name_en': 'Wrist Block',

    'probe': 'linear_high_freq',
    'frequency_mhz': (12, 18),
    'depth_cm': (0.5, 2),

    'targets': [
        {
            'name': 'MEDIAN_NERVE_WRIST',
            'display': 'N. Mediano (pulso)',
            'appearance': 'hypoechoic_honeycomb',
            'position': 'between_fds_fdp',  # Entre flexor superficial e profundo
            'typical_csa_mm2': (8, 12),  # CSA normal
            'csa_cts_threshold': 10,  # > 10mm² sugere STC
            'color': (0, 255, 255),
        },
        {
            'name': 'ULNAR_NERVE_WRIST',
            'display': 'N. Ulnar (pulso)',
            'appearance': 'hypoechoic_honeycomb',
            'position': 'medial_to_ulnar_artery',
            'color': (0, 255, 200),
        },
        {
            'name': 'RADIAL_NERVE_WRIST',
            'display': 'N. Radial Superficial',
            'appearance': 'small_hypoechoic',
            'position': 'lateral_subcutaneous',
            'color': (0, 200, 255),
        },
    ],

    'landmarks': [
        {
            'name': 'RADIAL_ARTERY',
            'display': 'A. Radial',
            'color': (0, 0, 255),
        },
        {
            'name': 'ULNAR_ARTERY',
            'display': 'A. Ulnar',
            'color': (0, 0, 255),
        },
        {
            'name': 'FLEXOR_TENDONS',
            'display': 'Tendões Flexores',
            'appearance': 'hyperechoic_fibrillar',
            'color': (200, 200, 200),
        },
    ],

    'clinical_note': 'CSA do mediano > 10mm² no túnel sugere STC',
}
```

---

# MEMBRO INFERIOR

---

## 6. BLOQUEIO DO NERVO FEMORAL

### Indicações
- Cirurgias de quadril, fêmur, joelho
- Analgesia pós-operatória

### Anatomia Ultrassonográfica

```
ESTRUTURAS A DETECTAR:
┌─────────────────────────────────────────────────────────────┐
│                         Pele                                │
│    ┌──────────────────────────────────────────────────┐     │
│    │                 Fascia Lata                      │     │
│    └──────────────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────┐     │
│    │                Fascia Ilíaca                     │     │
│    └──────────────────────────────────────────────────┘     │
│                                                             │
│    LATERAL                              MEDIAL              │
│                                                             │
│    ┌─────────┐                                              │
│    │   N.    │                                              │
│    │FEMORAL  │    ┌───────┐    ┌───────┐                    │
│    │(hipoec.)│    │   A.  │    │   V.  │                    │
│    └─────────┘    │FEMORAL│    │FEMORAL│                    │
│                   └───────┘    └───────┘                    │
│                                                             │
│    ┌──────────────────────────────────────────────────┐     │
│    │              M. Iliopsoas                        │     │
│    └──────────────────────────────────────────────────┘     │
│                                                             │
│    MNEMÔNICO (lateral para medial): N.A.V.E.L               │
│    Nervo - Artéria - Veia - Espaço - Linfáticos             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Configuração para IA

```python
FEMORAL_NERVE_BLOCK = {
    'id': 'FEMORAL',
    'name': 'Bloqueio do Nervo Femoral',
    'name_en': 'Femoral Nerve Block',

    'probe': 'linear',
    'frequency_mhz': (8, 14),
    'depth_cm': (2, 5),

    'targets': [
        {
            'name': 'FEMORAL_NERVE',
            'display': 'N. Femoral',
            'appearance': 'hypoechoic_triangular',
            'position': 'lateral_to_artery',
            'deep_to': 'fascia_iliaca',
            'color': (0, 255, 255),
        },
    ],

    'landmarks': [
        {
            'name': 'FEMORAL_ARTERY',
            'display': 'A. Femoral',
            'appearance': 'anechoic_pulsatile',
            'position': 'medial_to_nerve',
            'color': (0, 0, 255),
            'key_landmark': True,
        },
        {
            'name': 'FEMORAL_VEIN',
            'display': 'V. Femoral',
            'appearance': 'anechoic_compressible',
            'position': 'medial_to_artery',
            'color': (255, 100, 100),
        },
        {
            'name': 'FASCIA_ILIACA',
            'display': 'Fascia Ilíaca',
            'appearance': 'hyperechoic_line',
            'position': 'superficial_to_nerve',
            'color': (200, 200, 200),
        },
        {
            'name': 'ILIOPSOAS',
            'display': 'M. Iliopsoas',
            'appearance': 'muscle_texture',
            'position': 'deep',
            'color': (150, 100, 100),
        },
    ],

    'mnemonic': 'NAVEL (lateral to medial)',
    'patient_position': 'supine_leg_slightly_abducted',
    'dermatomes': ['L2', 'L3', 'L4'],
    'motor_coverage': ['quadriceps'],
    'warning': 'Causa fraqueza do quadríceps - risco de queda',
}
```

---

## 7. BLOQUEIO DA FASCIA ILÍACA (FICB)

### Indicações
- Fratura de quadril (analgesia de emergência)
- Cirurgias de quadril e fêmur

### Configuração para IA

```python
FASCIA_ILIACA_BLOCK = {
    'id': 'FASCIA_ILIACA',
    'name': 'Bloqueio da Fascia Ilíaca',
    'name_en': 'Fascia Iliaca Compartment Block',

    'variants': {
        'infrainguinal': {
            'probe_position': 'inguinal_crease',
            'depth_cm': (2, 4),
        },
        'suprainguinal': {
            'probe_position': 'above_inguinal_ligament',
            'depth_cm': (3, 6),
            'advantage': 'Melhor cobertura do obturador',
        },
    },

    'targets': [
        {
            'name': 'FASCIA_ILIACA_PLANE',
            'display': 'Plano da Fascia Ilíaca',
            'appearance': 'fascial_plane',
            'position': 'between_fascias',
            'injection_site': True,
        },
    ],

    'landmarks': [
        {
            'name': 'FASCIA_LATA',
            'display': 'Fascia Lata',
            'appearance': 'hyperechoic_superficial',
        },
        {
            'name': 'FASCIA_ILIACA',
            'display': 'Fascia Ilíaca',
            'appearance': 'hyperechoic_deep',
        },
        {
            'name': 'ILIACUS_MUSCLE',
            'display': 'M. Ilíaco',
            'position': 'deep_to_fascia_iliaca',
        },
        {
            'name': 'SARTORIUS',
            'display': 'M. Sartório',
            'position': 'lateral_reference',
        },
    ],

    'nerves_blocked': ['femoral', 'lateral_femoral_cutaneous', 'obturator_partial'],
    'volume_ml': (30, 40),
}
```

---

## 8. PENG BLOCK

### Indicações
- Fratura de quadril (alternativa moderna)
- Artroplastia de quadril
- Preserva força motora (vantagem sobre femoral)

### Anatomia Ultrassonográfica

```
ESTRUTURAS A DETECTAR:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    Nível: EIAS → EIAI → Eminência Iliopúbica                │
│                                                             │
│    ┌───────────────────────────────────────────────────┐    │
│    │          M. ILIOPSOAS                             │    │
│    └───────────────────────────────────────────────────┘    │
│                                                             │
│         ●                              ●                    │
│        EIAI                    Eminência Iliopúbica         │
│    (Espinha Ilíaca            (proeminência óssea)          │
│     Ântero-Inferior)                                        │
│                                                             │
│    ALVO: Plano entre o tendão do iliopsoas e o osso         │
│    Depositar anestésico próximo aos ramos articulares       │
│                                                             │
│    ▲ N. Cutâneo Femoral Lateral pode estar próximo          │
│      (cuidado com parestesia na injeção)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Configuração para IA

```python
PENG_BLOCK = {
    'id': 'PENG',
    'name': 'Bloqueio PENG',
    'name_en': 'Pericapsular Nerve Group Block',

    'probe': 'curvilinear',
    'frequency_mhz': (4, 8),
    'depth_cm': (4, 8),

    'targets': [
        {
            'name': 'PENG_PLANE',
            'display': 'Plano PENG',
            'position': 'between_psoas_tendon_and_bone',
            'injection_site': True,
        },
    ],

    'landmarks': [
        {
            'name': 'AIIS',
            'display': 'EIAI',
            'full_name': 'Espinha Ilíaca Ântero-Inferior',
            'appearance': 'hyperechoic_bony',
            'key_landmark': True,
        },
        {
            'name': 'ILIOPUBIC_EMINENCE',
            'display': 'Eminência Iliopúbica',
            'appearance': 'hyperechoic_bony',
            'key_landmark': True,
        },
        {
            'name': 'ILIOPSOAS',
            'display': 'M. Iliopsoas',
            'position': 'superficial',
        },
        {
            'name': 'FEMORAL_HEAD',
            'display': 'Cabeça Femoral',
            'appearance': 'hyperechoic_curved',
        },
    ],

    'warning': [
        {
            'structure': 'LATERAL_FEMORAL_CUTANEOUS',
            'description': 'Próximo ao local de injeção',
            'prevention': 'Verificar parestesia antes de injetar',
        },
    ],

    'advantages': [
        'Preserva força do quadríceps',
        'Analgesia articular específica',
        'Alternativa ao bloqueio femoral',
    ],
}
```

---

## 9. BLOQUEIO CIÁTICO - SUBGLÚTEO

### Indicações
- Cirurgias de perna, tornozelo, pé
- Amputações de membro inferior

### Configuração para IA

```python
SCIATIC_SUBGLUTEAL_BLOCK = {
    'id': 'SCIATIC_SUBGLUTEAL',
    'name': 'Bloqueio Ciático Subglúteo',
    'name_en': 'Subgluteal Sciatic Block',

    'probe': 'curvilinear',
    'frequency_mhz': (4, 8),
    'depth_cm': (4, 10),

    'targets': [
        {
            'name': 'SCIATIC_NERVE',
            'display': 'N. Ciático',
            'appearance': 'hypoechoic_large_oval',
            'typical_csa_mm2': (40, 80),
            'position': 'between_gt_and_it',
            'color': (0, 255, 255),
        },
    ],

    'landmarks': [
        {
            'name': 'GREATER_TROCHANTER',
            'display': 'Trocanter Maior',
            'abbreviation': 'GT',
            'appearance': 'hyperechoic_bony',
        },
        {
            'name': 'ISCHIAL_TUBEROSITY',
            'display': 'Tuberosidade Isquiática',
            'abbreviation': 'IT',
            'appearance': 'hyperechoic_bony',
        },
        {
            'name': 'GLUTEUS_MAXIMUS',
            'display': 'M. Glúteo Máximo',
            'position': 'superficial',
        },
        {
            'name': 'QUADRATUS_FEMORIS',
            'display': 'M. Quadrado Femoral',
            'position': 'deep_to_sciatic',
        },
    ],

    'patient_position': 'lateral_or_prone',
    'dermatomes': ['L4', 'L5', 'S1', 'S2', 'S3'],
}
```

---

## 10. BLOQUEIO CIÁTICO - POPLÍTEO

### Indicações
- Cirurgias de tornozelo e pé
- Nível mais comum para bloqueio do ciático

### Anatomia Ultrassonográfica

```
ESTRUTURAS A DETECTAR (Fossa Poplítea):
┌─────────────────────────────────────────────────────────────┐
│                         Pele                                │
│                                                             │
│    ┌───────────────────────────────────────────────────┐    │
│    │         M. Bíceps Femoral (lateral)               │    │
│    └───────────────────────────────────────────────────┘    │
│                                                             │
│         ┌─────────────────┐                                 │
│         │   N. CIÁTICO    │  → Bifurcação ~5-7cm acima      │
│         │   (proximal)    │     do joelho                   │
│         └────────┬────────┘                                 │
│                  │                                          │
│         ┌────────┴────────┐                                 │
│         │                 │                                 │
│    ┌────┴────┐      ┌─────┴─────┐                           │
│    │N.TIBIAL │      │N.PERONEAL │                           │
│    │(medial) │      │ COMUM     │                           │
│    │         │      │ (lateral) │                           │
│    └─────────┘      └───────────┘                           │
│                                                             │
│    ┌───────────────────────────────────────────────────┐    │
│    │         A. Poplítea (profunda)                    │    │
│    └───────────────────────────────────────────────────┘    │
│    ┌───────────────────────────────────────────────────┐    │
│    │         V. Poplítea (mais profunda)               │    │
│    └───────────────────────────────────────────────────┘    │
│                                                             │
│    ORDEM (superficial → profundo):                          │
│    Nervo → Veia → Artéria                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Configuração para IA

```python
SCIATIC_POPLITEAL_BLOCK = {
    'id': 'SCIATIC_POPLITEAL',
    'name': 'Bloqueio Ciático Poplíteo',
    'name_en': 'Popliteal Sciatic Block',

    'probe': 'linear',
    'frequency_mhz': (8, 14),
    'depth_cm': (2, 5),

    'targets': [
        {
            'name': 'SCIATIC_NERVE',
            'display': 'N. Ciático',
            'appearance': 'hypoechoic_honeycomb',
            'position': 'superficial_to_vessels',
            'color': (0, 255, 255),
        },
        {
            'name': 'TIBIAL_NERVE',
            'display': 'N. Tibial',
            'appearance': 'hypoechoic',
            'position': 'medial_after_bifurcation',
            'color': (0, 255, 200),
        },
        {
            'name': 'COMMON_PERONEAL_NERVE',
            'display': 'N. Peroneal Comum',
            'appearance': 'hypoechoic_smaller',
            'position': 'lateral_superficial',
            'color': (0, 200, 255),
        },
    ],

    'landmarks': [
        {
            'name': 'POPLITEAL_ARTERY',
            'display': 'A. Poplítea',
            'appearance': 'anechoic_pulsatile',
            'position': 'deep_to_nerve',
            'color': (0, 0, 255),
        },
        {
            'name': 'POPLITEAL_VEIN',
            'display': 'V. Poplítea',
            'appearance': 'anechoic_compressible',
            'position': 'between_nerve_and_artery',
            'color': (255, 100, 100),
        },
        {
            'name': 'BICEPS_FEMORIS',
            'display': 'M. Bíceps Femoral',
            'position': 'lateral',
        },
        {
            'name': 'SEMIMEMBRANOSUS',
            'display': 'M. Semimembranoso',
            'position': 'medial',
        },
    ],

    'technique_note': 'Bloquear proximal à bifurcação (5-7cm acima do joelho)',
    'patient_position': 'prone_or_lateral',
    'dermatomes': ['L4', 'L5', 'S1', 'S2'],
}
```

---

## 11. BLOQUEIO DO CANAL ADUTOR (Safeno)

### Indicações
- Cirurgias de joelho (preserva quadríceps)
- Analgesia pós-artroplastia
- Alternativa motor-sparing ao femoral

### Configuração para IA

```python
ADDUCTOR_CANAL_BLOCK = {
    'id': 'ADDUCTOR_CANAL',
    'name': 'Bloqueio do Canal Adutor',
    'name_en': 'Adductor Canal Block',

    'probe': 'linear',
    'frequency_mhz': (10, 15),
    'depth_cm': (2, 4),

    'targets': [
        {
            'name': 'SAPHENOUS_NERVE',
            'display': 'N. Safeno',
            'appearance': 'small_hypoechoic',
            'position': 'anterolateral_to_artery',
            'color': (0, 255, 255),
        },
    ],

    'landmarks': [
        {
            'name': 'FEMORAL_ARTERY',
            'display': 'A. Femoral (no canal)',
            'appearance': 'anechoic_pulsatile',
            'key_landmark': True,
            'color': (0, 0, 255),
        },
        {
            'name': 'SARTORIUS',
            'display': 'M. Sartório',
            'appearance': 'muscle_triangular',
            'position': 'roof_of_canal',
            'key_landmark': True,
        },
        {
            'name': 'VASTUS_MEDIALIS',
            'display': 'M. Vasto Medial',
            'position': 'lateral_wall',
        },
        {
            'name': 'ADDUCTOR_LONGUS',
            'display': 'M. Adutor Longo',
            'position': 'medial_wall',
        },
    ],

    'anatomy_note': 'Canal entre sartório (teto), vasto medial (lateral) e adutores (medial)',
    'advantage': 'Preserva força do quadríceps vs. bloqueio femoral',
    'dermatomes': ['L3', 'L4'],
}
```

---

## 12. BLOQUEIO DO NERVO OBTURADOR

### Indicações
- Prevenção de reflexo adutor em RTU vesical
- Suplementação de analgesia de joelho

### Configuração para IA

```python
OBTURATOR_NERVE_BLOCK = {
    'id': 'OBTURATOR',
    'name': 'Bloqueio do Nervo Obturador',
    'name_en': 'Obturator Nerve Block',

    'probe': 'linear',
    'frequency_mhz': (8, 14),
    'depth_cm': (2, 5),

    'targets': [
        {
            'name': 'ANTERIOR_DIVISION',
            'display': 'Divisão Anterior',
            'position': 'between_AL_and_AB',
            'color': (0, 255, 255),
        },
        {
            'name': 'POSTERIOR_DIVISION',
            'display': 'Divisão Posterior',
            'position': 'between_AB_and_AM',
            'color': (0, 200, 255),
        },
    ],

    'landmarks': [
        {
            'name': 'ADDUCTOR_LONGUS',
            'display': 'M. Adutor Longo',
            'abbreviation': 'AL',
            'position': 'superficial',
        },
        {
            'name': 'ADDUCTOR_BREVIS',
            'display': 'M. Adutor Curto',
            'abbreviation': 'AB',
            'position': 'middle',
        },
        {
            'name': 'ADDUCTOR_MAGNUS',
            'display': 'M. Adutor Magno',
            'abbreviation': 'AM',
            'position': 'deep',
        },
        {
            'name': 'PECTINEUS',
            'display': 'M. Pectíneo',
            'position': 'lateral',
        },
    ],

    'mnemonic': 'ALABAma = AL, AB, AMagnus (anterior → posterior)',
    'technique': 'Duas injeções interfasciais',
}
```

---

## 13. BLOQUEIO DO TORNOZELO

### Indicações
- Cirurgias de pé e dedos
- 5 nervos a bloquear

### Configuração para IA

```python
ANKLE_BLOCK = {
    'id': 'ANKLE',
    'name': 'Bloqueio do Tornozelo',
    'name_en': 'Ankle Block',

    'description': '5 nervos terminais do membro inferior',

    'targets': [
        # NERVOS PROFUNDOS (US obrigatório)
        {
            'name': 'TIBIAL_NERVE',
            'display': 'N. Tibial',
            'appearance': 'hypoechoic_honeycomb',
            'position': 'posterior_to_medial_malleolus',
            'landmark': 'posterior_tibial_artery',
            'innervation': 'sole_and_heel',
            'deep': True,
            'color': (0, 255, 255),
        },
        {
            'name': 'DEEP_PERONEAL_NERVE',
            'display': 'N. Peroneal Profundo',
            'appearance': 'small_hypoechoic',
            'position': 'lateral_to_anterior_tibial_artery',
            'innervation': 'first_web_space',
            'deep': True,
            'color': (0, 255, 200),
        },
        # NERVOS SUPERFICIAIS (podem ser bloqueados subcutâneo)
        {
            'name': 'SUPERFICIAL_PERONEAL_NERVE',
            'display': 'N. Peroneal Superficial',
            'appearance': 'small_hypoechoic',
            'position': 'anterolateral_subcutaneous',
            'innervation': 'dorsum_foot',
            'deep': False,
            'color': (0, 200, 255),
        },
        {
            'name': 'SURAL_NERVE',
            'display': 'N. Sural',
            'appearance': 'small_near_vein',
            'position': 'near_small_saphenous_vein',
            'innervation': 'lateral_foot',
            'deep': False,
            'color': (100, 255, 200),
        },
        {
            'name': 'SAPHENOUS_NERVE',
            'display': 'N. Safeno',
            'appearance': 'small_subcutaneous',
            'position': 'anteromedial',
            'innervation': 'medial_ankle',
            'deep': False,
            'note': 'Pode ser omitido para antepé',
            'color': (200, 255, 100),
        },
    ],

    'landmarks': [
        {
            'name': 'POSTERIOR_TIBIAL_ARTERY',
            'display': 'A. Tibial Posterior',
            'use_doppler': True,
        },
        {
            'name': 'ANTERIOR_TIBIAL_ARTERY',
            'display': 'A. Tibial Anterior',
            'use_doppler': True,
        },
        {
            'name': 'SMALL_SAPHENOUS_VEIN',
            'display': 'V. Safena Parva',
            'near': 'SURAL_NERVE',
        },
        {
            'name': 'MEDIAL_MALLEOLUS',
            'display': 'Maléolo Medial',
        },
        {
            'name': 'LATERAL_MALLEOLUS',
            'display': 'Maléolo Lateral',
        },
    ],

    'note': 'Nervos superficiais podem ser bloqueados com wheal subcutâneo se não visualizados',
}
```

---

## 14. NERVO CUTÂNEO FEMORAL LATERAL

### Configuração para IA

```python
LATERAL_FEMORAL_CUTANEOUS_BLOCK = {
    'id': 'LFCN',
    'name': 'Bloqueio do N. Cutâneo Femoral Lateral',
    'name_en': 'Lateral Femoral Cutaneous Nerve Block',

    'probe': 'linear',
    'frequency_mhz': (10, 15),
    'depth_cm': (1, 3),

    'targets': [
        {
            'name': 'LFCN',
            'display': 'N. Cut. Femoral Lateral',
            'appearance': 'small_hypoechoic',
            'position': 'medial_to_ASIS_under_fascia',
            'color': (0, 255, 255),
        },
    ],

    'landmarks': [
        {
            'name': 'ASIS',
            'display': 'EIAS',
            'full_name': 'Espinha Ilíaca Ântero-Superior',
        },
        {
            'name': 'INGUINAL_LIGAMENT',
            'display': 'Ligamento Inguinal',
        },
        {
            'name': 'SARTORIUS',
            'display': 'M. Sartório',
        },
    ],

    'indication': 'Meralgia parestésica, suplementação PENG/FICB',
    'dermatome': 'L2-L3 (coxa lateral)',
}
```

---

# TRONCO E PAREDE ABDOMINAL

---

## 15. TAP BLOCK

### Indicações
- Cirurgias abdominais (laparotomia, cesárea)
- Analgesia da parede abdominal

### Anatomia Ultrassonográfica

```
ESTRUTURAS A DETECTAR (Parede Abdominal):
┌─────────────────────────────────────────────────────────────┐
│                         Pele                                │
│    ┌──────────────────────────────────────────────────┐     │
│    │         1. M. OBLÍQUO EXTERNO                    │     │
│    └──────────────────────────────────────────────────┘     │
│    ════════════════════════════════════════════════════     │
│    ┌──────────────────────────────────────────────────┐     │
│    │         2. M. OBLÍQUO INTERNO                    │     │
│    │            (geralmente o mais espesso)           │     │
│    └──────────────────────────────────────────────────┘     │
│    ════════════════════════════════════════════════════     │
│    ┌──────────────────────────────────────────────────┐     │
│    │         3. M. TRANSVERSO ABDOMINAL               │     │
│    │            (frequentemente o mais fino)          │     │
│    └──────────────────────────────────────────────────┘     │
│    ════════════════════════════════════════════════════     │
│                                                             │
│    ░░░░░░░░░░░░░░ PERITÔNIO ░░░░░░░░░░░░░░░░░░░░░░          │
│                                                             │
│    ALVO: Plano entre oblíquo interno e transverso           │
│          (plano TAP)                                        │
│                                                             │
│    VARIANTES:                                               │
│    - Subcostal: para cirurgias acima do umbigo              │
│    - Lateral: para cirurgias abaixo do umbigo               │
│    - Posterior: extensão para flanco                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Configuração para IA

```python
TAP_BLOCK = {
    'id': 'TAP',
    'name': 'Bloqueio TAP',
    'name_en': 'Transversus Abdominis Plane Block',

    'probe': 'linear',
    'frequency_mhz': (8, 14),
    'depth_cm': (2, 5),

    'variants': {
        'lateral': {
            'position': 'linha_axilar_média',
            'coverage': 'T10-L1',
            'indication': 'cirurgias abaixo umbigo',
        },
        'subcostal': {
            'position': 'margem_costal',
            'coverage': 'T6-T10',
            'indication': 'cirurgias acima umbigo',
        },
        'posterior': {
            'position': 'triângulo_de_Petit',
            'coverage': 'T9-L1',
            'advantage': 'maior duração',
        },
    },

    'targets': [
        {
            'name': 'TAP_PLANE',
            'display': 'Plano TAP',
            'position': 'between_IO_and_TA',
            'injection_site': True,
        },
    ],

    'landmarks': [
        {
            'name': 'EXTERNAL_OBLIQUE',
            'display': 'M. Oblíquo Externo',
            'abbreviation': 'OE',
            'position': 'superficial',
            'color': (180, 100, 100),
        },
        {
            'name': 'INTERNAL_OBLIQUE',
            'display': 'M. Oblíquo Interno',
            'abbreviation': 'OI',
            'position': 'middle',
            'note': 'Geralmente o mais espesso',
            'color': (160, 100, 100),
        },
        {
            'name': 'TRANSVERSUS_ABDOMINIS',
            'display': 'M. Transverso Abdominal',
            'abbreviation': 'TA',
            'position': 'deep',
            'note': 'Frequentemente o mais fino',
            'color': (140, 100, 100),
        },
        {
            'name': 'PERITONEUM',
            'display': 'Peritônio',
            'position': 'deep_to_TA',
            'warning': True,
        },
    ],

    'nerves_blocked': ['T7-L1 intercostais'],
}
```

---

## 16. QUADRATUS LUMBORUM BLOCK

### Indicações
- Cirurgias abdominais extensas
- Nefrectomia, cesárea
- Cobertura visceral + somática

### Configuração para IA

```python
QUADRATUS_LUMBORUM_BLOCK = {
    'id': 'QL',
    'name': 'Bloqueio do Quadrado Lombar',
    'name_en': 'Quadratus Lumborum Block',

    'probe': 'curvilinear',
    'frequency_mhz': (4, 8),
    'depth_cm': (4, 10),

    'variants': {
        'QL1_lateral': {
            'injection': 'lateral_to_QL',
            'description': 'Lateral ao músculo QL',
        },
        'QL2_posterior': {
            'injection': 'posterior_to_QL',
            'description': 'Entre QL e eretor da espinha',
            'location': 'triângulo lombar interfascial',
        },
        'QL3_anterior': {
            'injection': 'anterior_to_QL',
            'description': 'Entre QL e psoas',
            'approach': 'transmuscular',
        },
    },

    'targets': [
        {
            'name': 'QL_PLANE',
            'display': 'Plano QL',
            'varies_by_variant': True,
        },
    ],

    'landmarks': [
        {
            'name': 'QUADRATUS_LUMBORUM',
            'display': 'M. Quadrado Lombar',
            'abbreviation': 'QL',
            'key_landmark': True,
            'color': (150, 100, 100),
        },
        {
            'name': 'ERECTOR_SPINAE',
            'display': 'M. Eretor da Espinha',
            'abbreviation': 'ES',
            'position': 'posterior_to_QL',
            'color': (130, 100, 100),
        },
        {
            'name': 'PSOAS_MAJOR',
            'display': 'M. Psoas Maior',
            'position': 'anterior_to_QL',
            'color': (170, 100, 100),
        },
        {
            'name': 'L4_TRANSVERSE_PROCESS',
            'display': 'Processo Transverso L4',
            'appearance': 'hyperechoic_shadow',
            'key_landmark': True,
        },
        {
            'name': 'THORACOLUMBAR_FASCIA',
            'display': 'Fáscia Toracolombar',
            'abbreviation': 'TLF',
        },
    ],

    'shamrock_sign': {
        'stem': 'L4_transverse_process',
        'leaves': ['erector_spinae', 'QL', 'psoas'],
    },

    'advantage': 'Spread visceral e paravertebral (vs TAP)',
}
```

---

## 17. BLOQUEIO DA BAINHA DO RETO

### Configuração para IA

```python
RECTUS_SHEATH_BLOCK = {
    'id': 'RECTUS_SHEATH',
    'name': 'Bloqueio da Bainha do Reto',
    'name_en': 'Rectus Sheath Block',

    'probe': 'linear',
    'frequency_mhz': (10, 15),
    'depth_cm': (1, 4),

    'targets': [
        {
            'name': 'POSTERIOR_RECTUS_SHEATH',
            'display': 'Bainha Posterior do Reto',
            'injection_site': True,
            'position': 'between_rectus_and_posterior_sheath',
        },
    ],

    'landmarks': [
        {
            'name': 'RECTUS_ABDOMINIS',
            'display': 'M. Reto Abdominal',
            'key_landmark': True,
        },
        {
            'name': 'LINEA_ALBA',
            'display': 'Linha Alba',
            'position': 'midline',
        },
        {
            'name': 'POSTERIOR_SHEATH',
            'display': 'Bainha Posterior',
        },
    ],

    'indication': 'Incisões medianas (laparotomia, umbilical)',
    'dermatomes': ['T9', 'T10', 'T11'],
    'bilateral': True,
}
```

---

## 18. ERECTOR SPINAE BLOCK

### Configuração para IA

```python
ERECTOR_SPINAE_BLOCK = {
    'id': 'ESP',
    'name': 'Bloqueio Erector Spinae',
    'name_en': 'Erector Spinae Plane Block',

    'probe': 'linear_or_curvilinear',
    'frequency_mhz': (6, 12),
    'depth_cm': (2, 6),

    'levels': {
        'thoracic': {
            'range': 'T2-T12',
            'indication': 'Toracotomia, mastectomia, costelas',
        },
        'lumbar': {
            'range': 'L1-L4',
            'indication': 'Cirurgias lombares, quadril',
        },
    },

    'targets': [
        {
            'name': 'ESP_PLANE',
            'display': 'Plano ESP',
            'position': 'deep_to_erector_spinae',
            'injection_site': True,
        },
    ],

    'landmarks': [
        {
            'name': 'TRANSVERSE_PROCESS',
            'display': 'Processo Transverso',
            'appearance': 'hyperechoic_flat_shadow',
            'key_landmark': True,
        },
        {
            'name': 'ERECTOR_SPINAE',
            'display': 'M. Eretor da Espinha',
            'position': 'superficial_to_TP',
        },
        {
            'name': 'TRAPEZIUS',
            'display': 'M. Trapézio',
            'position': 'superficial',
            'only_at': 'thoracic_levels',
        },
        {
            'name': 'RHOMBOID',
            'display': 'M. Romboide',
            'position': 'superficial',
            'only_at': 'thoracic_levels',
        },
    ],

    'mechanism': 'Spread paravertebral, epidural e intercostal',
    'advantage': 'Mais seguro que paravertebral (longe da pleura)',
}
```

---

## 19. SERRATUS ANTERIOR BLOCK

### Configuração para IA

```python
SERRATUS_ANTERIOR_BLOCK = {
    'id': 'SAPB',
    'name': 'Bloqueio do Serrátil Anterior',
    'name_en': 'Serratus Anterior Plane Block',

    'probe': 'linear',
    'frequency_mhz': (8, 14),
    'depth_cm': (1, 4),

    'variants': {
        'superficial': {
            'position': 'between_latissimus_and_serratus',
            'spread': 'lateral_cutaneous_branches',
        },
        'deep': {
            'position': 'deep_to_serratus',
            'spread': 'intercostal_nerves',
        },
    },

    'targets': [
        {
            'name': 'SERRATUS_PLANE',
            'display': 'Plano do Serrátil',
        },
    ],

    'landmarks': [
        {
            'name': 'SERRATUS_ANTERIOR',
            'display': 'M. Serrátil Anterior',
            'key_landmark': True,
        },
        {
            'name': 'LATISSIMUS_DORSI',
            'display': 'M. Grande Dorsal',
            'position': 'superficial',
        },
        {
            'name': 'RIBS',
            'display': 'Costelas',
            'appearance': 'hyperechoic_shadow',
        },
        {
            'name': 'THORACODORSAL_ARTERY',
            'display': 'A. Toracodorsal',
            'use_doppler': True,
        },
    ],

    'indication': 'Fraturas costais, toracotomia lateral, mama',
    'coverage': 'T2-T9 (parede torácica lateral)',
}
```

---

## 20. PECS BLOCK

### Configuração para IA

```python
PECS_BLOCK = {
    'id': 'PECS',
    'name': 'Bloqueio PECS',
    'name_en': 'Pectoral Nerve Block',

    'types': {
        'PECS_I': {
            'plane': 'between_pectoralis_major_and_minor',
            'nerves': ['pectoral_nerves'],
            'indication': 'Implante de marca-passo, porta',
        },
        'PECS_II': {
            'plane': 'between_pectoralis_minor_and_serratus',
            'nerves': ['pectoral', 'intercostal', 'intercostobrachial'],
            'indication': 'Mastectomia, cirurgia de mama',
        },
    },

    'landmarks': [
        {
            'name': 'PECTORALIS_MAJOR',
            'display': 'M. Peitoral Maior',
        },
        {
            'name': 'PECTORALIS_MINOR',
            'display': 'M. Peitoral Menor',
        },
        {
            'name': 'SERRATUS_ANTERIOR',
            'display': 'M. Serrátil Anterior',
        },
        {
            'name': 'THORACOACROMIAL_ARTERY',
            'display': 'A. Toracoacromial',
            'use_doppler': True,
        },
    ],
}
```

---

## 21. BLOQUEIO PARAVERTEBRAL

### Configuração para IA

```python
PARAVERTEBRAL_BLOCK = {
    'id': 'PVB',
    'name': 'Bloqueio Paravertebral Torácico',
    'name_en': 'Thoracic Paravertebral Block',

    'probe': 'linear_or_curvilinear',
    'frequency_mhz': (6, 12),
    'depth_cm': (2, 5),

    'skill_level': 'ADVANCED',

    'targets': [
        {
            'name': 'PARAVERTEBRAL_SPACE',
            'display': 'Espaço Paravertebral',
            'position': 'between_TP_and_pleura',
        },
    ],

    'landmarks': [
        {
            'name': 'TRANSVERSE_PROCESS',
            'display': 'Processo Transverso',
            'key_landmark': True,
        },
        {
            'name': 'COSTOTRANSVERSE_LIGAMENT',
            'display': 'Ligamento Costotransverso',
        },
        {
            'name': 'PLEURA',
            'display': 'Pleura',
            'warning': True,
            'complication': 'Pneumotórax',
        },
    ],

    'spread': ['paravertebral', 'epidural', 'intercostal'],
    'indication': 'Toracotomia, mastectomia, herpes zoster',
}
```

---

## 22. BLOQUEIO INTERCOSTAL

### Configuração para IA

```python
INTERCOSTAL_BLOCK = {
    'id': 'INTERCOSTAL',
    'name': 'Bloqueio Intercostal',
    'name_en': 'Intercostal Nerve Block',

    'probe': 'linear',
    'frequency_mhz': (10, 15),
    'depth_cm': (1, 3),

    'targets': [
        {
            'name': 'INTERCOSTAL_NERVE',
            'display': 'N. Intercostal',
            'position': 'inferior_rib_groove',
        },
    ],

    'landmarks': [
        {
            'name': 'RIB',
            'display': 'Costela',
            'appearance': 'hyperechoic_shadow',
        },
        {
            'name': 'INTERCOSTAL_SPACE',
            'display': 'Espaço Intercostal',
        },
        {
            'name': 'PLEURA',
            'display': 'Pleura',
            'warning': True,
        },
    ],

    'mnemonic': 'VAN (Veia-Artéria-Nervo) no sulco costal',
    'indication': 'Fraturas costais, dor pós-toracotomia',
    'duration': '8-12 horas',
}
```

---

# CABEÇA E PESCOÇO

---

## 23. PLEXO CERVICAL SUPERFICIAL

### Configuração para IA

```python
SUPERFICIAL_CERVICAL_PLEXUS_BLOCK = {
    'id': 'SCP',
    'name': 'Bloqueio do Plexo Cervical Superficial',
    'name_en': 'Superficial Cervical Plexus Block',

    'probe': 'linear',
    'frequency_mhz': (10, 15),
    'depth_cm': (1, 2),

    'targets': [
        {
            'name': 'SCP',
            'display': 'Plexo Cervical Superficial',
            'position': 'posterior_border_SCM',
            'level': 'mid_SCM',
        },
    ],

    'landmarks': [
        {
            'name': 'STERNOCLEIDOMASTOID',
            'display': 'M. Esternocleidomastóideo',
            'abbreviation': 'ECM',
            'key_landmark': True,
        },
        {
            'name': 'EXTERNAL_JUGULAR',
            'display': 'V. Jugular Externa',
        },
    ],

    'branches': [
        'N. Auricular Maior',
        'N. Cervical Transverso',
        'N. Occipital Menor',
        'Nn. Supraclaviculares',
    ],

    'indication': 'Endarterectomia carotídea, tireoide, traqueostomia',
}
```

---

## 24. PLEXO CERVICAL PROFUNDO

### Configuração para IA

```python
DEEP_CERVICAL_PLEXUS_BLOCK = {
    'id': 'DCP',
    'name': 'Bloqueio do Plexo Cervical Profundo',
    'name_en': 'Deep Cervical Plexus Block',

    'skill_level': 'ADVANCED',
    'probe': 'linear',
    'frequency_mhz': (8, 14),
    'depth_cm': (2, 4),

    'targets': [
        {
            'name': 'C2_C4_ROOTS',
            'display': 'Raízes C2-C4',
            'position': 'deep_to_prevertebral_fascia',
        },
    ],

    'landmarks': [
        {
            'name': 'SCM',
            'display': 'M. Esternocleidomastóideo',
        },
        {
            'name': 'LEVATOR_SCAPULAE',
            'display': 'M. Elevador da Escápula',
        },
        {
            'name': 'CAROTID_ARTERY',
            'display': 'A. Carótida',
            'warning': True,
        },
        {
            'name': 'INTERNAL_JUGULAR',
            'display': 'V. Jugular Interna',
            'warning': True,
        },
    ],

    'risks': [
        'Injeção intratecal',
        'Injeção na artéria vertebral',
        'Paralisia do frênico',
    ],
}
```

---

## 25. GÂNGLIO ESTRELADO

### Configuração para IA

```python
STELLATE_GANGLION_BLOCK = {
    'id': 'SGB',
    'name': 'Bloqueio do Gânglio Estrelado',
    'name_en': 'Stellate Ganglion Block',

    'skill_level': 'ADVANCED',
    'probe': 'linear',
    'frequency_mhz': (8, 14),
    'depth_cm': (2, 4),

    'targets': [
        {
            'name': 'STELLATE_GANGLION',
            'display': 'Gânglio Estrelado',
            'position': 'anterolateral_to_longus_colli',
            'level': 'C6-C7',
        },
    ],

    'landmarks': [
        {
            'name': 'CHASSAIGNAC_TUBERCLE',
            'display': 'Tubérculo de Chassaignac',
            'description': 'Tubérculo anterior de C6',
            'key_landmark': True,
        },
        {
            'name': 'LONGUS_COLLI',
            'display': 'M. Longo do Pescoço',
        },
        {
            'name': 'CAROTID_ARTERY',
            'display': 'A. Carótida',
            'position': 'lateral',
            'warning': True,
        },
        {
            'name': 'VERTEBRAL_ARTERY',
            'display': 'A. Vertebral',
            'warning': True,
        },
        {
            'name': 'THYROID',
            'display': 'Tireoide',
            'position': 'medial',
        },
    ],

    'success_sign': 'Síndrome de Horner (ptose, miose, anidrose)',
    'indication': 'SDRC, hiperidrose, dor vascular',
}
```

---

# PELVE E PERÍNEO

---

## 26. BLOQUEIO DO NERVO PUDENDO

### Configuração para IA

```python
PUDENDAL_NERVE_BLOCK = {
    'id': 'PUDENDAL',
    'name': 'Bloqueio do Nervo Pudendo',
    'name_en': 'Pudendal Nerve Block',

    'probe': 'curvilinear',
    'frequency_mhz': (2, 5),
    'depth_cm': (5, 10),

    'approach': 'transgluteal',
    'patient_position': 'prone',

    'targets': [
        {
            'name': 'PUDENDAL_NERVE',
            'display': 'N. Pudendo',
            'position': 'between_SSL_and_STL',
        },
    ],

    'landmarks': [
        {
            'name': 'ISCHIAL_SPINE',
            'display': 'Espinha Isquiática',
            'key_landmark': True,
        },
        {
            'name': 'SACROSPINOUS_LIGAMENT',
            'display': 'Lig. Sacroespinhoso',
            'abbreviation': 'SSL',
        },
        {
            'name': 'SACROTUBEROUS_LIGAMENT',
            'display': 'Lig. Sacrotuberal',
            'abbreviation': 'STL',
        },
        {
            'name': 'INTERNAL_PUDENDAL_ARTERY',
            'display': 'A. Pudenda Interna',
            'use_doppler': True,
        },
    ],

    'indication': 'Neuralgia pudenda, cirurgia perineal, hemorroidectomia',
    'innervation': 'Períneo, genitália externa, esfíncter anal',
}
```

---

## 27. IPACK BLOCK

### Configuração para IA

```python
IPACK_BLOCK = {
    'id': 'IPACK',
    'name': 'Bloqueio IPACK',
    'name_en': 'Infiltration between Popliteal Artery and Capsule of Knee',

    'probe': 'linear_or_curvilinear',
    'frequency_mhz': (6, 12),
    'depth_cm': (3, 6),

    'targets': [
        {
            'name': 'IPACK_SPACE',
            'display': 'Espaço IPACK',
            'position': 'between_popliteal_artery_and_femur',
        },
    ],

    'landmarks': [
        {
            'name': 'POPLITEAL_ARTERY',
            'display': 'A. Poplítea',
            'key_landmark': True,
        },
        {
            'name': 'FEMORAL_CONDYLES',
            'display': 'Côndilos Femorais',
        },
        {
            'name': 'POSTERIOR_CAPSULE',
            'display': 'Cápsula Posterior',
        },
    ],

    'nerves_blocked': [
        'Ramos articulares posteriores do joelho',
        'Ramo poplíteo do n. obturador',
    ],

    'indication': 'Artroplastia total de joelho, LCA',
    'advantage': 'Analgesia posterior sem bloqueio motor',
}
```

---

## 28. BLOQUEIO GENICULAR

### Configuração para IA

```python
GENICULAR_NERVE_BLOCK = {
    'id': 'GENICULAR',
    'name': 'Bloqueio dos Nervos Geniculares',
    'name_en': 'Genicular Nerve Block',

    'probe': 'linear',
    'frequency_mhz': (10, 15),
    'depth_cm': (1, 3),

    'targets': [
        {
            'name': 'SUPEROMEDIAL_GENICULAR',
            'display': 'N. Genicular Superomedial',
            'position': 'junction_shaft_condyle_medial',
        },
        {
            'name': 'SUPEROLATERAL_GENICULAR',
            'display': 'N. Genicular Superolateral',
            'position': 'junction_shaft_condyle_lateral',
        },
        {
            'name': 'INFEROMEDIAL_GENICULAR',
            'display': 'N. Genicular Inferomedial',
            'position': 'below_tibial_plateau_medial',
        },
    ],

    'landmarks': [
        {
            'name': 'ADDUCTOR_TUBERCLE',
            'display': 'Tubérculo Adutor',
        },
        {
            'name': 'FEMORAL_CONDYLES',
            'display': 'Côndilos Femorais',
        },
        {
            'name': 'TIBIAL_PLATEAU',
            'display': 'Platô Tibial',
        },
    ],

    'indication': 'Dor crônica de joelho, osteoartrite',
    'can_use_RF': True,  # Ablação por radiofrequência
}
```

---

# IMPLEMENTAÇÃO NO SISTEMA

## Estrutura de Dados Principal

```python
# src/block_database.py

ALL_NERVE_BLOCKS = {
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

def get_block_config(block_id: str) -> dict:
    """Retorna configuração de um bloqueio específico"""
    return ALL_NERVE_BLOCKS.get(block_id)

def get_blocks_by_region(region: str) -> list:
    """Retorna todos os bloqueios de uma região"""
    regions = {
        'upper_limb': ['INTERSCALENE', 'SUPRACLAVICULAR', 'INFRACLAVICULAR', 'AXILLARY', 'WRIST'],
        'lower_limb': ['FEMORAL', 'FASCIA_ILIACA', 'PENG', 'SCIATIC_SUBGLUTEAL', 'SCIATIC_POPLITEAL',
                       'ADDUCTOR_CANAL', 'OBTURATOR', 'ANKLE', 'LFCN'],
        'trunk': ['TAP', 'QL', 'RECTUS_SHEATH', 'ESP', 'SAPB', 'PECS', 'PVB', 'INTERCOSTAL'],
        'head_neck': ['SCP', 'DCP', 'SGB'],
        'pelvis': ['PUDENDAL', 'IPACK', 'GENICULAR'],
    }
    return [ALL_NERVE_BLOCKS[bid] for bid in regions.get(region, [])]

def get_structures_to_detect(block_id: str) -> list:
    """Retorna todas as estruturas que a IA deve detectar para um bloqueio"""
    block = ALL_NERVE_BLOCKS.get(block_id)
    if not block:
        return []

    structures = []
    structures.extend(block.get('targets', []))
    structures.extend(block.get('landmarks', []))

    return structures
```

---

## TOTAL: 28 BLOQUEIOS DOCUMENTADOS

| Categoria | Quantidade | Bloqueios |
|-----------|------------|-----------|
| Membro Superior | 5 | Interescalênico, Supraclavicular, Infraclavicular, Axilar, Pulso |
| Membro Inferior | 9 | Femoral, Fascia Ilíaca, PENG, Ciático Subglúteo, Ciático Poplíteo, Canal Adutor, Obturador, Tornozelo, LFCN |
| Tronco | 8 | TAP, QL, Rectus Sheath, ESP, SAPB, PECS, Paravertebral, Intercostal |
| Cabeça/Pescoço | 3 | Plexo Cervical Superficial, Profundo, Gânglio Estrelado |
| Pelve | 3 | Pudendo, IPACK, Genicular |

---

## REFERÊNCIAS

- [NYSORA - Regional Anesthesia](https://www.nysora.com/)
- [ASRA - American Society of Regional Anesthesia](https://www.asra.com/)
- [BJA Education](https://www.bjaed.org/)
- [StatPearls](https://www.ncbi.nlm.nih.gov/books/)

---

*Documento criado em: 2025-12-22*
*Para uso no NERVE TRACK v2.0 PREMIUM*
*Baseado em pesquisa extensiva de literatura médica e comercial*
