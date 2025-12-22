#!/usr/bin/env python3
"""
USG FLOW - Unified Dataset Manager
===================================
Sistema UNIFICADO de datasets para TODOS os plugins de IA.

Estratégia:
- Um grande dataset com imagens de ultrassom de várias fontes
- Múltiplas anotações por imagem (needle, segmentation, etc)
- Modelos podem ser treinados conjuntamente (transfer learning)

Plugins Suportados:
- NEEDLE: Detecção de ponta de agulha
- NERVE: Segmentação de nervos/vasos/músculos
- CARDIAC: Segmentação cardíaca
- FAST: Detecção de líquido livre
- ANATOMY: Identificação de estruturas
- LUNG: Detecção de linhas B
- BLADDER: Volume vesical
"""

import os
import sys
import json
import hashlib
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

try:
    import cv2
except ImportError:
    os.system("pip install opencv-python")
    import cv2

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

BASE_DIR = Path(__file__).parent
UNIFIED_DIR = BASE_DIR / "unified"
RAW_DIR = UNIFIED_DIR / "raw"
PROCESSED_DIR = UNIFIED_DIR / "processed"
EXPORTS_DIR = UNIFIED_DIR / "exports"

# Criar diretórios
for d in [UNIFIED_DIR, RAW_DIR, PROCESSED_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ESTRUTURAS DE DADOS
# =============================================================================

@dataclass
class Annotation:
    """Anotação genérica para uma imagem"""
    type: str  # "point", "bbox", "mask", "polygon", "class"
    data: Any  # Dados da anotação
    confidence: float = 1.0
    metadata: Dict = None


@dataclass
class ImageSample:
    """Uma amostra de imagem com múltiplas anotações"""
    id: str
    source: str
    modality: str  # "bmode", "doppler", "mmode"
    anatomy: str   # "neck", "arm", "abdomen", "cardiac", "lung"
    image_path: str
    width: int
    height: int
    annotations: Dict[str, Annotation]  # plugin -> annotation


@dataclass
class DatasetInfo:
    """Informações sobre um dataset"""
    name: str
    source_url: str
    n_samples: int
    annotations_available: List[str]
    description: str


# =============================================================================
# CATÁLOGO DE DATASETS PÚBLICOS
# =============================================================================

DATASET_CATALOG = {
    # =========================================================================
    # DATASETS PARA NEEDLE (detecção de agulha)
    # =========================================================================
    "kaggle_nerve": {
        "name": "Kaggle Ultrasound Nerve Segmentation",
        "url": "https://www.kaggle.com/c/ultrasound-nerve-segmentation",
        "kaggle_id": "ultrasound-nerve-segmentation",
        "type": "kaggle_competition",
        "samples": 5635,
        "anatomy": "neck",
        "annotations": ["nerve_mask"],
        "useful_for": ["NERVE", "NEEDLE"],  # Pode ser adaptado
        "description": "Imagens de plexo braquial - útil para nervos e contexto de needle"
    },

    "brachial_plexus": {
        "name": "Regional-US Brachial Plexus",
        "url": "https://github.com/Regional-US/brachial_plexus",
        "type": "github",
        "samples": 41000,
        "anatomy": "neck",
        "annotations": ["needle_tip", "nerve_mask"],
        "useful_for": ["NEEDLE", "NERVE"],
        "description": "41k frames com anotações de agulha - MELHOR para needle!"
    },

    # =========================================================================
    # DATASETS PARA NERVE (segmentação anatômica)
    # =========================================================================
    "musv_vascular": {
        "name": "Mus-V Multimodal Ultrasound Vascular",
        "url": "https://www.kaggle.com/datasets/among22/multimodal-ultrasound-vascular-segmentation",
        "kaggle_id": "among22/multimodal-ultrasound-vascular-segmentation",
        "type": "kaggle_dataset",
        "samples": 1000,
        "anatomy": "vascular",
        "annotations": ["artery_mask", "vein_mask"],
        "useful_for": ["NERVE"],
        "description": "Segmentação de artérias e veias"
    },

    "neck_anatomy": {
        "name": "Axial Neck Anatomy Dataset",
        "url": "http://e-space.mmu.ac.uk/624643/",
        "type": "academic",
        "samples": 2172,
        "anatomy": "neck",
        "annotations": ["multi_class_mask"],
        "useful_for": ["NERVE", "ANATOMY"],
        "description": "13 classes anatômicas do pescoço - EXCELENTE!"
    },

    "musculoskeletal": {
        "name": "Transverse Musculoskeletal Ultrasound",
        "url": "http://www.mdpi.com/2313-433X/4/2/29",
        "type": "academic",
        "samples": 3917,
        "anatomy": "limbs",
        "annotations": ["muscle_mask"],
        "useful_for": ["NERVE", "ANATOMY"],
        "description": "Músculos dos membros"
    },

    # =========================================================================
    # DATASETS PARA CARDIAC
    # =========================================================================
    "camus": {
        "name": "CAMUS Cardiac Dataset",
        "url": "https://www.creatis.insa-lyon.fr/Challenge/camus/",
        "type": "academic",
        "samples": 4000,
        "anatomy": "cardiac",
        "annotations": ["lv_mask", "la_mask", "myocardium_mask"],
        "useful_for": ["CARDIAC"],
        "description": "500 pacientes com segmentação cardíaca"
    },

    "echonet": {
        "name": "EchoNet-Dynamic",
        "url": "https://echonet.github.io/dynamic/",
        "type": "academic",
        "samples": 10000,
        "anatomy": "cardiac",
        "annotations": ["ef_value", "lv_mask"],
        "useful_for": ["CARDIAC"],
        "description": "10k vídeos com fração de ejeção"
    },

    # =========================================================================
    # DATASETS PARA FAST/ABDOMEN
    # =========================================================================
    "us_simulation": {
        "name": "US Simulation & Segmentation",
        "url": "https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm",
        "kaggle_id": "ignaciorlando/ussimandsegm",
        "type": "kaggle_dataset",
        "samples": 500,
        "anatomy": "abdomen",
        "annotations": ["organ_mask"],
        "useful_for": ["FAST", "ANATOMY"],
        "description": "Ultrassom abdominal com segmentação"
    },

    # =========================================================================
    # DATASETS PARA LUNG
    # =========================================================================
    "lung_ultrasound": {
        "name": "Lung Ultrasound COVID-19",
        "url": "https://github.com/jannisborn/covid19_ultrasound",
        "type": "github",
        "samples": 2000,
        "anatomy": "lung",
        "annotations": ["pathology_class", "blines_count"],
        "useful_for": ["LUNG"],
        "description": "Ultrassom pulmonar com classificação"
    },

    # =========================================================================
    # DATASETS GERAIS (MULTI-USO)
    # =========================================================================
    "breast_us": {
        "name": "Breast Ultrasound Images",
        "url": "https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset",
        "kaggle_id": "aryashah2k/breast-ultrasound-images-dataset",
        "type": "kaggle_dataset",
        "samples": 780,
        "anatomy": "breast",
        "annotations": ["lesion_mask", "class"],
        "useful_for": ["ANATOMY"],
        "description": "Ultrassom de mama com segmentação"
    },

    "thyroid_us": {
        "name": "DDTI Thyroid Ultrasound",
        "url": "https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images",
        "kaggle_id": "dasmehdixtr/ddti-thyroid-ultrasound-images",
        "type": "kaggle_dataset",
        "samples": 400,
        "anatomy": "neck",
        "annotations": ["nodule_mask"],
        "useful_for": ["ANATOMY", "NERVE"],
        "description": "Tireoide - contexto para pescoço"
    },

    "fetal_us": {
        "name": "Fetal Ultrasound Dataset",
        "url": "https://zenodo.org/record/3904280",
        "type": "zenodo",
        "samples": 14000,
        "anatomy": "obstetric",
        "annotations": ["structure_mask"],
        "useful_for": ["ANATOMY"],
        "description": "Ultrassom fetal"
    },
}


# =============================================================================
# FUNÇÕES DE DOWNLOAD
# =============================================================================

def check_kaggle_api() -> bool:
    """Verifica se a API do Kaggle está configurada"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def download_kaggle_competition(competition_id: str, dest_dir: Path) -> bool:
    """Baixa dataset de competição do Kaggle"""
    if not check_kaggle_api():
        return False

    try:
        import kaggle
        dest_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.competition_download_files(competition_id, path=str(dest_dir))

        # Extrair ZIP
        for zip_file in dest_dir.glob("*.zip"):
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(dest_dir)
            zip_file.unlink()

        return True
    except Exception as e:
        print(f"❌ Erro Kaggle: {e}")
        return False


def download_kaggle_dataset(dataset_id: str, dest_dir: Path) -> bool:
    """Baixa dataset público do Kaggle"""
    if not check_kaggle_api():
        return False

    try:
        import kaggle
        dest_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(dataset_id, path=str(dest_dir), unzip=True)
        return True
    except Exception as e:
        print(f"❌ Erro Kaggle: {e}")
        return False


def download_dataset(dataset_key: str) -> Optional[Path]:
    """Baixa um dataset do catálogo"""
    if dataset_key not in DATASET_CATALOG:
        print(f"❌ Dataset '{dataset_key}' não encontrado no catálogo")
        return None

    info = DATASET_CATALOG[dataset_key]
    dest_dir = RAW_DIR / dataset_key

    print(f"\n📥 Baixando: {info['name']}")
    print(f"   Fonte: {info['url']}")
    print(f"   Amostras: ~{info['samples']:,}")
    print(f"   Útil para: {', '.join(info['useful_for'])}")

    if dest_dir.exists() and any(dest_dir.rglob("*")):
        print("✅ Dataset já existe!")
        return dest_dir

    if info["type"] == "kaggle_competition":
        if download_kaggle_competition(info["kaggle_id"], dest_dir):
            print("✅ Download completo!")
            return dest_dir

    elif info["type"] == "kaggle_dataset":
        if download_kaggle_dataset(info["kaggle_id"], dest_dir):
            print("✅ Download completo!")
            return dest_dir

    elif info["type"] in ["github", "academic", "zenodo"]:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  DOWNLOAD MANUAL NECESSÁRIO                                  ║
╠══════════════════════════════════════════════════════════════╣
║  1. Acesse: {info['url'][:50]}...
║  2. Baixe o dataset
║  3. Extraia para: {str(dest_dir)[:40]}...
╚══════════════════════════════════════════════════════════════╝
        """)
        dest_dir.mkdir(parents=True, exist_ok=True)
        return None

    return None


# =============================================================================
# PROCESSAMENTO UNIFICADO
# =============================================================================

def process_for_plugin(dataset_key: str, target_plugin: str) -> Optional[Path]:
    """
    Processa um dataset para uso com um plugin específico

    Args:
        dataset_key: Chave do dataset no catálogo
        target_plugin: Plugin alvo (NEEDLE, NERVE, CARDIAC, etc)

    Returns:
        Caminho do dataset processado ou None
    """
    if dataset_key not in DATASET_CATALOG:
        return None

    info = DATASET_CATALOG[dataset_key]
    raw_dir = RAW_DIR / dataset_key
    output_dir = PROCESSED_DIR / target_plugin.lower() / dataset_key

    if not raw_dir.exists():
        print(f"❌ Dataset raw não encontrado: {raw_dir}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔄 Processando {info['name']} para {target_plugin}...")

    # Processar baseado no dataset e plugin
    if dataset_key == "kaggle_nerve":
        return _process_kaggle_nerve(raw_dir, output_dir, target_plugin)

    elif dataset_key == "musv_vascular":
        return _process_musv(raw_dir, output_dir, target_plugin)

    elif dataset_key == "camus":
        return _process_camus(raw_dir, output_dir, target_plugin)

    elif dataset_key in ["breast_us", "thyroid_us"]:
        return _process_generic_segmentation(raw_dir, output_dir, target_plugin)

    else:
        return _process_generic(raw_dir, output_dir, target_plugin)


def _process_kaggle_nerve(raw_dir: Path, output_dir: Path, plugin: str) -> Path:
    """Processa Kaggle Nerve para NEEDLE ou NERVE"""
    train_dir = raw_dir / "train"
    if not train_dir.exists():
        for subdir in raw_dir.iterdir():
            if (subdir / "train").exists():
                train_dir = subdir / "train"
                break

    images = []
    labels = []

    image_files = sorted([f for f in train_dir.glob("*.tif") if "_mask" not in f.stem])

    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        mask_path = train_dir / f"{img_path.stem}_mask.tif"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            continue

        # Redimensionar
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        if plugin == "NEEDLE":
            # Para NEEDLE: extrair centroide do nervo como ponto de referência
            if mask.max() > 0:
                moments = cv2.moments(mask)
                if moments["m00"] > 0:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]
                    images.append(img)
                    labels.append([cy / 256.0, cx / 256.0])

        elif plugin == "NERVE":
            # Para NERVE: usar máscara de segmentação
            mask_binary = (mask > 127).astype(np.uint8)
            images.append(img)
            labels.append(mask_binary)

    images = np.array(images, dtype=np.uint8)
    images = np.expand_dims(images, axis=-1)

    if plugin == "NEEDLE":
        labels = np.array(labels, dtype=np.float32)
        np.save(output_dir / "images.npy", images)
        np.save(output_dir / "labels.npy", labels)
    else:
        labels = np.array(labels, dtype=np.uint8)
        np.save(output_dir / "images.npy", images)
        np.save(output_dir / "masks.npy", labels)

    print(f"✅ Processado: {len(images)} amostras para {plugin}")
    return output_dir


def _process_musv(raw_dir: Path, output_dir: Path, plugin: str) -> Path:
    """Processa Mus-V Vascular"""
    images = []
    masks = []

    # Encontrar diretórios de imagem e máscara
    for img_path in raw_dir.rglob("*.png"):
        if "mask" in str(img_path).lower():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Procurar máscara
        mask = np.zeros_like(img)
        for mask_pattern in ["_mask", "_label", "_seg"]:
            mask_path = img_path.parent / f"{img_path.stem}{mask_pattern}{img_path.suffix}"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                break

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Converter para classes: 2=artéria, 3=veia
        mask_classes = np.zeros_like(mask)
        mask_classes[mask > 170] = 2
        mask_classes[(mask > 80) & (mask <= 170)] = 3

        images.append(img)
        masks.append(mask_classes)

    if len(images) > 0:
        images = np.array(images, dtype=np.uint8)
        images = np.expand_dims(images, axis=-1)
        masks = np.array(masks, dtype=np.uint8)

        np.save(output_dir / "images.npy", images)
        np.save(output_dir / "masks.npy", masks)

        print(f"✅ Processado: {len(images)} amostras")

    return output_dir


def _process_camus(raw_dir: Path, output_dir: Path, plugin: str) -> Path:
    """Processa CAMUS Cardiac"""
    try:
        import SimpleITK as sitk
    except ImportError:
        os.system("pip install SimpleITK")
        import SimpleITK as sitk

    images = []
    masks = []

    training_dir = raw_dir / "training"
    if not training_dir.exists():
        for subdir in raw_dir.rglob("training"):
            training_dir = subdir
            break

    if not training_dir.exists():
        print(f"⚠️ Diretório training não encontrado em {raw_dir}")
        return output_dir

    for patient_dir in sorted(training_dir.glob("patient*")):
        for view in ["2CH", "4CH"]:
            for phase in ["ED", "ES"]:
                img_file = patient_dir / f"{patient_dir.name}_{view}_{phase}.mhd"
                mask_file = patient_dir / f"{patient_dir.name}_{view}_{phase}_gt.mhd"

                if img_file.exists():
                    try:
                        img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_file)))
                        if mask_file.exists():
                            mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_file)))
                        else:
                            mask = np.zeros_like(img)

                        for i in range(img.shape[0]):
                            frame = cv2.resize(img[i].astype(np.uint8), (256, 256))
                            frame_mask = cv2.resize(mask[i].astype(np.uint8), (256, 256),
                                                   interpolation=cv2.INTER_NEAREST)
                            images.append(frame)
                            masks.append(frame_mask)
                    except Exception as e:
                        print(f"   Erro em {img_file.name}: {e}")

    if len(images) > 0:
        images = np.array(images, dtype=np.uint8)
        images = np.expand_dims(images, axis=-1)
        masks = np.array(masks, dtype=np.uint8)

        np.save(output_dir / "images.npy", images)
        np.save(output_dir / "masks.npy", masks)

        print(f"✅ Processado: {len(images)} frames cardíacos")

    return output_dir


def _process_generic_segmentation(raw_dir: Path, output_dir: Path, plugin: str) -> Path:
    """Processamento genérico para datasets de segmentação"""
    images = []
    masks = []

    for img_path in raw_dir.rglob("*.png"):
        if "mask" in str(img_path).lower():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Procurar máscara
        mask = None
        for pattern in ["_mask", "_seg", "_label", "_annotation"]:
            mask_path = img_path.parent / f"{img_path.stem}{pattern}{img_path.suffix}"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                break

        if mask is None:
            mask = np.zeros_like(img)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        images.append(img)
        masks.append((mask > 127).astype(np.uint8))

    if len(images) > 0:
        images = np.array(images, dtype=np.uint8)
        images = np.expand_dims(images, axis=-1)
        masks = np.array(masks, dtype=np.uint8)

        np.save(output_dir / "images.npy", images)
        np.save(output_dir / "masks.npy", masks)

        print(f"✅ Processado: {len(images)} imagens")

    return output_dir


def _process_generic(raw_dir: Path, output_dir: Path, plugin: str) -> Path:
    """Processamento genérico para qualquer dataset"""
    return _process_generic_segmentation(raw_dir, output_dir, plugin)


# =============================================================================
# GERAÇÃO DE DADOS SINTÉTICOS
# =============================================================================

def generate_synthetic_data(plugin: str, n_samples: int = 5000) -> Path:
    """Gera dados sintéticos para um plugin específico"""
    output_dir = PROCESSED_DIR / plugin.lower() / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔧 Gerando {n_samples} amostras sintéticas para {plugin}...")

    if plugin == "NEEDLE":
        return _generate_needle_synthetic(output_dir, n_samples)
    elif plugin == "NERVE":
        return _generate_nerve_synthetic(output_dir, n_samples)
    elif plugin == "CARDIAC":
        return _generate_cardiac_synthetic(output_dir, n_samples)
    else:
        return _generate_generic_synthetic(output_dir, n_samples)


def _generate_needle_synthetic(output_dir: Path, n: int) -> Path:
    """Gera dados sintéticos para detecção de agulha"""
    images = []
    labels = []

    for i in range(n):
        if (i + 1) % 1000 == 0:
            print(f"   Progresso: {i + 1}/{n}")

        # Base com textura de ultrassom
        img = np.random.randint(30, 70, (256, 256), dtype=np.uint8)
        img = (img * np.random.exponential(1.0, (256, 256))).clip(0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 1.5)

        # Adicionar estruturas de fundo
        for _ in range(np.random.randint(2, 5)):
            y = np.random.randint(50, 200)
            cv2.rectangle(img, (0, y), (256, y + np.random.randint(10, 30)),
                         np.random.randint(80, 150), -1)

        img = cv2.GaussianBlur(img, (7, 7), 2)

        # Agulha
        entry_x = np.random.randint(30, 150)
        entry_y = np.random.randint(5, 30)
        angle = np.radians(np.random.uniform(20, 70))
        length = np.random.randint(100, 200)

        tip_x = int(entry_x + length * np.cos(angle))
        tip_y = int(entry_y + length * np.sin(angle))
        tip_x = np.clip(tip_x, 10, 245)
        tip_y = np.clip(tip_y, 10, 245)

        cv2.line(img, (entry_x, entry_y), (tip_x, tip_y),
                np.random.randint(200, 255), np.random.randint(1, 3))

        # Ruído
        img = np.clip(img + np.random.normal(0, 8, img.shape), 0, 255).astype(np.uint8)

        images.append(img)
        labels.append([tip_y / 256.0, tip_x / 256.0])

    images = np.array(images, dtype=np.uint8)
    images = np.expand_dims(images, axis=-1)
    labels = np.array(labels, dtype=np.float32)

    np.save(output_dir / "images.npy", images)
    np.save(output_dir / "labels.npy", labels)

    print(f"✅ Gerado: {n} amostras para NEEDLE")
    return output_dir


def _generate_nerve_synthetic(output_dir: Path, n: int) -> Path:
    """Gera dados sintéticos para segmentação de estruturas"""
    images = []
    masks = []

    for i in range(n):
        if (i + 1) % 1000 == 0:
            print(f"   Progresso: {i + 1}/{n}")

        # Base
        img = np.random.randint(30, 60, (256, 256), dtype=np.uint8)
        img = (img * np.random.exponential(1.0, (256, 256))).clip(0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 1.5)

        mask = np.zeros((256, 256), dtype=np.uint8)

        # Músculo (classe 4)
        for _ in range(np.random.randint(1, 3)):
            y = np.random.randint(30, 200)
            h = np.random.randint(20, 50)
            cv2.rectangle(img, (0, y), (256, y + h), np.random.randint(60, 100), -1)
            mask[y:y+h, :] = 4

        img = cv2.GaussianBlur(img, (5, 5), 1.5)

        # Nervo (classe 1)
        if np.random.random() > 0.2:
            cx, cy = np.random.randint(50, 200), np.random.randint(60, 190)
            rx, ry = np.random.randint(12, 30), np.random.randint(8, 20)
            cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, np.random.randint(150, 200), -1)
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)

        # Artéria (classe 2)
        if np.random.random() > 0.3:
            cx, cy = np.random.randint(40, 210), np.random.randint(50, 200)
            r = np.random.randint(8, 16)
            cv2.circle(img, (cx, cy), r, np.random.randint(20, 50), -1)
            cv2.circle(img, (cx, cy), r, np.random.randint(160, 200), 2)
            cv2.circle(mask, (cx, cy), r, 2, -1)

        # Veia (classe 3)
        if np.random.random() > 0.3:
            cx, cy = np.random.randint(40, 210), np.random.randint(50, 200)
            rx, ry = np.random.randint(10, 22), np.random.randint(5, 12)
            cv2.ellipse(img, (cx, cy), (rx, ry), np.random.randint(-20, 20),
                       0, 360, np.random.randint(15, 35), -1)
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 3, -1)

        # Ruído
        img = np.clip(img + np.random.normal(0, 6, img.shape), 0, 255).astype(np.uint8)

        images.append(img)
        masks.append(mask)

    images = np.array(images, dtype=np.uint8)
    images = np.expand_dims(images, axis=-1)
    masks = np.array(masks, dtype=np.uint8)

    np.save(output_dir / "images.npy", images)
    np.save(output_dir / "masks.npy", masks)

    print(f"✅ Gerado: {n} amostras para NERVE")
    return output_dir


def _generate_cardiac_synthetic(output_dir: Path, n: int) -> Path:
    """Gera dados sintéticos para cardiac"""
    images = []
    masks = []

    for i in range(n):
        if (i + 1) % 1000 == 0:
            print(f"   Progresso: {i + 1}/{n}")

        # Base escura (cavity)
        img = np.random.randint(10, 30, (256, 256), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)

        # Ventrículo esquerdo (LV)
        cx, cy = 128 + np.random.randint(-20, 20), 140 + np.random.randint(-20, 20)
        rx, ry = np.random.randint(40, 60), np.random.randint(50, 70)
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, np.random.randint(5, 20), -1)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)  # LV

        # Miocárdio
        cv2.ellipse(img, (cx, cy), (rx + 15, ry + 15), 0, 0, 360,
                   np.random.randint(100, 150), 15)
        cv2.ellipse(mask, (cx, cy), (rx + 15, ry + 15), 0, 0, 360, 2, 15)  # Myocardium

        # Átrio
        cv2.ellipse(img, (cx, cy - ry - 20), (rx - 10, 25), 0, 0, 360,
                   np.random.randint(5, 20), -1)
        cv2.ellipse(mask, (cx, cy - ry - 20), (rx - 10, 25), 0, 0, 360, 3, -1)  # LA

        # Textura
        img = cv2.GaussianBlur(img, (5, 5), 1.5)
        noise = np.random.exponential(1.0, (256, 256)) * 20
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        images.append(img)
        masks.append(mask)

    images = np.array(images, dtype=np.uint8)
    images = np.expand_dims(images, axis=-1)
    masks = np.array(masks, dtype=np.uint8)

    np.save(output_dir / "images.npy", images)
    np.save(output_dir / "masks.npy", masks)

    print(f"✅ Gerado: {n} amostras para CARDIAC")
    return output_dir


def _generate_generic_synthetic(output_dir: Path, n: int) -> Path:
    """Gera dados sintéticos genéricos"""
    return _generate_nerve_synthetic(output_dir, n)


# =============================================================================
# COMBINAÇÃO E EXPORTAÇÃO
# =============================================================================

def combine_for_plugin(plugin: str) -> Path:
    """Combina todos os datasets processados para um plugin"""
    plugin_dir = PROCESSED_DIR / plugin.lower()

    if not plugin_dir.exists():
        print(f"❌ Nenhum dataset processado para {plugin}")
        return None

    all_images = []
    all_labels = []
    sources = []

    # Encontrar todos os datasets
    for dataset_dir in plugin_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        images_file = dataset_dir / "images.npy"
        labels_file = dataset_dir / "labels.npy"
        masks_file = dataset_dir / "masks.npy"

        if images_file.exists():
            images = np.load(images_file)
            all_images.append(images)

            if labels_file.exists():
                labels = np.load(labels_file)
                all_labels.append(labels)
            elif masks_file.exists():
                masks = np.load(masks_file)
                all_labels.append(masks)

            sources.append((dataset_dir.name, len(images)))
            print(f"   + {dataset_dir.name}: {len(images)} amostras")

    if len(all_images) == 0:
        print(f"❌ Nenhum dado encontrado para {plugin}")
        return None

    # Combinar
    combined_images = np.concatenate(all_images, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    # Shuffle
    indices = np.random.permutation(len(combined_images))
    combined_images = combined_images[indices]
    combined_labels = combined_labels[indices]

    # Dividir
    n = len(combined_images)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    # Salvar
    export_dir = EXPORTS_DIR / plugin.lower()
    export_dir.mkdir(parents=True, exist_ok=True)

    np.save(export_dir / "X_train.npy", combined_images[:train_end])
    np.save(export_dir / "Y_train.npy", combined_labels[:train_end])
    np.save(export_dir / "X_val.npy", combined_images[train_end:val_end])
    np.save(export_dir / "Y_val.npy", combined_labels[train_end:val_end])
    np.save(export_dir / "X_test.npy", combined_images[val_end:])
    np.save(export_dir / "Y_test.npy", combined_labels[val_end:])

    metadata = {
        "plugin": plugin,
        "total_samples": n,
        "train": train_end,
        "val": val_end - train_end,
        "test": n - val_end,
        "sources": sources,
        "created": datetime.now().isoformat()
    }
    with open(export_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Dataset combinado para {plugin}!")
    print(f"   Total: {n:,} amostras")
    print(f"   Treino: {train_end:,}")
    print(f"   Validação: {val_end - train_end:,}")
    print(f"   Teste: {n - val_end:,}")
    print(f"   Salvo em: {export_dir}")

    return export_dir


# =============================================================================
# INTERFACE
# =============================================================================

def show_catalog():
    """Mostra catálogo de datasets"""
    print("\n" + "=" * 70)
    print("📚 CATÁLOGO DE DATASETS DISPONÍVEIS")
    print("=" * 70)

    for key, info in DATASET_CATALOG.items():
        print(f"\n📦 {key}")
        print(f"   Nome: {info['name']}")
        print(f"   Amostras: ~{info['samples']:,}")
        print(f"   Anatomia: {info['anatomy']}")
        print(f"   Útil para: {', '.join(info['useful_for'])}")
        print(f"   Tipo: {info['type']}")


def show_status():
    """Mostra status dos datasets"""
    print("\n" + "=" * 70)
    print("📊 STATUS DOS DATASETS")
    print("=" * 70)

    # Raw
    print("\n📁 RAW (Downloads):")
    for key in DATASET_CATALOG:
        path = RAW_DIR / key
        if path.exists() and any(path.rglob("*")):
            n_files = len(list(path.rglob("*")))
            print(f"   ✅ {key}: {n_files} arquivos")
        else:
            print(f"   ❌ {key}: não baixado")

    # Processed
    print("\n📁 PROCESSED (Por plugin):")
    for plugin in ["needle", "nerve", "cardiac", "fast", "anatomy", "lung"]:
        plugin_dir = PROCESSED_DIR / plugin
        if plugin_dir.exists():
            total = 0
            for ds_dir in plugin_dir.iterdir():
                if (ds_dir / "images.npy").exists():
                    arr = np.load(ds_dir / "images.npy")
                    total += len(arr)
            if total > 0:
                print(f"   ✅ {plugin.upper()}: {total:,} amostras")
        else:
            print(f"   ❌ {plugin.upper()}: não processado")

    # Exports
    print("\n📁 EXPORTS (Prontos para treino):")
    for plugin_dir in EXPORTS_DIR.iterdir():
        if (plugin_dir / "X_train.npy").exists():
            train = np.load(plugin_dir / "X_train.npy")
            print(f"   ✅ {plugin_dir.name.upper()}: {len(train):,} treino")


def main():
    print("\n" + "=" * 70)
    print("  USG FLOW - Unified Dataset Manager")
    print("  Gerenciador de Datasets para TODOS os Plugins")
    print("=" * 70)

    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║  OPÇÕES                                                               ║
╠═══════════════════════════════════════════════════════════════════════╣
║  1. Ver catálogo de datasets                                          ║
║  2. Ver status atual                                                  ║
║  3. Baixar dataset específico                                         ║
║  4. Processar dataset para plugin                                     ║
║  5. Gerar dados sintéticos                                            ║
║  6. Combinar datasets para plugin                                     ║
║  7. SETUP COMPLETO (baixa + processa + combina tudo)                  ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)

    choice = input("Escolha (1-7) [7]: ").strip() or "7"

    if choice == "1":
        show_catalog()

    elif choice == "2":
        show_status()

    elif choice == "3":
        show_catalog()
        key = input("\nDataset para baixar: ").strip()
        download_dataset(key)

    elif choice == "4":
        print("\nDatasets disponíveis:", list(DATASET_CATALOG.keys()))
        key = input("Dataset: ").strip()
        plugin = input("Plugin (NEEDLE/NERVE/CARDIAC/FAST/ANATOMY/LUNG): ").strip().upper()
        process_for_plugin(key, plugin)

    elif choice == "5":
        plugin = input("Plugin (NEEDLE/NERVE/CARDIAC): ").strip().upper() or "NERVE"
        n = int(input("Número de amostras [5000]: ").strip() or "5000")
        generate_synthetic_data(plugin, n)

    elif choice == "6":
        plugin = input("Plugin: ").strip().upper()
        combine_for_plugin(plugin)

    elif choice == "7":
        print("\n🚀 SETUP COMPLETO")
        print("=" * 50)

        # Plugins principais
        plugins = ["NEEDLE", "NERVE", "CARDIAC"]

        for plugin in plugins:
            print(f"\n\n{'='*50}")
            print(f"📦 PROCESSANDO PARA: {plugin}")
            print("=" * 50)

            # Datasets relevantes para este plugin
            relevant = [k for k, v in DATASET_CATALOG.items() if plugin in v["useful_for"]]

            # Baixar e processar
            for ds_key in relevant:
                if download_dataset(ds_key):
                    process_for_plugin(ds_key, plugin)

            # Gerar sintético
            generate_synthetic_data(plugin, 5000)

            # Combinar
            combine_for_plugin(plugin)

        show_status()


if __name__ == "__main__":
    main()
