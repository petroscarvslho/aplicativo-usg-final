#!/usr/bin/env python3
"""
NERVE TRACK v2.0 - Dataset Downloader & Processor
==================================================
Download e preparação de datasets para treinamento do modelo de segmentação
de nervos, artérias, veias e músculos em ultrassom.

Datasets Suportados:
1. Kaggle Ultrasound Nerve Segmentation (5,635 imagens - Plexo Braquial)
2. Mus-V: Multimodal Ultrasound Vascular (artérias e veias)
3. Transverse Musculoskeletal Ultrasound (3,917 imagens - músculos)
4. Axial Neck Anatomy Dataset (2,172 imagens - 13 classes)
5. US-43d Collection (280,000+ imagens - 43 datasets)
6. Dataset Sintético (geração automática)

Fontes:
- Kaggle: https://www.kaggle.com
- GitHub: https://github.com/ziyangwang007/Awesome-Medical-Image-Segmentation-Dataset
- Papers With Code: https://paperswithcode.com/datasets
- UltraSam: https://arxiv.org/html/2411.16222v1
"""

import os
import sys
import json
import shutil
import hashlib
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import cv2
except ImportError:
    print("Instalando OpenCV...")
    os.system("pip install opencv-python")
    import cv2

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

BASE_DIR = Path(__file__).parent
NERVE_DATA_DIR = BASE_DIR / "nerve_track"
RAW_DIR = NERVE_DATA_DIR / "raw"
PROCESSED_DIR = NERVE_DATA_DIR / "processed"
COMBINED_DIR = NERVE_DATA_DIR / "combined"

# Subdiretórios por dataset
KAGGLE_NERVE_DIR = RAW_DIR / "kaggle_nerve"
MUSV_DIR = RAW_DIR / "mus_v_vascular"
MUSCLE_DIR = RAW_DIR / "musculoskeletal"
NECK_ANATOMY_DIR = RAW_DIR / "neck_anatomy"
US43D_DIR = RAW_DIR / "us43d"
SYNTHETIC_DIR = RAW_DIR / "synthetic"

# Classes para segmentação
CLASSES = {
    0: "background",
    1: "nerve",
    2: "artery",
    3: "vein",
    4: "muscle",
    5: "fascia",
    6: "bone",
    7: "other"
}

# =============================================================================
# UTILIDADES
# =============================================================================

def create_directories():
    """Cria estrutura de diretórios"""
    dirs = [
        NERVE_DATA_DIR, RAW_DIR, PROCESSED_DIR, COMBINED_DIR,
        KAGGLE_NERVE_DIR, MUSV_DIR, MUSCLE_DIR, NECK_ANATOMY_DIR,
        US43D_DIR, SYNTHETIC_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("✅ Estrutura de diretórios criada")


def check_kaggle_api() -> bool:
    """Verifica se a API do Kaggle está configurada"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("""
╔══════════════════════════════════════════════════════════════╗
║  ⚠️  CONFIGURAÇÃO DO KAGGLE NECESSÁRIA                       ║
╠══════════════════════════════════════════════════════════════╣
║  1. Crie conta em https://www.kaggle.com                     ║
║  2. Vá em Account > API > Create New Token                   ║
║  3. Salve kaggle.json em ~/.kaggle/                          ║
║  4. Execute: chmod 600 ~/.kaggle/kaggle.json                 ║
╚══════════════════════════════════════════════════════════════╝
        """)
        return False
    return True


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Baixa arquivo com barra de progresso"""
    try:
        print(f"📥 Baixando {desc or url}...")

        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 / total_size)
                sys.stdout.write(f"\r   Progresso: {percent:.1f}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, str(dest), progress_hook)
        print()
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path):
    """Extrai arquivo ZIP ou TAR"""
    print(f"📦 Extraindo {archive_path.name}...")

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, 'r') as z:
            z.extractall(dest_dir)
    elif archive_path.suffix in [".tar", ".gz", ".tgz"]:
        with tarfile.open(archive_path, 'r:*') as t:
            t.extractall(dest_dir)
    else:
        print(f"⚠️ Formato não suportado: {archive_path.suffix}")
        return False

    return True


# =============================================================================
# DATASET 1: KAGGLE ULTRASOUND NERVE SEGMENTATION
# =============================================================================

def download_kaggle_nerve():
    """
    Baixa o dataset Kaggle Ultrasound Nerve Segmentation

    - 5,635 imagens de treinamento
    - Plexo Braquial (nervos do pescoço/ombro)
    - Máscaras de segmentação incluídas

    URL: https://www.kaggle.com/c/ultrasound-nerve-segmentation
    """
    print("\n" + "=" * 60)
    print("📥 DATASET 1: KAGGLE NERVE SEGMENTATION")
    print("=" * 60)
    print("   Fonte: Kaggle Competition")
    print("   Imagens: 5,635 (plexo braquial)")
    print("   Classes: Nervo (foreground/background)")
    print("-" * 60)

    if not check_kaggle_api():
        print("\n📝 Download manual:")
        print("   1. Acesse: https://www.kaggle.com/c/ultrasound-nerve-segmentation/data")
        print("   2. Baixe 'train.zip'")
        print(f"   3. Extraia para: {KAGGLE_NERVE_DIR}")
        return False

    try:
        import kaggle

        kaggle.api.competition_download_files(
            'ultrasound-nerve-segmentation',
            path=str(KAGGLE_NERVE_DIR)
        )

        # Extrair
        zip_file = KAGGLE_NERVE_DIR / "ultrasound-nerve-segmentation.zip"
        if zip_file.exists():
            extract_archive(zip_file, KAGGLE_NERVE_DIR)
            zip_file.unlink()

        print("✅ Kaggle Nerve dataset baixado!")
        return True

    except ImportError:
        print("⚠️ Instale: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


def process_kaggle_nerve():
    """Processa o dataset Kaggle para formato de treinamento"""
    print("\n🔄 Processando Kaggle Nerve dataset...")

    train_dir = KAGGLE_NERVE_DIR / "train"
    if not train_dir.exists():
        print(f"❌ Dataset não encontrado em {train_dir}")
        return None

    output_dir = PROCESSED_DIR / "kaggle_nerve"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    masks = []

    # Encontrar pares imagem/máscara
    image_files = sorted([f for f in train_dir.glob("*.tif") if "_mask" not in f.stem])

    print(f"   Encontradas {len(image_files)} imagens")

    for i, img_path in enumerate(image_files):
        if (i + 1) % 500 == 0:
            print(f"   Progresso: {i + 1}/{len(image_files)}")

        # Carregar imagem
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Carregar máscara correspondente
        mask_path = train_dir / f"{img_path.stem}_mask.tif"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros_like(img)

        # Redimensionar para 256x256
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Converter máscara para classe 1 (nervo)
        mask = (mask > 127).astype(np.uint8)

        images.append(img)
        masks.append(mask)

    if len(images) == 0:
        print("❌ Nenhuma imagem processada")
        return None

    # Converter para arrays
    images = np.array(images, dtype=np.uint8)
    masks = np.array(masks, dtype=np.uint8)

    # Adicionar canal
    images = np.expand_dims(images, axis=-1)

    # Salvar
    np.save(output_dir / "images.npy", images)
    np.save(output_dir / "masks.npy", masks)

    # Metadados
    metadata = {
        "name": "Kaggle Ultrasound Nerve Segmentation",
        "n_samples": len(images),
        "image_shape": list(images.shape),
        "classes": {0: "background", 1: "nerve"},
        "source": "https://www.kaggle.com/c/ultrasound-nerve-segmentation"
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Processado: {len(images)} imagens")
    print(f"   Salvo em: {output_dir}")

    return output_dir


# =============================================================================
# DATASET 2: MUS-V VASCULAR SEGMENTATION
# =============================================================================

def download_musv_vascular():
    """
    Baixa o dataset Mus-V: Multimodal Ultrasound Vascular Segmentation

    - Segmentação de artérias e veias
    - Multimodal (B-mode + Doppler)

    URL: https://www.kaggle.com/datasets/among22/multimodal-ultrasound-vascular-segmentation
    """
    print("\n" + "=" * 60)
    print("📥 DATASET 2: MUS-V VASCULAR SEGMENTATION")
    print("=" * 60)
    print("   Fonte: Kaggle")
    print("   Classes: Artéria, Veia")
    print("   Modais: B-mode, Color Doppler")
    print("-" * 60)

    if not check_kaggle_api():
        print("\n📝 Download manual:")
        print("   1. Acesse: https://www.kaggle.com/datasets/among22/multimodal-ultrasound-vascular-segmentation")
        print(f"   2. Extraia para: {MUSV_DIR}")
        return False

    try:
        import kaggle

        kaggle.api.dataset_download_files(
            'among22/multimodal-ultrasound-vascular-segmentation',
            path=str(MUSV_DIR),
            unzip=True
        )

        print("✅ Mus-V Vascular dataset baixado!")
        return True

    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


def process_musv_vascular():
    """Processa o dataset Mus-V para formato de treinamento"""
    print("\n🔄 Processando Mus-V Vascular dataset...")

    if not MUSV_DIR.exists() or not any(MUSV_DIR.iterdir()):
        print(f"❌ Dataset não encontrado em {MUSV_DIR}")
        return None

    output_dir = PROCESSED_DIR / "musv_vascular"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    masks = []

    # Procurar imagens e máscaras
    for img_dir in ["images", "image", "data"]:
        img_path = MUSV_DIR / img_dir
        if img_path.exists():
            break

    for mask_dir in ["masks", "mask", "label", "labels", "annotation", "annotations"]:
        mask_path = MUSV_DIR / mask_dir
        if mask_path.exists():
            break

    if not img_path.exists():
        # Tentar encontrar automaticamente
        for p in MUSV_DIR.rglob("*.png"):
            if "mask" not in str(p).lower():
                img_path = p.parent
                break

    image_files = sorted(list(img_path.glob("*.png")) + list(img_path.glob("*.jpg")))
    print(f"   Encontradas {len(image_files)} imagens em {img_path}")

    for img_file in image_files:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Procurar máscara correspondente
        mask = None
        if mask_path.exists():
            for ext in [".png", ".jpg", ".tif"]:
                mask_file = mask_path / f"{img_file.stem}{ext}"
                if mask_file.exists():
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    break

        if mask is None:
            mask = np.zeros_like(img)

        # Redimensionar
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Converter máscara: assumir que valores diferentes são classes diferentes
        # 0 = background, vermelho/alto = artéria (2), azul/baixo = veia (3)
        mask_converted = np.zeros_like(mask)
        mask_converted[mask > 170] = 2  # Artéria
        mask_converted[(mask > 80) & (mask <= 170)] = 3  # Veia

        images.append(img)
        masks.append(mask_converted)

    if len(images) == 0:
        print("⚠️ Nenhuma imagem encontrada. Verifique a estrutura do dataset.")
        return None

    images = np.array(images, dtype=np.uint8)
    masks = np.array(masks, dtype=np.uint8)
    images = np.expand_dims(images, axis=-1)

    np.save(output_dir / "images.npy", images)
    np.save(output_dir / "masks.npy", masks)

    metadata = {
        "name": "Mus-V Multimodal Ultrasound Vascular",
        "n_samples": len(images),
        "image_shape": list(images.shape),
        "classes": {0: "background", 2: "artery", 3: "vein"},
        "source": "https://www.kaggle.com/datasets/among22/multimodal-ultrasound-vascular-segmentation"
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Processado: {len(images)} imagens")
    return output_dir


# =============================================================================
# DATASET 3: TRANSVERSE MUSCULOSKELETAL ULTRASOUND
# =============================================================================

def download_musculoskeletal():
    """
    Informações sobre o Transverse Musculoskeletal Ultrasound Dataset

    - 3,917 imagens
    - Músculos: biceps brachii, tibialis anterior, gastrocnemius medialis
    - 1,283 sujeitos

    URL: http://www.mdpi.com/2313-433X/4/2/29
    """
    print("\n" + "=" * 60)
    print("📥 DATASET 3: TRANSVERSE MUSCULOSKELETAL ULTRASOUND")
    print("=" * 60)
    print("   Fonte: MDPI Journal of Imaging")
    print("   Imagens: 3,917")
    print("   Músculos: Biceps, Tibialis, Gastrocnemius")
    print("-" * 60)

    print("""
╔══════════════════════════════════════════════════════════════╗
║  Este dataset requer download manual                         ║
╠══════════════════════════════════════════════════════════════╣
║  1. Acesse: http://www.mdpi.com/2313-433X/4/2/29            ║
║  2. Na seção "Supplementary Materials" ou "Data"            ║
║  3. Baixe o dataset                                          ║
║  4. Extraia para:                                            ║
║     {muscle_dir}
╚══════════════════════════════════════════════════════════════╝

Alternativa - Kaggle:
  Busque "musculoskeletal ultrasound" no Kaggle
    """.format(muscle_dir=MUSCLE_DIR))

    # Verificar se já existe
    if any(MUSCLE_DIR.glob("*.png")) or any(MUSCLE_DIR.glob("*.jpg")):
        print("✅ Algumas imagens já presentes!")
        return True

    return False


def process_musculoskeletal():
    """Processa o dataset musculoesquelético"""
    print("\n🔄 Processando Musculoskeletal dataset...")

    if not MUSCLE_DIR.exists() or not any(MUSCLE_DIR.rglob("*")):
        print(f"❌ Dataset não encontrado em {MUSCLE_DIR}")
        return None

    output_dir = PROCESSED_DIR / "musculoskeletal"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    masks = []

    # Procurar todas as imagens
    image_files = list(MUSCLE_DIR.rglob("*.png")) + list(MUSCLE_DIR.rglob("*.jpg"))
    image_files = [f for f in image_files if "mask" not in f.stem.lower()]

    print(f"   Encontradas {len(image_files)} imagens")

    for img_file in image_files:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Procurar máscara
        mask = None
        for pattern in [f"{img_file.stem}_mask", f"{img_file.stem}_seg", f"mask_{img_file.stem}"]:
            for ext in [".png", ".jpg"]:
                mask_file = img_file.parent / f"{pattern}{ext}"
                if mask_file.exists():
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    break

        if mask is None:
            mask = np.zeros_like(img)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Classe 4 = músculo
        mask = ((mask > 127) * 4).astype(np.uint8)

        images.append(img)
        masks.append(mask)

    if len(images) == 0:
        print("⚠️ Nenhuma imagem processada")
        return None

    images = np.array(images, dtype=np.uint8)
    masks = np.array(masks, dtype=np.uint8)
    images = np.expand_dims(images, axis=-1)

    np.save(output_dir / "images.npy", images)
    np.save(output_dir / "masks.npy", masks)

    metadata = {
        "name": "Transverse Musculoskeletal Ultrasound",
        "n_samples": len(images),
        "classes": {0: "background", 4: "muscle"},
        "source": "http://www.mdpi.com/2313-433X/4/2/29"
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Processado: {len(images)} imagens")
    return output_dir


# =============================================================================
# DATASET 4: AXIAL NECK ANATOMY
# =============================================================================

def download_neck_anatomy():
    """
    Informações sobre o Axial Neck Anatomy Dataset

    - 2,172 imagens totalmente anotadas
    - 13 classes anatômicas
    - 61 participantes

    URL: http://e-space.mmu.ac.uk/624643/
    """
    print("\n" + "=" * 60)
    print("📥 DATASET 4: AXIAL NECK ANATOMY")
    print("=" * 60)
    print("   Fonte: Manchester Metropolitan University")
    print("   Imagens: 2,172")
    print("   Classes: 13 estruturas anatômicas")
    print("-" * 60)

    print("""
╔══════════════════════════════════════════════════════════════╗
║  Este dataset é EXCELENTE para NERVE TRACK!                  ║
╠══════════════════════════════════════════════════════════════╣
║  Contém anotações para:                                      ║
║  - Carótida (artéria)                                        ║
║  - Jugular (veia)                                            ║
║  - Músculos do pescoço                                       ║
║  - Outras estruturas anatômicas                              ║
╠══════════════════════════════════════════════════════════════╣
║  Download:                                                   ║
║  1. Acesse: http://e-space.mmu.ac.uk/624643/                ║
║  2. Baixe o dataset (pode requerer registro)                 ║
║  3. Extraia para:                                            ║
║     {neck_dir}
╚══════════════════════════════════════════════════════════════╝
    """.format(neck_dir=NECK_ANATOMY_DIR))

    if any(NECK_ANATOMY_DIR.rglob("*")):
        print("✅ Dataset já presente!")
        return True

    return False


def process_neck_anatomy():
    """Processa o dataset de anatomia do pescoço"""
    print("\n🔄 Processando Neck Anatomy dataset...")

    if not NECK_ANATOMY_DIR.exists() or not any(NECK_ANATOMY_DIR.rglob("*")):
        print(f"❌ Dataset não encontrado em {NECK_ANATOMY_DIR}")
        return None

    output_dir = PROCESSED_DIR / "neck_anatomy"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    masks = []

    # Mapear classes do dataset original para nossas classes
    # Isso vai depender do formato específico do dataset
    class_mapping = {
        "carotid": 2,      # artéria
        "jugular": 3,      # veia
        "muscle": 4,       # músculo
        "nerve": 1,        # nervo
        "trachea": 7,      # outro
        "thyroid": 7,      # outro
    }

    image_files = list(NECK_ANATOMY_DIR.rglob("*.png")) + list(NECK_ANATOMY_DIR.rglob("*.jpg"))
    image_files = [f for f in image_files if "mask" not in f.stem.lower() and "label" not in f.stem.lower()]

    print(f"   Encontradas {len(image_files)} imagens")

    for img_file in image_files:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Procurar máscara
        mask = None
        for pattern in ["_mask", "_label", "_seg", "_annotation"]:
            mask_file = img_file.parent / f"{img_file.stem}{pattern}{img_file.suffix}"
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                break

        if mask is None:
            mask = np.zeros_like(img)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        images.append(img)
        masks.append(mask)

    if len(images) == 0:
        print("⚠️ Nenhuma imagem processada")
        return None

    images = np.array(images, dtype=np.uint8)
    masks = np.array(masks, dtype=np.uint8)
    images = np.expand_dims(images, axis=-1)

    np.save(output_dir / "images.npy", images)
    np.save(output_dir / "masks.npy", masks)

    metadata = {
        "name": "Axial Neck Anatomy Dataset",
        "n_samples": len(images),
        "classes": CLASSES,
        "source": "http://e-space.mmu.ac.uk/624643/"
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Processado: {len(images)} imagens")
    return output_dir


# =============================================================================
# DATASET 5: US-43D (ULTRASAM COLLECTION)
# =============================================================================

def download_us43d_info():
    """
    Informações sobre o US-43d / UltraSam Collection

    - 280,000+ imagens de 43 datasets
    - 50+ estruturas anatômicas
    - Coletado de múltiplas fontes públicas

    URL: https://arxiv.org/html/2411.16222v1
    """
    print("\n" + "=" * 60)
    print("📥 DATASET 5: US-43D (ULTRASAM COLLECTION)")
    print("=" * 60)
    print("   Fonte: UltraSam Paper (arXiv)")
    print("   Imagens: 280,000+")
    print("   Datasets: 43 combinados")
    print("   Estruturas: 50+ classes")
    print("-" * 60)

    print("""
╔══════════════════════════════════════════════════════════════╗
║  MEGA DATASET - 43 datasets de ultrassom combinados!         ║
╠══════════════════════════════════════════════════════════════╣
║  Fontes incluídas:                                           ║
║  - Papers With Code                                          ║
║  - Google Dataset Search                                     ║
║  - GitHub                                                    ║
║  - Kaggle                                                    ║
║  - Zenodo                                                    ║
║  - Mendeley                                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Para baixar:                                                ║
║  1. Acesse: https://arxiv.org/html/2411.16222v1             ║
║  2. Procure links para os datasets no paper                  ║
║  3. Ou use o script de coleta do GitHub:                     ║
║     https://github.com/... (ver paper)                       ║
╚══════════════════════════════════════════════════════════════╝

Datasets individuais mais relevantes para NERVE TRACK:

1. BUSI (Breast Ultrasound):
   https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

2. Thyroid Ultrasound:
   https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images

3. Musculoskeletal:
   https://www.kaggle.com/datasets/tommyngx/msk-us-bone-ultrasound

4. Fetal/OB-GYN:
   https://zenodo.org/record/3904280
    """)

    return False


# =============================================================================
# DATASET 6: SINTÉTICO (PARA NERVE TRACK)
# =============================================================================

def generate_synthetic_nerve_dataset(n_samples: int = 10000):
    """
    Gera dataset sintético de ultrassom com nervos, artérias, veias e músculos

    Útil para pré-treinamento quando não há dados reais suficientes
    """
    print("\n" + "=" * 60)
    print(f"🔧 GERANDO DATASET SINTÉTICO - {n_samples} imagens")
    print("=" * 60)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    images = []
    masks = []

    for i in range(n_samples):
        if (i + 1) % 1000 == 0:
            print(f"   Progresso: {i + 1}/{n_samples}")

        img, mask = generate_synthetic_ultrasound_image()
        images.append(img)
        masks.append(mask)

    images = np.array(images, dtype=np.uint8)
    masks = np.array(masks, dtype=np.uint8)
    images = np.expand_dims(images, axis=-1)

    output_dir = PROCESSED_DIR / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "images.npy", images)
    np.save(output_dir / "masks.npy", masks)

    metadata = {
        "name": "Synthetic Nerve Track Dataset",
        "n_samples": n_samples,
        "image_shape": list(images.shape),
        "classes": CLASSES,
        "generated": True
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Dataset sintético gerado: {n_samples} imagens")
    print(f"   Salvo em: {output_dir}")

    return output_dir


def generate_synthetic_ultrasound_image(size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Gera uma imagem sintética de ultrassom com estruturas anatômicas"""

    # Criar base com textura de ultrassom
    img = np.random.randint(30, 70, (size, size), dtype=np.uint8)

    # Adicionar ruído speckle
    speckle = np.random.exponential(1.0, (size, size))
    img = np.clip(img * speckle, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 1.5)

    # Criar máscara vazia
    mask = np.zeros((size, size), dtype=np.uint8)

    # Adicionar fundo muscular (camadas horizontais)
    n_layers = np.random.randint(2, 5)
    for _ in range(n_layers):
        y = np.random.randint(30, size - 30)
        thickness = np.random.randint(20, 50)
        brightness = np.random.randint(50, 100)
        cv2.rectangle(img, (0, y), (size, y + thickness), brightness, -1)
        mask[y:y+thickness, :] = 4  # Músculo

    img = cv2.GaussianBlur(img, (7, 7), 2)

    # Adicionar nervo (estrutura ovalada hiperecóica com padrão honeycomb)
    if np.random.random() > 0.2:
        nerve_x = np.random.randint(50, size - 50)
        nerve_y = np.random.randint(80, size - 80)
        nerve_w = np.random.randint(15, 35)
        nerve_h = np.random.randint(10, 25)

        # Desenhar nervo com textura
        cv2.ellipse(img, (nerve_x, nerve_y), (nerve_w, nerve_h),
                   np.random.randint(-20, 20), 0, 360,
                   np.random.randint(140, 200), -1)

        # Adicionar padrão honeycomb (fascículos)
        for _ in range(np.random.randint(3, 8)):
            fx = nerve_x + np.random.randint(-nerve_w//2, nerve_w//2)
            fy = nerve_y + np.random.randint(-nerve_h//2, nerve_h//2)
            fr = np.random.randint(2, 5)
            cv2.circle(img, (fx, fy), fr, np.random.randint(60, 100), -1)

        # Máscara do nervo
        cv2.ellipse(mask, (nerve_x, nerve_y), (nerve_w, nerve_h), 0, 0, 360, 1, -1)

    # Adicionar artéria (círculo pulsátil, hipoecóico)
    if np.random.random() > 0.3:
        art_x = np.random.randint(40, size - 40)
        art_y = np.random.randint(60, size - 60)
        art_r = np.random.randint(8, 18)

        # Artéria (escura por dentro, borda brilhante)
        cv2.circle(img, (art_x, art_y), art_r, np.random.randint(20, 50), -1)
        cv2.circle(img, (art_x, art_y), art_r, np.random.randint(150, 200), 2)

        cv2.circle(mask, (art_x, art_y), art_r, 2, -1)

    # Adicionar veia (oval compressível, mais escura)
    if np.random.random() > 0.3:
        vein_x = np.random.randint(40, size - 40)
        vein_y = np.random.randint(60, size - 60)
        vein_w = np.random.randint(10, 25)
        vein_h = np.random.randint(5, 15)

        # Veia (muito escura, paredes finas)
        cv2.ellipse(img, (vein_x, vein_y), (vein_w, vein_h),
                   np.random.randint(-30, 30), 0, 360,
                   np.random.randint(10, 30), -1)

        cv2.ellipse(mask, (vein_x, vein_y), (vein_w, vein_h), 0, 0, 360, 3, -1)

    # Adicionar fáscia (linha brilhante)
    if np.random.random() > 0.5:
        y = np.random.randint(50, size - 50)
        cv2.line(img, (0, y), (size, y + np.random.randint(-10, 10)),
                np.random.randint(180, 230), np.random.randint(1, 3))
        cv2.line(mask, (0, y), (size, y), 5, 2)

    # Adicionar mais ruído final
    noise = np.random.normal(0, 8, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img, mask


# =============================================================================
# COMBINAÇÃO DE DATASETS
# =============================================================================

def combine_all_datasets():
    """Combina todos os datasets processados em um único dataset de treinamento"""
    print("\n" + "=" * 60)
    print("🔀 COMBINANDO TODOS OS DATASETS")
    print("=" * 60)

    all_images = []
    all_masks = []
    sources = []

    # Lista de datasets processados
    datasets = [
        ("kaggle_nerve", "Kaggle Nerve"),
        ("musv_vascular", "Mus-V Vascular"),
        ("musculoskeletal", "Musculoskeletal"),
        ("neck_anatomy", "Neck Anatomy"),
        ("synthetic", "Synthetic"),
    ]

    for folder, name in datasets:
        dataset_dir = PROCESSED_DIR / folder
        images_file = dataset_dir / "images.npy"
        masks_file = dataset_dir / "masks.npy"

        if images_file.exists() and masks_file.exists():
            images = np.load(images_file)
            masks = np.load(masks_file)

            all_images.append(images)
            all_masks.append(masks)
            sources.append((name, len(images)))

            print(f"   + {name}: {len(images)} imagens")

    if len(all_images) == 0:
        print("❌ Nenhum dataset encontrado para combinar!")
        print("   Execute primeiro o download e processamento dos datasets.")
        return None

    # Concatenar
    combined_images = np.concatenate(all_images, axis=0)
    combined_masks = np.concatenate(all_masks, axis=0)

    # Shuffle
    indices = np.random.permutation(len(combined_images))
    combined_images = combined_images[indices]
    combined_masks = combined_masks[indices]

    # Dividir em train/val/test
    n = len(combined_images)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    # Salvar splits
    np.save(COMBINED_DIR / "X_train.npy", combined_images[:train_end])
    np.save(COMBINED_DIR / "Y_train.npy", combined_masks[:train_end])
    np.save(COMBINED_DIR / "X_val.npy", combined_images[train_end:val_end])
    np.save(COMBINED_DIR / "Y_val.npy", combined_masks[train_end:val_end])
    np.save(COMBINED_DIR / "X_test.npy", combined_images[val_end:])
    np.save(COMBINED_DIR / "Y_test.npy", combined_masks[val_end:])

    # Metadados
    metadata = {
        "name": "NERVE TRACK Combined Dataset",
        "total_samples": n,
        "train_samples": train_end,
        "val_samples": val_end - train_end,
        "test_samples": n - val_end,
        "sources": sources,
        "classes": CLASSES,
        "image_shape": list(combined_images.shape)
    }
    with open(COMBINED_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ DATASETS COMBINADOS!")
    print(f"   Total: {n} imagens")
    print(f"   Treino: {train_end}")
    print(f"   Validação: {val_end - train_end}")
    print(f"   Teste: {n - val_end}")
    print(f"   Salvo em: {COMBINED_DIR}")

    return COMBINED_DIR


# =============================================================================
# STATUS
# =============================================================================

def show_status():
    """Mostra status de todos os datasets"""
    print("\n" + "=" * 60)
    print("📊 STATUS DOS DATASETS - NERVE TRACK")
    print("=" * 60)

    datasets = [
        ("Kaggle Nerve (raw)", KAGGLE_NERVE_DIR / "train"),
        ("Mus-V Vascular (raw)", MUSV_DIR),
        ("Musculoskeletal (raw)", MUSCLE_DIR),
        ("Neck Anatomy (raw)", NECK_ANATOMY_DIR),
        ("", None),  # Separador
        ("Kaggle Nerve (processed)", PROCESSED_DIR / "kaggle_nerve" / "images.npy"),
        ("Mus-V (processed)", PROCESSED_DIR / "musv_vascular" / "images.npy"),
        ("Musculoskeletal (processed)", PROCESSED_DIR / "musculoskeletal" / "images.npy"),
        ("Neck Anatomy (processed)", PROCESSED_DIR / "neck_anatomy" / "images.npy"),
        ("Synthetic (processed)", PROCESSED_DIR / "synthetic" / "images.npy"),
        ("", None),  # Separador
        ("COMBINED (train)", COMBINED_DIR / "X_train.npy"),
    ]

    for name, path in datasets:
        if path is None:
            print("-" * 50)
            continue

        if path.exists():
            if path.suffix == ".npy":
                arr = np.load(path)
                status = f"✅ {len(arr):,} imagens"
            else:
                n_files = len(list(path.rglob("*")))
                status = f"✅ {n_files:,} arquivos"
        else:
            status = "❌ Não encontrado"

        print(f"   {name:30s}: {status}")

    print()


# =============================================================================
# MENU PRINCIPAL
# =============================================================================

def main():
    create_directories()

    print("\n" + "=" * 70)
    print("  NERVE TRACK v2.0 - Dataset Manager")
    print("  Download e Preparação de Datasets para Treinamento")
    print("=" * 70)

    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║  DATASETS DISPONÍVEIS                                                 ║
╠═══════════════════════════════════════════════════════════════════════╣
║  1. Kaggle Nerve Segmentation    - 5,635 imagens (plexo braquial)     ║
║  2. Mus-V Vascular               - Artérias e veias                   ║
║  3. Musculoskeletal Ultrasound   - 3,917 imagens (músculos)           ║
║  4. Axial Neck Anatomy           - 2,172 imagens (13 classes)         ║
║  5. US-43d Info                  - 280,000+ imagens (43 datasets)     ║
║  6. Gerar Sintético              - Dataset artificial para treino     ║
╠═══════════════════════════════════════════════════════════════════════╣
║  PROCESSAMENTO                                                         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  7. Combinar todos os datasets                                        ║
║  8. Ver status dos datasets                                           ║
║  9. Download + Processar TUDO (automático)                            ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)

    choice = input("Escolha uma opção (1-9) [9]: ").strip() or "9"

    if choice == "1":
        if download_kaggle_nerve():
            process_kaggle_nerve()

    elif choice == "2":
        if download_musv_vascular():
            process_musv_vascular()

    elif choice == "3":
        download_musculoskeletal()
        if any(MUSCLE_DIR.rglob("*")):
            process_musculoskeletal()

    elif choice == "4":
        download_neck_anatomy()
        if any(NECK_ANATOMY_DIR.rglob("*")):
            process_neck_anatomy()

    elif choice == "5":
        download_us43d_info()

    elif choice == "6":
        n = input("Número de imagens sintéticas [10000]: ").strip() or "10000"
        generate_synthetic_nerve_dataset(int(n))

    elif choice == "7":
        combine_all_datasets()

    elif choice == "8":
        show_status()

    elif choice == "9":
        print("\n🚀 DOWNLOAD E PROCESSAMENTO AUTOMÁTICO")
        print("=" * 50)

        # 1. Kaggle (se API configurada)
        if check_kaggle_api():
            if download_kaggle_nerve():
                process_kaggle_nerve()
            if download_musv_vascular():
                process_musv_vascular()

        # 2. Sintético (sempre funciona)
        generate_synthetic_nerve_dataset(10000)

        # 3. Combinar
        combine_all_datasets()

        # 4. Status final
        show_status()

    else:
        print("Opção inválida")


if __name__ == "__main__":
    main()
