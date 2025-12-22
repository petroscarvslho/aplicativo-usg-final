#!/usr/bin/env python3
"""
Script para download de datasets de ultrassom para treinamento da CNN VASST
NEEDLE PILOT v3.1 - Dataset Downloader COMPLETO

Datasets suportados:
1. Kaggle Ultrasound Nerve Segmentation (5,635 imagens)
2. CAMUS Cardiac Dataset (500 pacientes, ~4000 frames)
3. Dataset Sintético (geração automática)
4. Regional-US Brachial Plexus (41,000 frames) - requer download manual
5. OpenCV Ultrasound Collection (vários datasets abertos)
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import urllib.request
import zipfile
import json
import shutil
import hashlib

# Diretórios
BASE_DIR = Path(__file__).parent
KAGGLE_DIR = BASE_DIR / "kaggle_nerve"
CAMUS_DIR = BASE_DIR / "camus_cardiac"
SYNTHETIC_DIR = BASE_DIR / "synthetic_needle"
BRACHIAL_DIR = BASE_DIR / "brachial_plexus"
CLARIUS_DIR = BASE_DIR / "clarius_open"
PROCESSED_DIR = BASE_DIR / "processed"


# ============================================================================
# DATASET 1: KAGGLE NERVE SEGMENTATION
# ============================================================================

def check_kaggle_api():
    """Verifica se a API do Kaggle está configurada"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("=" * 60)
        print("⚠️  CONFIGURAÇÃO DO KAGGLE NECESSÁRIA")
        print("=" * 60)
        print("""
Para baixar o dataset do Kaggle:

1. Crie uma conta em https://www.kaggle.com
2. Vá em Account > API > Create New Token
3. Salve o arquivo kaggle.json em ~/.kaggle/
4. Execute: chmod 600 ~/.kaggle/kaggle.json
5. Rode este script novamente

Ou baixe manualmente de:
https://www.kaggle.com/c/ultrasound-nerve-segmentation/data
        """)
        return False
    return True


def download_kaggle_dataset():
    """Baixa o dataset de segmentação de nervos do Kaggle"""
    print("\n📥 KAGGLE - Ultrasound Nerve Segmentation")
    print("-" * 50)

    if not check_kaggle_api():
        return False

    try:
        import kaggle
        KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

        kaggle.api.competition_download_files(
            'ultrasound-nerve-segmentation',
            path=str(KAGGLE_DIR)
        )

        zip_file = KAGGLE_DIR / "ultrasound-nerve-segmentation.zip"
        if zip_file.exists():
            print("📦 Extraindo arquivos...")
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(KAGGLE_DIR)
            zip_file.unlink()

        print("✅ Dataset do Kaggle baixado com sucesso!")
        return True

    except ImportError:
        print("⚠️  Instale a API do Kaggle: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Erro ao baixar: {e}")
        return False


# ============================================================================
# DATASET 2: CAMUS CARDIAC
# ============================================================================

def download_camus_dataset():
    """
    Baixa o CAMUS Cardiac Ultrasound Dataset

    Fonte: CREATIS Lab - Université de Lyon
    Site: https://www.creatis.insa-lyon.fr/Challenge/camus/

    Este é um dataset público de ecocardiografia com 500 pacientes.
    Útil para transfer learning em ultrassom.
    """
    print("\n📥 CAMUS - Cardiac Ultrasound Dataset")
    print("-" * 50)

    CAMUS_DIR.mkdir(parents=True, exist_ok=True)

    print("""
╔══════════════════════════════════════════════════════════════╗
║  CAMUS - Cardiac Acquisitions for Multi-structure Ultrasound ║
╠══════════════════════════════════════════════════════════════╣
║  500 pacientes com múltiplos frames de ecocardiografia       ║
║  Ideal para transfer learning em ultrassom                   ║
╚══════════════════════════════════════════════════════════════╝

Para baixar o CAMUS dataset:

1. Acesse: https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html

2. Preencha o formulário de registro (uso acadêmico)

3. Você receberá um link de download por email

4. Extraia os arquivos para:
   {camus_dir}

5. Rode este script novamente com opção de processar CAMUS

Estrutura esperada após download:
{camus_dir}/
├── training/
│   ├── patient0001/
│   │   ├── patient0001_2CH_ED.mhd
│   │   ├── patient0001_2CH_ES.mhd
│   │   └── ...
│   └── ...
└── testing/
    """.format(camus_dir=CAMUS_DIR))

    # Verificar se já existe
    if (CAMUS_DIR / "training").exists():
        print("✅ Dataset CAMUS já presente!")
        return True

    return False


def process_camus_dataset():
    """Processa o CAMUS dataset para formato de treinamento"""
    print("\n🔄 Processando CAMUS dataset...")

    training_dir = CAMUS_DIR / "training"
    if not training_dir.exists():
        print("❌ CAMUS não encontrado. Execute download primeiro.")
        return False

    try:
        import SimpleITK as sitk
    except ImportError:
        print("⚠️  Instalando SimpleITK...")
        os.system("pip install SimpleITK")
        import SimpleITK as sitk

    camus_processed = PROCESSED_DIR / "camus"
    camus_processed.mkdir(parents=True, exist_ok=True)

    images = []

    # Processar cada paciente
    patient_dirs = sorted(training_dir.glob("patient*"))
    print(f"  Encontrados {len(patient_dirs)} pacientes")

    for patient_dir in patient_dirs:
        # Carregar imagens 2CH e 4CH
        for view in ["2CH", "4CH"]:
            for phase in ["ED", "ES"]:
                mhd_file = patient_dir / f"{patient_dir.name}_{view}_{phase}.mhd"
                if mhd_file.exists():
                    try:
                        img = sitk.ReadImage(str(mhd_file))
                        arr = sitk.GetArrayFromImage(img)

                        # Normalizar e redimensionar
                        for frame_idx in range(arr.shape[0]):
                            frame = arr[frame_idx]
                            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                            frame = cv2.resize(frame.astype(np.uint8), (256, 256))
                            images.append(frame)
                    except Exception as e:
                        print(f"  Erro em {mhd_file.name}: {e}")

    if len(images) == 0:
        print("❌ Nenhuma imagem processada")
        return False

    images = np.array(images)
    images = np.expand_dims(images, axis=-1)

    np.save(str(camus_processed / "camus_images.npy"), images)

    print(f"✅ CAMUS processado!")
    print(f"   Frames extraídos: {len(images)}")
    print(f"   Salvo em: {camus_processed}")

    return True


# ============================================================================
# DATASET 3: REGIONAL-US BRACHIAL PLEXUS (NEEDLE SPECIFIC!)
# ============================================================================

def download_brachial_plexus_info():
    """
    Informações sobre o Regional-US Brachial Plexus Dataset

    Este é O MELHOR dataset para treinamento de needle tracking!
    41,000 frames com anotações de agulha
    """
    print("\n📥 REGIONAL-US BRACHIAL PLEXUS - Needle Dataset")
    print("-" * 50)

    BRACHIAL_DIR.mkdir(parents=True, exist_ok=True)

    print("""
╔══════════════════════════════════════════════════════════════╗
║  ⭐ REGIONAL-US BRACHIAL PLEXUS - MELHOR PARA NEEDLE! ⭐    ║
╠══════════════════════════════════════════════════════════════╣
║  41,000 frames com anotações de ponta de agulha              ║
║  Perfeito para treinar detecção de needle tip                ║
╚══════════════════════════════════════════════════════════════╝

FONTES PARA DOWNLOAD:

1. IEEE DataPort (Oficial):
   https://ieee-dataport.org/
   Buscar: "ultrasound needle" ou "brachial plexus"

2. GitHub (Repositórios de pesquisa):
   https://github.com/topics/ultrasound-needle-tracking
   https://github.com/topics/ultrasound-guided-intervention

3. Papers With Code:
   https://paperswithcode.com/datasets?q=ultrasound+needle

4. Zenodo (Open Science):
   https://zenodo.org/search?q=ultrasound%20needle

5. PhysioNet (Dados médicos abertos):
   https://physionet.org/
   Buscar: ultrasound

ARTIGOS RELACIONADOS (com datasets):
- "Deep Learning for Needle Detection" - MICCAI 2020
- "Automatic Needle Tracking in 3D Ultrasound" - TMI 2019
- "Real-time Needle Tip Localization" - IJCARS 2021

Se encontrar o dataset, extraia para:
{brachial_dir}/

Formato esperado:
{brachial_dir}/
├── images/
│   ├── frame_00001.png
│   └── ...
└── annotations/
    └── needle_coords.csv (ou .json)
    """.format(brachial_dir=BRACHIAL_DIR))

    # Verificar se já existe
    if (BRACHIAL_DIR / "images").exists():
        print("✅ Dataset Brachial Plexus já presente!")
        return True

    # Criar arquivo de instruções
    instructions_file = BRACHIAL_DIR / "COMO_BAIXAR.txt"
    with open(instructions_file, "w") as f:
        f.write("""
REGIONAL-US BRACHIAL PLEXUS DATASET
===================================

Este dataset contém 41,000 frames de ultrassom com anotações de agulha.

ONDE PROCURAR:

1. Contato com autores de artigos:
   - Procure artigos sobre "ultrasound needle tracking" no Google Scholar
   - Entre em contato com os autores solicitando acesso ao dataset

2. IEEE DataPort:
   - Crie uma conta gratuita
   - Busque por "ultrasound needle" ou "regional anesthesia"
   - Alguns datasets requerem solicitação formal

3. Kaggle:
   - Busque: "ultrasound needle" ou "ultrasound guided"
   - Novos datasets são adicionados frequentemente

4. GitHub Issues:
   - https://github.com/neelanjanpalwork/Needle-Tip-Tracking
   - https://github.com/MahdiGilany/TIP-Ultrasound

5. Research Gate:
   - Procure autores de papers de needle tracking
   - Solicite acesso via mensagem

FORMATO DO DATASET:
- Imagens: PNG/JPG grayscale ou RGB
- Anotações: CSV ou JSON com coordenadas (x, y) da ponta da agulha

Após download, coloque aqui e execute o script de processamento.
        """)

    print(f"\n📄 Instruções salvas em: {instructions_file}")

    return False


def process_brachial_plexus():
    """Processa o dataset Brachial Plexus se disponível"""
    print("\n🔄 Processando Brachial Plexus dataset...")

    images_dir = BRACHIAL_DIR / "images"
    if not images_dir.exists():
        print("❌ Dataset não encontrado. Baixe primeiro seguindo as instruções.")
        return False

    # Procurar arquivos de anotação
    annotations_file = None
    for ext in ["csv", "json", "txt"]:
        for name in ["annotations", "labels", "needle_coords", "coordinates"]:
            candidate = BRACHIAL_DIR / "annotations" / f"{name}.{ext}"
            if candidate.exists():
                annotations_file = candidate
                break
            candidate = BRACHIAL_DIR / f"{name}.{ext}"
            if candidate.exists():
                annotations_file = candidate
                break

    if annotations_file is None:
        print("⚠️  Arquivo de anotações não encontrado automaticamente.")
        print("   Procurando por padrão alternativo...")

        # Tentar carregar apenas imagens
        image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        if len(image_files) == 0:
            print("❌ Nenhuma imagem encontrada")
            return False

        print(f"   Encontradas {len(image_files)} imagens (sem anotações)")
        # Usar para transfer learning apenas

    brachial_processed = PROCESSED_DIR / "brachial"
    brachial_processed.mkdir(parents=True, exist_ok=True)

    images = []
    labels = []

    image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))

    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (256, 256))
            images.append(img)

    if annotations_file:
        if annotations_file.suffix == ".csv":
            import csv
            with open(annotations_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Tentar diferentes formatos de coluna
                    x = float(row.get("x", row.get("tip_x", row.get("needle_x", 0))))
                    y = float(row.get("y", row.get("tip_y", row.get("needle_y", 0))))
                    labels.append([y / 256.0, x / 256.0])
        elif annotations_file.suffix == ".json":
            with open(annotations_file, "r") as f:
                data = json.load(f)
                for item in data:
                    x = float(item.get("x", item.get("tip_x", 0)))
                    y = float(item.get("y", item.get("tip_y", 0)))
                    labels.append([y / 256.0, x / 256.0])

    images = np.array(images)
    images = np.expand_dims(images, axis=-1)

    np.save(str(brachial_processed / "images.npy"), images)

    if len(labels) > 0:
        labels = np.array(labels)
        np.save(str(brachial_processed / "labels.npy"), labels)
        print(f"✅ Processado com anotações: {len(images)} imagens, {len(labels)} labels")
    else:
        print(f"✅ Processado (apenas imagens): {len(images)} frames")

    return True


# ============================================================================
# DATASET 4: CLARIUS OPEN DATASETS
# ============================================================================

def download_clarius_datasets():
    """Lista datasets abertos do Clarius e outros fabricantes"""
    print("\n📥 CLARIUS & OUTROS - Datasets Abertos")
    print("-" * 50)

    CLARIUS_DIR.mkdir(parents=True, exist_ok=True)

    print("""
╔══════════════════════════════════════════════════════════════╗
║  DATASETS ABERTOS DE FABRICANTES DE ULTRASSOM                ║
╚══════════════════════════════════════════════════════════════╝

1. BUTTERFLY NETWORK:
   - Alguns datasets de demonstração disponíveis
   - https://www.butterflynetwork.com/research

2. CLARIUS:
   - Imagens de exemplo para desenvolvedores
   - https://clarius.com/resources/

3. PHILIPS/SIEMENS:
   - Datasets de pesquisa mediante solicitação
   - Contatar departamento de pesquisa

4. DATASETS GERAIS DE ULTRASSOM:

   a) Fetal Ultrasound (14 mil imagens):
      https://zenodo.org/record/3904280

   b) Breast Ultrasound (780 imagens):
      https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

   c) Thyroid Ultrasound (diversos):
      https://www.kaggle.com/search?q=thyroid+ultrasound

5. GRAND CHALLENGES:
   https://grand-challenge.org/challenges/
   - Novos datasets de ultrassom regularmente
   - Buscar: ultrasound, needle, intervention

DOWNLOAD AUTOMÁTICO (Breast Ultrasound - exemplo):
    """)

    # Oferecer download do breast ultrasound (público no Kaggle)
    choice = input("\nBaixar Breast Ultrasound Dataset? (s/n) [n]: ").strip().lower()

    if choice == "s":
        if check_kaggle_api():
            try:
                import kaggle
                CLARIUS_DIR.mkdir(parents=True, exist_ok=True)

                kaggle.api.dataset_download_files(
                    'aryashah2k/breast-ultrasound-images-dataset',
                    path=str(CLARIUS_DIR / "breast_us")
                )
                print("✅ Breast Ultrasound baixado!")
                return True
            except Exception as e:
                print(f"❌ Erro: {e}")

    return False


# ============================================================================
# DATASET 5: SINTÉTICO (SEMPRE FUNCIONA!)
# ============================================================================

def generate_synthetic_dataset(n_samples: int = 5000):
    """
    Gera dataset sintético de agulhas em ultrassom
    Útil para pré-treinamento quando não há dados reais
    """
    print(f"\n🔧 SINTÉTICO - Gerando {n_samples} imagens...")
    print("-" * 50)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = SYNTHETIC_DIR / "images"
    labels_dir = SYNTHETIC_DIR / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    images = []
    labels = []

    for i in range(n_samples):
        if (i + 1) % 500 == 0:
            print(f"  Progresso: {i + 1}/{n_samples}")

        # Criar imagem base com textura de ultrassom
        img = np.random.randint(20, 80, (256, 256), dtype=np.uint8)

        # Adicionar ruído speckle (característico de ultrassom)
        noise = np.random.exponential(1.0, (256, 256))
        img = np.clip(img * noise, 0, 255).astype(np.uint8)

        # Suavizar para simular textura
        img = cv2.GaussianBlur(img, (5, 5), 1.5)

        # Adicionar estruturas anatômicas simuladas (faixas horizontais)
        for _ in range(np.random.randint(2, 5)):
            y = np.random.randint(50, 200)
            thickness = np.random.randint(5, 20)
            brightness = np.random.randint(100, 200)
            cv2.rectangle(img, (0, y), (256, y + thickness), brightness, -1)
            img = cv2.GaussianBlur(img, (7, 7), 2)

        # Gerar parâmetros da agulha
        entry_x = np.random.randint(50, 200)
        entry_y = np.random.randint(10, 40)
        angle = np.random.uniform(15, 75)
        angle_rad = np.radians(angle)
        length = np.random.randint(100, 200)

        # Calcular ponta da agulha
        tip_x = int(entry_x + length * np.cos(angle_rad))
        tip_y = int(entry_y + length * np.sin(angle_rad))
        tip_x = np.clip(tip_x, 10, 245)
        tip_y = np.clip(tip_y, 10, 245)

        # Desenhar agulha com aparência realística
        cv2.line(img, (entry_x, entry_y), (tip_x, tip_y),
                 np.random.randint(200, 255),
                 np.random.randint(1, 3))

        # Reverberação (artefato comum)
        if np.random.random() > 0.3:
            offset = np.random.randint(3, 8)
            cv2.line(img, (entry_x, entry_y + offset), (tip_x, tip_y + offset),
                     np.random.randint(150, 200), 1)

        # Sombra abaixo da agulha
        if np.random.random() > 0.5:
            shadow_y = tip_y + np.random.randint(5, 15)
            cv2.line(img, (entry_x, entry_y + 10), (tip_x, shadow_y),
                     np.random.randint(30, 60), 2)

        # Adicionar mais ruído
        noise2 = np.random.normal(0, 10, img.shape)
        img = np.clip(img + noise2, 0, 255).astype(np.uint8)

        images.append(img)
        label = np.array([tip_y / 256.0, tip_x / 256.0])
        labels.append(label)

        # Salvar imagem individual
        cv2.imwrite(str(images_dir / f"needle_{i:05d}.png"), img)

    # Converter para arrays numpy
    images = np.array(images)
    labels = np.array(labels)
    images = np.expand_dims(images, axis=-1)

    # Salvar arrays numpy
    np.save(str(SYNTHETIC_DIR / "images.npy"), images)
    np.save(str(SYNTHETIC_DIR / "labels.npy"), labels)

    # Salvar metadados
    metadata = {
        "n_samples": n_samples,
        "image_shape": [256, 256, 1],
        "label_format": "normalized [y, x] tip coordinates",
        "description": "Synthetic ultrasound needle dataset for pretraining"
    }
    with open(SYNTHETIC_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Dataset sintético gerado!")
    print(f"   Imagens: {images.shape}")
    print(f"   Labels: {labels.shape}")
    print(f"   Salvo em: {SYNTHETIC_DIR}")

    return True


# ============================================================================
# PROCESSAMENTO GERAL
# ============================================================================

def prepare_kaggle_for_training():
    """Converte dataset do Kaggle para formato de treinamento VASST"""
    print("\n🔄 Processando dataset do Kaggle...")

    train_dir = KAGGLE_DIR / "train"
    if not train_dir.exists():
        print("❌ Dataset do Kaggle não encontrado. Baixe primeiro.")
        return False

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    images = []
    labels = []

    image_files = sorted(train_dir.glob("*_*.tif"))
    mask_files = {f.stem.replace("_mask", ""): f for f in train_dir.glob("*_mask.tif")}

    print(f"  Encontradas {len(image_files)} imagens")

    for i, img_path in enumerate(image_files):
        if "_mask" in img_path.stem:
            continue

        if (i + 1) % 500 == 0:
            print(f"  Progresso: {i + 1}/{len(image_files)}")

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (256, 256))

        mask_path = mask_files.get(img_path.stem)
        if mask_path and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256, 256))

            if mask.max() > 0:
                moments = cv2.moments(mask)
                if moments["m00"] > 0:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]

                    images.append(img)
                    labels.append([cy / 256.0, cx / 256.0])

    if len(images) == 0:
        print("❌ Nenhuma imagem processada")
        return False

    images = np.array(images)
    labels = np.array(labels)
    images = np.expand_dims(images, axis=-1)

    n = len(images)
    indices = np.random.permutation(n)

    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    np.save(str(PROCESSED_DIR / "X_train.npy"), images[train_idx])
    np.save(str(PROCESSED_DIR / "Y_train.npy"), labels[train_idx])
    np.save(str(PROCESSED_DIR / "X_val.npy"), images[val_idx])
    np.save(str(PROCESSED_DIR / "Y_val.npy"), labels[val_idx])
    np.save(str(PROCESSED_DIR / "X_test.npy"), images[test_idx])
    np.save(str(PROCESSED_DIR / "Y_test.npy"), labels[test_idx])

    print(f"✅ Dataset processado!")
    print(f"   Treino: {len(train_idx)} amostras")
    print(f"   Validação: {len(val_idx)} amostras")
    print(f"   Teste: {len(test_idx)} amostras")

    return True


def combine_all_datasets():
    """Combina todos os datasets disponíveis para treino"""
    print("\n🔀 Combinando todos os datasets disponíveis...")

    all_images = []
    all_labels = []

    # 1. Sintético
    if (SYNTHETIC_DIR / "images.npy").exists():
        images = np.load(SYNTHETIC_DIR / "images.npy")
        labels = np.load(SYNTHETIC_DIR / "labels.npy")
        all_images.append(images)
        all_labels.append(labels)
        print(f"   + Sintético: {len(images)} imagens")

    # 2. Kaggle processado
    if (PROCESSED_DIR / "X_train.npy").exists():
        for split in ["train", "val"]:
            images = np.load(PROCESSED_DIR / f"X_{split}.npy")
            labels = np.load(PROCESSED_DIR / f"Y_{split}.npy")
            all_images.append(images)
            all_labels.append(labels)
            print(f"   + Kaggle {split}: {len(images)} imagens")

    # 3. Brachial Plexus
    brachial_processed = PROCESSED_DIR / "brachial"
    if (brachial_processed / "images.npy").exists() and (brachial_processed / "labels.npy").exists():
        images = np.load(brachial_processed / "images.npy")
        labels = np.load(brachial_processed / "labels.npy")
        all_images.append(images)
        all_labels.append(labels)
        print(f"   + Brachial Plexus: {len(images)} imagens")

    if len(all_images) == 0:
        print("❌ Nenhum dataset encontrado!")
        return False

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

    combined_dir = PROCESSED_DIR / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    np.save(combined_dir / "X_train.npy", combined_images[:train_end])
    np.save(combined_dir / "Y_train.npy", combined_labels[:train_end])
    np.save(combined_dir / "X_val.npy", combined_images[train_end:val_end])
    np.save(combined_dir / "Y_val.npy", combined_labels[train_end:val_end])
    np.save(combined_dir / "X_test.npy", combined_images[val_end:])
    np.save(combined_dir / "Y_test.npy", combined_labels[val_end:])

    print(f"\n✅ Datasets combinados!")
    print(f"   Total: {n} imagens")
    print(f"   Treino: {train_end}")
    print(f"   Validação: {val_end - train_end}")
    print(f"   Teste: {n - val_end}")
    print(f"   Salvo em: {combined_dir}")

    return True


# ============================================================================
# MENU PRINCIPAL
# ============================================================================

def main():
    print("=" * 70)
    print("  NEEDLE PILOT v3.1 - Dataset Manager")
    print("  Sistema Completo para Download e Preparação de Datasets")
    print("=" * 70)

    print("""
╔═════════════════════════════════════════════════════════════════════╗
║  DATASETS DISPONÍVEIS                                               ║
╠═════════════════════════════════════════════════════════════════════╣
║  1. Kaggle Nerve Segmentation   - 5,635 imagens (público)          ║
║  2. CAMUS Cardiac               - 4,000+ frames (registro grátis)  ║
║  3. Brachial Plexus (NEEDLE!)   - 41,000 frames (buscar manualmente)║
║  4. Outros (Breast US, etc)     - Vários datasets menores          ║
║  5. Sintético                   - Gerado automaticamente           ║
╠═════════════════════════════════════════════════════════════════════╣
║  OPÇÕES                                                              ║
╠═════════════════════════════════════════════════════════════════════╣
║  6. Combinar todos os datasets                                      ║
║  7. Ver status dos datasets                                         ║
╚═════════════════════════════════════════════════════════════════════╝
    """)

    choice = input("Escolha uma opção (1-7) [5]: ").strip() or "5"

    if choice == "1":
        download_kaggle_dataset()
        if input("\nProcessar para treinamento? (s/n) [s]: ").lower() != "n":
            prepare_kaggle_for_training()

    elif choice == "2":
        if download_camus_dataset():
            if input("\nProcessar para treinamento? (s/n) [s]: ").lower() != "n":
                process_camus_dataset()

    elif choice == "3":
        download_brachial_plexus_info()
        if (BRACHIAL_DIR / "images").exists():
            if input("\nProcessar dataset existente? (s/n) [s]: ").lower() != "n":
                process_brachial_plexus()

    elif choice == "4":
        download_clarius_datasets()

    elif choice == "5":
        n = input("Número de amostras sintéticas [5000]: ").strip() or "5000"
        generate_synthetic_dataset(int(n))

    elif choice == "6":
        combine_all_datasets()

    elif choice == "7":
        print("\n📊 STATUS DOS DATASETS:")
        print("-" * 50)

        datasets = [
            ("Kaggle Nerve", KAGGLE_DIR / "train"),
            ("CAMUS Cardiac", CAMUS_DIR / "training"),
            ("Brachial Plexus", BRACHIAL_DIR / "images"),
            ("Sintético", SYNTHETIC_DIR / "images.npy"),
            ("Processado", PROCESSED_DIR / "X_train.npy"),
            ("Combinado", PROCESSED_DIR / "combined" / "X_train.npy"),
        ]

        for name, path in datasets:
            status = "✅ Presente" if path.exists() else "❌ Não encontrado"
            print(f"  {name:20s}: {status}")

        print()

    else:
        print("Opção inválida")


if __name__ == "__main__":
    main()
