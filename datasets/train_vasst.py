#!/usr/bin/env python3
"""
Script de Treinamento da CNN VASST para Detecção de Agulhas
NEEDLE PILOT v3.1 - Training Pipeline

Este script treina o modelo VASST PyTorch para detectar
a ponta da agulha em imagens de ultrassom.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
import cv2

# Adicionar src ao path para importar o modelo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Diretórios
BASE_DIR = Path(__file__).parent
SYNTHETIC_DIR = BASE_DIR / "synthetic_needle"
PROCESSED_DIR = BASE_DIR / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"


class VASSTPyTorch(nn.Module):
    """
    Rede Neural Convolucional para detecção de ponta de agulha
    Arquitetura baseada no VASST (TensorFlow) convertida para PyTorch

    Input: Imagem grayscale 256x256
    Output: Coordenadas normalizadas [y, x] da ponta da agulha
    """

    def __init__(self, input_shape=(256, 256)):
        super(VASSTPyTorch, self).__init__()
        self.input_shape = input_shape

        # Camadas convolucionais (mesmo padrão do VASST original)
        self.conv0 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0)

        # MaxPooling
        self.pool = nn.MaxPool2d(2, 2)

        # LeakyReLU
        self.leaky = nn.LeakyReLU(0.01)

        # Calcular tamanho após convoluções
        self._calculate_flatten_size()

        # Camadas densas
        self.fc0 = nn.Linear(self.flatten_size, 1024)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 16)
        self.output = nn.Linear(16, 2)

        # Dropout para regularização
        self.dropout = nn.Dropout(0.3)

    def _calculate_flatten_size(self):
        """Calcula o tamanho do tensor após as convoluções"""
        x = torch.zeros(1, 1, self.input_shape[0], self.input_shape[1])

        x = self.pool(self.leaky(self.conv0(x)))
        x = self.pool(self.leaky(self.conv1(x)))
        x = self.pool(self.leaky(self.conv2(x)))
        x = self.pool(self.leaky(self.conv3(x)))
        x = self.pool(self.leaky(self.conv4(x)))

        self.flatten_size = x.view(1, -1).size(1)

    def forward(self, x):
        # Convoluções com LeakyReLU e MaxPool
        x = self.pool(self.leaky(self.conv0(x)))
        x = self.pool(self.leaky(self.conv1(x)))
        x = self.pool(self.leaky(self.conv2(x)))
        x = self.pool(self.leaky(self.conv3(x)))
        x = self.pool(self.leaky(self.conv4(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Camadas densas
        x = self.leaky(self.fc0(x))
        x = self.dropout(x)
        x = self.leaky(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky(self.fc2(x))
        x = self.output(x)  # Saída linear

        return x


class NeedleDataset(Dataset):
    """Dataset PyTorch para imagens de ultrassom com agulhas"""

    def __init__(self, images, labels, augment=False):
        self.images = torch.FloatTensor(images).permute(0, 3, 1, 2) / 255.0
        self.labels = torch.FloatTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.augment:
            # Data augmentation
            if torch.rand(1) > 0.5:
                # Flip horizontal
                image = torch.flip(image, [2])
                label = label.clone()
                label[1] = 1.0 - label[1]  # Inverter coordenada x

            # Adicionar ruído
            if torch.rand(1) > 0.5:
                noise = torch.randn_like(image) * 0.05
                image = torch.clamp(image + noise, 0, 1)

            # Ajuste de brilho
            if torch.rand(1) > 0.5:
                brightness = torch.rand(1) * 0.4 + 0.8  # 0.8 - 1.2
                image = torch.clamp(image * brightness, 0, 1)

        return image, label


def load_data(data_dir: Path = None):
    """Carrega dados de treinamento"""

    # Tentar carregar dados processados primeiro
    if PROCESSED_DIR.exists():
        try:
            X_train = np.load(PROCESSED_DIR / "X_train.npy")
            Y_train = np.load(PROCESSED_DIR / "Y_train.npy")
            X_val = np.load(PROCESSED_DIR / "X_val.npy")
            Y_val = np.load(PROCESSED_DIR / "Y_val.npy")
            print(f"✅ Dados processados carregados de {PROCESSED_DIR}")
            return X_train, Y_train, X_val, Y_val
        except:
            pass

    # Tentar carregar dados sintéticos
    if SYNTHETIC_DIR.exists():
        try:
            images = np.load(SYNTHETIC_DIR / "images.npy")
            labels = np.load(SYNTHETIC_DIR / "labels.npy")

            # Dividir em treino/validação
            n = len(images)
            indices = np.random.permutation(n)
            train_end = int(0.85 * n)

            X_train = images[indices[:train_end]]
            Y_train = labels[indices[:train_end]]
            X_val = images[indices[train_end:]]
            Y_val = labels[indices[train_end:]]

            print(f"✅ Dados sintéticos carregados de {SYNTHETIC_DIR}")
            return X_train, Y_train, X_val, Y_val
        except:
            pass

    raise FileNotFoundError(
        "Nenhum dataset encontrado! Execute download_datasets.py primeiro."
    )


def train_model(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    patience: int = 15,
    save_path: str = None
):
    """
    Treina o modelo VASST

    Args:
        epochs: Número máximo de épocas
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado inicial
        patience: Paciência para early stopping
        save_path: Caminho para salvar o modelo
    """

    print("=" * 60)
    print("  NEEDLE PILOT - Treinamento VASST CNN")
    print("=" * 60)

    # Configurar device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n📱 Device: {device}")

    # Carregar dados
    print("\n📂 Carregando dados...")
    X_train, Y_train, X_val, Y_val = load_data()

    print(f"   Treino: {X_train.shape[0]} amostras")
    print(f"   Validação: {X_val.shape[0]} amostras")

    # Criar datasets e dataloaders
    train_dataset = NeedleDataset(X_train, Y_train, augment=True)
    val_dataset = NeedleDataset(X_val, Y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Criar modelo
    print("\n🧠 Criando modelo...")
    model = VASSTPyTorch(input_shape=(256, 256))
    model = model.to(device)

    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parâmetros: {total_params:,}")

    # Configurar otimizador e loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.L1Loss()  # MAE (igual ao VASST original)

    # Variáveis para tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    # Diretório para checkpoints
    if save_path is None:
        save_path = MODELS_DIR / "vasst_needle.pt"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n🎯 Iniciando treinamento...")
    print(f"   Épocas: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Salvando em: {save_path}")
    print("-" * 60)

    for epoch in range(epochs):
        # TREINO
        model.train()
        train_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # VALIDAÇÃO
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Atualizar learning rate
        scheduler.step(val_loss)

        # Calcular erro em pixels (assumindo imagem 256x256)
        pixel_error = val_loss * 256

        # Log
        print(f"Época {epoch + 1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Erro: ~{pixel_error:.1f}px")

        # Early stopping e checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Salvar melhor modelo
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'input_shape': (256, 256)
            }, save_path)
            print(f"         💾 Modelo salvo! (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping após {epoch + 1} épocas")
                break

    print("-" * 60)
    print(f"\n✅ Treinamento concluído!")
    print(f"   Melhor Val Loss: {best_val_loss:.6f}")
    print(f"   Erro aproximado: ~{best_val_loss * 256:.1f} pixels")
    print(f"   Modelo salvo em: {save_path}")

    # Salvar gráfico de loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss (MAE)')
    plt.title('VASST Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path.parent / "training_plot.png")
    print(f"   Gráfico salvo em: {save_path.parent / 'training_plot.png'}")

    return model, train_losses, val_losses


def evaluate_model(model_path: str = None):
    """Avalia o modelo em dados de teste"""

    if model_path is None:
        model_path = MODELS_DIR / "vasst_needle.pt"

    print("\n📊 Avaliando modelo...")

    # Carregar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = VASSTPyTorch()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Carregar dados de teste
    if (PROCESSED_DIR / "X_test.npy").exists():
        X_test = np.load(PROCESSED_DIR / "X_test.npy")
        Y_test = np.load(PROCESSED_DIR / "Y_test.npy")
    else:
        # Usar parte dos dados sintéticos
        images = np.load(SYNTHETIC_DIR / "images.npy")
        labels = np.load(SYNTHETIC_DIR / "labels.npy")
        n = len(images)
        X_test = images[int(0.9 * n):]
        Y_test = labels[int(0.9 * n):]

    # Preparar dados
    test_dataset = NeedleDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Avaliar
    total_error = 0.0
    pixel_errors = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            errors = torch.abs(outputs - labels) * 256  # Erro em pixels
            pixel_errors.extend(errors.cpu().numpy().flatten())

    mean_error = np.mean(pixel_errors)
    std_error = np.std(pixel_errors)

    print(f"\n📈 Resultados no conjunto de teste:")
    print(f"   Erro médio: {mean_error:.2f} ± {std_error:.2f} pixels")
    print(f"   Erro máximo: {np.max(pixel_errors):.2f} pixels")
    print(f"   Erro mínimo: {np.min(pixel_errors):.2f} pixels")

    # Visualizar algumas predições
    visualize_predictions(model, X_test[:9], Y_test[:9], device)


def visualize_predictions(model, images, labels, device):
    """Visualiza predições do modelo"""

    model.eval()
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            break

        img = images[i]
        true_label = labels[i]

        # Preparar input
        x = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0) / 255.0
        x = x.to(device)

        # Predição
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0]

        # Desnormalizar
        true_y, true_x = true_label * 256
        pred_y, pred_x = pred * 256

        # Plotar
        ax.imshow(img.squeeze(), cmap='gray')
        ax.scatter(true_x, true_y, c='green', s=100, marker='o', label='Real')
        ax.scatter(pred_x, pred_y, c='red', s=100, marker='x', label='Predição')
        ax.set_title(f'Erro: {np.sqrt((true_x-pred_x)**2 + (true_y-pred_y)**2):.1f}px')
        ax.axis('off')

    plt.legend()
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "predictions_sample.png")
    print(f"\n📸 Amostras salvas em: {MODELS_DIR / 'predictions_sample.png'}")


def main():
    print("=" * 60)
    print("  NEEDLE PILOT - Sistema de Treinamento VASST")
    print("=" * 60)

    print("""
Opções:
  1. Treinar modelo (padrão)
  2. Avaliar modelo existente
  3. Treinar + Avaliar
    """)

    choice = input("Escolha (1-3) [1]: ").strip() or "1"

    if choice == "1":
        epochs = input("Número de épocas [100]: ").strip() or "100"
        batch_size = input("Batch size [32]: ").strip() or "32"
        train_model(epochs=int(epochs), batch_size=int(batch_size))
    elif choice == "2":
        evaluate_model()
    elif choice == "3":
        epochs = input("Número de épocas [100]: ").strip() or "100"
        train_model(epochs=int(epochs))
        evaluate_model()


if __name__ == "__main__":
    main()
