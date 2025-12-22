#!/usr/bin/env python3
"""
USG FLOW - Unified Training Script
===================================
Script de treinamento unificado para TODOS os plugins de IA.

Suporta:
- NEEDLE: Detecção de ponta de agulha (regressão)
- NERVE: Segmentação de nervos/vasos/músculos
- CARDIAC: Segmentação cardíaca
- FAST: Detecção de líquido livre
- ANATOMY: Segmentação geral
- LUNG: Classificação de linhas B

Uso:
    python train_unified.py --plugin NERVE --epochs 100
    python train_unified.py --plugin NEEDLE --epochs 50 --batch_size 32
    python train_unified.py --plugin CARDIAC --resume checkpoint.pth
"""

import os
import sys
import json
import argparse
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

# Adicionar diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))
from plugin_registry import get_model_spec

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR.parent / "datasets" / "unified" / "exports"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
CONFIGS_DIR = BASE_DIR / "configs"

for d in [CHECKPOINTS_DIR, LOGS_DIR, CONFIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Configurações padrão por plugin
PLUGIN_CONFIGS = {
    "NEEDLE": {
        "task": "regression",
        "output_dim": 2,  # (y, x) da ponta
        "loss": "mse",
        "metrics": ["mae", "euclidean_distance"],
        "model": "vasst",  # CNN específica
        "input_size": (256, 256),
    },
    "NERVE": {
        "task": "segmentation",
        "num_classes": 8,  # background, nerve, artery, vein, muscle, fascia, bone, other
        "loss": "dice_focal",
        "metrics": ["dice", "iou"],
        "model": "unetpp",  # U-Net++
        "input_size": (256, 256),
    },
    "CARDIAC": {
        "task": "segmentation",
        "num_classes": 4,  # background, LV, myocardium, LA
        "loss": "dice_focal",
        "metrics": ["dice", "iou", "ef_error"],
        "model": "unetpp",
        "input_size": (256, 256),
    },
    "FAST": {
        "task": "segmentation",
        "num_classes": 3,  # background, organ, fluid
        "loss": "dice_focal",
        "metrics": ["dice", "iou"],
        "model": "unetpp",
        "input_size": (256, 256),
    },
    "ANATOMY": {
        "task": "segmentation",
        "num_classes": 10,
        "loss": "dice_focal",
        "metrics": ["dice", "iou"],
        "model": "unetpp",
        "input_size": (256, 256),
    },
    "LUNG": {
        "task": "classification",
        "num_classes": 4,  # normal, A-lines, few B-lines, many B-lines
        "loss": "cross_entropy",
        "metrics": ["accuracy", "f1"],
        "model": "efficientnet",
        "input_size": (256, 256),
    },
}


def resolve_config(plugin: str, model_key: Optional[str] = None) -> Tuple[Dict, Dict]:
    """
    Resolve configuracao dinamica com base no plugin_registry.
    Retorna (model_spec, config) para treino.
    """
    model_spec = get_model_spec(plugin, model_key)
    if not model_spec:
        raise ValueError(f"Plugin/modelo desconhecido: {plugin}/{model_key}")

    task = model_spec["task"]
    label_type = model_spec.get("label_type")

    if task == "regression":
        if label_type == "ef_esv_edv":
            output_dim = 3
        else:
            output_dim = 2
        loss = "mae"
        metrics = ["mae", "euclidean_distance"]
    elif task == "segmentation":
        output_dim = None
        loss = "cross_entropy"
        metrics = ["dice", "iou"]
    elif task == "classification":
        output_dim = None
        loss = "cross_entropy"
        metrics = ["accuracy", "f1"]
    else:
        output_dim = None
        loss = "mse"
        metrics = []

    config = {
        "task": task,
        "output_dim": output_dim,
        "loss": loss,
        "metrics": metrics,
        "model": model_spec.get("arch"),
        "input_size": model_spec.get("input_size", (256, 256)),
        "num_classes": model_spec.get("num_classes"),
        "label_type": label_type,
        "label_scale": model_spec.get("label_scale"),
        "label_order": "yx" if label_type == "point_yx" else None,
        "channels": model_spec.get("channels", 1),
        "expected_path": model_spec.get("expected_path"),
    }

    return model_spec, config


# =============================================================================
# IMPORTS PYTORCH (com fallback)
# =============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch não instalado. Instalando...")
    os.system("pip install torch torchvision")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
    TORCH_AVAILABLE = True

try:
    import cv2
except ImportError:
    os.system("pip install opencv-python")
    import cv2


# =============================================================================
# DATASET
# =============================================================================

class USGDataset(Dataset):
    """Dataset unificado para todos os plugins"""

    def __init__(
        self,
        plugin: str,
        split: str = "train",
        augment: bool = True,
        data_dir: Path = None,
        config: Dict = None
    ):
        self.plugin = plugin.upper()
        self.split = split
        self.augment = augment and (split == "train")
        self.config = config or PLUGIN_CONFIGS[self.plugin]

        # Carregar dados
        data_dir = data_dir or DATASETS_DIR / plugin.lower()

        x_file = data_dir / f"X_{split}.npy"
        y_file = data_dir / f"Y_{split}.npy"

        if not x_file.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {x_file}")

        self.images = np.load(x_file)
        self.labels = np.load(y_file)

        print(f"📊 {plugin} {split}: {len(self.images)} amostras")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        label = self.labels[idx].copy()

        # Garantir shape correto
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # Redimensionar para input_size
        target_h, target_w = self.config["input_size"]
        if image.shape[0] != target_h or image.shape[1] != target_w:
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            if self.config["task"] == "segmentation":
                label = cv2.resize(label, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Ajustar canais
        if self.config.get("channels", 1) == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif self.config.get("channels", 1) == 1 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=-1)

        # Augmentação
        if self.augment:
            image, label = self._augment(image, label)

        # Normalizar imagem para [0, 1]
        image = image.astype(np.float32) / 255.0

        # Converter para tensor
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        if self.config["task"] == "regression":
            label = label.astype(np.float32)
            if self.config.get("label_scale") == "neg1_1":
                label = label * 2.0 - 1.0
            label = torch.from_numpy(label)
        elif self.config["task"] == "segmentation":
            label = torch.from_numpy(label.astype(np.int64))
        elif self.config["task"] == "classification":
            label = torch.tensor(label, dtype=torch.long)

        return image, label

    def _augment(self, image, label):
        """Augmentação de dados"""
        h, w = image.shape[:2]

        # Flip horizontal
        if np.random.random() > 0.5:
            image = np.fliplr(image).copy()
            if self.config["task"] == "regression":
                label[1] = 1.0 - label[1]  # Inverter x
            elif self.config["task"] == "segmentation":
                label = np.fliplr(label).copy()

        # Flip vertical
        if np.random.random() > 0.5:
            image = np.flipud(image).copy()
            if self.config["task"] == "regression":
                label[0] = 1.0 - label[0]  # Inverter y
            elif self.config["task"] == "segmentation":
                label = np.flipud(label).copy()

        # Rotação pequena
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            image = cv2.warpAffine(image, M, (w, h))
            if self.config["task"] == "segmentation":
                label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST)

        # Brilho/contraste
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contraste
            beta = np.random.uniform(-20, 20)    # Brilho
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

        # Ruído gaussiano
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 10, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        return image, label


# =============================================================================
# MODELOS
# =============================================================================

class ConvBlock(nn.Module):
    """Bloco convolucional básico"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
        )
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel attention
        avg = self.avg_pool(x).view(b, c)
        max_p = self.max_pool(x).view(b, c)
        ca = torch.sigmoid(self.fc(avg) + self.fc(max_p)).view(b, c, 1, 1)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * sa

        return x


class UNetPP(nn.Module):
    """U-Net++ para segmentação"""
    def __init__(self, in_channels=1, num_classes=8, features=[32, 64, 128, 256, 512]):
        super().__init__()
        self.num_classes = num_classes

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(in_ch, f))
            in_ch = f

        # Decoder
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.upconvs.append(nn.ConvTranspose2d(features[i], features[i-1], 2, 2))
            self.decoders.append(ConvBlock(features[i-1] * 2, features[i-1]))

        # CBAM
        self.cbam = nn.ModuleList([CBAM(f) for f in features])

        # Pool e final
        self.pool = nn.MaxPool2d(2, 2)
        self.final = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        # Encoder
        enc_features = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            x = self.cbam[i](x)
            enc_features.append(x)
            if i < len(self.encoders) - 1:
                x = self.pool(x)

        # Decoder
        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            skip = enc_features[-(i+2)]
            # Ajustar tamanho se necessário
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.final(x)


class VASSTPyTorch(nn.Module):
    """VASST CNN compatível com a inferência do app"""
    def __init__(self, input_shape: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.input_shape = input_shape

        self.conv0 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0)

        self.pool = nn.MaxPool2d(2, 2)
        self.leaky = nn.LeakyReLU(0.01)

        self._calculate_flatten_size()

        self.fc0 = nn.Linear(self.flatten_size, 1024)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 16)
        self.output = nn.Linear(16, 2)

    def _calculate_flatten_size(self):
        x = torch.zeros(1, 1, self.input_shape[0], self.input_shape[1])
        x = self.pool(self.leaky(self.conv0(x)))
        x = self.pool(self.leaky(self.conv1(x)))
        x = self.pool(self.leaky(self.conv2(x)))
        x = self.pool(self.leaky(self.conv3(x)))
        x = self.pool(self.leaky(self.conv4(x)))
        self.flatten_size = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(self.leaky(self.conv0(x)))
        x = self.pool(self.leaky(self.conv1(x)))
        x = self.pool(self.leaky(self.conv2(x)))
        x = self.pool(self.leaky(self.conv3(x)))
        x = self.pool(self.leaky(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.leaky(self.fc0(x))
        x = self.leaky(self.fc1(x))
        x = self.leaky(self.fc2(x))
        return self.output(x)


class EchoNetRegression(nn.Module):
    """EchoNet-like regression para EF/ESV/EDV (compatível com inferência)."""
    def __init__(self):
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet18(weights=None)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class EfficientNetClassifier(nn.Module):
    """Classificador baseado em EfficientNet-like"""
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def create_model(config: Dict, device: str = "cpu") -> nn.Module:
    """Cria modelo para o plugin especificado com base no registry."""
    arch = config["model"]

    if arch == "smp_unet":
        try:
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=config.get("channels", 1),
                classes=config.get("num_classes", 2),
                activation=None,
            )
        except ImportError as e:
            raise RuntimeError("segmentation_models_pytorch nao instalado") from e
    elif arch == "vasst":
        model = VASSTPyTorch(input_shape=config.get("input_size", (256, 256)))
    elif arch == "echonet":
        model = EchoNetRegression()
    elif arch == "efficientnet":
        model = EfficientNetClassifier(
            in_channels=config.get("channels", 1),
            num_classes=config.get("num_classes", 4),
        )
    elif arch == "nerve_track":
        try:
            from nerve_track.nerve_model import NerveSegmentationModel
            base_model = NerveSegmentationModel(
                num_classes=config.get("num_classes", 6),
                use_temporal=True,
            )
            class _Wrapper(nn.Module):
                def __init__(self, inner: nn.Module):
                    super().__init__()
                    self.inner = inner

                def forward(self, x):
                    return self.inner(x)["logits"]

            model = _Wrapper(base_model)
        except Exception as e:
            raise RuntimeError(f"NerveTrack model indisponivel: {e}") from e
    else:
        raise ValueError(f"Modelo desconhecido: {arch}")

    return model.to(device)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class DiceLoss(nn.Module):
    """Dice Loss para segmentação"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss para classes desbalanceadas"""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


class DiceFocalLoss(nn.Module):
    """Combinação de Dice + Focal Loss"""
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        return self.dice_weight * self.dice(pred, target) + \
               self.focal_weight * self.focal(pred, target)


def create_loss(config: Dict) -> nn.Module:
    """Cria função de loss para o plugin"""
    loss = config.get("loss", "mae")

    if loss == "mse":
        return nn.MSELoss()
    if loss == "mae":
        return nn.L1Loss()
    if loss == "dice_focal":
        return DiceFocalLoss()
    if loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Loss desconhecida: {loss}")


# =============================================================================
# MÉTRICAS
# =============================================================================

def compute_metrics(pred, target, config: Dict) -> Dict[str, float]:
    """Calcula métricas para o plugin"""
    metrics: Dict[str, float] = {}

    if config["task"] == "regression":
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # MAE
        metrics["mae"] = np.mean(np.abs(pred_np - target_np))

        # Distância euclidiana
        if pred_np.shape[1] == 2:
            dist = np.sqrt(np.sum((pred_np - target_np) ** 2, axis=1))
            pixel_scale = config.get("input_size", (256, 256))[0]
            metrics["euclidean_distance"] = np.mean(dist) * pixel_scale

    elif config["task"] == "segmentation":
        pred_np = pred.argmax(dim=1).detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # Dice por classe
        dice_scores = []
        iou_scores = []
        for c in range(1, config["num_classes"]):  # Ignorar background
            pred_c = (pred_np == c)
            target_c = (target_np == c)

            intersection = (pred_c & target_c).sum()
            union = pred_c.sum() + target_c.sum()

            if union > 0:
                dice = 2 * intersection / union
                iou = intersection / (union - intersection + 1e-6)
                dice_scores.append(dice)
                iou_scores.append(iou)

        metrics["dice"] = np.mean(dice_scores) if dice_scores else 0.0
        metrics["iou"] = np.mean(iou_scores) if iou_scores else 0.0

    elif config["task"] == "classification":
        pred_np = pred.argmax(dim=1).detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        metrics["accuracy"] = (pred_np == target_np).mean()

    return metrics


# =============================================================================
# TREINAMENTO
# =============================================================================

class Trainer:
    """Treinador unificado"""

    def __init__(
        self,
        plugin: str,
        model: nn.Module,
        device: str,
        config: Dict,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        self.plugin = plugin.upper()
        self.model = model
        self.device = device
        self.config = config

        # Loss e otimizador
        self.criterion = create_loss(config)
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Histórico
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_state_dict = None

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Treina uma época"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.append(outputs.detach())
            all_targets.append(labels.detach())

        avg_loss = total_loss / len(dataloader)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_preds, all_targets, self.config)

        return avg_loss, metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Valida o modelo"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.append(outputs)
            all_targets.append(labels)

        avg_loss = total_loss / len(dataloader)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_preds, all_targets, self.config)

        return avg_loss, metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 10,
        save_dir: Path = None,
        export_path: Optional[str] = None,
    ):
        """Loop de treinamento completo"""
        save_dir = save_dir or CHECKPOINTS_DIR / self.plugin.lower()
        save_dir.mkdir(parents=True, exist_ok=True)

        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)

        print(f"\n{'='*60}")
        print(f"🚀 TREINAMENTO: {self.plugin}")
        print(f"{'='*60}")
        print(f"   Épocas: {epochs}")
        print(f"   Dispositivo: {self.device}")
        print(f"   Modelo: {self.config['model']}")
        print(f"   Loss: {self.config['loss']}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            # Treinar
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["train_metrics"].append(train_metrics)

            # Validar
            val_loss, val_metrics = self.validate(val_loader)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)

            # Scheduler
            scheduler.step(val_loss)

            # Log
            main_metric = list(val_metrics.values())[0] if val_metrics else 0
            print(f"Época {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val {list(val_metrics.keys())[0] if val_metrics else 'metric'}: {main_metric:.4f}")

            # Early stopping e checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Salvar melhor modelo
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "config": self.config,
                }
                torch.save(checkpoint, save_dir / "best_model.pth")
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                print(f"   ✅ Melhor modelo salvo! (loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n⚠️ Early stopping na época {epoch+1}")
                    break

        # Salvar histórico
        with open(save_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # Salvar modelo final
        torch.save(self.model.state_dict(), save_dir / "final_model.pth")

        # Export para inferência (state_dict + metadata)
        if export_path:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            state_dict = self.best_state_dict or self.model.state_dict()
            torch.save(state_dict, export_path)

            meta = {
                "plugin": self.plugin,
                "task": self.config.get("task"),
                "label_type": self.config.get("label_type"),
                "label_scale": self.config.get("label_scale"),
                "label_order": self.config.get("label_order"),
                "input_size": self.config.get("input_size"),
                "num_classes": self.config.get("num_classes"),
                "exported_at": datetime.now().isoformat(),
            }
            meta_path = export_path.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"   Exportado para inferência: {export_path}")

        print(f"\n{'='*60}")
        print(f"✅ TREINAMENTO CONCLUÍDO!")
        print(f"   Melhor Val Loss: {self.best_val_loss:.4f}")
        print(f"   Checkpoints em: {save_dir}")
        print(f"{'='*60}")

        return self.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="USG Flow - Unified Training")
    parser.add_argument("--plugin", type=str, required=True,
                       choices=["NEEDLE", "NERVE", "CARDIAC", "FAST", "ANATOMY", "LUNG"],
                       help="Plugin para treinar")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/mps/auto)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint para resumir")
    parser.add_argument("--data_dir", type=str, default=None, help="Diretório de dados")
    parser.add_argument("--model", type=str, default=None, help="Modelo especifico do plugin")

    args = parser.parse_args()

    # Determinar dispositivo
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"\n🖥️ Usando dispositivo: {device}")

    # Config e validação
    model_spec, config = resolve_config(args.plugin, args.model)
    if config["task"] == "detection":
        print("\n❌ Este treinamento nao suporta deteccao (YOLO).")
        print("   Use: python training/train_yolo.py --plugin FAST/NEEDLE")
        return

    # Carregar dados
    data_dir = Path(args.data_dir) if args.data_dir else DATASETS_DIR / args.plugin.lower()

    try:
        train_dataset = USGDataset(
            args.plugin,
            "train",
            augment=True,
            data_dir=data_dir,
            config=config,
        )
        val_dataset = USGDataset(
            args.plugin,
            "val",
            augment=False,
            data_dir=data_dir,
            config=config,
        )
    except FileNotFoundError as e:
        print(f"\n❌ ERRO: {e}")
        print(f"\nPara treinar o plugin {args.plugin}, primeiro execute:")
        print(f"   python datasets/unified_dataset_manager.py")
        print(f"\nE escolha a opção 7 para setup completo.")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Criar modelo
    model = create_model(config, device)
    print(f"\n📦 Modelo criado: {config['model']}")
    print(f"   Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    # Resumir treinamento
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Checkpoint carregado: {args.resume}")

    # Treinar
    trainer = Trainer(args.plugin, model, device, config, learning_rate=args.lr)
    trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        patience=args.patience,
        export_path=config.get("expected_path"),
    )


if __name__ == "__main__":
    main()
