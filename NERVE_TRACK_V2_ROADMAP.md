# 🧠 NERVE TRACK v2.0 PREMIUM - ROADMAP COMPLETO

## Objetivo: Criar o melhor sistema de detecção de nervos do mundo

Baseado em pesquisa extensiva de: Clarius, Butterfly iQ, GE cNerve, ScanNav PNB,
Nerveblox, papers científicos 2024-2025, GitHub, Kaggle.

---

# 📋 ÍNDICE

1. [Visão Geral da Arquitetura](#1-visão-geral-da-arquitetura)
2. [Fase 1: Preparação e Infraestrutura](#2-fase-1-preparação-e-infraestrutura)
3. [Fase 2: Modelo de Segmentação Premium](#3-fase-2-modelo-de-segmentação-premium)
4. [Fase 3: Tracking Temporal Anti-Flicker](#4-fase-3-tracking-temporal-anti-flicker)
5. [Fase 4: Classificação Nervo/Artéria/Veia](#5-fase-4-classificação-nervoartériaveia)
6. [Fase 5: Identificação de Nervos Específicos](#6-fase-5-identificação-de-nervos-específicos)
7. [Fase 6: Medição Automática de CSA](#7-fase-6-medição-automática-de-csa)
8. [Fase 7: Visual Premium](#8-fase-7-visual-premium)
9. [Fase 8: Testes e Validação](#9-fase-8-testes-e-validação)
10. [Checklist Final](#10-checklist-final)

---

# 1. VISÃO GERAL DA ARQUITETURA

## 1.1 Arquitetura Atual vs Proposta

```
ATUAL (v1.0)                          PROPOSTA (v2.0 PREMIUM)
─────────────────                     ─────────────────────────
Frame                                 Frame
  │                                     │
  ▼                                     ▼
U-Net básico                          ┌─────────────────────────┐
(ResNet34, 3 classes)                 │ PRÉ-PROCESSAMENTO       │
  │                                   │ ├─ CLAHE adaptativo     │
  ▼                                   │ ├─ Speckle reduction    │
HoughCircles fallback                 │ └─ ROI detection        │
  │                                   └─────────────────────────┘
  ▼                                     │
Desenho direto                          ▼
(sem tracking)                        ┌─────────────────────────┐
  │                                   │ ENCODER                 │
  ▼                                   │ EfficientNet-B4         │
Escala fixa 0.3mm/px                  │ (pretrained ImageNet)   │
                                      │ + CBAM attention        │
                                      └─────────────────────────┘
                                        │
                                        ▼
                                      ┌─────────────────────────┐
                                      │ TEMPORAL MODULE         │
                                      │ ConvLSTM bottleneck     │
                                      │ + Memory Bank           │
                                      └─────────────────────────┘
                                        │
                                        ▼
                                      ┌─────────────────────────┐
                                      │ DECODER U-Net++         │
                                      │ Skip connections        │
                                      │ aninhados + attention   │
                                      └─────────────────────────┘
                                        │
                                        ▼
                                      ┌─────────────────────────┐
                                      │ MULTI-HEAD OUTPUT       │
                                      │ ├─ Segmentação (4 cls)  │
                                      │ ├─ Nerve ID (8 tipos)   │
                                      │ └─ CSA regression       │
                                      └─────────────────────────┘
                                        │
                                        ▼
                                      ┌─────────────────────────┐
                                      │ NERVE TRACKER           │
                                      │ ├─ Kalman Filter        │
                                      │ ├─ Temporal smoothing   │
                                      │ └─ Confidence decay     │
                                      └─────────────────────────┘
                                        │
                                        ▼
                                      ┌─────────────────────────┐
                                      │ POST-PROCESSING         │
                                      │ ├─ CRF refinement       │
                                      │ ├─ Morphological ops    │
                                      │ └─ Connected components │
                                      └─────────────────────────┘
                                        │
                                        ▼
                                      ┌─────────────────────────┐
                                      │ VISUAL PREMIUM          │
                                      │ ├─ CSA display          │
                                      │ ├─ Confidence overlay   │
                                      │ ├─ Nerve identification │
                                      │ └─ Quality indicator    │
                                      └─────────────────────────┘
```

## 1.2 Classes de Segmentação

```python
CLASSES = {
    0: "BACKGROUND",    # Fundo
    1: "NERVE",         # Nervo (amarelo)
    2: "ARTERY",        # Artéria (vermelho)
    3: "VEIN",          # Veia (azul)
}

NERVE_TYPES = {
    0: "UNKNOWN",           # Nervo não identificado
    1: "MEDIAN",            # Nervo Mediano
    2: "ULNAR",             # Nervo Ulnar
    3: "RADIAL",            # Nervo Radial
    4: "SCIATIC",           # Nervo Ciático
    5: "FEMORAL",           # Nervo Femoral
    6: "BRACHIAL_PLEXUS",   # Plexo Braquial
    7: "TIBIAL",            # Nervo Tibial
    8: "PERONEAL",          # Nervo Peroneal
}
```

---

# 2. FASE 1: PREPARAÇÃO E INFRAESTRUTURA

## 2.1 Dependências a Instalar

```bash
# Ativar ambiente virtual
source /Users/priscoleao/usgapp/.venv/bin/activate

# Instalar novas dependências
pip install segmentation-models-pytorch  # Arquiteturas de segmentação
pip install timm                          # Encoders pretrained
pip install albumentations                # Data augmentation avançado
pip install pydensecrf                    # CRF post-processing
pip install filterpy                      # Kalman filter (se não instalado)
pip install scikit-image                  # Morphological operations
pip install einops                        # Tensor operations
```

## 2.2 Estrutura de Arquivos a Criar

```
aplicativo-usg-final/
├── src/
│   ├── ai_processor.py          # Modificar (adicionar NerveTrackV2)
│   ├── nerve_tracker.py         # NOVO: Tracker temporal
│   ├── nerve_classifier.py      # NOVO: Classificação de nervos
│   └── nerve_postprocess.py     # NOVO: Post-processing
├── models/
│   ├── nerve_segmentation/      # NOVO: Pasta para modelos
│   │   ├── nerve_unetpp.pt      # Modelo principal
│   │   ├── nerve_classifier.pt  # Classificador de tipo
│   │   └── config.json          # Configurações do modelo
│   └── ...
├── datasets/
│   ├── nerve_segmentation/      # NOVO: Dataset para treino
│   │   ├── images/
│   │   ├── masks/
│   │   └── annotations.json
│   └── ...
└── training/
    ├── train_nerve_segmentation.py  # NOVO: Script de treino
    ├── augmentations.py             # NOVO: Augmentations
    └── losses.py                    # NOVO: Loss functions
```

## 2.3 Configurações em config.py

```python
# ══════════════════════════════════════════════════════════════════════════════
# NERVE TRACK v2.0 PREMIUM - CONFIGURAÇÕES
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# MODELO DE SEGMENTAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
NERVE_MODEL_ENCODER = "efficientnet-b4"      # Encoder backbone
NERVE_MODEL_WEIGHTS = "imagenet"             # Pretrained weights
NERVE_MODEL_CLASSES = 4                      # Background, Nerve, Artery, Vein
NERVE_INPUT_SIZE = (256, 256)                # Tamanho de entrada do modelo
NERVE_RESOLUTION_SCALE = 0.5                 # Escala para inferência

# ─────────────────────────────────────────────────────────────────────────────
# TRACKING TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────
NERVE_TRACKING_ENABLED = True                # Habilitar tracking
NERVE_MAX_LOST_FRAMES = 15                   # Frames para manter predição
NERVE_SMOOTHING_FACTOR = 0.7                 # Suavização temporal (0-1)
NERVE_MIN_CONFIDENCE = 0.3                   # Confiança mínima para exibir
NERVE_KALMAN_PROCESS_NOISE = 0.1             # Ruído do processo Kalman
NERVE_KALMAN_MEASUREMENT_NOISE = 5.0         # Ruído da medição Kalman

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICAÇÃO DE ESTRUTURAS
# ─────────────────────────────────────────────────────────────────────────────
NERVE_CLASSIFY_ENABLED = True                # Habilitar classificação
NERVE_PULSATILITY_FRAMES = 30                # Frames para detectar pulsação
NERVE_COMPRESSIBILITY_THRESHOLD = 0.3        # Threshold de compressibilidade
NERVE_MIN_AREA_PX = 100                      # Área mínima para considerar

# ─────────────────────────────────────────────────────────────────────────────
# MEDIÇÃO DE CSA
# ─────────────────────────────────────────────────────────────────────────────
NERVE_CSA_ENABLED = True                     # Habilitar medição CSA
NERVE_CSA_NORMAL_MAX = 10.0                  # CSA normal máximo (mm²)
NERVE_CSA_CTS_MODERATE = 6.0                 # Diferença para CTS moderado
NERVE_CSA_CTS_SEVERE = 9.0                   # Diferença para CTS severo

# ─────────────────────────────────────────────────────────────────────────────
# IDENTIFICAÇÃO DE NERVOS
# ─────────────────────────────────────────────────────────────────────────────
NERVE_IDENTIFICATION_ENABLED = True          # Habilitar identificação
NERVE_ID_CONFIDENCE_THRESHOLD = 0.6          # Confiança mínima para ID

# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
NERVE_CRF_ENABLED = True                     # Habilitar CRF
NERVE_MORPHOLOGY_ENABLED = True              # Habilitar operações morfológicas
NERVE_MIN_CONTOUR_AREA = 50                  # Área mínima de contorno

# ─────────────────────────────────────────────────────────────────────────────
# VISUAL
# ─────────────────────────────────────────────────────────────────────────────
NERVE_SHOW_CONFIDENCE = True                 # Mostrar confiança
NERVE_SHOW_CSA = True                        # Mostrar CSA
NERVE_SHOW_TRACKING_STATUS = True            # Mostrar status do tracking
NERVE_SHOW_QUALITY_INDICATOR = True          # Mostrar qualidade do scan
NERVE_TRAIL_LENGTH = 30                      # Comprimento do trail
NERVE_OVERLAY_ALPHA = 0.4                    # Transparência do overlay

# ─────────────────────────────────────────────────────────────────────────────
# CORES (BGR)
# ─────────────────────────────────────────────────────────────────────────────
NERVE_COLOR_NERVE = (0, 255, 255)            # Amarelo
NERVE_COLOR_ARTERY = (0, 0, 255)             # Vermelho
NERVE_COLOR_VEIN = (255, 100, 100)           # Azul
NERVE_COLOR_TRACKING = (0, 255, 0)           # Verde (tracking ativo)
NERVE_COLOR_PREDICTING = (0, 165, 255)       # Laranja (predição)
NERVE_COLOR_LOST = (128, 128, 128)           # Cinza (perdido)
```

---

# 3. FASE 2: MODELO DE SEGMENTAÇÃO PREMIUM

## 3.1 Criar Arquivo: src/models/nerve_model.py

```python
"""
NERVE TRACK v2.0 - Modelo de Segmentação Premium

Arquitetura: U-Net++ com EfficientNet-B4 encoder e CBAM attention
Baseado em: segmentation_models_pytorch + attention customizado
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, Tuple, Dict, List
import numpy as np


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Combina Channel Attention + Spatial Attention

    Paper: https://arxiv.org/abs/1807.06521
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att

        return x


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell para processamento temporal

    Mantém estado entre frames para consistência temporal
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()

        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,  # i, f, o, g gates
            kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x: torch.Tensor,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if state is None:
            h = torch.zeros(x.size(0), self.hidden_channels, x.size(2), x.size(3),
                          device=x.device, dtype=x.dtype)
            c = torch.zeros_like(h)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, (h_new, c_new)


class NerveSegmentationModel(nn.Module):
    """
    Modelo principal de segmentação de nervos

    Features:
    - U-Net++ com EfficientNet-B4 encoder
    - CBAM attention em cada nível do decoder
    - ConvLSTM para consistência temporal
    - Multi-head output (segmentação + classificação + CSA)
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        num_classes: int = 4,
        num_nerve_types: int = 9,
        use_temporal: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()

        self.use_temporal = use_temporal
        self.use_attention = use_attention
        self.num_classes = num_classes
        self.num_nerve_types = num_nerve_types

        # Backbone: U-Net++ com encoder pretrained
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,  # Grayscale ultrasound
            classes=num_classes,
            activation=None,  # Aplicamos depois
        )

        # Obter número de canais do encoder
        encoder_channels = self.backbone.encoder.out_channels

        # CBAM attention para cada nível do decoder
        if use_attention:
            self.attention_blocks = nn.ModuleList([
                CBAM(ch) for ch in encoder_channels[1:]  # Excluir primeiro (input)
            ])

        # ConvLSTM para processamento temporal
        if use_temporal:
            # Aplicar no bottleneck (feature mais profunda)
            bottleneck_channels = encoder_channels[-1]
            self.temporal_lstm = ConvLSTMCell(
                input_channels=bottleneck_channels,
                hidden_channels=bottleneck_channels,
                kernel_size=3
            )
            self.lstm_state = None

        # Head de classificação de tipo de nervo
        self.nerve_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_nerve_types),
        )

        # Head de regressão de CSA (Cross-Sectional Area)
        self.csa_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.ReLU(),  # CSA sempre positivo
        )

    def reset_temporal_state(self):
        """Reset do estado temporal (para novo vídeo/sequência)"""
        self.lstm_state = None

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor (B, 1, H, W)
            return_features: Se True, retorna features intermediárias

        Returns:
            Dict com:
            - 'segmentation': (B, num_classes, H, W)
            - 'nerve_type': (B, num_nerve_types)
            - 'csa': (B, 1)
            - 'features': (opcional) features do encoder
        """
        # Encoder
        features = self.backbone.encoder(x)

        # Aplicar CBAM attention
        if self.use_attention:
            enhanced_features = [features[0]]  # Input sem modificação
            for i, (feat, attn) in enumerate(zip(features[1:], self.attention_blocks)):
                enhanced_features.append(attn(feat))
            features = enhanced_features

        # Aplicar ConvLSTM no bottleneck
        if self.use_temporal:
            bottleneck = features[-1]
            temporal_out, self.lstm_state = self.temporal_lstm(bottleneck, self.lstm_state)
            features = list(features[:-1]) + [temporal_out]

        # Decoder
        decoder_output = self.backbone.decoder(*features)
        segmentation = self.backbone.segmentation_head(decoder_output)

        # Classificação de tipo de nervo
        nerve_type = self.nerve_classifier(features[-1])

        # Regressão de CSA
        csa = self.csa_regressor(features[-1])

        output = {
            'segmentation': segmentation,
            'nerve_type': nerve_type,
            'csa': csa,
        }

        if return_features:
            output['features'] = features

        return output

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Predição com pós-processamento

        Args:
            x: Input tensor
            threshold: Threshold para binarização

        Returns:
            Dict com predições numpy
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)

            # Segmentação
            seg_probs = F.softmax(output['segmentation'], dim=1)
            seg_pred = seg_probs.argmax(dim=1)

            # Tipo de nervo
            nerve_probs = F.softmax(output['nerve_type'], dim=1)
            nerve_pred = nerve_probs.argmax(dim=1)
            nerve_conf = nerve_probs.max(dim=1).values

            # CSA
            csa_pred = output['csa']

        return {
            'segmentation': seg_pred.cpu().numpy(),
            'seg_probs': seg_probs.cpu().numpy(),
            'nerve_type': nerve_pred.cpu().numpy(),
            'nerve_confidence': nerve_conf.cpu().numpy(),
            'csa': csa_pred.cpu().numpy(),
        }


class NerveSegmentationLoss(nn.Module):
    """
    Loss function combinada para segmentação de nervos

    Combina:
    - Dice Loss (overlap)
    - Focal Loss (hard examples)
    - BCE Loss (pixel accuracy)
    - Cross Entropy (classificação)
    - MSE (CSA regression)
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.3,
        bce_weight: float = 0.2,
        class_weight: float = 0.1,
        csa_weight: float = 0.1,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.class_weight = class_weight
        self.csa_weight = csa_weight
        self.focal_gamma = focal_gamma

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Dice loss para segmentação"""
        pred = F.softmax(pred, dim=1)

        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal loss para exemplos difíceis"""
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.focal_gamma) * ce
        return focal.mean()

    def forward(
        self,
        pred_seg: torch.Tensor,
        pred_class: torch.Tensor,
        pred_csa: torch.Tensor,
        target_seg: torch.Tensor,
        target_class: torch.Tensor,
        target_csa: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula loss total

        Returns:
            Dict com loss total e componentes individuais
        """
        # Segmentação losses
        dice = self.dice_loss(pred_seg, target_seg)
        focal = self.focal_loss(pred_seg, target_seg)
        bce = F.cross_entropy(pred_seg, target_seg)

        seg_loss = (
            self.dice_weight * dice +
            self.focal_weight * focal +
            self.bce_weight * bce
        )

        # Classificação loss
        class_loss = self.ce_loss(pred_class, target_class)

        # CSA regression loss
        csa_loss = self.mse_loss(pred_csa.squeeze(), target_csa)

        # Total
        total_loss = (
            seg_loss +
            self.class_weight * class_loss +
            self.csa_weight * csa_loss
        )

        return {
            'total': total_loss,
            'segmentation': seg_loss,
            'dice': dice,
            'focal': focal,
            'classification': class_loss,
            'csa': csa_loss,
        }


def create_nerve_model(
    encoder: str = "efficientnet-b4",
    pretrained: bool = True,
    temporal: bool = True,
    attention: bool = True,
) -> NerveSegmentationModel:
    """
    Factory function para criar modelo

    Args:
        encoder: Nome do encoder (efficientnet-b4, resnet50, etc)
        pretrained: Usar weights pretrained
        temporal: Habilitar ConvLSTM temporal
        attention: Habilitar CBAM attention

    Returns:
        Modelo configurado
    """
    weights = "imagenet" if pretrained else None

    model = NerveSegmentationModel(
        encoder_name=encoder,
        encoder_weights=weights,
        num_classes=4,
        num_nerve_types=9,
        use_temporal=temporal,
        use_attention=attention,
    )

    return model
```

## 3.2 Criar Arquivo: training/augmentations.py

```python
"""
Data Augmentation para Ultrassom de Nervos

Baseado em pesquisa de papers 2024 sobre augmentation para US
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2


def get_training_augmentations(image_size: tuple = (256, 256)) -> A.Compose:
    """
    Augmentations para treino

    Inclui:
    - Transformações geométricas
    - Transformações de intensidade (específicas para US)
    - Elastic transform (simula deformação tecidual)
    """
    return A.Compose([
        # Resize
        A.Resize(image_size[0], image_size[1]),

        # ═══════════════════════════════════════════════════════════
        # GEOMÉTRICAS
        # ═══════════════════════════════════════════════════════════
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),

        A.ShiftScaleRotate(
            shift_limit=0.2,      # Translação ±20%
            scale_limit=0.2,      # Escala 80-120%
            rotate_limit=10,      # Rotação ±10°
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.8
        ),

        A.Affine(
            shear=(-2, 2),        # Shear ±2°
            p=0.3
        ),

        # Elastic transform (simula deformação de tecido)
        A.ElasticTransform(
            alpha=50,
            sigma=10,
            alpha_affine=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3
        ),

        # ═══════════════════════════════════════════════════════════
        # INTENSIDADE (CRÍTICO PARA ULTRASSOM)
        # ═══════════════════════════════════════════════════════════

        # CLAHE - melhora contraste local
        A.CLAHE(
            clip_limit=(1.0, 4.0),
            tile_grid_size=(8, 8),
            p=0.5
        ),

        # Gamma correction
        A.RandomGamma(
            gamma_limit=(70, 130),  # 0.7 - 1.3
            p=0.5
        ),

        # Brightness e contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        # Speckle noise (específico para US)
        A.GaussNoise(
            var_limit=(10, 50),
            mean=0,
            p=0.4
        ),

        # Multiplicative noise (simula speckle melhor)
        A.MultiplicativeNoise(
            multiplier=(0.9, 1.1),
            per_channel=False,
            p=0.3
        ),

        # Blur (simula foco ruim)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ], p=0.2),

        # ═══════════════════════════════════════════════════════════
        # NORMALIZAÇÃO
        # ═══════════════════════════════════════════════════════════
        A.Normalize(
            mean=[0.485],  # ImageNet mean (grayscale)
            std=[0.229],   # ImageNet std
            max_pixel_value=255.0,
        ),

        ToTensorV2(),
    ])


def get_validation_augmentations(image_size: tuple = (256, 256)) -> A.Compose:
    """
    Augmentations para validação (apenas resize e normalize)
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485],
            std=[0.229],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_inference_augmentations(image_size: tuple = (256, 256)) -> A.Compose:
    """
    Augmentations para inferência (apenas resize e normalize)
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485],
            std=[0.229],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_tta_augmentations(image_size: tuple = (256, 256)) -> list:
    """
    Test Time Augmentation (TTA)

    Retorna lista de augmentations para TTA
    """
    base = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    return [
        base,
        A.Compose([A.HorizontalFlip(p=1.0)] + base.transforms),
        A.Compose([A.VerticalFlip(p=1.0)] + base.transforms),
        A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.RandomGamma(gamma_limit=(80, 80), p=1.0),
            A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=255.0),
            ToTensorV2(),
        ]),
    ]
```

---

# 4. FASE 3: TRACKING TEMPORAL ANTI-FLICKER

## 4.1 Criar Arquivo: src/nerve_tracker.py

```python
"""
NERVE TRACKER - Sistema de Tracking Temporal para Nervos

Combina:
- Kalman Filter para suavização e predição
- Temporal smoothing para consistência
- Confidence decay quando perde detecção
- Memory de estruturas para re-identificação

Similar ao NeedleTracker do NEEDLE PILOT v3.1
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import cv2


@dataclass
class TrackedStructure:
    """Representa uma estrutura rastreada (nervo, artéria ou veia)"""
    id: int
    structure_type: str  # 'NERVE', 'ARTERY', 'VEIN'
    center: Tuple[int, int]
    contour: np.ndarray
    area: float
    diameter_mm: float
    confidence: float
    nerve_type: Optional[str] = None  # 'MEDIAN', 'ULNAR', etc
    csa_mm2: Optional[float] = None
    frames_tracked: int = 0
    frames_lost: int = 0
    kalman: Optional[KalmanFilter] = None
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    color: Tuple[int, int, int] = (0, 255, 255)


class NerveTracker:
    """
    Sistema de tracking temporal para estruturas neurovasculares

    Features:
    - Kalman Filter para cada estrutura
    - Associação de detecções entre frames (Hungarian algorithm)
    - Predição quando estrutura desaparece
    - Suavização temporal de posições
    - Confidence decay
    """

    def __init__(
        self,
        max_lost_frames: int = 15,
        smoothing_factor: float = 0.7,
        min_confidence: float = 0.3,
        iou_threshold: float = 0.3,
        process_noise: float = 0.1,
        measurement_noise: float = 5.0,
    ):
        self.max_lost_frames = max_lost_frames
        self.smoothing_factor = smoothing_factor
        self.min_confidence = min_confidence
        self.iou_threshold = iou_threshold
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.tracked_structures: Dict[int, TrackedStructure] = {}
        self.next_id = 0
        self.frame_count = 0

    def _create_kalman(self, initial_state: Tuple[float, float, float, float]) -> KalmanFilter:
        """
        Cria Kalman Filter para tracking de estrutura

        Estado: [cx, cy, area, diameter, vcx, vcy]
        Medição: [cx, cy, area, diameter]
        """
        kf = KalmanFilter(dim_x=6, dim_z=4)

        dt = 1/30  # ~30 FPS

        # Matriz de transição
        kf.F = np.array([
            [1, 0, 0, 0, dt, 0],   # cx
            [0, 1, 0, 0, 0, dt],   # cy
            [0, 0, 1, 0, 0, 0],    # area
            [0, 0, 0, 1, 0, 0],    # diameter
            [0, 0, 0, 0, 1, 0],    # vcx
            [0, 0, 0, 0, 0, 1],    # vcy
        ])

        # Matriz de medição
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ])

        # Ruído do processo
        kf.Q = np.eye(6) * self.process_noise
        kf.Q[4, 4] = 0.5  # Velocidade pode variar mais
        kf.Q[5, 5] = 0.5

        # Ruído da medição
        kf.R = np.eye(4) * self.measurement_noise

        # Covariância inicial
        kf.P *= 100

        # Estado inicial
        cx, cy, area, diameter = initial_state
        kf.x = np.array([[cx], [cy], [area], [diameter], [0], [0]])

        return kf

    def _compute_iou(self, contour1: np.ndarray, contour2: np.ndarray,
                     img_shape: Tuple[int, int]) -> float:
        """Calcula IoU entre dois contornos"""
        mask1 = np.zeros(img_shape[:2], dtype=np.uint8)
        mask2 = np.zeros(img_shape[:2], dtype=np.uint8)

        cv2.drawContours(mask1, [contour1], -1, 255, -1)
        cv2.drawContours(mask2, [contour2], -1, 255, -1)

        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        return intersection / (union + 1e-6)

    def _match_detections(
        self,
        detections: List[Dict],
        img_shape: Tuple[int, int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associa detecções atuais com estruturas rastreadas

        Usa IoU e distância para matching

        Returns:
            matches: Lista de (track_id, detection_idx)
            unmatched_tracks: IDs não associados
            unmatched_detections: Índices não associados
        """
        if not self.tracked_structures or not detections:
            return [], list(self.tracked_structures.keys()), list(range(len(detections)))

        # Matriz de custo (negativo do IoU)
        n_tracks = len(self.tracked_structures)
        n_dets = len(detections)
        cost_matrix = np.ones((n_tracks, n_dets)) * 1e6

        track_ids = list(self.tracked_structures.keys())

        for i, track_id in enumerate(track_ids):
            track = self.tracked_structures[track_id]
            for j, det in enumerate(detections):
                # Mesmo tipo de estrutura
                if track.structure_type == det['type']:
                    # Calcular IoU
                    iou = self._compute_iou(track.contour, det['contour'], img_shape)

                    # Distância do centro
                    dist = np.sqrt(
                        (track.center[0] - det['center'][0])**2 +
                        (track.center[1] - det['center'][1])**2
                    )

                    # Custo combinado (menor é melhor)
                    if iou > self.iou_threshold:
                        cost_matrix[i, j] = (1 - iou) + dist * 0.01

        # Greedy matching (pode usar Hungarian para melhor resultado)
        matches = []
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(n_dets))

        for _ in range(min(n_tracks, n_dets)):
            if cost_matrix.min() >= 1e6:
                break

            i, j = np.unravel_index(cost_matrix.argmin(), cost_matrix.shape)

            if cost_matrix[i, j] < 1e6:
                matches.append((track_ids[i], j))
                unmatched_tracks.discard(track_ids[i])
                unmatched_dets.discard(j)
                cost_matrix[i, :] = 1e6
                cost_matrix[:, j] = 1e6

        return matches, list(unmatched_tracks), list(unmatched_dets)

    def update(
        self,
        detections: List[Dict],
        img_shape: Tuple[int, int],
        mm_per_pixel: float = 0.3
    ) -> List[TrackedStructure]:
        """
        Atualiza tracking com novas detecções

        Args:
            detections: Lista de dicts com:
                - 'type': 'NERVE', 'ARTERY', 'VEIN'
                - 'center': (cx, cy)
                - 'contour': np.ndarray
                - 'area': float
                - 'confidence': float
                - 'nerve_type': str (opcional)
            img_shape: Shape da imagem
            mm_per_pixel: Escala para conversão

        Returns:
            Lista de estruturas rastreadas ativas
        """
        self.frame_count += 1

        # Match detections com tracks existentes
        matches, unmatched_tracks, unmatched_dets = self._match_detections(
            detections, img_shape
        )

        # Atualizar tracks matched
        for track_id, det_idx in matches:
            track = self.tracked_structures[track_id]
            det = detections[det_idx]

            # Calcular métricas
            diameter_px = np.sqrt(4 * det['area'] / np.pi)
            diameter_mm = diameter_px * mm_per_pixel
            csa_mm2 = det['area'] * (mm_per_pixel ** 2)

            # Atualizar Kalman
            measurement = np.array([
                [det['center'][0]],
                [det['center'][1]],
                [det['area']],
                [diameter_px],
            ])

            track.kalman.predict()
            track.kalman.update(measurement)

            # Suavização temporal
            state = track.kalman.x
            smooth_cx = int(self.smoothing_factor * state[0, 0] +
                          (1 - self.smoothing_factor) * det['center'][0])
            smooth_cy = int(self.smoothing_factor * state[1, 0] +
                          (1 - self.smoothing_factor) * det['center'][1])

            # Atualizar estrutura
            track.center = (smooth_cx, smooth_cy)
            track.contour = det['contour']
            track.area = det['area']
            track.diameter_mm = diameter_mm
            track.csa_mm2 = csa_mm2
            track.confidence = det.get('confidence', 1.0)
            track.nerve_type = det.get('nerve_type', track.nerve_type)
            track.frames_tracked += 1
            track.frames_lost = 0
            track.history.append((smooth_cx, smooth_cy, self.frame_count))

        # Predição para tracks não matched
        for track_id in unmatched_tracks:
            track = self.tracked_structures[track_id]
            track.frames_lost += 1

            if track.frames_lost > self.max_lost_frames:
                # Remover track perdido
                del self.tracked_structures[track_id]
            else:
                # Predição do Kalman
                track.kalman.predict()
                state = track.kalman.x

                track.center = (int(state[0, 0]), int(state[1, 0]))
                track.confidence *= 0.9  # Decay de confiança

        # Criar novos tracks para detecções não matched
        for det_idx in unmatched_dets:
            det = detections[det_idx]

            diameter_px = np.sqrt(4 * det['area'] / np.pi)
            diameter_mm = diameter_px * mm_per_pixel
            csa_mm2 = det['area'] * (mm_per_pixel ** 2)

            # Criar Kalman
            kalman = self._create_kalman((
                det['center'][0],
                det['center'][1],
                det['area'],
                diameter_px,
            ))

            # Cor baseada no tipo
            color_map = {
                'NERVE': (0, 255, 255),    # Amarelo
                'ARTERY': (0, 0, 255),      # Vermelho
                'VEIN': (255, 100, 100),    # Azul
            }

            # Criar estrutura
            structure = TrackedStructure(
                id=self.next_id,
                structure_type=det['type'],
                center=det['center'],
                contour=det['contour'],
                area=det['area'],
                diameter_mm=diameter_mm,
                confidence=det.get('confidence', 1.0),
                nerve_type=det.get('nerve_type'),
                csa_mm2=csa_mm2,
                kalman=kalman,
                color=color_map.get(det['type'], (255, 255, 255)),
            )

            self.tracked_structures[self.next_id] = structure
            self.next_id += 1

        # Retornar estruturas ativas
        active = [
            s for s in self.tracked_structures.values()
            if s.confidence >= self.min_confidence
        ]

        return sorted(active, key=lambda x: x.area, reverse=True)

    def get_tracking_status(self, structure_id: int) -> str:
        """
        Retorna status do tracking

        Returns:
            'TRACKING': Detecção ativa
            'PREDICTING': Usando predição Kalman
            'LOST': Perdido (confidence baixa)
        """
        if structure_id not in self.tracked_structures:
            return 'LOST'

        structure = self.tracked_structures[structure_id]

        if structure.frames_lost == 0:
            return 'TRACKING'
        elif structure.frames_lost <= self.max_lost_frames // 2:
            return 'PREDICTING'
        else:
            return 'SEARCHING'

    def reset(self):
        """Reset completo do tracker"""
        self.tracked_structures.clear()
        self.next_id = 0
        self.frame_count = 0
```

---

# 5. FASE 4: CLASSIFICAÇÃO NERVO/ARTÉRIA/VEIA

## 5.1 Criar Arquivo: src/nerve_classifier.py

```python
"""
NERVE CLASSIFIER - Classificação de Estruturas Neurovasculares

Técnicas baseadas em pesquisa:
1. Características de ecogenicidade
2. Análise de compressibilidade (temporal)
3. Detecção de pulsatilidade (temporal)
4. Padrão "honeycomb" para nervos
5. Morfologia (formato, proporções)
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass


@dataclass
class StructureFeatures:
    """Features extraídas de uma estrutura"""
    # Morfologia
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    solidity: float

    # Ecogenicidade
    mean_intensity: float
    std_intensity: float
    texture_entropy: float
    has_honeycomb: bool

    # Temporal (preenchido ao longo do tempo)
    compressibility: Optional[float] = None
    pulsatility: Optional[float] = None
    area_variation: Optional[float] = None


class StructureClassifier:
    """
    Classificador de estruturas neurovasculares

    Combina features estáticas (morfologia, textura) com
    features temporais (compressibilidade, pulsatilidade)
    """

    def __init__(
        self,
        pulsatility_frames: int = 30,
        compressibility_threshold: float = 0.3,
        min_confidence: float = 0.5,
    ):
        self.pulsatility_frames = pulsatility_frames
        self.compressibility_threshold = compressibility_threshold
        self.min_confidence = min_confidence

        # Histórico de áreas por estrutura ID
        self.area_history: Dict[int, deque] = {}

    def extract_features(
        self,
        contour: np.ndarray,
        gray_image: np.ndarray,
        structure_id: int
    ) -> StructureFeatures:
        """
        Extrai features de uma estrutura

        Args:
            contour: Contorno da estrutura
            gray_image: Imagem em escala de cinza
            structure_id: ID para tracking temporal

        Returns:
            StructureFeatures com todas as métricas
        """
        # ═══════════════════════════════════════════════════════════
        # MORFOLOGIA
        # ═══════════════════════════════════════════════════════════
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Circularity: 1.0 = círculo perfeito
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)

        # Aspect ratio do bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / (h + 1e-6)

        # Solidity: área / área do convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)

        # ═══════════════════════════════════════════════════════════
        # ECOGENICIDADE
        # ═══════════════════════════════════════════════════════════
        mask = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        roi = gray_image[mask > 0]

        if len(roi) > 0:
            mean_intensity = np.mean(roi)
            std_intensity = np.std(roi)
        else:
            mean_intensity = 0
            std_intensity = 0

        # Entropy (textura)
        if len(roi) > 10:
            hist, _ = np.histogram(roi, bins=64, range=(0, 256))
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            texture_entropy = -np.sum(hist * np.log2(hist))
        else:
            texture_entropy = 0

        # Padrão honeycomb (característico de nervos)
        has_honeycomb = self._detect_honeycomb_pattern(gray_image, contour, mask)

        # ═══════════════════════════════════════════════════════════
        # TEMPORAL (histórico de área)
        # ═══════════════════════════════════════════════════════════
        if structure_id not in self.area_history:
            self.area_history[structure_id] = deque(maxlen=self.pulsatility_frames)

        self.area_history[structure_id].append(area)

        compressibility = None
        pulsatility = None
        area_variation = None

        if len(self.area_history[structure_id]) >= 10:
            areas = np.array(self.area_history[structure_id])

            # Variação de área
            area_variation = (areas.max() - areas.min()) / (areas.mean() + 1e-6)

            # Pulsatilidade: FFT para detectar padrão cíclico
            if len(areas) >= self.pulsatility_frames:
                pulsatility = self._compute_pulsatility(areas)

            # Compressibilidade: mudança brusca de área
            compressibility = self._compute_compressibility(areas)

        return StructureFeatures(
            area=area,
            perimeter=perimeter,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            texture_entropy=texture_entropy,
            has_honeycomb=has_honeycomb,
            compressibility=compressibility,
            pulsatility=pulsatility,
            area_variation=area_variation,
        )

    def _detect_honeycomb_pattern(
        self,
        gray: np.ndarray,
        contour: np.ndarray,
        mask: np.ndarray
    ) -> bool:
        """
        Detecta padrão "honeycomb" característico de nervos

        Nervos têm fascículos hipoecóicos separados por septa hiperecóicas
        """
        # Extrair ROI
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]

        if roi.size < 100:
            return False

        # Aplicar threshold adaptativo
        thresh = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Aplicar máscara
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

        # Detectar contornos internos (fascículos)
        contours_internal, _ = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filtrar contornos muito pequenos ou muito grandes
        valid_contours = [
            c for c in contours_internal
            if 20 < cv2.contourArea(c) < (w * h * 0.3)
        ]

        # Nervos típicos têm múltiplos fascículos
        return len(valid_contours) >= 3

    def _compute_pulsatility(self, areas: np.ndarray) -> float:
        """
        Calcula índice de pulsatilidade usando FFT

        Artérias têm pulsação regular (~1-2 Hz)
        Veias e nervos não pulsam
        """
        # Remover tendência
        areas_detrended = areas - np.mean(areas)

        # FFT
        fft = np.fft.fft(areas_detrended)
        freqs = np.fft.fftfreq(len(areas), d=1/30)  # 30 FPS

        # Procurar pico na faixa de frequência cardíaca (0.8-3 Hz)
        cardiac_mask = (np.abs(freqs) >= 0.8) & (np.abs(freqs) <= 3.0)

        if cardiac_mask.sum() > 0:
            cardiac_power = np.abs(fft[cardiac_mask]).max()
            total_power = np.abs(fft).sum()

            return cardiac_power / (total_power + 1e-6)

        return 0.0

    def _compute_compressibility(self, areas: np.ndarray) -> float:
        """
        Calcula compressibilidade baseado em mudanças bruscas de área

        Veias são altamente compressíveis
        Artérias e nervos são pouco compressíveis
        """
        if len(areas) < 5:
            return 0.0

        # Calcular mudanças entre frames consecutivos
        changes = np.abs(np.diff(areas))

        # Normalizar pela área média
        relative_changes = changes / (np.mean(areas) + 1e-6)

        # Compressibilidade = máxima mudança relativa
        return float(relative_changes.max())

    def classify(self, features: StructureFeatures) -> Tuple[str, float, Dict]:
        """
        Classifica estrutura como NERVE, ARTERY ou VEIN

        Args:
            features: Features extraídas

        Returns:
            (type, confidence, scores): Tipo, confiança e scores detalhados
        """
        scores = {
            'NERVE': 0.0,
            'ARTERY': 0.0,
            'VEIN': 0.0,
        }

        # ═══════════════════════════════════════════════════════════
        # REGRAS BASEADAS EM ECOGENICIDADE
        # ═══════════════════════════════════════════════════════════

        # Nervos: hipoecóico com septa hiperecóicas (padrão honeycomb)
        if features.has_honeycomb:
            scores['NERVE'] += 0.4

        # Alta entropia de textura sugere nervo
        if features.texture_entropy > 3.0:
            scores['NERVE'] += 0.2

        # Intensidade média
        if features.mean_intensity < 60:
            # Anecóico/hipoecóico - vaso
            scores['ARTERY'] += 0.2
            scores['VEIN'] += 0.2
        elif features.mean_intensity > 120:
            # Hiperecóico - pode ser nervo fibrosado
            scores['NERVE'] += 0.1

        # ═══════════════════════════════════════════════════════════
        # REGRAS BASEADAS EM MORFOLOGIA
        # ═══════════════════════════════════════════════════════════

        # Artérias tendem a ser mais circulares
        if features.circularity > 0.8:
            scores['ARTERY'] += 0.2

        # Nervos têm formato mais irregular
        if features.circularity < 0.6 and features.solidity > 0.8:
            scores['NERVE'] += 0.2

        # Veias são mais ovais e compressíveis
        if features.aspect_ratio > 1.5 or features.aspect_ratio < 0.67:
            scores['VEIN'] += 0.15

        # ═══════════════════════════════════════════════════════════
        # REGRAS TEMPORAIS (se disponíveis)
        # ═══════════════════════════════════════════════════════════

        if features.pulsatility is not None:
            # Alta pulsatilidade = artéria
            if features.pulsatility > 0.3:
                scores['ARTERY'] += 0.4
            elif features.pulsatility < 0.1:
                scores['VEIN'] += 0.1
                scores['NERVE'] += 0.1

        if features.compressibility is not None:
            # Alta compressibilidade = veia
            if features.compressibility > self.compressibility_threshold:
                scores['VEIN'] += 0.4
            elif features.compressibility < 0.1:
                scores['ARTERY'] += 0.1
                scores['NERVE'] += 0.2

        if features.area_variation is not None:
            # Nervos têm área estável
            if features.area_variation < 0.1:
                scores['NERVE'] += 0.15

        # ═══════════════════════════════════════════════════════════
        # NORMALIZAR E DECIDIR
        # ═══════════════════════════════════════════════════════════

        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] /= total

        # Classificação final
        structure_type = max(scores, key=scores.get)
        confidence = scores[structure_type]

        return structure_type, confidence, scores

    def clear_history(self, structure_id: Optional[int] = None):
        """Limpa histórico temporal"""
        if structure_id is not None:
            self.area_history.pop(structure_id, None)
        else:
            self.area_history.clear()
```

---

# 6. FASE 5: IDENTIFICAÇÃO DE NERVOS ESPECÍFICOS

## 6.1 Criar Arquivo: src/nerve_identifier.py

```python
"""
NERVE IDENTIFIER - Identificação de Nervos Específicos

Identifica o tipo de nervo baseado em:
1. Localização na imagem (região do corpo)
2. Estruturas adjacentes (artérias, músculos)
3. Características morfológicas
4. Contexto anatômico configurado pelo usuário
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class BodyRegion(Enum):
    """Regiões do corpo para contexto"""
    UNKNOWN = 0
    WRIST = 1           # Pulso (nervo mediano, ulnar)
    FOREARM = 2         # Antebraço
    ELBOW = 3           # Cotovelo (nervo ulnar)
    UPPER_ARM = 4       # Braço superior
    NECK = 5            # Pescoço (plexo braquial)
    AXILLA = 6          # Axila
    THIGH = 7           # Coxa (nervo femoral)
    POPLITEAL = 8       # Região poplítea (ciático)
    ANKLE = 9           # Tornozelo (tibial)


@dataclass
class NerveProfile:
    """Perfil de características de um tipo de nervo"""
    name: str
    display_name: str
    typical_csa_mm2: Tuple[float, float]  # (min, max) normal
    typical_depth_mm: Tuple[float, float]
    typical_diameter_mm: Tuple[float, float]
    adjacent_structures: List[str]  # ['artery_radial', 'tendon_fcr']
    body_regions: List[BodyRegion]
    echogenicity: str  # 'hypoechoic', 'hyperechoic', 'mixed'
    shape: str  # 'round', 'oval', 'triangular', 'irregular'
    fascicle_pattern: bool  # True se tem padrão honeycomb visível


# Base de dados de perfis de nervos
NERVE_PROFILES = {
    'MEDIAN': NerveProfile(
        name='MEDIAN',
        display_name='Nervo Mediano',
        typical_csa_mm2=(6.0, 10.0),  # Normal < 10mm² no pulso
        typical_depth_mm=(3.0, 15.0),
        typical_diameter_mm=(3.0, 5.0),
        adjacent_structures=['flexor_tendons', 'carpal_tunnel'],
        body_regions=[BodyRegion.WRIST, BodyRegion.FOREARM, BodyRegion.ELBOW],
        echogenicity='hypoechoic',
        shape='oval',
        fascicle_pattern=True,
    ),
    'ULNAR': NerveProfile(
        name='ULNAR',
        display_name='Nervo Ulnar',
        typical_csa_mm2=(4.0, 8.0),
        typical_depth_mm=(2.0, 10.0),
        typical_diameter_mm=(2.5, 4.0),
        adjacent_structures=['ulnar_artery', 'fcu_muscle'],
        body_regions=[BodyRegion.WRIST, BodyRegion.FOREARM, BodyRegion.ELBOW],
        echogenicity='hypoechoic',
        shape='round',
        fascicle_pattern=True,
    ),
    'RADIAL': NerveProfile(
        name='RADIAL',
        display_name='Nervo Radial',
        typical_csa_mm2=(3.0, 6.0),
        typical_depth_mm=(2.0, 8.0),
        typical_diameter_mm=(2.0, 3.5),
        adjacent_structures=['radial_artery', 'brachioradialis'],
        body_regions=[BodyRegion.FOREARM, BodyRegion.UPPER_ARM],
        echogenicity='hypoechoic',
        shape='oval',
        fascicle_pattern=True,
    ),
    'BRACHIAL_PLEXUS': NerveProfile(
        name='BRACHIAL_PLEXUS',
        display_name='Plexo Braquial',
        typical_csa_mm2=(10.0, 30.0),
        typical_depth_mm=(10.0, 30.0),
        typical_diameter_mm=(5.0, 10.0),
        adjacent_structures=['subclavian_artery', 'scalene_muscles'],
        body_regions=[BodyRegion.NECK, BodyRegion.AXILLA],
        echogenicity='hypoechoic',
        shape='irregular',
        fascicle_pattern=True,
    ),
    'FEMORAL': NerveProfile(
        name='FEMORAL',
        display_name='Nervo Femoral',
        typical_csa_mm2=(15.0, 40.0),
        typical_depth_mm=(15.0, 40.0),
        typical_diameter_mm=(6.0, 12.0),
        adjacent_structures=['femoral_artery', 'femoral_vein', 'iliacus_muscle'],
        body_regions=[BodyRegion.THIGH],
        echogenicity='hypoechoic',
        shape='triangular',
        fascicle_pattern=True,
    ),
    'SCIATIC': NerveProfile(
        name='SCIATIC',
        display_name='Nervo Ciático',
        typical_csa_mm2=(30.0, 80.0),
        typical_depth_mm=(30.0, 80.0),
        typical_diameter_mm=(10.0, 20.0),
        adjacent_structures=['popliteal_artery', 'biceps_femoris'],
        body_regions=[BodyRegion.POPLITEAL, BodyRegion.THIGH],
        echogenicity='hypoechoic',
        shape='oval',
        fascicle_pattern=True,
    ),
    'TIBIAL': NerveProfile(
        name='TIBIAL',
        display_name='Nervo Tibial',
        typical_csa_mm2=(8.0, 20.0),
        typical_depth_mm=(10.0, 30.0),
        typical_diameter_mm=(4.0, 8.0),
        adjacent_structures=['posterior_tibial_artery', 'flexor_tendons'],
        body_regions=[BodyRegion.POPLITEAL, BodyRegion.ANKLE],
        echogenicity='hypoechoic',
        shape='round',
        fascicle_pattern=True,
    ),
    'PERONEAL': NerveProfile(
        name='PERONEAL',
        display_name='Nervo Peroneal',
        typical_csa_mm2=(5.0, 15.0),
        typical_depth_mm=(5.0, 15.0),
        typical_diameter_mm=(3.0, 6.0),
        adjacent_structures=['fibular_head', 'peroneus_muscle'],
        body_regions=[BodyRegion.POPLITEAL],
        echogenicity='hypoechoic',
        shape='round',
        fascicle_pattern=True,
    ),
}


class NerveIdentifier:
    """
    Identificador de tipo de nervo

    Usa contexto anatômico + características para identificar
    """

    def __init__(self, default_region: BodyRegion = BodyRegion.UNKNOWN):
        self.current_region = default_region
        self.profiles = NERVE_PROFILES

    def set_body_region(self, region: BodyRegion):
        """Define região do corpo atual"""
        self.current_region = region

    def identify(
        self,
        csa_mm2: float,
        depth_mm: float,
        diameter_mm: float,
        has_fascicles: bool,
        adjacent_artery: bool = False,
        confidence_threshold: float = 0.5,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Identifica o tipo de nervo

        Args:
            csa_mm2: Cross-sectional area em mm²
            depth_mm: Profundidade em mm
            diameter_mm: Diâmetro em mm
            has_fascicles: Se tem padrão de fascículos
            adjacent_artery: Se há artéria adjacente
            confidence_threshold: Threshold mínimo

        Returns:
            (nerve_type, confidence, all_scores)
        """
        scores = {}

        # Filtrar por região do corpo (se conhecida)
        candidates = self.profiles
        if self.current_region != BodyRegion.UNKNOWN:
            candidates = {
                k: v for k, v in self.profiles.items()
                if self.current_region in v.body_regions
            }

        if not candidates:
            candidates = self.profiles

        # Calcular score para cada candidato
        for name, profile in candidates.items():
            score = 0.0

            # CSA match
            if profile.typical_csa_mm2[0] <= csa_mm2 <= profile.typical_csa_mm2[1]:
                score += 0.3
            else:
                # Penalidade proporcional à distância
                dist = min(
                    abs(csa_mm2 - profile.typical_csa_mm2[0]),
                    abs(csa_mm2 - profile.typical_csa_mm2[1])
                )
                score += max(0, 0.3 - dist * 0.02)

            # Depth match
            if profile.typical_depth_mm[0] <= depth_mm <= profile.typical_depth_mm[1]:
                score += 0.2

            # Diameter match
            if profile.typical_diameter_mm[0] <= diameter_mm <= profile.typical_diameter_mm[1]:
                score += 0.2

            # Fascicle pattern
            if has_fascicles == profile.fascicle_pattern:
                score += 0.2

            # Região correta
            if self.current_region in profile.body_regions:
                score += 0.1

            scores[name] = score

        # Normalizar
        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] /= total

        # Melhor match
        if scores:
            best = max(scores, key=scores.get)
            confidence = scores[best]

            if confidence >= confidence_threshold:
                return best, confidence, scores

        return 'UNKNOWN', 0.0, scores

    def get_reference_values(self, nerve_type: str) -> Optional[NerveProfile]:
        """Retorna valores de referência para um tipo de nervo"""
        return self.profiles.get(nerve_type)

    def get_csa_status(self, nerve_type: str, csa_mm2: float) -> Tuple[str, str]:
        """
        Retorna status do CSA comparado com valores normais

        Returns:
            (status, description)
            status: 'NORMAL', 'ENLARGED', 'REDUCED'
        """
        profile = self.profiles.get(nerve_type)

        if not profile:
            return 'UNKNOWN', 'Tipo de nervo desconhecido'

        min_csa, max_csa = profile.typical_csa_mm2

        if csa_mm2 < min_csa:
            return 'REDUCED', f'CSA abaixo do normal (<{min_csa:.1f}mm²)'
        elif csa_mm2 > max_csa:
            # Para nervo mediano no túnel do carpo
            if nerve_type == 'MEDIAN' and csa_mm2 > 10:
                if csa_mm2 > 15:
                    return 'ENLARGED', 'CSA aumentado - Sugestivo de CTS severo'
                else:
                    return 'ENLARGED', 'CSA aumentado - Sugestivo de CTS'
            return 'ENLARGED', f'CSA acima do normal (>{max_csa:.1f}mm²)'
        else:
            return 'NORMAL', f'CSA dentro da normalidade ({min_csa:.1f}-{max_csa:.1f}mm²)'
```

---

# 7. FASE 6: MEDIÇÃO AUTOMÁTICA DE CSA

## 7.1 Adicionar em ai_processor.py

```python
def _calculate_csa(
    self,
    contour: np.ndarray,
    mm_per_pixel: float
) -> Dict[str, float]:
    """
    Calcula Cross-Sectional Area e métricas relacionadas

    Args:
        contour: Contorno da estrutura
        mm_per_pixel: Escala de conversão

    Returns:
        Dict com:
        - csa_mm2: Área em mm²
        - diameter_mm: Diâmetro equivalente
        - perimeter_mm: Perímetro em mm
        - circularity: Índice de circularidade
        - aspect_ratio: Razão de aspecto
    """
    # Área em pixels
    area_px = cv2.contourArea(contour)

    # Perímetro
    perimeter_px = cv2.arcLength(contour, True)

    # Converter para mm
    mm2_per_pixel2 = mm_per_pixel ** 2
    csa_mm2 = area_px * mm2_per_pixel2
    perimeter_mm = perimeter_px * mm_per_pixel

    # Diâmetro equivalente (círculo de mesma área)
    diameter_mm = np.sqrt(4 * csa_mm2 / np.pi)

    # Circularidade
    circularity = 4 * np.pi * area_px / (perimeter_px ** 2 + 1e-6)

    # Aspect ratio (do bounding rect)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / (h + 1e-6)

    # Eixos da elipse ajustada
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (ma, MA), angle = ellipse
        major_axis_mm = MA * mm_per_pixel
        minor_axis_mm = ma * mm_per_pixel
    else:
        major_axis_mm = diameter_mm
        minor_axis_mm = diameter_mm

    return {
        'csa_mm2': csa_mm2,
        'diameter_mm': diameter_mm,
        'perimeter_mm': perimeter_mm,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'major_axis_mm': major_axis_mm,
        'minor_axis_mm': minor_axis_mm,
    }
```

---

# 8. FASE 7: VISUAL PREMIUM

## 8.1 Novo _process_nerve_v2() completo

```python
def _process_nerve_v2(self, frame: np.ndarray) -> np.ndarray:
    """
    NERVE TRACK v2.0 PREMIUM

    Sistema avançado de segmentação neurovascular

    Features:
    - Segmentação com U-Net++ EfficientNet
    - Tracking temporal com Kalman
    - Classificação Nervo/Artéria/Veia
    - Identificação de nervo específico
    - Medição automática de CSA
    - Visual premium estilo Clarius
    """
    output = frame.copy()
    h, w = frame.shape[:2]

    # ═══════════════════════════════════════════════════════════════
    # ETAPA 1: PRÉ-PROCESSAMENTO
    # ═══════════════════════════════════════════════════════════════
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE para melhor contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Calibrar escala
    mm_per_pixel = self._calibrate_scale(h)

    # ═══════════════════════════════════════════════════════════════
    # ETAPA 2: SEGMENTAÇÃO (Modelo ou Fallback)
    # ═══════════════════════════════════════════════════════════════
    detections = []

    if self.nerve_model is not None:
        # Usar modelo de deep learning
        detections = self._segment_with_model(enhanced)
    else:
        # Fallback: HoughCircles + classificação
        detections = self._segment_with_cv(enhanced, gray)

    # ═══════════════════════════════════════════════════════════════
    # ETAPA 3: TRACKING TEMPORAL
    # ═══════════════════════════════════════════════════════════════
    tracked_structures = self.nerve_tracker.update(
        detections,
        (h, w),
        mm_per_pixel
    )

    # ═══════════════════════════════════════════════════════════════
    # ETAPA 4: DESENHO DAS ESTRUTURAS
    # ═══════════════════════════════════════════════════════════════
    for structure in tracked_structures:
        self._draw_structure_premium(output, structure, mm_per_pixel)

    # ═══════════════════════════════════════════════════════════════
    # ETAPA 5: LINHAS DE DISTÂNCIA
    # ═══════════════════════════════════════════════════════════════
    self._draw_distance_lines(output, tracked_structures, mm_per_pixel)

    # ═══════════════════════════════════════════════════════════════
    # ETAPA 6: PAINEL LATERAL PREMIUM
    # ═══════════════════════════════════════════════════════════════
    self._draw_nerve_panel_v2(output, tracked_structures, mm_per_pixel)

    # ═══════════════════════════════════════════════════════════════
    # ETAPA 7: ESCALA DE PROFUNDIDADE
    # ═══════════════════════════════════════════════════════════════
    self._draw_depth_scale(output, h, mm_per_pixel)

    # ═══════════════════════════════════════════════════════════════
    # ETAPA 8: INDICADOR DE QUALIDADE
    # ═══════════════════════════════════════════════════════════════
    quality = self._assess_image_quality(gray)
    self._draw_quality_indicator(output, quality)

    return output


def _draw_structure_premium(
    self,
    output: np.ndarray,
    structure: TrackedStructure,
    mm_per_pixel: float
) -> None:
    """Desenha estrutura com visual premium"""
    cx, cy = structure.center
    contour = structure.contour

    # Cor baseada no tipo e status do tracking
    tracking_status = self.nerve_tracker.get_tracking_status(structure.id)

    if tracking_status == 'TRACKING':
        color = structure.color
        alpha = 0.4
    elif tracking_status == 'PREDICTING':
        color = (0, 165, 255)  # Laranja
        alpha = 0.25
    else:
        color = (128, 128, 128)  # Cinza
        alpha = 0.15

    # Preenchimento semi-transparente
    overlay = output.copy()
    cv2.drawContours(overlay, [contour], -1, color, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # Contorno com glow
    dark_color = tuple(c // 2 for c in color)
    cv2.drawContours(output, [contour], -1, dark_color, 3)
    cv2.drawContours(output, [contour], -1, color, 1)

    # Centro com crosshair
    cv2.drawMarker(output, (cx, cy), color, cv2.MARKER_CROSS, 8, 1)

    # Trail (histórico de posições)
    if len(structure.history) > 2:
        points = [(p[0], p[1]) for p in structure.history]
        for i in range(1, len(points)):
            # Cor fade
            fade = i / len(points)
            trail_color = tuple(int(c * fade) for c in color)
            cv2.line(output, points[i-1], points[i], trail_color, 1)

    # ─────────────────────────────────────────────────────────────
    # LABEL
    # ─────────────────────────────────────────────────────────────
    label_y = cy - 25

    # Tipo de estrutura
    type_label = structure.structure_type
    if structure.nerve_type and structure.nerve_type != 'UNKNOWN':
        type_label = structure.nerve_type

    # Background do label
    (tw, th), _ = cv2.getTextSize(type_label, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
    cv2.rectangle(
        output,
        (cx - tw//2 - 3, label_y - th - 3),
        (cx + tw//2 + 3, label_y + 3),
        (20, 20, 30), -1
    )
    cv2.rectangle(
        output,
        (cx - tw//2 - 3, label_y - th - 3),
        (cx + tw//2 + 3, label_y + 3),
        color, 1
    )

    # Texto do tipo
    cv2.putText(
        output, type_label,
        (cx - tw//2, label_y),
        cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1, cv2.LINE_AA
    )

    # CSA abaixo
    if structure.csa_mm2:
        csa_label = f"{structure.csa_mm2:.1f}mm²"
        cv2.putText(
            output, csa_label,
            (cx - 20, cy + 25),
            cv2.FONT_HERSHEY_DUPLEX, 0.35, (180, 180, 200), 1, cv2.LINE_AA
        )

    # Indicador de confiança
    conf_width = 30
    conf_x = cx - conf_width // 2
    conf_y = cy + 35

    # Background
    cv2.rectangle(output, (conf_x, conf_y), (conf_x + conf_width, conf_y + 4), (40, 40, 40), -1)

    # Fill
    fill_w = int(conf_width * structure.confidence)
    conf_color = (0, 255, 0) if structure.confidence > 0.7 else (0, 200, 255) if structure.confidence > 0.4 else (0, 0, 255)
    cv2.rectangle(output, (conf_x, conf_y), (conf_x + fill_w, conf_y + 4), conf_color, -1)

    # Status do tracking
    if tracking_status != 'TRACKING':
        status_label = tracking_status
        cv2.putText(
            output, status_label,
            (cx - 30, cy + 50),
            cv2.FONT_HERSHEY_DUPLEX, 0.25, (0, 165, 255), 1, cv2.LINE_AA
        )


def _draw_nerve_panel_v2(
    self,
    output: np.ndarray,
    structures: List[TrackedStructure],
    mm_per_pixel: float
) -> None:
    """Desenha painel lateral estilo Clarius"""
    h, w = output.shape[:2]

    panel_w = 180
    panel_x = w - panel_w - 10
    panel_y = 10
    panel_h = 50 + len(structures) * 55 + 60
    panel_h = min(panel_h, h - 20)

    # Background
    overlay = output.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (25, 25, 20), -1)
    cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)

    # Borda
    cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 200, 200), 1)

    # Título
    cv2.putText(
        output, "NERVE TRACK v2.0",
        (panel_x + 10, panel_y + 22),
        cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA
    )

    # Linha separadora
    cv2.line(output, (panel_x + 5, panel_y + 32), (panel_x + panel_w - 5, panel_y + 32), (60, 60, 60), 1)

    # Lista de estruturas
    y_off = panel_y + 50

    for i, s in enumerate(structures[:5]):
        # Círculo colorido
        cv2.circle(output, (panel_x + 15, y_off), 6, s.color, -1)
        cv2.circle(output, (panel_x + 15, y_off), 6, (255, 255, 255), 1)

        # Tipo
        type_label = s.nerve_type if s.nerve_type and s.nerve_type != 'UNKNOWN' else s.structure_type
        cv2.putText(
            output, type_label,
            (panel_x + 28, y_off + 4),
            cv2.FONT_HERSHEY_DUPLEX, 0.35, s.color, 1, cv2.LINE_AA
        )

        # CSA
        if s.csa_mm2:
            csa_text = f"CSA: {s.csa_mm2:.1f}mm²"
            cv2.putText(
                output, csa_text,
                (panel_x + 28, y_off + 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.3, (150, 150, 170), 1, cv2.LINE_AA
            )

        # Diâmetro
        if s.diameter_mm:
            diam_text = f"⌀ {s.diameter_mm:.1f}mm"
            cv2.putText(
                output, diam_text,
                (panel_x + 100, y_off + 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.3, (150, 150, 170), 1, cv2.LINE_AA
            )

        # Barra de confiança
        bar_x = panel_x + 28
        bar_y = y_off + 30
        bar_w = panel_w - 40

        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + 4), (40, 50, 40), -1)
        fill_w = int(bar_w * s.confidence)
        conf_color = (0, 255, 0) if s.confidence > 0.7 else (0, 200, 255) if s.confidence > 0.4 else (0, 0, 255)
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + fill_w, bar_y + 4), conf_color, -1)

        y_off += 50

    # Contagem total
    nerve_count = sum(1 for s in structures if s.structure_type == 'NERVE')
    artery_count = sum(1 for s in structures if s.structure_type == 'ARTERY')
    vein_count = sum(1 for s in structures if s.structure_type == 'VEIN')

    y_off = panel_y + panel_h - 25
    cv2.putText(
        output, f"N:{nerve_count} A:{artery_count} V:{vein_count}",
        (panel_x + 10, y_off),
        cv2.FONT_HERSHEY_DUPLEX, 0.35, (120, 120, 140), 1, cv2.LINE_AA
    )

    # Escala
    cv2.putText(
        output, f"Scale: {mm_per_pixel:.2f}mm/px",
        (panel_x + 10, y_off + 15),
        cv2.FONT_HERSHEY_DUPLEX, 0.25, (80, 100, 80), 1, cv2.LINE_AA
    )


def _draw_quality_indicator(self, output: np.ndarray, quality: float) -> None:
    """Desenha indicador de qualidade do scan (estilo Nerveblox)"""
    h, w = output.shape[:2]

    # Posição
    ind_x = w - 50
    ind_y = h - 50
    radius = 20

    # Cor baseada na qualidade
    if quality >= 0.7:
        color = (0, 255, 0)  # Verde
        text = "GOOD"
    elif quality >= 0.4:
        color = (0, 200, 255)  # Amarelo
        text = "FAIR"
    else:
        color = (0, 0, 255)  # Vermelho
        text = "POOR"

    # Círculo de fundo
    cv2.circle(output, (ind_x, ind_y), radius + 2, (30, 30, 30), -1)

    # Arco de progresso
    angle = int(quality * 360)
    cv2.ellipse(output, (ind_x, ind_y), (radius, radius), -90, 0, angle, color, 3)

    # Círculo interno
    cv2.circle(output, (ind_x, ind_y), radius - 5, (40, 40, 40), -1)

    # Texto
    cv2.putText(
        output, text,
        (ind_x - 15, ind_y + 5),
        cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1, cv2.LINE_AA
    )

    # Label
    cv2.putText(
        output, "SCAN",
        (ind_x - 15, ind_y + 35),
        cv2.FONT_HERSHEY_DUPLEX, 0.25, (100, 100, 100), 1, cv2.LINE_AA
    )


def _assess_image_quality(self, gray: np.ndarray) -> float:
    """
    Avalia qualidade da imagem de ultrassom

    Considera:
    - Contraste
    - Nitidez
    - Ruído
    """
    # Contraste (desvio padrão)
    contrast = np.std(gray) / 128.0
    contrast = min(1.0, contrast)

    # Nitidez (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var() / 1000.0
    sharpness = min(1.0, sharpness)

    # Ruído (estimativa)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = np.abs(gray.astype(float) - blur.astype(float)).mean() / 30.0
    noise_score = max(0.0, 1.0 - noise)

    # Score combinado
    quality = 0.4 * contrast + 0.4 * sharpness + 0.2 * noise_score

    return min(1.0, max(0.0, quality))
```

---

# 9. FASE 8: TESTES E VALIDAÇÃO

## 9.1 Criar Arquivo: tests/test_nerve_track.py

```python
"""
Testes para NERVE TRACK v2.0
"""

import pytest
import numpy as np
import cv2
import sys
sys.path.append('/Users/priscoleao/aplicativo-usg-final')

from src.nerve_tracker import NerveTracker, TrackedStructure
from src.nerve_classifier import StructureClassifier, StructureFeatures
from src.nerve_identifier import NerveIdentifier, BodyRegion


class TestNerveTracker:
    """Testes do sistema de tracking"""

    def setup_method(self):
        self.tracker = NerveTracker(max_lost_frames=10)

    def test_new_detection(self):
        """Teste de nova detecção"""
        contour = np.array([[[100, 100]], [[150, 100]], [[150, 150]], [[100, 150]]])
        detection = {
            'type': 'NERVE',
            'center': (125, 125),
            'contour': contour,
            'area': 2500,
            'confidence': 0.9,
        }

        structures = self.tracker.update([detection], (500, 500))

        assert len(structures) == 1
        assert structures[0].structure_type == 'NERVE'
        assert structures[0].confidence == 0.9

    def test_tracking_continuity(self):
        """Teste de continuidade do tracking"""
        contour1 = np.array([[[100, 100]], [[150, 100]], [[150, 150]], [[100, 150]]])
        contour2 = np.array([[[105, 105]], [[155, 105]], [[155, 155]], [[105, 155]]])

        det1 = {'type': 'NERVE', 'center': (125, 125), 'contour': contour1, 'area': 2500, 'confidence': 0.9}
        det2 = {'type': 'NERVE', 'center': (130, 130), 'contour': contour2, 'area': 2500, 'confidence': 0.9}

        self.tracker.update([det1], (500, 500))
        structures = self.tracker.update([det2], (500, 500))

        assert len(structures) == 1
        assert structures[0].frames_tracked == 2

    def test_prediction_on_lost(self):
        """Teste de predição quando perde detecção"""
        contour = np.array([[[100, 100]], [[150, 100]], [[150, 150]], [[100, 150]]])
        detection = {'type': 'NERVE', 'center': (125, 125), 'contour': contour, 'area': 2500, 'confidence': 0.9}

        self.tracker.update([detection], (500, 500))
        structures = self.tracker.update([], (500, 500))  # Sem detecção

        assert len(structures) == 1
        assert structures[0].frames_lost == 1
        assert self.tracker.get_tracking_status(structures[0].id) == 'PREDICTING'


class TestStructureClassifier:
    """Testes do classificador de estruturas"""

    def setup_method(self):
        self.classifier = StructureClassifier()

    def test_nerve_classification(self):
        """Teste de classificação de nervo"""
        features = StructureFeatures(
            area=1000,
            perimeter=150,
            circularity=0.5,
            aspect_ratio=1.2,
            solidity=0.9,
            mean_intensity=80,
            std_intensity=30,
            texture_entropy=3.5,
            has_honeycomb=True,
            compressibility=0.05,
            pulsatility=0.05,
            area_variation=0.05,
        )

        structure_type, confidence, scores = self.classifier.classify(features)

        assert structure_type == 'NERVE'
        assert confidence > 0.4

    def test_artery_classification(self):
        """Teste de classificação de artéria"""
        features = StructureFeatures(
            area=800,
            perimeter=100,
            circularity=0.9,
            aspect_ratio=1.0,
            solidity=0.95,
            mean_intensity=40,
            std_intensity=10,
            texture_entropy=1.5,
            has_honeycomb=False,
            compressibility=0.05,
            pulsatility=0.5,  # Alta pulsatilidade
            area_variation=0.1,
        )

        structure_type, confidence, scores = self.classifier.classify(features)

        assert structure_type == 'ARTERY'

    def test_vein_classification(self):
        """Teste de classificação de veia"""
        features = StructureFeatures(
            area=1200,
            perimeter=160,
            circularity=0.6,
            aspect_ratio=1.8,  # Oval
            solidity=0.85,
            mean_intensity=35,
            std_intensity=8,
            texture_entropy=1.2,
            has_honeycomb=False,
            compressibility=0.5,  # Alta compressibilidade
            pulsatility=0.02,
            area_variation=0.3,
        )

        structure_type, confidence, scores = self.classifier.classify(features)

        assert structure_type == 'VEIN'


class TestNerveIdentifier:
    """Testes do identificador de nervos"""

    def setup_method(self):
        self.identifier = NerveIdentifier(default_region=BodyRegion.WRIST)

    def test_median_nerve_identification(self):
        """Teste de identificação do nervo mediano"""
        nerve_type, confidence, scores = self.identifier.identify(
            csa_mm2=8.0,
            depth_mm=5.0,
            diameter_mm=4.0,
            has_fascicles=True,
        )

        assert nerve_type in ['MEDIAN', 'ULNAR']  # Ambos possíveis no pulso
        assert confidence > 0.3

    def test_csa_status(self):
        """Teste de status do CSA"""
        status, desc = self.identifier.get_csa_status('MEDIAN', 12.0)

        assert status == 'ENLARGED'
        assert 'CTS' in desc or 'aumentado' in desc.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

# 10. CHECKLIST FINAL

## 10.1 Checklist de Implementação

```
FASE 1: PREPARAÇÃO
[ ] Instalar dependências (segmentation-models-pytorch, timm, albumentations)
[ ] Criar estrutura de pastas
[ ] Adicionar configurações em config.py
[ ] Testar imports

FASE 2: MODELO DE SEGMENTAÇÃO
[ ] Criar nerve_model.py com NerveSegmentationModel
[ ] Criar CBAM attention module
[ ] Criar ConvLSTMCell para temporal
[ ] Criar loss function combinada
[ ] Criar augmentations.py
[ ] Testar modelo standalone

FASE 3: TRACKING TEMPORAL
[ ] Criar nerve_tracker.py
[ ] Implementar Kalman Filter por estrutura
[ ] Implementar matching de detecções
[ ] Implementar predição quando perde
[ ] Implementar suavização temporal
[ ] Testar tracking com sequência

FASE 4: CLASSIFICAÇÃO
[ ] Criar nerve_classifier.py
[ ] Implementar extração de features
[ ] Implementar detecção honeycomb
[ ] Implementar análise de pulsatilidade
[ ] Implementar análise de compressibilidade
[ ] Testar classificação

FASE 5: IDENTIFICAÇÃO
[ ] Criar nerve_identifier.py
[ ] Criar base de dados de perfis
[ ] Implementar matching por região
[ ] Implementar matching por CSA
[ ] Testar identificação

FASE 6: MEDIÇÃO CSA
[ ] Implementar _calculate_csa
[ ] Integrar com calibração mm/pixel
[ ] Calcular métricas derivadas
[ ] Testar medições

FASE 7: VISUAL PREMIUM
[ ] Implementar _process_nerve_v2
[ ] Implementar _draw_structure_premium
[ ] Implementar _draw_nerve_panel_v2
[ ] Implementar _draw_quality_indicator
[ ] Implementar trail de histórico
[ ] Testar visual completo

FASE 8: TESTES
[ ] Criar testes unitários
[ ] Testar com imagens reais
[ ] Validar métricas (Dice, IoU)
[ ] Testar performance (FPS)
[ ] Documentar resultados

FASE 9: INTEGRAÇÃO
[ ] Integrar no ai_processor.py
[ ] Adicionar ao main.py
[ ] Testar fluxo completo
[ ] Otimizar performance
[ ] Documentar uso
```

## 10.2 Métricas de Sucesso

| Métrica | Target | Estado |
|---------|--------|--------|
| Dice Score (segmentação) | > 0.80 | [ ] |
| IoU (segmentação) | > 0.70 | [ ] |
| Accuracy (classificação) | > 85% | [ ] |
| FPS (inference) | > 25 | [ ] |
| Tracking continuity | > 90% | [ ] |
| CSA error | < 10% | [ ] |

---

# 📚 REFERÊNCIAS

## Papers
1. [U-Net++: A Nested U-Net Architecture](https://arxiv.org/abs/1807.10165)
2. [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
3. [DeepNerve: CNN for Median Nerve Segmentation](https://pubmed.ncbi.nlm.nih.gov/32527593/)
4. [Unified Focal Loss](https://github.com/mlyg/unified-focal-loss)

## Repositórios
1. [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
2. [Kaggle Nerve Segmentation](https://www.kaggle.com/c/ultrasound-nerve-segmentation)
3. [timm](https://github.com/huggingface/pytorch-image-models)

## Produtos Comerciais
1. [Clarius Median Nerve AI](https://clarius.com/)
2. [GE Healthcare cNerve](https://www.gehealthcare.com/)
3. [ScanNav Anatomy PNB](https://www.intelligentultrasound.com/)
4. [Nerveblox](https://www.nerveblox.com/)

---

*Documento criado em: 2025-12-22*
*Baseado em pesquisa extensiva de estado da arte mundial*
*NERVE TRACK v2.0 PREMIUM - O melhor sistema de detecção de nervos*
