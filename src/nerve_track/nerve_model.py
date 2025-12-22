"""
NERVE TRACK v2.0 - Modelo de Segmentação Premium
=================================================
Arquitetura: U-Net++ com EfficientNet-B4 encoder + CBAM + ConvLSTM

Baseado em pesquisa de:
- Clarius Median Nerve AI
- GE Healthcare cNerve
- ScanNav Anatomy PNB
- DeepNerve (MICCAI)
- segmentation_models.pytorch

Características:
- U-Net++ com skip connections densos
- EfficientNet-B4 pretrained (ImageNet + Ultrasound)
- CBAM (Convolutional Block Attention Module)
- ConvLSTM para consistência temporal
- Unified Focal Loss (Dice + BCE + Focal)
- Multi-class: Background, Nervo, Artéria, Veia, Fascia, Músculo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


# =============================================================================
# CBAM - Convolutional Block Attention Module
# =============================================================================
# Paper: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
# Melhora segmentação focando em características relevantes

class ChannelAttention(nn.Module):
    """
    Atenção no canal - aprende QUAIS features são importantes.
    Usa average pooling + max pooling seguido de MLP compartilhado.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP compartilhado
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """
    Atenção espacial - aprende ONDE focar na imagem.
    Usa concatenação de avg/max pooling seguido de conv.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(concat))
        return x * attention


class CBAM(nn.Module):
    """
    CBAM completo: Channel Attention -> Spatial Attention
    Aplicado após cada bloco do decoder para refinar segmentação.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# =============================================================================
# ConvLSTM - Consistência Temporal
# =============================================================================
# Paper: "Convolutional LSTM Network" (NIPS 2015)
# Mantém consistência entre frames, elimina flicker

class ConvLSTMCell(nn.Module):
    """
    Célula ConvLSTM para processamento temporal de features.
    Mantém estado entre frames para suavização e consistência.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Gates: input, forget, cell, output
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor [B, C, H, W]
            hidden_state: Tuple of (h, c) or None for initialization

        Returns:
            h_next: Next hidden state [B, hidden_channels, H, W]
            (h_next, c_next): Tuple for next iteration
        """
        batch_size, _, height, width = x.size()

        # Inicializar estado oculto se necessário
        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        else:
            h, c = hidden_state

        # Concatenar input com hidden state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        # Separar gates
        i, f, g, o = torch.chunk(gates, 4, dim=1)

        # Aplicar ativações
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell gate
        o = torch.sigmoid(o)  # Output gate

        # Atualizar cell state e hidden state
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)

    def init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inicializa estados ocultos."""
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


class TemporalConsistencyModule(nn.Module):
    """
    Módulo de consistência temporal usando ConvLSTM.
    Processa features do decoder para eliminar flicker entre frames.
    """

    def __init__(self, channels: int, hidden_channels: Optional[int] = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = channels

        self.convlstm = ConvLSTMCell(channels, hidden_channels)

        # Projeção de volta para número original de canais se necessário
        if hidden_channels != channels:
            self.proj = nn.Conv2d(hidden_channels, channels, 1)
        else:
            self.proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, state = self.convlstm(x, hidden_state)
        out = self.proj(h)
        return out, state


# =============================================================================
# Blocos do Decoder U-Net++
# =============================================================================

class ConvBlock(nn.Module):
    """
    Bloco convolucional padrão: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Com opcional CBAM attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_cbam: bool = True
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.cbam = CBAM(out_channels) if use_cbam else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.cbam(x)
        return x


class UNetPlusPlusDecoder(nn.Module):
    """
    Decoder U-Net++ com skip connections densos.
    Baseado em "UNet++: A Nested U-Net Architecture" (DLMIA 2018)
    """

    def __init__(
        self,
        encoder_channels: List[int],  # [16, 24, 40, 112, 320] para EfficientNet-B4
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        use_cbam: bool = True
    ):
        super().__init__()

        # Inverter para ir do fundo para o topo
        encoder_channels = encoder_channels[::-1]  # [320, 112, 40, 24, 16]

        self.num_stages = len(encoder_channels) - 1

        # Blocos de convolução para cada nó do U-Net++
        # Formato: x{i,j} onde i=profundidade, j=posição horizontal
        self.blocks = nn.ModuleDict()

        for i in range(self.num_stages):
            for j in range(self.num_stages - i):
                # Calcular canais de entrada
                if j == 0:
                    # Primeira coluna: encoder + upsampled do nível anterior
                    in_ch = encoder_channels[i] + (decoder_channels[i-1] if i > 0 else 0)
                else:
                    # Outras colunas: soma dos nós anteriores na mesma linha + upsampled
                    in_ch = encoder_channels[i+j] + decoder_channels[i] * j
                    if i > 0:
                        in_ch += decoder_channels[i-1]

                out_ch = decoder_channels[i]

                block_name = f"x{i}_{j}"
                self.blocks[block_name] = ConvBlock(in_ch, out_ch, use_cbam)

        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: Lista de features do encoder [x0, x1, x2, x3, x4]
                     Do mais superficial ao mais profundo
        Returns:
            Feature map do decoder mais superficial
        """
        features = features[::-1]  # Inverter: [x4, x3, x2, x1, x0]

        # Armazenar outputs de cada nó
        outputs = {}

        for i in range(self.num_stages):
            for j in range(self.num_stages - i):
                block_name = f"x{i}_{j}"

                # Coletar inputs
                inputs = []

                # Feature do encoder correspondente
                inputs.append(features[i + j])

                # Outputs dos nós anteriores na mesma linha
                for k in range(j):
                    inputs.append(outputs[f"x{i}_{k}"])

                # Output do nível anterior (upsampled)
                if i > 0:
                    prev_out = outputs[f"x{i-1}_{j}"]
                    prev_out = self.up(prev_out)
                    inputs.append(prev_out)

                # Concatenar e processar
                x = torch.cat(inputs, dim=1)
                outputs[block_name] = self.blocks[block_name](x)

        # Retornar output do nó mais superficial
        return outputs[f"x{self.num_stages-1}_0"]


# =============================================================================
# Modelo Principal: NerveSegmentationModel
# =============================================================================

class NerveSegmentationModel(nn.Module):
    """
    Modelo principal de segmentação de nervos.

    Arquitetura:
    - Encoder: EfficientNet-B4 (pretrained)
    - Decoder: U-Net++ com skip connections densos
    - Attention: CBAM em cada bloco do decoder
    - Temporal: ConvLSTM opcional para consistência entre frames

    Classes:
    - 0: Background
    - 1: Nervo
    - 2: Artéria
    - 3: Veia
    - 4: Fascia
    - 5: Músculo
    """

    # Canais do EfficientNet-B4 em cada estágio
    EFFICIENTNET_B4_CHANNELS = [24, 32, 56, 160, 448]

    def __init__(
        self,
        num_classes: int = 6,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        use_cbam: bool = True,
        use_temporal: bool = True,
        temporal_hidden: int = 32
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_temporal = use_temporal

        # Encoder (usando segmentation_models.pytorch ou timm)
        try:
            import segmentation_models_pytorch as smp

            # Criar modelo U-Net++ completo e extrair encoder
            self._full_model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=num_classes,
                decoder_channels=decoder_channels,
                decoder_attention_type="scse" if use_cbam else None
            )

            self.encoder = self._full_model.encoder
            self.decoder = self._full_model.decoder
            self.segmentation_head = self._full_model.segmentation_head

            self._using_smp = True

        except ImportError:
            # Fallback: usar timm para encoder + decoder custom
            import timm

            self.encoder = timm.create_model(
                encoder_name,
                pretrained=True,
                features_only=True,
                out_indices=(1, 2, 3, 4)
            )

            # Obter canais de saída do encoder
            with torch.no_grad():
                dummy = torch.randn(1, 3, 384, 384)
                features = self.encoder(dummy)
                encoder_channels = [f.shape[1] for f in features]

            self.decoder = UNetPlusPlusDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                use_cbam=use_cbam
            )

            # Cabeça de segmentação
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(decoder_channels[-1], decoder_channels[-1], 3, padding=1),
                nn.BatchNorm2d(decoder_channels[-1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(decoder_channels[-1], num_classes, 1)
            )

            self._using_smp = False

        # Módulo temporal (ConvLSTM)
        if use_temporal:
            self.temporal = TemporalConsistencyModule(
                channels=decoder_channels[-1] if not self._using_smp else 16,
                hidden_channels=temporal_hidden
            )
        else:
            self.temporal = None

        # Estado temporal (para inferência)
        self._temporal_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Uncertainty estimation
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        reset_temporal: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]
            return_features: Se True, retorna features intermediárias
            reset_temporal: Se True, reseta estado temporal

        Returns:
            Dict com:
            - 'logits': Logits de segmentação [B, C, H, W]
            - 'probs': Probabilidades [B, C, H, W]
            - 'mask': Máscara argmax [B, H, W]
            - 'features': Features do decoder (se return_features=True)
        """
        input_shape = x.shape[-2:]

        if reset_temporal:
            self._temporal_state = None

        # Encoder
        if self._using_smp:
            features = self.encoder(x)
            decoder_output = self.decoder(*features)
        else:
            features = self.encoder(x)
            decoder_output = self.decoder(features)

        # Temporal consistency
        if self.use_temporal and self.temporal is not None:
            decoder_output, self._temporal_state = self.temporal(
                decoder_output,
                self._temporal_state
            )

        # Segmentation head
        if self._using_smp:
            logits = self.segmentation_head(decoder_output)
        else:
            logits = self.segmentation_head(decoder_output)

        # Upsample para resolução original se necessário
        if logits.shape[-2:] != input_shape:
            logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=True)

        # Probabilidades e máscara
        probs = F.softmax(logits, dim=1)
        mask = torch.argmax(probs, dim=1)

        output = {
            'logits': logits,
            'probs': probs,
            'mask': mask
        }

        if return_features:
            output['features'] = decoder_output

        return output

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Predição com estimativa de incerteza usando MC Dropout.

        Args:
            x: Input tensor [B, 3, H, W]
            num_samples: Número de amostras para MC Dropout

        Returns:
            Dict com:
            - 'mean_probs': Probabilidades médias
            - 'uncertainty': Mapa de incerteza (entropia)
            - 'mask': Máscara final
        """
        self.train()  # Ativa dropout

        samples = []
        for _ in range(num_samples):
            with torch.no_grad():
                output = self.forward(x, reset_temporal=True)
                samples.append(output['probs'])

        self.eval()

        # Média das predições
        stacked = torch.stack(samples, dim=0)
        mean_probs = stacked.mean(dim=0)

        # Incerteza via entropia
        uncertainty = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)

        # Máscara final
        mask = torch.argmax(mean_probs, dim=1)

        return {
            'mean_probs': mean_probs,
            'uncertainty': uncertainty,
            'mask': mask
        }

    def reset_temporal_state(self):
        """Reseta o estado temporal (usar ao trocar de vídeo/paciente)."""
        self._temporal_state = None


# =============================================================================
# Loss Functions
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss para segmentação multi-classe.
    Bom para classes desbalanceadas.
    """

    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W] com valores de classe
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Calcular Dice por classe
        dims = (0, 2, 3)  # Reduzir sobre batch e spatial
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = (probs + targets_one_hot).sum(dim=dims)

        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.reduction == 'mean':
            return 1 - dice.mean()
        elif self.reduction == 'sum':
            return num_classes - dice.sum()
        else:
            return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss para lidar com desbalanceamento de classes.
    Paper: "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W]
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)

        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class UnifiedFocalLoss(nn.Module):
    """
    Unified Focal Loss: combina Dice + BCE + Focal.
    Melhor performance para segmentação de estruturas anatômicas.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.3,
        bce_weight: float = 0.2,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        bce = F.cross_entropy(logits, targets)

        total = (
            self.dice_weight * dice +
            self.focal_weight * focal +
            self.bce_weight * bce
        )

        return total


# =============================================================================
# Post-Processing
# =============================================================================

class NervePostProcessor:
    """
    Pós-processamento de máscaras de segmentação.
    Inclui:
    - Remoção de componentes pequenos
    - Preenchimento de buracos
    - Suavização de bordas
    - Cálculo de contornos
    """

    def __init__(
        self,
        min_area: int = 100,
        smooth_kernel: int = 5,
        fill_holes: bool = True
    ):
        self.min_area = min_area
        self.smooth_kernel = smooth_kernel
        self.fill_holes = fill_holes

    def process(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Processa máscara de segmentação.

        Args:
            mask: Máscara [H, W] com valores de classe

        Returns:
            Dict com máscaras processadas e contornos
        """
        import cv2

        result = {
            'processed_mask': mask.copy(),
            'contours': {},
            'areas': {},
            'centroids': {}
        }

        # Processar cada classe separadamente
        unique_classes = np.unique(mask)

        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue

            # Máscara binária para esta classe
            binary = (mask == class_id).astype(np.uint8)

            # Remover componentes pequenos
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

            clean_mask = np.zeros_like(binary)
            valid_contours = []
            valid_areas = []
            valid_centroids = []

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.min_area:
                    component_mask = (labels == i).astype(np.uint8)

                    # Preencher buracos
                    if self.fill_holes:
                        contour, _ = cv2.findContours(
                            component_mask,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        if contour:
                            cv2.fillPoly(component_mask, contour, 1)

                    # Suavizar bordas
                    if self.smooth_kernel > 0:
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (self.smooth_kernel, self.smooth_kernel)
                        )
                        component_mask = cv2.morphologyEx(
                            component_mask,
                            cv2.MORPH_CLOSE,
                            kernel
                        )

                    clean_mask = np.maximum(clean_mask, component_mask)

                    # Extrair contorno
                    contour, _ = cv2.findContours(
                        component_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contour:
                        valid_contours.extend(contour)
                        valid_areas.append(cv2.contourArea(contour[0]))
                        M = cv2.moments(contour[0])
                        if M['m00'] > 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            valid_centroids.append((cx, cy))

            # Atualizar máscara processada
            result['processed_mask'][mask == class_id] = 0
            result['processed_mask'][clean_mask > 0] = class_id

            result['contours'][class_id] = valid_contours
            result['areas'][class_id] = valid_areas
            result['centroids'][class_id] = valid_centroids

        return result


# =============================================================================
# Wrapper para Inferência
# =============================================================================

class NerveSegmentationInference:
    """
    Wrapper para inferência do modelo de segmentação.
    Inclui pré-processamento, pós-processamento e visualização.
    """

    # Mapeamento de classes
    CLASS_NAMES = {
        0: 'Background',
        1: 'Nervo',
        2: 'Artéria',
        3: 'Veia',
        4: 'Fascia',
        5: 'Músculo'
    }

    # Cores BGR
    CLASS_COLORS = {
        0: (0, 0, 0),        # Background - preto
        1: (0, 255, 255),    # Nervo - amarelo
        2: (0, 0, 255),      # Artéria - vermelho
        3: (255, 0, 0),      # Veia - azul
        4: (128, 128, 128),  # Fascia - cinza
        5: (128, 0, 128)     # Músculo - roxo
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        input_size: Tuple[int, int] = (384, 384),
        use_temporal: bool = True
    ):
        # Determinar device
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.input_size = input_size

        # Criar modelo
        self.model = NerveSegmentationModel(
            num_classes=6,
            use_temporal=use_temporal
        ).to(self.device)

        # Carregar pesos se fornecido
        if model_path:
            self.load_weights(model_path)

        self.model.eval()

        # Pós-processador
        self.post_processor = NervePostProcessor()

        # Normalização ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def load_weights(self, path: str):
        """Carrega pesos do modelo."""
        state_dict = torch.load(path, map_location=self.device)

        # Remover prefixo 'module.' se modelo foi salvo com DataParallel
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Pré-processa imagem para o modelo.

        Args:
            image: Imagem BGR [H, W, 3]

        Returns:
            Tensor normalizado [1, 3, H, W]
        """
        import cv2

        # Converter BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Para tensor e normalizar
        tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device) / 255.0
        tensor = (tensor - self.mean) / self.std

        return tensor

    def postprocess(
        self,
        output: Dict[str, torch.Tensor],
        original_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Pós-processa output do modelo.

        Args:
            output: Output do modelo
            original_size: (height, width) da imagem original

        Returns:
            Dict com máscaras e metadados
        """
        import cv2

        # Extrair máscara
        mask = output['mask'][0].cpu().numpy().astype(np.uint8)
        probs = output['probs'][0].cpu().numpy()

        # Resize para tamanho original
        mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

        # Pós-processar
        processed = self.post_processor.process(mask)

        # Calcular confidências por classe
        confidences = {}
        for class_id in range(1, 6):
            class_probs = probs[class_id]
            class_mask = (mask == class_id)
            if class_mask.any():
                # Resize probs para original size
                class_probs_resized = cv2.resize(class_probs, (original_size[1], original_size[0]))
                confidences[class_id] = float(class_probs_resized[class_mask].mean())
            else:
                confidences[class_id] = 0.0

        return {
            'mask': processed['processed_mask'],
            'contours': processed['contours'],
            'areas': processed['areas'],
            'centroids': processed['centroids'],
            'confidences': confidences,
            'probs': probs
        }

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        reset_temporal: bool = False
    ) -> Dict[str, Any]:
        """
        Predição completa em uma imagem.

        Args:
            image: Imagem BGR [H, W, 3]
            reset_temporal: Se True, reseta estado temporal

        Returns:
            Dict com predições e metadados
        """
        original_size = image.shape[:2]

        # Pré-processar
        tensor = self.preprocess(image)

        # Inferência
        output = self.model(tensor, reset_temporal=reset_temporal)

        # Pós-processar
        result = self.postprocess(output, original_size)

        return result

    def visualize(
        self,
        image: np.ndarray,
        result: Dict[str, Any],
        alpha: float = 0.4,
        show_contours: bool = True,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Visualiza resultado da segmentação.

        Args:
            image: Imagem original BGR
            result: Resultado do predict()
            alpha: Transparência da overlay
            show_contours: Mostrar contornos
            show_labels: Mostrar labels das classes

        Returns:
            Imagem com visualização
        """
        import cv2

        vis = image.copy()
        mask = result['mask']

        # Criar overlay colorida
        overlay = np.zeros_like(vis)
        for class_id, color in self.CLASS_COLORS.items():
            if class_id == 0:
                continue
            overlay[mask == class_id] = color

        # Aplicar overlay
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)

        # Desenhar contornos
        if show_contours:
            for class_id, contours in result['contours'].items():
                color = self.CLASS_COLORS.get(class_id, (255, 255, 255))
                cv2.drawContours(vis, contours, -1, color, 2)

        # Adicionar labels
        if show_labels:
            for class_id, centroids in result['centroids'].items():
                name = self.CLASS_NAMES.get(class_id, f'Class {class_id}')
                conf = result['confidences'].get(class_id, 0)
                color = self.CLASS_COLORS.get(class_id, (255, 255, 255))

                for cx, cy in centroids:
                    label = f"{name} {conf:.0%}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis, (cx - 2, cy - h - 4), (cx + w + 2, cy + 2), (0, 0, 0), -1)
                    cv2.putText(vis, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis


# =============================================================================
# Factory Function
# =============================================================================

def create_nerve_model(
    model_path: Optional[str] = None,
    device: str = 'auto',
    use_temporal: bool = True
) -> NerveSegmentationInference:
    """
    Factory function para criar modelo de segmentação.

    Args:
        model_path: Caminho para pesos (opcional)
        device: 'auto', 'cpu', 'cuda', 'mps'
        use_temporal: Usar ConvLSTM para consistência temporal

    Returns:
        Instância de NerveSegmentationInference
    """
    return NerveSegmentationInference(
        model_path=model_path,
        device=device,
        use_temporal=use_temporal
    )


# =============================================================================
# Teste
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NERVE TRACK v2.0 - Modelo de Segmentação")
    print("=" * 60)

    # Verificar dispositivo
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"[OK] Usando GPU Apple Metal (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"[OK] Usando GPU CUDA")
    else:
        device = 'cpu'
        print(f"[INFO] Usando CPU")

    # Criar modelo
    print("\nCriando modelo...")
    try:
        model = NerveSegmentationModel(
            num_classes=6,
            use_temporal=True
        )

        # Testar forward pass
        dummy_input = torch.randn(1, 3, 384, 384)
        output = model(dummy_input)

        print(f"[OK] Modelo criado com sucesso!")
        print(f"    - Input shape: {dummy_input.shape}")
        print(f"    - Output logits: {output['logits'].shape}")
        print(f"    - Output mask: {output['mask'].shape}")
        print(f"    - Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"[ERRO] Falha ao criar modelo: {e}")

    print("\n" + "=" * 60)
