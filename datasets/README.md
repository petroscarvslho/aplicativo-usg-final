# USG FLOW - Sistema de Datasets e Treinamento

## Visao Geral

Este diretorio contem os scripts necessarios para:
1. Download de datasets de ultrassom
2. Processamento e preparacao de dados (unificado por plugin)
3. Export de dados para treinamento
4. Treinamento de modelos compatíveis com a inferência

## Estrutura de Diretorios

```
datasets/
├── download_datasets.py    # Script principal de download
├── train_vasst.py          # Script de treinamento da CNN
├── unified_dataset_manager.py  # Gerenciador unificado (todos os plugins)
├── README.md               # Este arquivo
│
├── kaggle_nerve/           # Dataset Kaggle (5,635 imagens)
├── camus_cardiac/          # Dataset CAMUS (4,000+ frames)
├── brachial_plexus/        # Dataset Brachial Plexus (41,000 frames)
├── synthetic_needle/       # Dataset sintetico gerado
├── clarius_open/           # Outros datasets abertos
└── processed/              # Dados processados para treinamento
    ├── X_train.npy
    ├── Y_train.npy
    ├── X_val.npy
    ├── Y_val.npy
    ├── X_test.npy
    ├── Y_test.npy
    └── combined/           # Datasets combinados
```

## Como Usar (Unificado)

### 1. Preparar Datasets por Plugin

```bash
cd datasets
python unified_dataset_manager.py
# Opcao 7 (setup completo) ou use opcoes 3-6 manualmente
```

### 2. Treinar Modelos (plugin_registry)

```bash
python training/train_unified.py --plugin NEEDLE
python training/train_unified.py --plugin NERVE --model unet
python training/train_unified.py --plugin CARDIAC
```

### 3. Treinar YOLO (deteccao)

```bash
python training/train_yolo.py --plugin FAST
python training/train_yolo.py --plugin NEEDLE
```

Os modelos treinados sao exportados automaticamente para `models/` com `.meta.json`.

## Datasets Disponiveis

| Dataset | Imagens | Tipo | Acesso |
|---------|---------|------|--------|
| Kaggle Nerve | 5,635 | Segmentacao | Publico (API) |
| CAMUS Cardiac | 4,000+ | Ecocardiografia | Registro gratis |
| Brachial Plexus | 41,000 | Needle Tracking | Busca manual |
| Sintetico | Ilimitado | Needle Tracking | Gerado localmente |

## Parametros de Treinamento

Configuracoes padrao do `train_vasst.py`:

- **Epocas**: 100 (com early stopping)
- **Batch size**: 32
- **Learning rate**: 0.0001 (com decay)
- **Loss function**: MAE (Mean Absolute Error)
- **Optimizer**: Adam
- **Input shape**: 256x256 grayscale

## Links Uteis

- Kaggle: https://www.kaggle.com/c/ultrasound-nerve-segmentation
- CAMUS: https://www.creatis.insa-lyon.fr/Challenge/camus/
- IEEE DataPort: https://ieee-dataport.org/
- Papers With Code: https://paperswithcode.com/datasets?q=ultrasound
- Zenodo: https://zenodo.org/search?q=ultrasound

## Proximos Passos

1. Execute `python download_datasets.py` para baixar/gerar dados
2. Execute `python train_vasst.py` para treinar o modelo
3. O modelo treinado sera carregado automaticamente pelo NEEDLE PILOT
