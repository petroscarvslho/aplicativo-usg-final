# NEEDLE PILOT v3.1 - Sistema de Datasets e Treinamento

## Visao Geral

Este diretorio contem os scripts necessarios para:
1. Download de datasets de ultrassom
2. Processamento e preparacao de dados
3. Treinamento da CNN VASST para deteccao de agulhas

## Estrutura de Diretorios

```
datasets/
├── download_datasets.py    # Script principal de download
├── train_vasst.py          # Script de treinamento da CNN
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

## Como Usar

### 1. Gerar Dataset Sintetico (Recomendado para comecar)

```bash
cd datasets
python download_datasets.py
# Escolha opcao 5
# Digite numero de amostras (ex: 5000)
```

### 2. Baixar Dataset do Kaggle

```bash
# Primeiro configure a API do Kaggle:
# 1. Crie conta em kaggle.com
# 2. Va em Account > API > Create New Token
# 3. Salve kaggle.json em ~/.kaggle/
# 4. chmod 600 ~/.kaggle/kaggle.json

python download_datasets.py
# Escolha opcao 1
```

### 3. Treinar a CNN VASST

```bash
python train_vasst.py
# Escolha opcao 1 para treinar
# Configure epocas e batch size conforme necessario
```

O modelo treinado sera salvo em: `models/vasst_needle.pt`

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
