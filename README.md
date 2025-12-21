# APLICATIVO USG FINAL

App 100% Python/OpenCV para captura e processamento de imagens de ultrassom.

**Zero conversao de imagem** - Os pixels exibidos sao IDENTICOS aos que vem do iPhone.

## Requisitos

- macOS (testado em M3)
- Python 3.11+
- QuickTime Player (para espelhamento do iPhone)
- iPhone com app de ultrassom (Butterfly iQ, Clarius, etc.)

## Instalacao

```bash
# Ativar ambiente virtual
source /Users/priscoleao/usgapp/.venv/bin/activate

# Instalar dependencias (se necessario)
pip install opencv-python numpy torch ultralytics segmentation-models-pytorch pyobjc-framework-Quartz
```

## Como Usar

### 1. Preparar iPhone
1. Conecte o iPhone ao Mac via cabo
2. Abra QuickTime Player
3. Arquivo > Nova Gravacao de Filme
4. Selecione iPhone como fonte (seta ao lado do botao vermelho)

### 2. Executar App
```bash
cd /Users/priscoleao/aplicativo-usg-final
source /Users/priscoleao/usgapp/.venv/bin/activate
python3 main.py
```

Ou use o atalho:
```bash
/Users/priscoleao/Desktop/Start_USG_FINAL.command
```

## Atalhos

| Tecla | Funcao |
|-------|--------|
| 1-6 | Selecionar modo |
| M | Proximo modo |
| F | Congelar imagem |
| R | Gravar video |
| S | Screenshot |
| A | Ligar/desligar IA |
| T | Mostrar/esconder sidebar |
| H | Ajuda |
| Q | Sair |

## Modos

1. **B-MODE** - Imagem padrao
2. **AGULHA** - Deteccao de agulha (YOLO)
3. **NERVO** - Segmentacao de nervo (U-Net)
4. **CARDIACO** - Modo cardiaco
5. **FAST** - Protocolo FAST
6. **PULMAO** - Ultrassom pulmonar

## Configuracao

Edite `config.py` para ajustar:

```python
# Performance
UI_SIMPLE = True         # True = interface rapida

# Qualidade
KEEP_ORIGINAL_RESOLUTION = True  # Sem resize
PRESERVE_ORIGINAL_IMAGE = True   # Sem filtros

# Janela
WINDOW_WIDTH = 2300
WINDOW_HEIGHT = 1200
FULLSCREEN = False

# Gravacao
VIDEO_QUALITY = "high"   # lossless, high, medium, light
```

## Arquivos

```
aplicativo-usg-final/
├── main.py              # App principal
├── config.py            # Configuracoes
├── src/
│   ├── capture.py       # Captura de video
│   ├── window_capture.py # Captura macOS
│   ├── ai_processor.py  # IA (YOLO, U-Net)
│   └── design_system.py # Cores
├── models/
│   └── best.pt          # Modelo YOLO
└── captures/            # Screenshots/videos
```

## Licenca

Projeto privado.
