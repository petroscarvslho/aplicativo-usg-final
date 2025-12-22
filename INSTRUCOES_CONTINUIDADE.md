# INSTRUCOES OBRIGATORIAS PARA CONTINUAR EM UM NOVO TERMINAL (CLAUDE CODE)

Este texto define as regras oficiais para continuidade do trabalho em um novo chat / terminal usando Claude Code.

====================================================================
## PROMPT PARA COPIAR NO NOVO CHAT
====================================================================

```
Estou continuando o desenvolvimento de um projeto de software.

LOCALIZACAO DO PROJETO: /Users/priscoleao/aplicativo-usg-final
REPOSITORIO GIT: https://github.com/petroscarvslho/aplicativo-usg-final

AO INICIAR, VOCE DEVE OBRIGATORIAMENTE:

1. Ler TODOS os arquivos de documentacao relevantes do projeto
   - README.md
   - INSTRUCOES_CONTINUIDADE.md (este arquivo)
   - REFERENCIAS.md
   - DIARIO.md (se existir)
   - config.py (para entender configuracoes)

2. Verificar o estado atual do repositorio:
   - Executar `git status`
   - Executar `git log --oneline -10`

3. Analisar o estado atual do projeto e entao:
   - Resumir o que ja foi feito
   - Identificar o que esta em andamento
   - Sugerir claramente os proximos passos

Somente apos isso, continuar o desenvolvimento.
```

====================================================================
## PASSO A PASSO PARA CONTINUAR (SEM TREINAMENTO)
====================================================================

1. Ir para o projeto e ativar o ambiente:
   - `cd /Users/priscoleao/aplicativo-usg-final`
   - `source /Users/priscoleao/usgapp/.venv/bin/activate`
2. Ler documentos obrigatorios:
   - `README.md`, `INSTRUCOES_CONTINUIDADE.md`, `REFERENCIAS.md`, `DIARIO.md`, `config.py`
3. Verificar estado do repo:
   - `git status`
   - `git log --oneline -10`
4. Confirmar que o treinamento de datasets esta rodando em outro terminal.
   - NAO iniciar treinamento neste terminal.
5. Se precisar validar UI, abrir o app:
   - `python3 main.py`
6. Foco atual de melhoria (ordem):
   - FAST Protocol (melhorar confianca e workflow)
   - LUNG / BLADDER / CARDIAC (refinar overlays e criterios)
7. Antes de encerrar a sessao:
   - Atualizar `DIARIO.md` e `INSTRUCOES_CONTINUIDADE.md`
   - Commitar apenas o que foi feito nesta sessao

====================================================================
## REGRA ABSOLUTA ANTES DE FECHAR QUALQUER CHAT / SESSAO
====================================================================

ESTA REGRA E OBRIGATORIA E NAO PODE SER IGNORADA!

ANTES DE ENCERRAR QUALQUER SESSAO, VOCE (CLAUDE CODE) DEVE EXECUTAR TODOS OS PASSOS ABAIXO, SEM EXCECAO E SEM PERGUNTAR NADA AO USUARIO:

### ETAPA 1 - ATUALIZACAO DE DOCUMENTACAO

1. Atualizar TODOS os arquivos de documentacao do projeto que registrem estado ou progresso, incluindo:
   - DIARIO.md (criar se nao existir)
   - Este arquivo de instrucoes (INSTRUCOES_CONTINUIDADE.md)
   - Qualquer outro arquivo que descreva estado atual do projeto

Essas atualizacoes DEVEM refletir fielmente tudo o que foi feito na sessao:
- Alteracoes de codigo
- Correcoes
- Decisoes tecnicas
- Melhorias
- Pendencias conhecidas
- Proximos passos recomendados

### ETAPA 2 - VERIFICACAO DO REPOSITORIO

2. Garantir que o repositorio esteja consistente:

```bash
git status
git add .
git commit -m "Atualizacao de sessao - [DATA] - [RESUMO]"
git push
```

### ETAPA 3 - RESUMO FINAL

3. Apresentar ao usuario um resumo claro do que foi feito na sessao.

====================================================================
## ESTRUTURA DO PROJETO
====================================================================

```
aplicativo-usg-final/
├── main.py                 # Aplicativo principal (interface completa)
├── config.py               # TODAS as configuracoes
├── src/
│   ├── capture.py          # Captura de video
│   ├── window_capture.py   # Captura de janela macOS (pixel-perfect)
│   ├── ai_processor.py     # Processamento de IA OTIMIZADO
│   ├── clip_recorder.py    # Gravacao de video
│   └── design_system.py    # Sistema de cores
├── models/
│   └── best.pt             # Modelo YOLO para agulha
├── captures/               # Screenshots e gravacoes
├── REFERENCIAS.md          # Referencias tecnicas
├── INSTRUCOES_CONTINUIDADE.md  # Este arquivo
└── DIARIO.md               # Diario de desenvolvimento
```

====================================================================
## COMO EXECUTAR O APP
====================================================================

```bash
cd /Users/priscoleao/aplicativo-usg-final
source /Users/priscoleao/usgapp/.venv/bin/activate
python3 main.py
```

Ou usar o atalho:
```bash
/Users/priscoleao/Desktop/Start_USG_FINAL.command
```

====================================================================
## REQUISITOS PARA FUNCIONAR
====================================================================

1. QuickTime Player aberto com iPhone espelhando
2. Virtual environment ativado (/Users/priscoleao/usgapp/.venv)
3. Dependencias instaladas (opencv, numpy, torch, ultralytics)

====================================================================
## CONFIGURACOES IMPORTANTES (config.py)
====================================================================

- VIDEO_SOURCE = "AIRPLAY" - Fonte de video
- KEEP_ORIGINAL_RESOLUTION = True - Sem resize (qualidade maxima)
- PRESERVE_ORIGINAL_IMAGE = True - Sem filtros na imagem original
- UI_SIMPLE = True - Interface rapida
- VIDEO_QUALITY = "high" - Qualidade de gravacao
- USE_MPS = True - Usar GPU Apple Silicon

====================================================================
## ATALHOS DO TECLADO
====================================================================

FONTE (grupo exclusivo - so 1 ativo):
- 1: B-MODE (sem IA)
- C: IA ZONE (selecionar regiao)
- A: IA FULL (IA em toda tela)

PLUGINS DE IA (2-9, 0, V):
- 2: NEEDLE (Needle Pilot)
- 3: NERVE (Nerve Track)
- 4: CARDIAC (Cardiac AI)
- 5: FAST (Trauma)
- 6: ANATOMY (ID estruturas)
- 7: M-MODE (Movimento temporal)
- 8: COLOR (Color Doppler)
- 9: POWER (Power Doppler)
- 0: LUNG (Linhas-B)
- V: BLADDER (Volume Vesical)

CONTROLES:
- F: Congelar imagem
- R: Iniciar/parar gravacao
- S: Screenshot
- B: Toggle Biplane

VISUALIZACAO:
- T: Mostrar/esconder sidebar
- H: Mostrar/esconder overlays
- I: Mostrar/esconder instrucoes
- +/-: Zoom In/Out
- Setas: Pan (mover imagem)
- 0 (em modo zoom): Reset View
- ?: Ajuda (modal)
- F11: Fullscreen
- Q/ESC: Sair

====================================================================
## OTIMIZACOES IMPLEMENTADAS
====================================================================

Baseado em pesquisa de: Butterfly iQ3, Clarius, SUPRA, EchoNet

1. **MPS/GPU Acceleration** - Usa GPU Apple Silicon (Metal)
2. **Lazy Loading** - Modelos carregados apenas quando necessario
3. **Smart Frame Skip** - Pula frames baseado em movimento
4. **Resolution Scaling** - Processa IA em resolucao menor, exibe em original
5. **Temporal Smoothing** - Suaviza deteccoes para reduzir jitter
6. **ROI Processing** - Processa apenas regiao de interesse
7. **Preset-Based Config** - Configuracoes otimizadas por modo

====================================================================
## REFERENCIAS DE PESQUISA
====================================================================

Projetos analisados:
- Butterfly iQ3: NeedleViz, Auto B-lines, Auto Bladder, Cardiac Output
- Clarius: T-Mode, MSK AI, Voice Controls, Median Nerve AI
- SUPRA: Pipeline real-time, beamforming
- EchoNet: Fracao de ejecao automatica
- USFM: Modelo fundacional para ultrassom
- YOLOv8: Deteccao em tempo real

Arquivos de pesquisa em:
- /Users/priscoleao/usgapp/research/
- /Users/priscoleao/Documents/PlataformaUSG/research/

====================================================================
## ESTADO ATUAL DO PROJETO (ATUALIZAR A CADA SESSAO)
====================================================================

### Ultima atualizacao: 2025-12-22 (Sessao 17)

- Registro central `plugin_registry.py` com nomes de pesos, formatos de label e modelos por plugin.
- Unified dataset manager expandido (FAST/ANATOMY/BLADDER/LUNG) + novos sinteticos (bladder/lung/fast).
- Treinamento unificado alinhado com inferencia: VASST compat, EchoNet regression, Unet SMP.
- Export automatico de pesos para `models/` com metadata `.meta.json`.
- Novo script `training/train_yolo.py` para YOLO (NEEDLE/FAST).
- VASST no app agora le metadata e suporta label order/scale automaticamente.
- REFERENCIAS.md expandido com pesquisa externa (apps pagos, AI comerciais,
  papers/whitepapers, datasets, repos GitHub e threads do Reddit).
- REFERENCIAS.md ampliado com FAST/eFAST (DL + view classification) e IQA (scan quality).
- LUNG AI fallback CV com Auto B-lines (densidade + clusters) e pleura_y suavizado.
- BLADDER AI fallback CV com dual-view (sag/trans), score de qualidade e suavizacao.
- CARDIAC AI fallback CV com AutoEF mais robusto (percentis + suavizacao).
- Auto Gain + Scan Quality overlay integrado em FAST/LUNG/BLADDER/CARDIAC.
- Treinamento de datasets ocorre em outro terminal (nao iniciar aqui).
- Repo inclui `datasets/unified/` e `training/checkpoints/` (grande, ~785MB).

### Proximos passos recomendados (ordem sugerida)
1. FAST: usar SCAN Q como gating para reduzir falso positivo + opcional auto-check de janela.
2. LUNG/BLADDER/CARDIAC: refinar overlays e criterios de qualidade/estabilidade.
3. Revisar mudancas locais de treino/datasets antes de commitar.

====================================================================
## PROJETOS RELACIONADOS
====================================================================

### ULTRASOUND NEEDLE TRAINER (Projeto Separado)
Repositorio dedicado ao treinamento da CNN para o plugin NEEDLE PILOT.

- **GitHub**: https://github.com/petroscarvslho/ultrasound-needle-trainer
- **Local**: /Users/priscoleao/ultrasound-needle-trainer
- **Funcao**: Baixar datasets e treinar modelo VASST CNN
- **Output**: models/vasst_needle.pt

**Como usar o modelo treinado no NEEDLE PILOT:**
1. Treinar modelo no projeto ultrasound-needle-trainer
2. Copiar models/vasst_needle.pt para aplicativo-usg-final/models/
3. O NEEDLE PILOT carrega automaticamente

```bash
# Apos treinar no projeto separado:
cp /Users/priscoleao/ultrasound-needle-trainer/models/vasst_needle.pt \
   /Users/priscoleao/aplicativo-usg-final/models/
```

====================================================================
## ARQUITETURA DO PROJETO
====================================================================

```
aplicativo-usg-final/           <- PROJETO PRINCIPAL (este)
├── main.py                     <- Aplicativo completo
├── src/ai_processor.py         <- Todos os plugins de IA
│   ├── NEEDLE PILOT v3.1       <- Plugin de agulha (usa VASST CNN)
│   ├── NERVE TRACK             <- Plugin de nervo
│   ├── CARDIAC AI              <- Plugin cardiaco
│   └── ... (10 plugins)
├── models/
│   ├── best.pt                 <- YOLO para agulha
│   ├── vasst_needle.pt         <- CNN treinada (do projeto separado)
│   └── ...
└── datasets/                   <- Scripts locais (copia do trainer)

ultrasound-needle-trainer/      <- PROJETO DE TREINAMENTO (separado)
├── download_datasets.py        <- Baixar/gerar datasets
├── train_vasst.py              <- Treinar CNN VASST
└── models/vasst_needle.pt      <- Modelo treinado (copiar para ca)
```

====================================================================

### SISTEMA DE DATASETS E TREINAMENTO (Sessao 10):
Criado sistema completo para download, preparacao e treinamento da CNN VASST:

**Arquivos criados em datasets/:**
- `download_datasets.py` - Script principal de download e processamento
- `train_vasst.py` - Script de treinamento da CNN PyTorch
- `README.md` - Documentacao do sistema

**Datasets suportados:**
1. Kaggle Nerve Segmentation (5,635 imagens) - Download via API
2. CAMUS Cardiac (4,000+ frames) - Registro gratis
3. Regional-US Brachial Plexus (41,000 frames) - Instrucoes de busca
4. Dataset Sintetico (ilimitado) - Geracao automatica
5. Outros (Breast US, etc) - Links e instrucoes

**Como usar:**
```bash
cd datasets
python download_datasets.py  # Opcao 5 para sintetico
python train_vasst.py        # Treinar modelo
# Modelo salvo em: models/vasst_needle.pt
```

### SISTEMA UNIFICADO (Sessao 12):
- `plugin_registry.py` centraliza nomes de pesos, formato de labels e modelos por plugin
- `datasets/unified_dataset_manager.py` gera/exporta dados para TODOS os plugins
- `training/train_unified.py` treina modelos compativeis com a inferencia
- `training/train_yolo.py` treina deteccao YOLO (NEEDLE/FAST)
- Export automatico em `models/` + metadata `.meta.json`

**Fluxo recomendado:**
```bash
# 1) Preparar datasets (kaggle/SimpleITK no venv)
python datasets/unified_dataset_manager.py

# 2) Treinar regressao/segmentacao/classificacao
python training/train_unified.py --plugin NEEDLE
python training/train_unified.py --plugin NERVE --model unet
python training/train_unified.py --plugin CARDIAC

# 3) Treinar deteccao YOLO
python training/train_yolo.py --plugin FAST
python training/train_yolo.py --plugin NEEDLE
```

### NEEDLE PILOT v3.1 PREMIUM (Sessao 9):
- **FASE 1 - Kalman Filter**: Suavizacao e predicao de posicao
  - Tracking continuo mesmo quando agulha some (ate 10 frames)
  - Modos visuais: TRACKING (verde), PREDICTING (laranja), SEARCHING (cinza)
- **FASE 2 - RANSAC**: Deteccao robusta de linha
  - Remove outliers (deteccoes falsas)
  - CLAHE para melhor contraste em regioes escuras
- **FASE 3 - Calibracao mm/pixel**: Conversao pixel ↔ milimetros
  - Escala lateral calibrada com marcacoes reais
  - Profundidade ajustavel: [ = -10mm, ] = +10mm
  - Config: ULTRASOUND_DEPTH_MM, CALIBRATION_MODE
- **FASE 4 - VASST CNN**: Refinamento de centroide
  - Classe VASSTPyTorch (CNN em PyTorch compativel com projeto)
  - Pronto para weights quando disponiveis (models/vasst_needle.pt)
  - Fallback CV com Sobel + CLAHE quando CNN indisponivel
  - Refinamento integrado no pipeline de deteccao
- **FASE 5 - Visual Premium Polish**:
  - Elipse de incerteza baseada na covariancia do Kalman
  - Info de calibracao no painel
  - Trail com cores por confianca
  - Versao v3.1 no painel

### O que foi feito:
- App 100% Python/OpenCV funcionando
- Interface premium com header, sidebar, footer
- **Sistema exclusivo de modos: B-MODE, IA ZONE, IA FULL**
- **Sistema de ROI profissional e interativo**
- **Fullscreen nativo** (cv2.setWindowProperty)
- **Visual premium para todos os plugins de IA**
- **Animacoes de transicao entre modos**
- 10 plugins de IA (NEEDLE ate BLADDER)
- Captura de tela do iPhone via AIRPLAY (pixel-perfect)
- Deteccao de agulha com YOLO + trajetoria animada
- Segmentacao de nervo com U-Net ResNet34
- Screenshot PNG lossless
- Gravacao com opcoes de qualidade
- Modo Biplane (lado a lado)
- Zoom/Pan funcionando
- Modal de ajuda com atalhos
- Instrucoes contextuais por modo
- TODOS os modelos AI de qualidade maxima criados

### Otimizacoes Premium (Sessao 8):
- **13 melhorias implementadas** (5 criticas, 4 altas, 3 medias, 1 baixa)
- **Correcoes de bugs criticos**: bare except, division by zero, memory leak, exponential backoff
- **Validacao de config no startup**: validate_config() e print_config_status()
- **Logging estruturado**: logger USG_FLOW com formato timestamp
- **Type hints**: CapturaThread, AIThread, AIProcessor
- **Constantes padronizadas**: MIN_ROI_SIZE=20, MIN_ROI_GRID=100, AI_MODE_MAP
- **UI/UX melhorado**: Indicador loading AI, timer gravacao com barra progresso
- **Error handling robusto**: Try/except com protecao OOM na AI

### Sistema de ROI Interativo (Sessao 7):
- **Redimensionar pelos handles**: Arrastar cantos para ajustar tamanho
- **Mover pelo centro**: Arrastar dentro da ROI para mover
- **Presets rapidos**: P=50%, O=75%, L=100%, K=Quadrado
- **Visual premium**: overlay escuro, marching ants, handles, crosshair, grid tercos
- **Controles**: ENTER confirma, ESC cancela
- **Barra de status** inferior com presets e instrucoes

### Visual Premium para Plugins (Sessao 7):
- **NEEDLE**: Trajetoria gradiente, projecao tracejada, painel com angulo
- **CARDIAC**: Painel EF% premium, barra colorida, legenda camaras
- **FAST**: Painel checkboxes, status NEGATIVE/POSITIVE
- **LUNG**: Display B-lines, barra severidade, linha pleura
- **BLADDER**: Volume mL grande, contorno glow, formula exibida

### Animacoes (Sessao 7):
- Fade suave ao trocar de modo
- Pulse nos botoes ativos
- Transicoes visuais premium

### Otimizacoes aplicadas:
- Fonte DUPLEX em toda interface
- Sidebar 280px, botoes 30px altura
- LINE_AA nos textos principais
- Fullscreen nativo OpenCV
- AIProcessor com todas otimizacoes
- Smart Frame Skipper + Temporal Smoother
- Lazy Loading + MPS/GPU

### Funcionalidades OK (testadas):
- ✅ Interface completa com sistema exclusivo
- ✅ Sistema ROI profissional e interativo
- ✅ Visual premium para todos plugins AI
- ✅ Animacoes de transicao
- ✅ Fullscreen nativo (sem faixa branca)
- ✅ Captura de tela iPhone via AIRPLAY
- ✅ Todos os 10 modos AI funcionando
- ✅ Gravacao de video MP4
- ✅ Screenshot PNG
- ✅ Zoom/Pan
- ✅ Biplane
- ✅ MPS/GPU Apple Silicon

### Modelos Disponiveis (335 MB total):
- ✅ `models/best.pt` - 6.2 MB - YOLOv8n (agulha)
- ✅ `models/fast_detector.pt` - 6.2 MB - YOLOv8n (FAST)
- ✅ `models/unet_nerve.pt` - 93.4 MB - U-Net ResNet34 (nervo)
- ✅ `models/bladder_seg.pt` - 93.4 MB - U-Net ResNet34 (bexiga)
- ✅ `models/usfm_segment.pt` - 93.4 MB - U-Net ResNet34 (anatomia)
- ✅ `models/echonet.pt` - 42.7 MB - ResNet18 (cardiaco)

====================================================================
## PROXIMAS MELHORIAS AGENDADAS (PRIORIDADE)
====================================================================

### 1. Treinar modelos especificos
- Treinar YOLO especifico para agulhas de ultrassom
- Fine-tune U-Net com dados reais de nervos
- Treinar EchoNet com dados cardiacos reais

### 2. Controles por voz
- Implementar voice controls para hands-free
- Comandos: "freeze", "record", "screenshot", modos

### 3. Geracao de relatorios
- Gerar PDF com capturas e medicoes
- Template profissional de laudo

### 4. Historico e persistencia
- Salvar historico de medicoes
- Sessoes com timeline
- Export de dados

### 5. Integracao PACS
- Export DICOM
- Conexao com sistemas hospitalares

### 6. Melhorias visuais adicionais
- Temas (dark/light)
- Customizacao de cores por usuario
- Layouts alternativos

####################################################################
####################################################################
##                                                                ##
##     CRONOGRAMA COMPLETO: OTIMIZACAO DO NEEDLE PILOT            ##
##     Objetivo: Tornar o plugin o mais perfeito possivel         ##
##     Data de criacao: 2024-12-22                                ##
##                                                                ##
####################################################################
####################################################################

============================================================
VISAO GERAL DO PLANO
============================================================

O NEEDLE PILOT atual funciona assim:

    FRAME → YOLO/Hough detecta → Guarda historico → Desenha

O NEEDLE PILOT OTIMIZADO vai funcionar assim:

    FRAME → YOLO/Hough detecta → RANSAC refina → Kalman suaviza
          → Calibracao converte → Desenha com precisao real

============================================================
FASE 1: KALMAN FILTER (Suavizacao e Predicao)
============================================================

### O QUE E:
O Kalman Filter e um algoritmo que:
- SUAVIZA posicoes ruidosas (agulha para de tremer)
- PREDIZ posicao quando a deteccao falha (agulha "some")
- USA FISICA do movimento para estimar posicoes futuras

### COMO FUNCIONA (simplificado):

    Frame 1: Detectou em (100, 200) → Kalman aceita
    Frame 2: Detectou em (102, 203) → Kalman suaviza para (101, 201)
    Frame 3: NAO detectou           → Kalman prediz (103, 204)
    Frame 4: Detectou em (120, 250) → Kalman diz "impossivel, deve ser (105, 206)"

### ARQUIVOS A MODIFICAR:
- src/ai_processor.py (classe AIProcessor)

### CODIGO A ADICIONAR:

```python
# No inicio do arquivo ai_processor.py, adicionar import:
from filterpy.kalman import KalmanFilter
import numpy as np

# Na classe AIProcessor, no __init__, adicionar:
def __init__(self):
    # ... codigo existente ...

    # ══════════════════════════════════════════════════════════
    # KALMAN FILTER PARA NEEDLE TRACKING
    # ══════════════════════════════════════════════════════════
    self.needle_kalman = self._create_needle_kalman()
    self.kalman_initialized = False
    self.frames_without_detection = 0
    self.max_prediction_frames = 10  # Maximo de frames para predizer sem deteccao

def _create_needle_kalman(self):
    """
    Cria Kalman Filter para tracking de agulha.

    Estado: [x, y, vx, vy, angle, v_angle]
    - x, y: posicao da ponta da agulha
    - vx, vy: velocidade da ponta
    - angle: angulo da agulha
    - v_angle: velocidade angular

    Medicao: [x, y, angle]
    """
    kf = KalmanFilter(dim_x=6, dim_z=3)

    # Matriz de transicao de estado (F)
    # Assume movimento com velocidade constante
    dt = 1/30  # ~30 FPS
    kf.F = np.array([
        [1, 0, dt, 0,  0,  0],   # x = x + vx*dt
        [0, 1, 0,  dt, 0,  0],   # y = y + vy*dt
        [0, 0, 1,  0,  0,  0],   # vx = vx
        [0, 0, 0,  1,  0,  0],   # vy = vy
        [0, 0, 0,  0,  1,  dt],  # angle = angle + v_angle*dt
        [0, 0, 0,  0,  0,  1],   # v_angle = v_angle
    ])

    # Matriz de medicao (H) - medimos x, y, angle
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
    ])

    # Covariancia do processo (Q) - incerteza do modelo
    kf.Q = np.eye(6) * 0.1
    kf.Q[2, 2] = 0.5  # Velocidade x pode variar mais
    kf.Q[3, 3] = 0.5  # Velocidade y pode variar mais

    # Covariancia da medicao (R) - incerteza da deteccao
    kf.R = np.array([
        [5, 0, 0],    # Incerteza em x (pixels)
        [0, 5, 0],    # Incerteza em y (pixels)
        [0, 0, 2],    # Incerteza em angulo (graus)
    ])

    # Covariancia inicial (P)
    kf.P *= 100

    return kf

def _update_needle_kalman(self, detected, tip_x, tip_y, angle):
    """
    Atualiza o Kalman Filter com nova medicao ou faz predicao.

    Args:
        detected: True se agulha foi detectada
        tip_x, tip_y: posicao da ponta (se detectada)
        angle: angulo da agulha (se detectada)

    Returns:
        (x, y, angle, confidence): posicao suavizada/predita
    """
    if detected:
        # Resetar contador de frames sem deteccao
        self.frames_without_detection = 0

        if not self.kalman_initialized:
            # Primeira deteccao - inicializar estado
            self.needle_kalman.x = np.array([
                tip_x, tip_y, 0, 0, angle, 0
            ]).reshape(6, 1)
            self.kalman_initialized = True
            return tip_x, tip_y, angle, 1.0

        # Predicao
        self.needle_kalman.predict()

        # Atualizacao com medicao
        z = np.array([tip_x, tip_y, angle]).reshape(3, 1)
        self.needle_kalman.update(z)

        # Extrair estado suavizado
        x = self.needle_kalman.x
        return float(x[0]), float(x[1]), float(x[4]), 1.0

    else:
        # Sem deteccao - usar predicao
        self.frames_without_detection += 1

        if not self.kalman_initialized:
            return None, None, None, 0.0

        if self.frames_without_detection > self.max_prediction_frames:
            # Muitos frames sem deteccao - perder tracking
            self.kalman_initialized = False
            return None, None, None, 0.0

        # Apenas predicao (sem update)
        self.needle_kalman.predict()

        # Confianca decai com o tempo
        confidence = 1.0 - (self.frames_without_detection / self.max_prediction_frames)

        x = self.needle_kalman.x
        return float(x[0]), float(x[1]), float(x[4]), confidence
```

### MODIFICAR _process_needle():

```python
def _process_needle(self, frame):
    # ... codigo de deteccao existente (YOLO ou Hough) ...

    # ANTES de desenhar, passar pelo Kalman:
    if detected_needle and needle_tip is not None:
        # Calcular angulo da deteccao
        if len(self.needle_history) >= 1:
            _, _, x1, y1, x2, y2 = self.needle_history[-1]
            raw_angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
        else:
            raw_angle = 0

        # Atualizar Kalman com deteccao
        smooth_x, smooth_y, smooth_angle, kalman_conf = self._update_needle_kalman(
            True, needle_tip[0], needle_tip[1], raw_angle
        )

        # Usar valores suavizados
        needle_tip = (int(smooth_x), int(smooth_y))
        needle_angle = smooth_angle
        confidence = confidence * kalman_conf
    else:
        # Sem deteccao - tentar predicao
        smooth_x, smooth_y, smooth_angle, kalman_conf = self._update_needle_kalman(
            False, 0, 0, 0
        )

        if smooth_x is not None:
            # Usar predicao do Kalman
            needle_tip = (int(smooth_x), int(smooth_y))
            needle_angle = smooth_angle
            confidence = kalman_conf * 0.7  # Menor confianca para predicao
            detected_needle = True  # Marcar como "detectado" via predicao

    # ... resto do codigo de desenho ...
```

### DEPENDENCIA A INSTALAR:
```bash
pip install filterpy
```

### RESULTADO ESPERADO:
- Agulha para de "tremer" entre frames
- Tracking continua por ate 10 frames mesmo sem deteccao
- Movimentos bruscos sao suavizados

============================================================
FASE 2: RANSAC (Robustez ao Ruido)
============================================================

### O QUE E:
RANSAC (Random Sample Consensus) e um algoritmo que:
- Encontra a MELHOR LINHA entre pontos ruidosos
- IGNORA outliers (deteccoes falsas)
- Muito usado em visao computacional

### COMO FUNCIONA:

    Pontos detectados: (100,200), (105,210), (300,50), (110,220), (115,230)
                                              ↑
                                         OUTLIER (erro)

    RANSAC: "A melhor linha passa por (100,200)→(115,230), ignoro o (300,50)"

### CODIGO A ADICIONAR:

```python
# No inicio do arquivo, adicionar:
from sklearn.linear_model import RANSACRegressor

# Na classe AIProcessor, adicionar metodo:
def _ransac_fit_needle(self, points, min_samples=3):
    """
    Usa RANSAC para encontrar a melhor linha entre pontos detectados.

    Args:
        points: lista de (x, y) dos pontos detectados
        min_samples: minimo de pontos para considerar valido

    Returns:
        (x1, y1, x2, y2, inliers_count): linha ajustada e quantidade de inliers
        ou None se nao conseguir ajustar
    """
    if len(points) < min_samples:
        return None

    # Separar X e Y
    X = np.array([p[0] for p in points]).reshape(-1, 1)
    y = np.array([p[1] for p in points])

    try:
        # RANSAC para regressao linear
        ransac = RANSACRegressor(
            min_samples=min_samples,
            residual_threshold=10,  # pixels de tolerancia
            max_trials=100
        )
        ransac.fit(X, y)

        # Obter inliers (pontos que fazem parte da linha)
        inlier_mask = ransac.inlier_mask_
        inliers_count = sum(inlier_mask)

        if inliers_count < min_samples:
            return None

        # Calcular extremos da linha
        X_inliers = X[inlier_mask]
        x_min, x_max = X_inliers.min(), X_inliers.max()

        # Prever Y nos extremos
        y_min = ransac.predict([[x_min]])[0]
        y_max = ransac.predict([[x_max]])[0]

        return (int(x_min), int(y_min), int(x_max), int(y_max), inliers_count)

    except Exception:
        return None

def _detect_needle_enhanced(self, frame):
    """
    Deteccao de agulha melhorada com RANSAC.

    1. Detecta edges com Canny
    2. Encontra linhas com HoughLinesP
    3. Coleta todos os pontos das linhas candidatas
    4. Usa RANSAC para encontrar a melhor linha
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar CLAHE para melhorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Detectar edges
    edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)

    # Encontrar linhas candidatas
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=40, maxLineGap=15)

    if lines is None:
        return None

    # Coletar pontos de linhas com angulo de agulha (15-75 graus)
    needle_points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)

        # Filtrar por angulo tipico de agulha
        if 15 < angle < 75 or 105 < angle < 165:
            # Adicionar pontos ao longo da linha
            num_points = max(5, int(np.sqrt((x2-x1)**2 + (y2-y1)**2) / 10))
            for t in np.linspace(0, 1, num_points):
                px = int(x1 + t * (x2 - x1))
                py = int(y1 + t * (y2 - y1))
                needle_points.append((px, py))

    if len(needle_points) < 5:
        return None

    # Aplicar RANSAC
    result = self._ransac_fit_needle(needle_points)

    return result
```

### MODIFICAR _process_needle() PARA USAR RANSAC:

```python
# No fallback CV (quando nao tem YOLO), substituir:
else:
    # Fallback CV com RANSAC
    ransac_result = self._detect_needle_enhanced(frame)

    if ransac_result is not None:
        x1, y1, x2, y2, inliers = ransac_result
        detected_needle = True
        confidence = min(0.95, inliers / 20)  # Mais inliers = mais confianca
        needle_tip = (x2, y2) if y2 > y1 else (x1, y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        self.needle_history.append((cx, cy, x1, y1, x2, y2))
```

### DEPENDENCIA:
```bash
pip install scikit-learn  # Ja deve estar instalado
```

### RESULTADO ESPERADO:
- Deteccao mais robusta em imagens ruidosas
- Ignora artefatos que parecem linhas
- Melhor precisao na orientacao da agulha

============================================================
FASE 3: CALIBRACAO mm/pixel (Medidas Reais)
============================================================

### O QUE E:
Sistema para converter pixels em milimetros reais baseado na
profundidade configurada no ultrassom.

### PROBLEMA ATUAL:
Seu codigo usa `0.3 mm/pixel` FIXO, que so e correto para UMA
configuracao especifica de profundidade.

### SOLUCAO:
Criar sistema de calibracao que:
1. Permite definir profundidade total da imagem
2. Calcula automaticamente mm/pixel
3. Permite calibracao manual com pontos conhecidos

### CODIGO A ADICIONAR NO config.py:

```python
# ══════════════════════════════════════════════════════════
# CALIBRACAO DE ESCALA
# ══════════════════════════════════════════════════════════

# Profundidade total da imagem em mm (ajustar conforme probe)
ULTRASOUND_DEPTH_MM = 80  # 8cm de profundidade

# Modo de calibracao
# "auto" - calcula baseado na altura da imagem e profundidade
# "manual" - usa valor fixo de MM_PER_PIXEL
CALIBRATION_MODE = "auto"

# Valor manual (usado se CALIBRATION_MODE = "manual")
MM_PER_PIXEL = 0.3

# Offset vertical (pixels do topo que nao fazem parte da imagem US)
# Util se houver header/barra no topo da imagem
IMAGE_TOP_OFFSET = 0
```

### CODIGO A ADICIONAR NO ai_processor.py:

```python
# Na classe AIProcessor, adicionar:

def __init__(self):
    # ... codigo existente ...

    # ══════════════════════════════════════════════════════════
    # SISTEMA DE CALIBRACAO
    # ══════════════════════════════════════════════════════════
    self.calibration = {
        'mode': config.CALIBRATION_MODE,
        'depth_mm': config.ULTRASOUND_DEPTH_MM,
        'mm_per_pixel': config.MM_PER_PIXEL,
        'top_offset': config.IMAGE_TOP_OFFSET,
        'image_height': None,  # Sera definido no primeiro frame
    }

def _calibrate_scale(self, frame_height):
    """
    Calcula a escala mm/pixel baseado na configuracao.

    Args:
        frame_height: altura do frame em pixels

    Returns:
        mm_per_pixel: escala calculada
    """
    self.calibration['image_height'] = frame_height

    if self.calibration['mode'] == 'auto':
        # Altura util (descontando offset)
        useful_height = frame_height - self.calibration['top_offset']

        # Calcular escala
        mm_per_pixel = self.calibration['depth_mm'] / useful_height
        self.calibration['mm_per_pixel'] = mm_per_pixel

        return mm_per_pixel
    else:
        return self.calibration['mm_per_pixel']

def _pixel_to_mm(self, pixel_y):
    """
    Converte posicao Y em pixels para profundidade em mm.

    Args:
        pixel_y: posicao Y em pixels (do topo)

    Returns:
        depth_mm: profundidade em milimetros
    """
    adjusted_y = pixel_y - self.calibration['top_offset']
    return max(0, adjusted_y * self.calibration['mm_per_pixel'])

def _mm_to_pixel(self, depth_mm):
    """
    Converte profundidade em mm para posicao Y em pixels.

    Args:
        depth_mm: profundidade em milimetros

    Returns:
        pixel_y: posicao Y em pixels
    """
    return int(depth_mm / self.calibration['mm_per_pixel']) + self.calibration['top_offset']
```

### MODIFICAR _process_needle() PARA USAR CALIBRACAO:

```python
def _process_needle(self, frame):
    # No inicio da funcao, calibrar escala
    h, w = frame.shape[:2]
    self._calibrate_scale(h)

    # ... codigo de deteccao ...

    # SUBSTITUIR a linha:
    # needle_depth = needle_tip[1] * 0.3  # mm

    # POR:
    if needle_tip:
        needle_depth = self._pixel_to_mm(needle_tip[1])

    # ... resto do codigo ...
```

### MODIFICAR ESCALA VISUAL (linhas 831-839):

```python
# Escala de profundidade calibrada
scale_x = 25
mm_per_pixel = self.calibration['mm_per_pixel']
step_pixels = int(10 / mm_per_pixel)  # Marcacao a cada 10mm

for i in range(0, h, step_pixels):
    depth_mm = self._pixel_to_mm(i)
    cv2.line(output, (scale_x - 5, i), (scale_x + 5, i), (0, 150, 0), 1)

    # Mostrar valor a cada 20mm
    if int(depth_mm) % 20 == 0:
        cv2.putText(output, f"{int(depth_mm)}", (scale_x + 8, i + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 150, 0), 1, cv2.LINE_AA)

cv2.line(output, (scale_x, 0), (scale_x, h), (0, 100, 0), 1)

# Mostrar info de calibracao no painel
cv2.putText(output, f"Scale: {mm_per_pixel:.2f}mm/px", (panel_x + 10, panel_y + panel_h - 10),
           cv2.FONT_HERSHEY_DUPLEX, 0.25, (80, 100, 80), 1, cv2.LINE_AA)
```

### ADICIONAR ATALHO PARA AJUSTAR PROFUNDIDADE:

No main.py, adicionar handler para teclas [ e ]:
```python
elif key == ord('['):
    # Diminuir profundidade
    config.ULTRASOUND_DEPTH_MM = max(20, config.ULTRASOUND_DEPTH_MM - 10)
    print(f"Profundidade: {config.ULTRASOUND_DEPTH_MM}mm")

elif key == ord(']'):
    # Aumentar profundidade
    config.ULTRASOUND_DEPTH_MM = min(200, config.ULTRASOUND_DEPTH_MM + 10)
    print(f"Profundidade: {config.ULTRASOUND_DEPTH_MM}mm")
```

### RESULTADO ESPERADO:
- Profundidade exibida em mm REAIS
- Escala lateral mostra marcacoes corretas
- Ajustavel em tempo real com [ e ]

============================================================
FASE 4: INTEGRACAO VASST (Opcional - Precisao Extra)
============================================================

### O QUE E:
O VASST CNN e uma rede neural treinada especificamente para
localizar o centroide de agulhas em ultrassom.

### QUANDO USAR:
- Para agulhas OUT-OF-PLANE (perpendiculares ao probe)
- Quando YOLO/Hough nao consegue detectar bem
- Para maior precisao sub-pixel

### PASSOS PARA INTEGRAR:

1. CLONAR REPOSITORIO:
```bash
cd /Users/priscoleao/aplicativo-usg-final/models
git clone https://github.com/VASST/AECAI.CNN-US-Needle-Segmentation.git vasst_needle
```

2. ANALISAR ARQUITETURA (model.py do VASST):
- Entender estrutura da CNN
- Verificar formato de entrada esperado
- Adaptar para nosso pipeline

3. TREINAR/USAR MODELO:
- Se houver weights pre-treinados, usar diretamente
- Se nao, treinar com dados de agulha de US

4. INTEGRAR NO ai_processor.py:
```python
def _load_vasst_model(self):
    """Carrega modelo VASST para deteccao de centroide."""
    # Implementar baseado na arquitetura do VASST
    pass

def _vasst_detect_centroid(self, frame):
    """Usa VASST para detectar centroide da agulha."""
    # Implementar inferencia
    pass
```

### NOTA:
Esta fase e OPCIONAL e mais complexa. As fases 1-3 ja darao
uma melhoria significativa no tracking.

============================================================
FASE 5: VISUALIZACAO PREMIUM (Polish Final)
============================================================

### MELHORIAS VISUAIS A IMPLEMENTAR:

1. **Indicador de modo Kalman**:
```python
# Mostrar se esta usando deteccao ou predicao
if self.frames_without_detection > 0:
    cv2.putText(output, "PREDICTING", (panel_x + 25, status_y + 4),
               cv2.FONT_HERSHEY_DUPLEX, 0.35, (0, 200, 255), 1, cv2.LINE_AA)
```

2. **Trail com cores por confianca**:
```python
# Trail colorido baseado na confianca de cada ponto
for i, (point, conf) in enumerate(self.needle_trail_with_confidence):
    color = (0, int(255 * conf), int(255 * (1-conf)))  # Verde->Amarelo
    cv2.circle(output, point, 3, color, -1)
```

3. **Zona de incerteza**:
```python
# Desenhar elipse de incerteza ao redor da ponta
if self.kalman_initialized:
    # Extrair covariancia da posicao
    P = self.needle_kalman.P
    cov_xy = P[:2, :2]

    # Calcular eixos da elipse (2 sigma)
    eigenvalues, eigenvectors = np.linalg.eig(cov_xy)
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
    width = int(2 * 2 * np.sqrt(eigenvalues[0]))
    height = int(2 * 2 * np.sqrt(eigenvalues[1]))

    cv2.ellipse(output, needle_tip, (width, height), angle, 0, 360,
                (0, 255, 0, 128), 1)
```

4. **Barra de confianca do Kalman**:
```python
# Barra mostrando confianca da predicao
kalman_conf_y = panel_y + 155
cv2.putText(output, "TRACK", (panel_x + 10, kalman_conf_y),
           cv2.FONT_HERSHEY_DUPLEX, 0.3, (100, 120, 100), 1, cv2.LINE_AA)
cv2.rectangle(output, (panel_x + 50, kalman_conf_y - 8),
              (panel_x + 50 + bar_w, kalman_conf_y + 2), (40, 50, 40), -1)
fill_w = int(bar_w * kalman_confidence)
cv2.rectangle(output, (panel_x + 50, kalman_conf_y - 8),
              (panel_x + 50 + fill_w, kalman_conf_y + 2), (0, 255, 200), -1)
```

============================================================
ORDEM DE IMPLEMENTACAO RECOMENDADA
============================================================

    FASE 1: Kalman Filter
       │
       │    Resultado: Agulha estavel, sem tremor
       │    Tempo estimado: 2-3 horas
       │    Impacto: ALTO
       │
       ▼
    FASE 3: Calibracao mm/pixel
       │
       │    Resultado: Medidas reais em mm
       │    Tempo estimado: 1-2 horas
       │    Impacto: ALTO
       │
       ▼
    FASE 2: RANSAC
       │
       │    Resultado: Deteccao mais robusta
       │    Tempo estimado: 1-2 horas
       │    Impacto: MEDIO
       │
       ▼
    FASE 5: Visualizacao Premium
       │
       │    Resultado: Interface polida
       │    Tempo estimado: 1-2 horas
       │    Impacto: MEDIO
       │
       ▼
    FASE 4: VASST (opcional)

       Resultado: Precisao maxima
       Tempo estimado: 4-6 horas
       Impacto: BAIXO-MEDIO (ja tera bom resultado sem isso)

============================================================
CHECKLIST DE IMPLEMENTACAO (Atualizado: 2024-12-22)
============================================================

[X] FASE 1 - KALMAN FILTER ✅ COMPLETO
    [X] Instalar filterpy: pip install filterpy
    [X] Adicionar imports no ai_processor.py
    [X] Criar metodo _create_needle_kalman()
    [X] Criar metodo _update_needle_kalman()
    [X] Criar metodo _reset_needle_kalman()
    [X] Modificar _process_needle() para usar Kalman
    [X] Testar suavizacao
    [X] Testar predicao quando agulha some

[X] FASE 2 - RANSAC ✅ COMPLETO
    [X] Instalar scikit-learn: pip install scikit-learn
    [X] Criar metodo _ransac_fit_needle()
    [X] Criar metodo _detect_needle_enhanced()
    [X] Modificar fallback CV para usar RANSAC + CLAHE
    [X] Testar com imagens ruidosas

[X] FASE 3 - CALIBRACAO ✅ COMPLETO
    [X] Adicionar configs em config.py (ULTRASOUND_DEPTH_MM, CALIBRATION_MODE, etc)
    [X] Criar metodo _calibrate_scale()
    [X] Criar metodo _pixel_to_mm()
    [X] Criar metodo _mm_to_pixel()
    [X] Criar metodo set_ultrasound_depth()
    [X] Modificar calculo de profundidade
    [X] Modificar escala visual com marcacoes calibradas
    [X] Adicionar atalhos [ e ] no main.py
    [X] Testar com diferentes profundidades

[X] FASE 4 - VASST CNN ✅ COMPLETO
    [X] Clonar repositorio VASST (github.com/VASST/AECAI.CNN-US-Needle-Segmentation)
    [X] Analisar arquitetura (TensorFlow -> convertido para PyTorch)
    [X] Criar classe VASSTPyTorch (modelo CNN em PyTorch)
    [X] Criar _try_load_vasst() - carrega weights quando disponiveis
    [X] Criar _vasst_predict_centroid() - predicao com CNN
    [X] Criar _cv_refine_centroid() - fallback CV com Sobel + CLAHE
    [X] Criar _refine_needle_position() - metodo principal
    [X] Integrar no _process_needle() (Etapa 1.5)
    [X] Testar compilacao

[X] FASE 5 - VISUALIZACAO ✅ COMPLETO
    [X] Indicador de modo (TRACKING/PREDICTING/SEARCHING)
    [X] Trail colorido por confianca
    [X] Elipse de incerteza (baseada na covariancia do Kalman)
    [X] Barra de confianca do Kalman
    [X] Info de calibracao no painel
    [X] Atalhos de ajuste de profundidade no painel
    [X] Versao atualizada para v3.0 PREMIUM

============================================================
REFERENCIAS TECNICAS
============================================================

Ver arquivo REFERENCIAS.md secao "NEEDLE TRACKING / DETECTION"
para links de papers, repositorios e documentacao.

============================================================
FIM DO CRONOGRAMA NEEDLE PILOT
============================================================

====================================================================

====================================================================
