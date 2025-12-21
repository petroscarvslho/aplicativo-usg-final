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

### Ultima atualizacao: 2025-12-21 (Sessao 6)

### O que foi feito:
- App 100% Python/OpenCV funcionando
- Interface premium com header, sidebar, footer
- **Sistema exclusivo de modos: B-MODE, IA ZONE, IA FULL**
- **Sistema de ROI profissional** (estilo Photoshop/Figma)
- **Fullscreen nativo** (cv2.setWindowProperty)
- 10 plugins de IA (NEEDLE ate BLADDER)
- Captura de tela do iPhone via AIRPLAY (pixel-perfect)
- Deteccao de agulha com YOLO + trajetoria
- Segmentacao de nervo com U-Net ResNet34
- Screenshot PNG lossless
- Gravacao com opcoes de qualidade
- Modo Biplane (lado a lado)
- Zoom/Pan funcionando
- Modal de ajuda com atalhos
- Instrucoes contextuais por modo
- TODOS os modelos AI de qualidade maxima criados

### Sistema de ROI (Sessao 6):
- **Visual premium**: overlay escuro, marching ants, handles, crosshair, grid tercos
- **Controles**: ENTER confirma, ESC cancela
- **Barra de status** inferior com botoes estilizados
- **Dimensoes** exibidas em tempo real
- **Metodo**: `_desenhar_roi_selection()` em main.py (linhas 731-923)

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
- ✅ Sistema ROI profissional
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

### 1. Redimensionar ROI pelos handles
- Arrastar os cantos brancos para ajustar tamanho
- Detectar clique nos handles (distancia < 15px)
- Modo de arraste: `roi_drag_handle` = 'tl', 'tr', 'bl', 'br'
- Atualizar `process_roi` durante arraste

### 2. Mover ROI arrastando o centro
- Clicar dentro da ROI (nao nos handles) permite mover
- `roi_drag_handle` = 'move'
- Manter tamanho, apenas mudar posicao

### 3. Presets de ROI (botoes rapidos)
- Adicionar botoes na barra inferior durante selecao:
  - "Centro 50%" - ROI centralizada com 50% da tela
  - "Centro 75%" - ROI centralizada com 75% da tela
  - "Tela toda" - ROI cobrindo 100%
  - "Quadrado central" - ROI quadrada no centro

### 4. Visual premium para cada plugin de IA
- Cada modo ter visual unico e profissional
- NEEDLE: linha de trajetoria animada, angulo exibido
- NERVE: contorno com gradiente, label anatomico
- CARDIAC: overlay de camaras coloridas, EF% grande
- FAST: areas de liquido destacadas com cor
- Etc para cada um dos 10 modos

### 5. Animacoes de transicao
- Fade suave ao trocar de modo (200ms)
- Transicao na sidebar ao selecionar
- Pulse nos botoes ativos

### 6. Outras melhorias pendentes
- Treinar YOLO especifico para agulhas de ultrassom
- Fine-tune modelos com dados reais
- Implementar voice controls
- Geracao de relatorios PDF
- Historico de medicoes
- Export de dados para PACS

====================================================================

====================================================================
