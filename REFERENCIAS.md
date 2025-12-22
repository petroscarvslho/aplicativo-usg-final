# Referencias - APLICATIVO USG FINAL

## Bibliotecas Python Utilizadas

### OpenCV (cv2)
- **Uso**: Captura de video, processamento de imagem, interface grafica
- **Documentacao**: https://docs.opencv.org/4.x/
- **GitHub**: https://github.com/opencv/opencv-python

### NumPy
- **Uso**: Manipulacao de arrays de pixels, operacoes matematicas
- **Documentacao**: https://numpy.org/doc/
- **GitHub**: https://github.com/numpy/numpy

### PyTorch
- **Uso**: Inferencia de modelos de IA (YOLO, U-Net)
- **Documentacao**: https://pytorch.org/docs/
- **GitHub**: https://github.com/pytorch/pytorch

### Ultralytics YOLO
- **Uso**: Deteccao de agulha em tempo real
- **Documentacao**: https://docs.ultralytics.com/
- **GitHub**: https://github.com/ultralytics/ultralytics

### Segmentation Models PyTorch (SMP)
- **Uso**: Segmentacao de nervos com U-Net
- **Documentacao**: https://smp.readthedocs.io/
- **GitHub**: https://github.com/qubvel/segmentation_models.pytorch

### PyObjC / Quartz
- **Uso**: Captura de janela nativa do macOS
- **Documentacao**: https://pyobjc.readthedocs.io/
- **GitHub**: https://github.com/ronaldoussoren/pyobjc

---

## Referencias de Design

### Apple Human Interface Guidelines
- **Uso**: Inspiracao para UI premium
- **Link**: https://developer.apple.com/design/human-interface-guidelines/

### Butterfly iQ3 App
- **Uso**: Referencia para cores e layout de app de ultrassom
- **Link**: https://www.butterflynetwork.com/

---

## Referencias Tecnicas - Captura de Tela macOS

### CGWindowList API
- **Documentacao**: https://developer.apple.com/documentation/coregraphics/quartz_window_services
- **Funcoes usadas**:
  - `CGWindowListCopyWindowInfo` - Lista janelas
  - `CGWindowListCreateImage` - Captura screenshot
  - `kCGWindowImageNominalResolution` - Evita scaling Retina

### Artigos Uteis
- Screen Capture on macOS: https://stackoverflow.com/questions/12978846/python-get-screen-pixel-value-in-os-x
- PyObjC Window Capture: https://gist.github.com/ronaldoussoren/pyobjc-examples

---

## Referencias de IA para Ultrassom

### YOLO para Deteccao Medica
- YOLOv8 Medical: https://github.com/ultralytics/ultralytics
- Needle Detection Papers: https://arxiv.org/search/?query=needle+detection+ultrasound

### U-Net para Segmentacao
- Paper Original: https://arxiv.org/abs/1505.04597
- U-Net Medical Imaging: https://github.com/milesial/Pytorch-UNet

### Datasets de Ultrassom
- Nerve Segmentation: https://www.kaggle.com/c/ultrasound-nerve-segmentation
- POCUS Dataset: https://github.com/jannisborn/covid19_pocus_ultrasound

---

## Projetos de Referencia no GitHub

### Ultrasound AI
- https://github.com/ultralytics/yolov5 (YOLO base)
- https://github.com/qubvel/segmentation_models.pytorch (U-Net)
- https://github.com/jannisborn/covid19_pocus_ultrasound (POCUS AI)

### Screen Capture Python
- https://github.com/BoboTiG/python-mss (Multi-platform)
- https://github.com/ponty/pyscreenshot (Cross-platform)

### OpenCV Real-time Processing
- https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV
- https://learnopencv.com/real-time-object-detection-using-yolov5/

---

## Otimizacoes de Performance

### OpenCV Performance
- https://docs.opencv.org/4.x/dc/d71/tutorial_py_optimization.html
- https://learnopencv.com/speeding-up-dlib-facial-landmark-detector/

### NumPy Performance
- https://numpy.org/doc/stable/user/quickstart.html#performance
- Vectorization: https://realpython.com/numpy-array-programming/

### Metal Performance Shaders (MPS) no Mac
- https://pytorch.org/docs/stable/notes/mps.html
- https://developer.apple.com/documentation/metalperformanceshaders

---

## Comunidades e Discussoes

### Reddit
- r/computervision
- r/MachineLearning
- r/ultrasound

### Reddit - Threads relevantes (handheld/POCUS)
- https://www.reddit.com/r/medicine/comments/jmamoo/handheld_us_butterfly_vs_clarius/
- https://www.reddit.com/r/medicine/comments/t8yi7h/best_handheld_ultrasound_for_use_in_hospital/
- https://www.reddit.com/r/Radiology/comments/1cnzuem/pocket_ultrasound_butterfly_etc/
- https://www.reddit.com/r/emergencymedicine/comments/14qz7sv/handheld_ultrasounds_for_emergency_physicians/
- https://www.reddit.com/r/Ultrasound/comments/yje7kd/which_portable_ultrasound_device_to_purchase/

### Stack Overflow Tags
- [opencv]
- [python]
- [macos]
- [pytorch]

---

## Apps pagos e sistemas comerciais (POCUS/handheld)

### Comparativos e reviews
- https://drsono.com/blogs/news/vscan-air-vs-clarius-vs-butterfly-iq-vs-drsono/
- https://journalfeed.org/article-a-day/2024/six-handheld-pocus-devices-compete-who-wins/
- https://ultrasoundfanatic.com/best-portable-ultrasound/
- https://www.gehealthcare.com/-/jssmedia/global/files/vscan-air/competetive-comparison-handheld-ultrasound-devices.pdf

### Plataformas e apps oficiais
- Butterfly iQ: https://www.butterflynetwork.com/ + https://apps.apple.com/us/app/butterfly-iq-ultrasound/id1183035589
- Clarius: https://clarius.com/ + https://apps.apple.com/us/app/clarius-ultrasound-app/id1140165095
- Philips Lumify: https://www.usa.philips.com/healthcare/sites/lumify-handheld-ultrasound/ + https://apps.apple.com/us/app/lumify-handheld-ultrasound/id6446034492
- GE Vscan Air: https://www.gehealthcare.com/products/ultrasound/handheld-ultrasound/vscan-air
- Mindray (smart bladder/handheld): https://www.mindray.com/

---

## Ferramentas AI comerciais (inspiracao de features)

### Clarius AI (T-Mode, Auto Preset, Auto Heart Rate, Auto Gain, Vessel Depth, MSK/OB/Prostate/Bladder)
- https://clarius.com/technology/clarius-ai/
- https://clarius.com/technology/bladder-ai/

### Butterfly Auto B-lines (contagem automatica de B-lines)
- https://www.butterflynetwork.com/press-releases/fda-clearance-butterfly-auto-b-lines-ai-tool

### GE Caption AI / AutoEF / Auto Bladder (Vscan Air)
- AutoEF whitepaper: https://www.gehealthcare.com/-/jssmedia/gehc/us/files/products/ultrasound/handheld-ultrasound/caption-ai-autoef-whitepaper_jb30471xx_v3.pdf
- Auto Bladder whitepaper: https://www.gehealthcare.com/-/jssmedia/gehc/us/files/products/ultrasound/handheld-ultrasound/cl/whitepaper-vscan-air-auto-bladder-volume-november-2025-jb35636xx.pdf

### Outros exemplos de AI (Auto B-lines / AutoEF)
- https://www.standardultrasound.com/ultrasound-tools/auto-b-lines
- https://www.standardultrasound.com/ultrasound-tools/ai-based-autoef
- Philips AI ultrasound tech: https://www.philips.ae/healthcare/technology/ai-ultrasound

---

## Papers e whitepapers utilitarios (FAST, B-lines, EF, Bladder)
- Deep learning free fluid (FAST/Morrison): https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2021.707437/full
- B-lines reliability: https://pubmed.ncbi.nlm.nih.gov/23269716/
- Lung US review: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7929643/
- Auto B-lines FDA clearance (Butterfly): https://www.butterflynetwork.com/press-releases/fda-clearance-butterfly-auto-b-lines-ai-tool
- Bladder US practice evidence: https://pubmed.ncbi.nlm.nih.gov/37721338/

---

## Datasets e benchmarks (ultrassom)
- EchoNet-Dynamic: https://echonet.github.io/dynamic/ + https://github.com/echonet/dynamic
- CAMUS: https://www.creatis.insa-lyon.fr/Challenge/camus/
- Kaggle Nerve Segmentation: https://www.kaggle.com/c/ultrasound-nerve-segmentation
- POCUS/COVID-19 US: https://github.com/jannisborn/covid19_ultrasound
- Open Ultrasound Projects (hardware + software): https://kelu124.github.io/openultrasoundprojects/
- MedSegBench (benchmark): https://medsegbench.github.io/

---

## Repositorios GitHub relevantes (IA para US)
- Needle tracking: https://github.com/DeeplearningBILAB/Ultrasound-guided-needle-tracking-with-deep-learning
- Needles in ultrasound: https://github.com/Misitlab/needles-in-ultrasound
- B-lines detection: https://github.com/RTLucassen/B-line_detection
- Ultrasound segmentation: https://github.com/luke-ck/ultrasound-segmentation
- UltraSAM (foundation model US): https://github.com/CAMMA-public/UltraSam
- POCUS COVID dataset code: https://github.com/jannisborn/covid19_ultrasound

---

## Ideias de features para deixar o app "premium"
- Auto Preset/Auto Gain (inspiracao Clarius AI) para reduzir ajustes manuais
- Auto Heart Rate e Vessel Depth AI (HUD automatico durante o exame)
- Auto B-lines (contagem automatica + score por janela)
- Auto Bladder Volume com validacao por duas vistas (sagital + transversal)
- AutoEF com tracking temporal do LV (clip curto + media robusta)
- Workflow guiado FAST (checklist + auto-sugestao de janela)
- T-Mode educacional (overlay anatomico para treinamento)
- Scan quality score (aviso de foco/ganho/ruido)
- Report generator (PDF/DICOM + export de medidas)

## Versoes Utilizadas

```
Python: 3.11+
OpenCV: 4.8+
NumPy: 1.24+
PyTorch: 2.0+
Ultralytics: 8.0+
segmentation-models-pytorch: 0.3+
pyobjc-framework-Quartz: 9.0+
```

---

## ═══════════════════════════════════════════════════════════════════════════════
## NEEDLE TRACKING / DETECTION - PESQUISA COMPLETA (Dezembro 2024)
## ═══════════════════════════════════════════════════════════════════════════════

### Estado da Arte em Needle Tracking (2024-2025)

---

## 1. PAPERS ACADEMICOS - STATE OF THE ART

### MambaXCTrack (2024) - MELHOR PERFORMANCE ATUAL
- **Paper**: "MambaXCTrack: Mamba-based Tracker with SSM Cross-correlation and Motion Prompt for Ultrasound Needle Tracking"
- **Autores**: Kunming University of Science and Technology
- **Performance**: 34.9 FPS, erro medio 0.34±0.55 mm
- **Arquitetura**: ResNet-50 backbone + 4 Mamba heads + SSMX-Corr
- **Dataset**: 87,330 frames de 108 procedimentos
- **Status codigo**: NAO DISPONIVEL PUBLICAMENTE (em desenvolvimento)
- **arXiv**: https://arxiv.org/html/2411.08395
- **IEEE**: https://ieeexplore.ieee.org/abstract/document/10950072/
- **Nota**: Primeiro uso de Mamba para needle tracking em US

### Motion-aware Needle Segmentation (CVPR 2024 Workshop)
- **Paper**: "Motion-aware Needle Segmentation in Ultrasound Images"
- **Autores**: Carnegie Mellon University (Raghavv Goel et al.)
- **Resultados**: 15% reducao em erro de ponta, 8% reducao em erro de comprimento
- **Tecnica**: CNN + Kalman Filter inspirado
- **PDF**: https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/papers/Goel_Motion-aware_Needle_Segmentation_in_Ultrasound_Images_CVPRW_2024_paper.pdf
- **arXiv**: https://arxiv.org/abs/2312.01239
- **Status codigo**: NAO DISPONIVEL

### UIU-Net com Ground Truth Photoacustico (2023)
- **Paper**: "Ultrasound-guided needle tracking with deep learning: A novel approach with photoacoustic ground truth"
- **Autores**: Nanyang Technological University, Singapore
- **Performance**: MHD ~3.73, targeting error ~2.03
- **Inovacao**: Usa PA imaging como ground truth (sem anotacao manual)
- **ScienceDirect**: https://www.sciencedirect.com/science/article/pii/S2213597923001283
- **PMC**: https://pmc.ncbi.nlm.nih.gov/articles/PMC10761306/

### Needle Tracking em 3D US com Low Resolution (2024)
- **Paper**: "Needle tracking in low-resolution ultrasound volumes using deep learning"
- **Autores**: Hamburg University of Technology
- **PMC**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11442564/

### CNN para Deteccao de Agulha em Tempo Real (2018)
- **Paper**: "Convolution neural networks for real-time needle detection and localization in 2D ultrasound"
- **Resultados**: IoU 0.986, F1 0.768, RMSE angulo 3.73°
- **Dataset**: 619 imagens 2D de 91 casos de cancer de mama
- **ResearchGate**: https://www.researchgate.net/publication/323595552

### Deteccao Robusta em 3D com CNN Ortogonal (2018)
- **Paper**: "Robust and semantic needle detection in 3D ultrasound using orthogonal-plane convolutional neural networks"
- **PMC**: https://pmc.ncbi.nlm.nih.gov/articles/PMC6132402/

---

## 2. REPOSITORIOS GITHUB - CODIGO DISPONIVEL

### ⭐⭐⭐ TOP PICKS - MAIS RELEVANTES PARA O PROJETO ⭐⭐⭐

### DeeplearningBILAB/Ultrasound-guided-needle-tracking-with-deep-learning ⭐⭐⭐ ALTAMENTE RECOMENDADO
- **Descricao**: UIU-Net para needle tracking com deep learning
- **Inovacao**: Usa imagem photoacustica como ground truth (sem anotacao manual!)
- **Resultados**: MHD ~3.73, targeting error ~2.03
- **Dataset**: Disponivel no Open Science Framework (OSF)
- **GitHub**: https://github.com/DeeplearningBILAB/Ultrasound-guided-needle-tracking-with-deep-learning
- **Paper**: https://www.sciencedirect.com/science/article/pii/S2213597923001283
- **Nota**: Codigo fonte do UIU-Net disponivel, pronto para testar!

### mobarakol/US_Needle_Segmentation ⭐⭐⭐ ALTAMENTE RECOMENDADO
- **Descricao**: Segmentacao de agulha + PREDICAO DE TRAJETORIA
- **Arquitetura**: Excitation Network
- **Publicacao**: Int J CARS, 2020, Springer
- **GitHub**: https://github.com/mobarakol/US_Needle_Segmentation
- **Nota**: Tem logica de predicao de caminho - pode melhorar tracking!

### VASST/AECAI.CNN-US-Needle-Segmentation ⭐⭐ RECOMENDADO
- **Descricao**: CNN para segmentacao de agulha out-of-plane em ultrassom
- **Arquivos**: model.py, prediction.py, segmentation.py, generate_data.py
- **Resultados**: RMSE 0.62mm (axial), 0.74mm (lateral)
- **Licenca**: MIT
- **GitHub**: https://github.com/VASST/AECAI.CNN-US-Needle-Segmentation
- **Paper associado**: "Deep learning approach for automatic out-of-plane needle localisation"

### jocicmarko/ultrasound-nerve-segmentation
- **Descricao**: Tutorial U-Net para segmentacao de nervos em US (Kaggle)
- **Framework**: Keras/TensorFlow
- **Score**: ~0.57 no leaderboard
- **GitHub**: https://github.com/jocicmarko/ultrasound-nerve-segmentation

### OverFlow7/Ultrasound-Nerve-Segmentation
- **Descricao**: U-Net em TensorFlow para segmentacao de nervos
- **Score**: Dice 0.70115, rank 38/973
- **GitHub**: https://github.com/OverFlow7/Ultrasound-Nerve-Segmentation

### ardamavi/Ultrasound-Nerve-Segmentation
- **Descricao**: Segmentacao de nervos com deep learning
- **GitHub**: https://github.com/ardamavi/Ultrasound-Nerve-Segmentation

### RootLeo00/ultrasound-segmentation
- **Descricao**: Segmentacao semantica multiclasse de imagens US
- **Acuracia**: >90% em treino e teste
- **GitHub**: https://github.com/RootLeo00/ultrasound-segmentation

### tqxli/breast_ultrasound_lesion_segmentation_PyTorch
- **Descricao**: Varios modelos PyTorch para segmentacao em US de mama
- **Dataset**: BUSI (780 imagens)
- **GitHub**: https://github.com/tqxli/breast_ultrasound_lesion_segmentation_PyTorch

### kelu124/openultrasoundprojects ⭐ LISTA CURADA
- **Descricao**: Lista curada de projetos open source de ultrassom
- **GitHub**: https://github.com/kelu124/openultrasoundprojects
- **Website**: https://kelu124.github.io/openultrasoundprojects/

### kelu124/echomods
- **Descricao**: Kit DIY de ultrassom open source (<$500)
- **GitHub**: https://github.com/kelu124/echomods

### joshrbaxter/ultrasound_tracking
- **Descricao**: Tracking em ultrassom
- **GitHub**: https://github.com/joshrbaxter/ultrasound_tracking

### falcondai/py-ransac
- **Descricao**: Implementacao Python de RANSAC para line/plane fitting
- **GitHub**: https://github.com/falcondai/py-ransac

### ═══════════════════════════════════════════════════════════════
### RECURSOS ADICIONAIS VERIFICADOS (Dezembro 2024)
### ═══════════════════════════════════════════════════════════════

### MisitLab/needles-in-ultrasound ⭐ DATASET E FERRAMENTAS
- **Descricao**: Quantificacao de visibilidade e echogenicity de agulhas em US
- **Conteudo**: Dados, codigo MATLAB/Arduino, modelos CAD 3D
- **Uso**: Criar dataset proprio ou entender fatores de deteccao
- **GitHub**: https://github.com/MisitLab/needles-in-ultrasound
- **Paper**: "A Methodical Quantification of Needle Visibility and Echogenicity in Ultrasound Images"
- **PDF**: https://sintef.brage.unit.no/sintef-xmlui/bitstream/handle/11250/2628720/

### adamtupper/ultrabench ⭐ BENCHMARK PADRONIZADO
- **Descricao**: Benchmark de 14 tarefas de classificacao/segmentacao em US
- **Datasets**: 10 datasets publicos, 11 regioes do corpo
- **Uso**: Comparar modelos, medir impacto de melhorias
- **GitHub**: https://github.com/adamtupper/ultrabench
- **Instalacao**: pip install ultrabench
- **Paper**: "Revisiting Data Augmentation for Ultrasound Images"

### SlicerIGT/aigt ⭐ FRAMEWORK COMPLETO
- **Descricao**: Modulos DL para procedimentos guiados por imagem
- **Funcoes**: Gravacao, anotacao, treino, deploy em tempo real
- **Modulo principal**: SegmentationUNet para US em tempo real
- **Framework**: TensorFlow, com suporte PyTorch via MONAI
- **GitHub**: https://github.com/SlicerIGT/aigt
- **Uso**: Prototipar plugins, workflow completo

### CMU Transformer Needle Tracking (Projeto de Pesquisa)
- **Descricao**: Primeiro uso de Transformer para needle tracking em US
- **Arquiteturas**: U-Net, U-Transformer, Axial-U-Trans
- **Link**: https://sites.google.com/andrew.cmu.edu/16824-project/home
- **Nota**: Projeto academico, pode ter codigo disponivel

### VibNet - Vibration-Boosted Needle Detection
- **Descricao**: Framework DL que usa informacao de vibracao para deteccao
- **Inovacao**: Primeiro end-to-end que usa vibracao
- **arXiv**: https://arxiv.org/html/2403.14523
- **Nota**: Abordagem unica, pode inspirar melhorias

---

## 2.5 PAPERS MAMBA-BASED (Estado da Arte 2024-2025)

### MrTrack (MICCAI 2025) ⭐ NOVISSIMO
- **Paper**: "MrTrack: Register Mamba for Needle Tracking with Rapid Reciprocating Motion"
- **Foco**: Tracking durante biopsia por aspiracao (movimento rapido)
- **Inovacao**: Mamba-based register extractor + retriever
- **Resultado**: State-of-the-art em biopsias roboticas e manuais
- **arXiv**: https://arxiv.org/abs/2505.09450
- **MICCAI**: https://papers.miccai.org/miccai-2025/paper/0251_paper.pdf
- **Codigo**: NAO DISPONIVEL (ainda)

### MambaXCTrack (IEEE 2024) ⭐ MELHOR PERFORMANCE GERAL
- **Paper**: "MambaXCTrack: Mamba-based Tracker with SSM Cross-correlation"
- **Performance**: 34.9 FPS, erro 0.34mm
- **arXiv**: https://arxiv.org/abs/2411.08395
- **Codigo**: NAO DISPONIVEL (em desenvolvimento)

---

## 3. TECNOLOGIAS COMERCIAIS - REFERENCIA

### Butterfly iQ3 - NeedleViz
- **Nome**: Needle Viz (in-plane)
- **Como funciona**: Realce visual da agulha em azul brilhante
- **Angulos**: 20°, 30°, 40°
- **Modos**: MSK, MSK-Soft Tissue, Nerve, Vascular Access
- **Biplane**: Suporte integrado
- **Documentacao**: https://support.butterflynetwork.com/hc/en-us/articles/16910213468315-Using-the-Needle-Viz-Tool
- **Info**: https://www.butterflynetwork.com/iq3-anesthesiology-perioperative-care

### Clarius - Needle Enhance 8.0
- **Nome**: Needle Enhance
- **Como funciona**: Multi-angulo + AI para deteccao automatica de lado/angulo
- **Frame rate**: Ate 30 FPS
- **Tecnologia**: Patent-pending, combina AI com multiplas imagens
- **Documentacao**: https://clarius.com/scanners/needle-enhance/
- **Blog tecnico**: https://clarius.com/ca/blog/clarius-technical-spotlight-different-way-visualizing-needle-tissue/

### Mindray - eSpecialNavi
- **Tipo**: Sistema de navegacao de agulha estilo GPS

### GE - Needle Assist
- **Tipo**: Assistente de visualizacao de agulha

---

## 4. TECNICAS DE TRACKING - ALGORITMOS

### Kalman Filter para Needle Tracking
- **Paper IEEE**: "Real-time Needle Tip Localization in 2D Ultrasound Images using Kalman Filter"
- **Link**: https://ieeexplore.ieee.org/document/8868799/

### Kalman Filter Adaptativo
- **Paper IEEE**: "Needle Tip Tracking in 2D Ultrasound Based on Improved Compressive Tracking and Adaptive Kalman Filter"
- **Link**: https://ieeexplore.ieee.org/document/9366991/

### RANSAC + Kalman em 3D US
- **Paper**: "Automatic needle detection and tracking in 3D ultrasound using an ROI-based RANSAC and Kalman method"
- **PubMed**: https://pubmed.ncbi.nlm.nih.gov/24081726/

### Neural Network + Kalman Filter Hibrido
- **Paper**: "Neural Network Kalman Filtering for 3-D Object Tracking From Linear Array Ultrasound Data"
- **PubMed**: https://pubmed.ncbi.nlm.nih.gov/35324438/

### Multi-scale RANSAC
- **Paper**: "Multi-scale RANSAC algorithm for needle localization in 3D ultrasound guided puncture surgery"
- **IEEE**: https://ieeexplore.ieee.org/document/7591631/

### Hough Transform (OpenCV)
- **Documentacao**: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
- **Tutorial**: https://learnopencv.com/hough-transform-with-opencv-c-python/

---

## 5. DATASETS DISPONIVEIS

### Kaggle - Ultrasound Nerve Segmentation
- **Tamanho**: ~5,600 imagens
- **Tipo**: Segmentacao de plexo braquial
- **Link**: https://www.kaggle.com/c/ultrasound-nerve-segmentation

### BUSI - Breast Ultrasound Images Dataset
- **Tamanho**: 780 imagens PNG
- **Classes**: Normal, Benigno, Maligno
- **Pacientes**: 600 mulheres

### Dataset de Needle Tracking (MambaXCTrack)
- **Tamanho**: 87,330 frames de 108 procedimentos
- **Resolucao**: 492×856 pixels
- **Condicoes**: 3 angulos (0°, 30°, 60°), 3 velocidades
- **Nota**: Dataset proprietario, nao disponivel publicamente

### POCUS COVID-19
- **GitHub**: https://github.com/jannisborn/covid19_pocus_ultrasound

---

## 6. FRAMEWORKS E FERRAMENTAS

### PLUS (Open-source Toolkit for US-Guided Intervention)
- **Paper PMC**: https://pmc.ncbi.nlm.nih.gov/articles/PMC4437531/
- **Uso**: Toolkit completo para sistemas de intervencao guiada por US

### 3D Slicer - SlicerAIGT Extension
- **Modulo**: Segmentation U-Net (real-time)
- **Suporte**: TensorFlow, PyTorch (via MONAI)

### pyransac3d (PyPI)
- **Uso**: RANSAC para fitting de primitivas 3D (cilindros, linhas)
- **Link**: https://pypi.org/project/pyransac3d/

### scikit-learn RANSACRegressor
- **Uso**: RANSAC para regressao linear robusta
- **Docs**: https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html

---

## 7. PROGRAMAS DE PESQUISA

### DARPA POCUS AI Program
- **Objetivo**: AI para ultrassom portatil com dados limitados (15-30 imagens)
- **Parceiro**: Carnegie Mellon University
- **Aplicacao**: Validacao de posicao de agulha em nerve blocks
- **Link**: https://www.cmu.edu/news/stories/archives/2021/june/ai-portable-ultrasound.html

---

## 8. REVIEWS E SURVEYS

### Deep Learning in Medical Ultrasound Analysis: A Review
- **ScienceDirect**: https://www.sciencedirect.com/science/article/pii/S2095809918301887

### Machine Learning for Medical Ultrasound: Status, Methods, and Future
- **PMC**: https://pmc.ncbi.nlm.nih.gov/articles/PMC5886811/

### AI Applications for POCUS in Low-Resource Settings
- **PMC**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11312308/

### Survey on Deep Learning in Medical Ultrasound Imaging (2024)
- **Frontiers**: https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2024.1398393/full

### Deep Learning-Based Medical US Image/Video Segmentation Methods (2025)
- **MDPI**: https://www.mdpi.com/1424-8220/25/8/2361

---

## 9. RECOMENDACOES PARA IMPLEMENTACAO

### Abordagem Recomendada para Needle Pilot:

1. **Deteccao Base**:
   - YOLO (rapido, bounding box) OU
   - U-Net (segmentacao precisa)

2. **Smoothing Temporal**:
   - Kalman Filter para reducao de jitter
   - Implementar em Python com filterpy ou manual

3. **Fallback CV**:
   - HoughLinesP para deteccao de linhas
   - RANSAC para fitting robusto
   - Canny edge detection

4. **Metricas a calcular**:
   - Angulo de insercao
   - Profundidade da ponta
   - Trajetoria projetada
   - Confianca da deteccao

5. **Calibracao**:
   - Sistema para calibrar mm/pixel
   - Usar escala do probe ou referencia conhecida

---

## Licencas

- OpenCV: Apache 2.0
- NumPy: BSD
- PyTorch: BSD
- Ultralytics YOLO: AGPL-3.0
- SMP: MIT
- PyObjC: MIT
