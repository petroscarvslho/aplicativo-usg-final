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

### Stack Overflow Tags
- [opencv]
- [python]
- [macos]
- [pytorch]

---

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

## Licencas

- OpenCV: Apache 2.0
- NumPy: BSD
- PyTorch: BSD
- Ultralytics YOLO: AGPL-3.0
- SMP: MIT
- PyObjC: MIT
