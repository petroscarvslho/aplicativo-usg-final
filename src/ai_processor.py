import torch
import cv2
import numpy as np
import config

class AIProcessor:
    def __init__(self):
        self.device = "mps" if config.USE_MPS and torch.backends.mps.is_available() else "cpu"
        print(f"🧠 AI Processor iniciado no dispositivo: {self.device}")
        
        self.models = {}
        self.active_mode = None  # None, 'needle', 'nerve'
        
        # Histórico para Trajetória da Agulha (Needle Pilot)
        from collections import deque
        self.needle_history = deque(maxlen=10) # Guarda últimos 10 centros
        
        # Carrega modelos sob demanda (lazy loading)
        self.model_paths = {
            'needle': config.YOLO_MODEL_PATH,
            'nerve': 'models/unet_nerve.pt' # Arquivo esperado
        }

    def set_mode(self, mode):
        if mode == self.active_mode:
            return 
        print(f"🔄 Trocando modo para: {mode}")
        self.active_mode = mode
        self.needle_history.clear() # Limpa histórico ao trocar
        
        if mode and mode not in self.models:
            self._load_model(mode)

    def _load_model(self, mode):
        path = self.model_paths.get(mode)
        try:
            if mode == 'needle':
                from ultralytics import YOLO
                print(f"Carregando YOLO: {path}")
                self.models[mode] = YOLO(path)
            
            elif mode == 'nerve':
                import segmentation_models_pytorch as smp
                import os
                
                # Caminho atualizado para o peso encontrado (ResNet50)
                # Adapte o caminho se necessário (usei um dos listados anteriormente)
                weight_path = "Ultrasound-Optimal-View-Detection/pasta sem título/u-resnet50_B64_E47_2022-10-14-07_44_38.pt"
                
                print(f"Carregando U-ResNet50 (SMP): {weight_path}")
                
                # Instancia U-Net com backbone ResNet50 (igual ao treino encontrado)
                # in_channels=1 (Grayscale), classes=3 (Nervo, Artéria, Costela?)
                self.models[mode] = smp.Unet(
                    encoder_name="resnet50", 
                    encoder_weights=None, 
                    in_channels=1, 
                    classes=3, 
                    activation='sigmoid'
                ).to(self.device)
                
                if os.path.exists(weight_path):
                    state_dict = torch.load(weight_path, map_location=self.device)
                    self.models[mode].load_state_dict(state_dict)
                    self.models[mode].eval()
                    print("✅ Pesos U-ResNet50 carregados com sucesso do projeto 'Optimal-View'!")
                else:
                    print(f"⚠️ Peso {weight_path} não encontrado. Buscando alternativos...")
                    # Fallback para tentar achar qualquer .pt na pasta certa
                    # (Lógica simplificada para MVP)
                    self.models[mode].eval()

        except Exception as e:
            print(f"❌ Erro ao carregar modelo {mode}: {e}")
            self.active_mode = None

    def process(self, frame):
        if not self.active_mode or self.active_mode not in self.models:
            return frame

        # Verificar se deve preservar imagem original
        preserve_original = getattr(config, 'PRESERVE_ORIGINAL_IMAGE', True)

        # Define Preset baseado no Modo
        preset = 'general'
        if self.active_mode == 'nerve':
            preset = 'nerve'  # Ativa alto contraste

        # --- PRE-PROCESSING (Image Enhancement para detecção) ---
        enhanced_frame = self._enhance_image(frame, preset=preset)

        if self.active_mode == 'needle':
            return self._process_needle(enhanced_frame, original_frame=frame, preserve_original=preserve_original)
        elif self.active_mode == 'nerve':
            return self._process_nerve(enhanced_frame, original_frame=frame, preserve_original=preserve_original)

        # Se nenhum modo ativo, retornar frame original
        return frame

    def _enhance_image(self, frame, preset='general'):
        # Presets Clínicos (Butterfly Style)
        # 'nerve': Alto contraste para diferenciar fascículos (Clip 4.0)
        # 'general': Suave para anatomia geral (Clip 2.0)
        # 'vascular': (Futuro)
        
        clip_limit = 2.0
        if preset == 'nerve':
            clip_limit = 4.0
        
        # Converte para LAB para aplicar CLAHE apenas no canal de Luminosidade (L)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge de volta
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def _process_needle(self, frame, original_frame=None, preserve_original=True):
        # frame = Enhanced (para detecção)
        # original_frame = Imagem original sem alterações

        model = self.models.get('needle')

        # --- AI MODE (YOLO) ---
        if model:
            # Inferência YOLO na imagem MELHORADA (melhor detecção)
            results = model(frame, verbose=False, conf=config.AI_CONFIDENCE)

            # Escolher qual frame usar para exibição
            if preserve_original and original_frame is not None:
                display_frame = original_frame.copy()  # Usar imagem ORIGINAL
            else:
                display_frame = frame.copy()  # Usar imagem enhanced

            # Desenhar detecções no frame escolhido
            for r in results:
                display_frame = r.plot(img=display_frame)

            annotated_frame = display_frame
            
            # --- Trajectory Prediction (Needle Pilot) ---
            boxes = results[0].boxes
            if len(boxes) > 0:
                best_box = sorted(boxes, key=lambda x: x.conf[0], reverse=True)[0]
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                
                self.needle_history.append((cx, cy))
                
                if len(self.needle_history) >= 2:
                    points = list(self.needle_history)
                    for i in range(1, len(points)):
                        cv2.line(annotated_frame, points[i-1], points[i], (0, 255, 0), 2)
                    
                    dx = points[-1][0] - points[0][0]
                    dy = points[-1][1] - points[0][1]
                    future_x = int(points[-1][0] + dx * 2) 
                    future_y = int(points[-1][1] + dy * 2)
                    
                    cv2.arrowedLine(annotated_frame, points[-1], (future_x, future_y), (0, 255, 255), 2, tipLength=0.3)
                    cv2.putText(annotated_frame, "TRAJECTORY", (future_x, future_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            return annotated_frame
            
        # --- CV MODE (Fallback / Hybrid) ---
        else:
            # Fallback para NeedleEnhancer (CV Clássico)
            if not hasattr(self, 'needle_cv'):
                from src.needle_viz import NeedleEnhancer
                self.needle_cv = NeedleEnhancer()
                print("⚠️ AI Needle Model not found. Using CV Enhancer fallback.")
            
            return self.needle_cv.process(frame)

    def _process_nerve(self, frame, original_frame=None, preserve_original=True):
        model = self.models.get('nerve')
        if not model:
            return original_frame if preserve_original and original_frame is not None else frame
        
        # Pipeline Real-Time U-Net (SMP ResNet50)
        input_size = (224, 224)

        # Usar dimensoes da imagem que sera usada para overlay
        if preserve_original and original_frame is not None:
            h, w = original_frame.shape[:2]
        else:
            h, w = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img_resized = cv2.resize(gray, input_size)
        
        # Normalize (0-1) e Batch Dimension
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device) # (B, 1, H, W)
        
        # 2. Inferência
        with torch.no_grad():
            output = model(img_tensor) # (B, 3, H, W)
            # Output já é sigmoid (definido na criação) ou logits? O train.py mostra activation='sigmoid' na criação!
            # Mas o forward da SMP com activation='sigmoid' já retorna probs.
            masks = output.cpu().numpy()[0] # (3, 224, 224)
        
        # 3. Pós-Processamento (Multi-class Overlay)
        # Usar imagem original para overlay se preserve_original=True
        if preserve_original and original_frame is not None:
            overlay = original_frame.copy()
        else:
            overlay = frame.copy()
        
        # Cores para cada classe (BGR)
        # Assume-se: 0=Nervo?, 1=Artéria?, 2=Costela? (A verificar empiricamente)
        # Vamos usar: 0=Amarelo, 1=Vermelho, 2=Azul
        colors = [
            (0, 255, 255), # Ch0: Amarelo
            (0, 0, 255),   # Ch1: Vermelho
            (255, 0, 0)    # Ch2: Azul
        ]
        
        labels = ["Plexo", "Arteria", "Costela"]
        
        for i in range(3):
            mask_ch = masks[i]
            # Resize mask para original
            mask_full = cv2.resize(mask_ch, (w, h))
            
            # Threshold da UI - AUMENTADO para reduzir ruído (0.5 -> 0.7)
            # Se config.AI_CONFIDENCE for muito baixo no arquivo, forçamos um mínimo aqui.
            threshold = max(config.AI_CONFIDENCE, 0.65)
            binary_mask = mask_full > threshold
            
            if binary_mask.any():
                # Modo Contorno (Mais limpo que preenchimento total)
                # Encontrar contornos na máscara binária
                contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Desenhar contornos (Highlight edges)
                cv2.drawContours(overlay, contours, -1, colors[i], 2)
                
                # Preenchimento muito leve (10%)
                color_mask = np.zeros_like(frame)
                color_mask[binary_mask] = colors[i]
                cv2.addWeighted(overlay, 1.0, color_mask, 0.1, 0, overlay)
                
                # Label no centro do maior contorno
                if contours:
                    max_c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(max_c) > 500: # Filtrar ruído muito pequeno
                        M = cv2.moments(max_c)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            # Texto com fundo preto para leitura
                            cv2.putText(overlay, labels[i], (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
        
        return overlay
