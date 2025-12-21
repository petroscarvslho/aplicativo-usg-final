"""
Video Clip Recorder - Gravação de clips de ultrassom
"""
import cv2
import os
import threading
import numpy as np
from datetime import datetime
from collections import deque

class ClipRecorder:
    """Gravação de clips de vídeo com buffer circular"""
    
    def __init__(self, output_dir=None, buffer_seconds=5, fps=25):
        if output_dir is None:
            output_dir = os.path.expanduser("~/Documents/POCUS_Clips")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.fps = fps
        self.buffer_size = buffer_seconds * fps
        self.frame_buffer = deque(maxlen=self.buffer_size)
        
        self.is_recording = False
        self.recording_frames = []
        self.recording_start_time = None
        
        # Settings
        self.max_duration_seconds = 30
        self.codec = 'mp4v'  # or 'avc1' for H.264
        
        # Stats
        self.last_clip_path = None
        self.total_clips_recorded = 0
    
    def add_frame(self, frame):
        """Adiciona frame ao buffer circular"""
        if frame is not None:
            # Store copy to avoid reference issues
            self.frame_buffer.append(frame.copy())
            
            # If recording, also store in recording buffer
            if self.is_recording:
                elapsed = (datetime.now() - self.recording_start_time).total_seconds()
                if elapsed < self.max_duration_seconds:
                    self.recording_frames.append(frame.copy())
                else:
                    # Auto-stop after max duration
                    self.stop_recording()
    
    def start_recording(self):
        """Inicia gravação"""
        if self.is_recording:
            return False
        
        self.is_recording = True
        self.recording_frames = list(self.frame_buffer)  # Include buffer
        self.recording_start_time = datetime.now()
        print("🔴 Gravação iniciada")
        return True
    
    def stop_recording(self):
        """Para gravação e salva clip"""
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        if len(self.recording_frames) == 0:
            print("⚠️ Nenhum frame gravado")
            return None
        
        # Save in background thread
        frames = self.recording_frames.copy()
        self.recording_frames = []
        
        thread = threading.Thread(target=self._save_clip, args=(frames,))
        thread.start()
        
        return "saving"
    
    def save_instant_clip(self, duration_seconds=5):
        """Salva clip dos últimos N segundos do buffer"""
        frames_to_save = int(duration_seconds * self.fps)
        frames = list(self.frame_buffer)[-frames_to_save:]
        
        if len(frames) == 0:
            print("⚠️ Buffer vazio")
            return None
        
        thread = threading.Thread(target=self._save_clip, args=(frames,))
        thread.start()
        
        return "saving"
    
    def _save_clip(self, frames):
        """Salva frames como vídeo (thread interna)"""
        if len(frames) == 0:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"POCUS_Clip_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        out = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))
        
        for frame in frames:
            # Ensure correct size
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
        
        out.release()
        
        self.last_clip_path = filepath
        self.total_clips_recorded += 1
        
        print(f"✅ Clip salvo: {filepath} ({len(frames)} frames, {len(frames)/self.fps:.1f}s)")
        return filepath
    
    def get_recording_status(self):
        """Retorna status da gravação"""
        if self.is_recording:
            elapsed = (datetime.now() - self.recording_start_time).total_seconds()
            return {
                "recording": True,
                "elapsed_seconds": elapsed,
                "frames_recorded": len(self.recording_frames),
                "max_duration": self.max_duration_seconds
            }
        return {
            "recording": False,
            "buffer_size": len(self.frame_buffer),
            "buffer_seconds": len(self.frame_buffer) / self.fps
        }


class LoopRecorder:
    """Gravação em loop contínuo (útil para revisão)"""
    
    def __init__(self, loop_seconds=60, fps=25):
        self.fps = fps
        self.loop_size = loop_seconds * fps
        self.buffer = deque(maxlen=self.loop_size)
        self.enabled = True
    
    def add_frame(self, frame):
        """Adiciona frame ao loop"""
        if self.enabled and frame is not None:
            self.buffer.append(frame.copy())
    
    def get_frame_at(self, seconds_ago):
        """Retorna frame de N segundos atrás"""
        if len(self.buffer) == 0:
            return None
        
        frames_ago = int(seconds_ago * self.fps)
        if frames_ago >= len(self.buffer):
            return self.buffer[0]
        
        return self.buffer[-(frames_ago + 1)]
    
    def export_range(self, start_seconds_ago, end_seconds_ago, output_path):
        """Exporta range específico do buffer"""
        start_idx = min(int(start_seconds_ago * self.fps), len(self.buffer) - 1)
        end_idx = min(int(end_seconds_ago * self.fps), len(self.buffer) - 1)
        
        if start_idx <= end_idx:
            # Swap if reversed
            start_idx, end_idx = end_idx, start_idx
        
        frames = list(self.buffer)[-(start_idx + 1):-(end_idx + 1) if end_idx > 0 else None]
        
        if len(frames) == 0:
            return None
        
        # Save
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        return output_path


class AnnotationOverlay:
    """Overlay de anotações em frames"""
    
    def __init__(self):
        self.annotations = []  # List of (type, data, timestamp)
        self.active_annotation = None
    
    def add_text(self, x, y, text, color=(0, 255, 255)):
        """Adiciona anotação de texto"""
        self.annotations.append({
            'type': 'text',
            'x': x, 'y': y,
            'text': text,
            'color': color,
            'timestamp': datetime.now()
        })
    
    def add_arrow(self, x1, y1, x2, y2, color=(0, 255, 0)):
        """Adiciona seta anotativa"""
        self.annotations.append({
            'type': 'arrow',
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'color': color,
            'timestamp': datetime.now()
        })
    
    def add_circle(self, x, y, radius, color=(255, 0, 255)):
        """Adiciona círculo de marcação"""
        self.annotations.append({
            'type': 'circle',
            'x': x, 'y': y,
            'radius': radius,
            'color': color,
            'timestamp': datetime.now()
        })
    
    def clear_all(self):
        """Limpa todas as anotações"""
        self.annotations = []
    
    def clear_last(self):
        """Remove última anotação"""
        if self.annotations:
            self.annotations.pop()
    
    def draw_on_frame(self, frame):
        """Desenha anotações no frame"""
        if frame is None or len(self.annotations) == 0:
            return frame
        
        result = frame.copy()
        
        for ann in self.annotations:
            if ann['type'] == 'text':
                cv2.putText(result, ann['text'], 
                           (ann['x'], ann['y']), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           ann['color'], 2)
            
            elif ann['type'] == 'arrow':
                cv2.arrowedLine(result, 
                               (ann['x1'], ann['y1']), 
                               (ann['x2'], ann['y2']), 
                               ann['color'], 2, tipLength=0.3)
            
            elif ann['type'] == 'circle':
                cv2.circle(result, (ann['x'], ann['y']), 
                          ann['radius'], ann['color'], 2)
        
        return result
    
    def get_annotations_count(self):
        """Retorna quantidade de anotações"""
        return len(self.annotations)
