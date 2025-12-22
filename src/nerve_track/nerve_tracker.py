"""
NERVE TRACK v2.0 - Sistema de Tracking Temporal
================================================
Tracking de estruturas anatômicas com Kalman Filter para consistência temporal.

Características:
- Kalman Filter para suavização de posições
- Associação Húngara para matching entre frames
- IOU (Intersection over Union) para tracking de máscaras
- Anti-flicker: elimina detecções instáveis
- Histórico de trajetórias para visualização

Baseado em:
- SORT (Simple Online Realtime Tracking)
- Deep SORT com aparência
- Técnicas de tracking médico
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import cv2
from scipy.optimize import linear_sum_assignment
from enum import IntEnum


# =============================================================================
# Estruturas de Dados
# =============================================================================

class StructureType(IntEnum):
    """Tipos de estruturas anatômicas."""
    BACKGROUND = 0
    NERVE = 1
    ARTERY = 2
    VEIN = 3
    FASCIA = 4
    MUSCLE = 5
    BONE = 6
    PLEURA = 7


@dataclass
class Detection:
    """
    Detecção de uma estrutura em um frame.
    """
    class_id: int                    # Tipo da estrutura
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    centroid: Tuple[float, float]    # Centro (cx, cy)
    contour: np.ndarray              # Contorno OpenCV
    area: float                      # Área em pixels
    confidence: float                # Confiança da detecção
    mask: Optional[np.ndarray] = None  # Máscara binária (opcional)

    @classmethod
    def from_contour(
        cls,
        contour: np.ndarray,
        class_id: int,
        confidence: float,
        mask: Optional[np.ndarray] = None
    ) -> 'Detection':
        """Cria Detection a partir de contorno OpenCV."""
        x, y, w, h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            cx, cy = x + w / 2, y + h / 2

        area = cv2.contourArea(contour)

        return cls(
            class_id=class_id,
            bbox=(x, y, w, h),
            centroid=(cx, cy),
            contour=contour,
            area=area,
            confidence=confidence,
            mask=mask
        )


@dataclass
class TrackedStructure:
    """
    Estrutura sendo rastreada ao longo do tempo.
    """
    track_id: int                    # ID único do track
    class_id: int                    # Tipo da estrutura
    state: np.ndarray                # Estado atual [x, y, vx, vy, w, h]
    covariance: np.ndarray           # Matriz de covariância
    hits: int = 0                    # Frames com detecção
    age: int = 0                     # Frames desde última detecção
    time_since_update: int = 0       # Frames desde update
    confidence: float = 0.0          # Confiança média
    contour: Optional[np.ndarray] = None  # Último contorno
    area: float = 0.0                # Área média
    trajectory: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def centroid(self) -> Tuple[float, float]:
        """Retorna centroide atual."""
        return (self.state[0], self.state[1])

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Retorna bounding box atual."""
        x = int(self.state[0] - self.state[4] / 2)
        y = int(self.state[1] - self.state[5] / 2)
        w = int(self.state[4])
        h = int(self.state[5])
        return (x, y, w, h)

    @property
    def velocity(self) -> Tuple[float, float]:
        """Retorna velocidade atual."""
        return (self.state[2], self.state[3])

    @property
    def is_confirmed(self) -> bool:
        """Track está confirmado (suficientes detecções)."""
        return self.hits >= 3

    @property
    def is_lost(self) -> bool:
        """Track está perdido (muito tempo sem detecção)."""
        return self.time_since_update > 10


# =============================================================================
# Kalman Filter para Tracking
# =============================================================================

class KalmanBoxTracker:
    """
    Kalman Filter para tracking de estruturas.
    Estado: [x, y, vx, vy, w, h]
    """

    # Contador global de IDs
    _id_counter = 0

    def __init__(
        self,
        detection: Detection,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1
    ):
        """
        Inicializa tracker com primeira detecção.

        Args:
            detection: Primeira detecção da estrutura
            process_noise: Ruído do processo (menor = mais suave)
            measurement_noise: Ruído da medição
        """
        self.track_id = KalmanBoxTracker._id_counter
        KalmanBoxTracker._id_counter += 1

        self.class_id = detection.class_id
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.confidence = detection.confidence
        self.contour = detection.contour
        self.area = detection.area
        self.trajectory = deque(maxlen=50)

        # Estado inicial: [x, y, vx, vy, w, h]
        cx, cy = detection.centroid
        x, y, w, h = detection.bbox

        self.state = np.array([cx, cy, 0, 0, w, h], dtype=np.float32)

        # Dimensões do estado e medição
        self.dim_x = 6  # [x, y, vx, vy, w, h]
        self.dim_z = 4  # [x, y, w, h]

        # Matriz de transição de estado (modelo de velocidade constante)
        # x' = x + vx * dt
        # y' = y + vy * dt
        self.F = np.eye(self.dim_x, dtype=np.float32)
        self.F[0, 2] = 1  # x += vx
        self.F[1, 3] = 1  # y += vy

        # Matriz de observação (medimos x, y, w, h)
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        self.H[2, 4] = 1  # w
        self.H[3, 5] = 1  # h

        # Covariância do processo Q
        self.Q = np.eye(self.dim_x, dtype=np.float32) * process_noise
        self.Q[2, 2] = 0.1  # Velocidade x
        self.Q[3, 3] = 0.1  # Velocidade y

        # Covariância da medição R
        self.R = np.eye(self.dim_z, dtype=np.float32) * measurement_noise

        # Covariância do estado P
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10
        self.P[2, 2] = 100  # Alta incerteza inicial em velocidade
        self.P[3, 3] = 100

        # Adicionar posição inicial à trajetória
        self.trajectory.append((cx, cy))

    def predict(self) -> np.ndarray:
        """
        Predição do próximo estado.

        Returns:
            Estado predito [x, y, vx, vy, w, h]
        """
        # Predição do estado
        self.state = self.F @ self.state

        # Predição da covariância
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1

        return self.state

    def update(self, detection: Detection):
        """
        Atualiza o tracker com nova detecção.

        Args:
            detection: Nova detecção
        """
        # Medição: [x, y, w, h]
        cx, cy = detection.centroid
        x, y, w, h = detection.bbox
        z = np.array([cx, cy, w, h], dtype=np.float32)

        # Resíduo (innovation)
        y_residual = z - self.H @ self.state

        # Covariância do resíduo
        S = self.H @ self.P @ self.H.T + self.R

        # Ganho de Kalman
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Atualização do estado
        self.state = self.state + K @ y_residual

        # Atualização da covariância
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        # Atualizar metadados
        self.hits += 1
        self.time_since_update = 0
        self.confidence = 0.9 * self.confidence + 0.1 * detection.confidence
        self.contour = detection.contour
        self.area = 0.9 * self.area + 0.1 * detection.area

        # Adicionar à trajetória
        self.trajectory.append((self.state[0], self.state[1]))

    def get_state(self) -> np.ndarray:
        """Retorna estado atual."""
        return self.state.copy()

    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Retorna bounding box."""
        cx, cy, _, _, w, h = self.state
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        return (x, y, int(w), int(h))

    def get_centroid(self) -> Tuple[float, float]:
        """Retorna centroide."""
        return (float(self.state[0]), float(self.state[1]))

    def to_tracked_structure(self) -> TrackedStructure:
        """Converte para TrackedStructure."""
        return TrackedStructure(
            track_id=self.track_id,
            class_id=self.class_id,
            state=self.state.copy(),
            covariance=self.P.copy(),
            hits=self.hits,
            age=self.age,
            time_since_update=self.time_since_update,
            confidence=self.confidence,
            contour=self.contour,
            area=self.area,
            trajectory=self.trajectory.copy()
        )


# =============================================================================
# NerveTracker - Sistema Principal de Tracking
# =============================================================================

class NerveTracker:
    """
    Sistema de tracking de estruturas anatômicas.

    Características:
    - Multi-objeto: rastreia múltiplas estruturas simultaneamente
    - Multi-classe: cada classe tem tracks separados
    - Kalman Filter: suavização temporal
    - Associação Húngara: matching ótimo entre detecções e tracks
    - Anti-flicker: filtra detecções instáveis
    """

    def __init__(
        self,
        max_age: int = 10,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_distance: float = 50.0,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1
    ):
        """
        Args:
            max_age: Frames máximos sem detecção antes de remover track
            min_hits: Hits mínimos para confirmar track
            iou_threshold: IOU mínimo para associação
            max_distance: Distância máxima para associação (pixels)
            process_noise: Ruído do processo Kalman
            measurement_noise: Ruído de medição Kalman
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Tracks ativos por classe
        self.tracks: Dict[int, List[KalmanBoxTracker]] = {}

        # Frame counter
        self.frame_count = 0

    def reset(self):
        """Reseta todos os tracks."""
        self.tracks.clear()
        self.frame_count = 0
        KalmanBoxTracker._id_counter = 0

    def update(self, detections: List[Detection]) -> List[TrackedStructure]:
        """
        Atualiza tracks com novas detecções.

        Args:
            detections: Lista de detecções do frame atual

        Returns:
            Lista de estruturas rastreadas confirmadas
        """
        self.frame_count += 1

        # Agrupar detecções por classe
        detections_by_class: Dict[int, List[Detection]] = {}
        for det in detections:
            if det.class_id not in detections_by_class:
                detections_by_class[det.class_id] = []
            detections_by_class[det.class_id].append(det)

        # Processar cada classe
        all_classes = set(self.tracks.keys()) | set(detections_by_class.keys())

        for class_id in all_classes:
            class_detections = detections_by_class.get(class_id, [])
            self._update_class(class_id, class_detections)

        # Retornar tracks confirmados
        return self.get_confirmed_tracks()

    def _update_class(self, class_id: int, detections: List[Detection]):
        """Atualiza tracks de uma classe específica."""

        # Inicializar lista de tracks se necessário
        if class_id not in self.tracks:
            self.tracks[class_id] = []

        tracks = self.tracks[class_id]

        # Predição para todos os tracks
        for track in tracks:
            track.predict()

        # Se não há detecções, apenas retornar
        if len(detections) == 0:
            self._cleanup_tracks(class_id)
            return

        # Se não há tracks, criar novos
        if len(tracks) == 0:
            for det in detections:
                self._create_track(class_id, det)
            return

        # Calcular matriz de custo (distância + IOU)
        cost_matrix = self._compute_cost_matrix(tracks, detections)

        # Associação Húngara
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:
            row_indices, col_indices = [], []

        # Processar matches
        matched_tracks = set()
        matched_detections = set()

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1.0:  # Threshold de custo
                tracks[row].update(detections[col])
                matched_tracks.add(row)
                matched_detections.add(col)

        # Criar tracks para detecções não associadas
        for i, det in enumerate(detections):
            if i not in matched_detections:
                self._create_track(class_id, det)

        # Limpar tracks perdidos
        self._cleanup_tracks(class_id)

    def _compute_cost_matrix(
        self,
        tracks: List[KalmanBoxTracker],
        detections: List[Detection]
    ) -> np.ndarray:
        """
        Computa matriz de custo para associação.
        Combina distância euclidiana e IOU.
        """
        n_tracks = len(tracks)
        n_dets = len(detections)

        cost_matrix = np.ones((n_tracks, n_dets), dtype=np.float32)

        for i, track in enumerate(tracks):
            track_centroid = track.get_centroid()
            track_bbox = track.get_bbox()

            for j, det in enumerate(detections):
                # Distância euclidiana normalizada
                dist = np.sqrt(
                    (track_centroid[0] - det.centroid[0]) ** 2 +
                    (track_centroid[1] - det.centroid[1]) ** 2
                )
                dist_cost = min(dist / self.max_distance, 1.0)

                # IOU
                iou = self._compute_iou(track_bbox, det.bbox)
                iou_cost = 1.0 - iou

                # Combinar custos (média ponderada)
                cost_matrix[i, j] = 0.6 * dist_cost + 0.4 * iou_cost

        return cost_matrix

    @staticmethod
    def _compute_iou(
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calcula Intersection over Union entre duas bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Coordenadas da interseção
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        # Área da interseção
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Áreas individuais
        area1 = w1 * h1
        area2 = w2 * h2

        # União
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _create_track(self, class_id: int, detection: Detection):
        """Cria novo track para detecção."""
        track = KalmanBoxTracker(
            detection,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise
        )
        self.tracks[class_id].append(track)

    def _cleanup_tracks(self, class_id: int):
        """Remove tracks perdidos."""
        if class_id not in self.tracks:
            return

        self.tracks[class_id] = [
            track for track in self.tracks[class_id]
            if track.time_since_update <= self.max_age
        ]

    def get_confirmed_tracks(self) -> List[TrackedStructure]:
        """Retorna tracks confirmados (com hits suficientes)."""
        confirmed = []
        for class_id, tracks in self.tracks.items():
            for track in tracks:
                if track.hits >= self.min_hits:
                    confirmed.append(track.to_tracked_structure())
        return confirmed

    def get_all_tracks(self) -> List[TrackedStructure]:
        """Retorna todos os tracks (incluindo não confirmados)."""
        all_tracks = []
        for class_id, tracks in self.tracks.items():
            for track in tracks:
                all_tracks.append(track.to_tracked_structure())
        return all_tracks

    def get_tracks_by_class(self, class_id: int) -> List[TrackedStructure]:
        """Retorna tracks de uma classe específica."""
        if class_id not in self.tracks:
            return []
        return [track.to_tracked_structure() for track in self.tracks[class_id]]


# =============================================================================
# Visualização de Tracking
# =============================================================================

class TrackingVisualizer:
    """Visualizador de tracking de estruturas."""

    # Cores por tipo de estrutura (BGR)
    COLORS = {
        StructureType.NERVE: (0, 255, 255),     # Amarelo
        StructureType.ARTERY: (0, 0, 255),      # Vermelho
        StructureType.VEIN: (255, 0, 0),        # Azul
        StructureType.FASCIA: (128, 128, 128),  # Cinza
        StructureType.MUSCLE: (128, 0, 128),    # Roxo
        StructureType.BONE: (255, 255, 255),    # Branco
        StructureType.PLEURA: (0, 165, 255),    # Laranja
    }

    # Nomes das estruturas
    NAMES = {
        StructureType.NERVE: "Nervo",
        StructureType.ARTERY: "Arteria",
        StructureType.VEIN: "Veia",
        StructureType.FASCIA: "Fascia",
        StructureType.MUSCLE: "Musculo",
        StructureType.BONE: "Osso",
        StructureType.PLEURA: "Pleura",
    }

    def __init__(
        self,
        show_trajectory: bool = True,
        show_id: bool = True,
        show_confidence: bool = True,
        trajectory_length: int = 30
    ):
        self.show_trajectory = show_trajectory
        self.show_id = show_id
        self.show_confidence = show_confidence
        self.trajectory_length = trajectory_length

    def draw(
        self,
        image: np.ndarray,
        tracks: List[TrackedStructure],
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Desenha tracks na imagem.

        Args:
            image: Imagem BGR
            tracks: Lista de TrackedStructure
            alpha: Transparência

        Returns:
            Imagem com visualização
        """
        vis = image.copy()
        overlay = image.copy()

        for track in tracks:
            color = self.COLORS.get(track.class_id, (255, 255, 255))
            name = self.NAMES.get(track.class_id, f"Classe {track.class_id}")

            # Desenhar contorno se disponível
            if track.contour is not None:
                cv2.drawContours(overlay, [track.contour], -1, color, -1)
                cv2.drawContours(vis, [track.contour], -1, color, 2)

            # Desenhar trajetória
            if self.show_trajectory and len(track.trajectory) > 1:
                points = list(track.trajectory)[-self.trajectory_length:]
                for i in range(1, len(points)):
                    # Fade baseado na antiguidade
                    intensity = int(255 * (i / len(points)))
                    traj_color = (
                        int(color[0] * intensity / 255),
                        int(color[1] * intensity / 255),
                        int(color[2] * intensity / 255)
                    )
                    cv2.line(
                        vis,
                        (int(points[i-1][0]), int(points[i-1][1])),
                        (int(points[i][0]), int(points[i][1])),
                        traj_color,
                        2
                    )

            # Desenhar centroide
            cx, cy = track.centroid
            cv2.circle(vis, (int(cx), int(cy)), 5, color, -1)

            # Label
            label_parts = []
            if self.show_id:
                label_parts.append(f"#{track.track_id}")
            label_parts.append(name)
            if self.show_confidence:
                label_parts.append(f"{track.confidence:.0%}")

            label = " ".join(label_parts)

            # Desenhar label
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x = int(cx - w / 2)
            label_y = int(cy - 15)

            cv2.rectangle(
                vis,
                (label_x - 2, label_y - h - 2),
                (label_x + w + 2, label_y + 2),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                vis,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        # Aplicar overlay com alpha
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

        return vis


# =============================================================================
# Factory Function
# =============================================================================

def create_nerve_tracker(
    max_age: int = 10,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    max_distance: float = 50.0,
    process_noise: float = 0.01,
    measurement_noise: float = 0.1
) -> NerveTracker:
    """
    Factory function para criar tracker de nervos.

    Args:
        max_age: Frames máximos sem detecção
        min_hits: Hits mínimos para confirmar
        iou_threshold: IOU mínimo para associação
        max_distance: Distância máxima para associação
        process_noise: Ruído do processo Kalman
        measurement_noise: Ruído de medição Kalman

    Returns:
        Instância de NerveTracker
    """
    return NerveTracker(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        max_distance=max_distance,
        process_noise=process_noise,
        measurement_noise=measurement_noise
    )


# =============================================================================
# Teste
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NERVE TRACK v2.0 - Sistema de Tracking")
    print("=" * 60)

    # Criar tracker
    tracker = create_nerve_tracker(
        max_age=10,
        min_hits=3,
        max_distance=50.0
    )

    # Simular detecções
    print("\nSimulando tracking...")

    # Frame 1: 2 nervos detectados
    detections = [
        Detection(
            class_id=StructureType.NERVE,
            bbox=(100, 100, 50, 50),
            centroid=(125, 125),
            contour=np.array([[[100, 100]], [[150, 100]], [[150, 150]], [[100, 150]]]),
            area=2500,
            confidence=0.9
        ),
        Detection(
            class_id=StructureType.ARTERY,
            bbox=(200, 200, 30, 30),
            centroid=(215, 215),
            contour=np.array([[[200, 200]], [[230, 200]], [[230, 230]], [[200, 230]]]),
            area=900,
            confidence=0.85
        )
    ]

    # Processar múltiplos frames
    for frame in range(10):
        # Simular movimento
        for det in detections:
            cx, cy = det.centroid
            det.centroid = (cx + 5, cy + 2)
            x, y, w, h = det.bbox
            det.bbox = (x + 5, y + 2, w, h)

        tracks = tracker.update(detections)
        print(f"  Frame {frame + 1}: {len(tracks)} tracks confirmados")

    # Mostrar resultado final
    print("\nTracks finais:")
    for track in tracker.get_all_tracks():
        print(f"  Track #{track.track_id}:")
        print(f"    - Classe: {StructureType(track.class_id).name}")
        print(f"    - Posição: {track.centroid}")
        print(f"    - Confiança: {track.confidence:.2%}")
        print(f"    - Hits: {track.hits}")
        print(f"    - Trajetória: {len(track.trajectory)} pontos")

    print("\n" + "=" * 60)
    print("[OK] Sistema de tracking funcionando!")
    print("=" * 60)
