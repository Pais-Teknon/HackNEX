"""
Object and Human Pose Detection Module
Wraps Ultralytics YOLOv8 with dual-mode support (standard bbox detection and pose keypoints).
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import cv2
from ultralytics import YOLO

class DetectionResult:
    """Encapsulates a single detected entity in a frame"""
    def __init__(
        self,
        bbox: Tuple[float, float, float, float], # (x1, y1, x2, y2)
        confidence: float,
        class_id: int,
        class_name: str,
        keypoints: Optional[np.ndarray] = None, # shape: (17, 3) [x, y, conf]
        track_id: Optional[int] = None
    ):
        self.bbox = bbox # (x1, y1, x2, y2)
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = class_name
        self.keypoints = keypoints
        self.track_id = track_id

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Feet / ground contact location of the entity"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def width(self) -> float:
        return max(1.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])

    @property
    def aspect_ratio(self) -> float:
        """Width / Height. Near 0.3-0.5 is upright person; > 0.8 is fallen/horizontal."""
        return self.width / self.height

class VisionDetector:
    """High-performance vision detector with tracking integration"""
    def __init__(
        self,
        model_path: str,
        use_pose: bool = True,
        conf_thresh: float = 0.35,
        device: str = "cpu"
    ):
        self.model_path = model_path
        self.use_pose = use_pose
        self.conf_thresh = conf_thresh
        self.device = device
        
        # Load model
        self.model = YOLO(model_path)
        self.model.to(device)

    def detect_and_track(
        self,
        frame: np.ndarray,
        persist: bool = True,
        tracker: str = "bytetrack.yaml"
    ) -> List[DetectionResult]:
        """
        Runs detection and tracking on a single frame.
        Returns a list of DetectionResult objects with persistent track_ids.
        """
        # Run tracking through YOLO's integrated tracker
        results = self.model.track(
            source=frame,
            persist=persist,
            tracker=tracker,
            conf=self.conf_thresh,
            classes=[0], # Class 0 is 'person' in COCO
            verbose=False,
            device=self.device
        )
        
        detections: List[DetectionResult] = []
        if not results or len(results) == 0:
            return detections
            
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # Extract boxes, scores, and track IDs
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        track_ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)
        
        # Keypoints extraction if pose model is active
        kpts_data = None
        if hasattr(r, 'keypoints') and r.keypoints is not None and r.keypoints.data is not None:
            kpts_data = r.keypoints.data.cpu().numpy() # shape: (N, 17, 3)

        for i in range(len(boxes)):
            tid = int(track_ids[i]) if track_ids[i] is not None else None
            kpts = kpts_data[i] if kpts_data is not None and i < len(kpts_data) else None
            
            # Keypoint plausibility filter: avoid phantom object detections (chairs, bags)
            if kpts is not None and len(kpts) >= 17:
                # Top upper body keypoints (head & shoulders)
                upper_conf = max(kpts[0][2], kpts[1][2], kpts[2][2], kpts[5][2], kpts[6][2])
                if upper_conf < 0.35 and confs[i] < 0.65:
                    continue

            det = DetectionResult(
                bbox=tuple(xyxy[i]),
                confidence=float(confs[i]),
                class_id=int(clss[i]),
                class_name="person",
                keypoints=kpts,
                track_id=tid
            )
            detections.append(det)

        return detections
