"""
Entity Tracking and Temporal State Management
Tracks individual entities across video frames, maintains spatial history,
calculates physical dynamics (velocity, acceleration, directional changes), and dwell times.
"""

from typing import List, Dict, Tuple, Optional, Deque
from collections import deque
import math
import numpy as np
from .detector import DetectionResult

class TrackedEntity:
    """Represents a persistently tracked entity across frames"""
    def __init__(
        self,
        track_id: int,
        initial_detection: DetectionResult,
        frame_idx: int,
        timestamp: float,
        max_history: int = 60
    ):
        self.track_id = track_id
        self.first_seen_frame = frame_idx
        self.first_seen_time = timestamp
        self.last_seen_frame = frame_idx
        self.last_seen_time = timestamp
        
        # Geometry and pose
        self.current_bbox = initial_detection.bbox
        self.current_confidence = initial_detection.confidence
        self.current_keypoints = initial_detection.keypoints
        
        # Spatial history: deque of (x, y, timestamp) where (x, y) is bottom_center/feet
        self.history: Deque[Tuple[float, float, float]] = deque(maxlen=max_history)
        self.history.append((initial_detection.bottom_center[0], initial_detection.bottom_center[1], timestamp))
        
        # Dynamic metrics
        self.speed: float = 0.0          # Pixels per second
        self.smoothed_speed: float = 0.0 # Exponentially smoothed speed
        self.acceleration: float = 0.0   # Speed delta / time
        self.direction_angle: float = 0.0 # Heading in degrees
        self.heading_variation: float = 0.0 # Measure of erratic direction changes
        
        # Behavior & Zone state
        self.current_action: str = "STANDING"
        self.action_confidence: float = 0.8
        self.action_history: Deque[str] = deque(maxlen=15)
        self.action_history.append("STANDING")
        
        self.is_anomalous: bool = False
        self.anomaly_score: float = 0.0 # 0.0 to 1.0
        self.active_anomalies: List[str] = []
        self.severity: str = "NORMAL" # NORMAL, WARNING, CRITICAL
        
        # Dwell & loitering metrics
        self.current_zones: List[str] = []
        self.zone_dwell_times: Dict[str, float] = {} # zone_name -> seconds inside
        self.stationary_start_time: Optional[float] = timestamp
        self.stationary_duration: float = 0.0
        self.reference_stationary_pt: Tuple[float, float] = initial_detection.bottom_center

        # Alerts debouncing & action logging
        self.last_alert_timestamps: Dict[str, float] = {}
        self.last_logged_action: Optional[str] = None
        self.last_action_change_time: float = timestamp

    @property
    def total_active_time(self) -> float:
        return max(0.0, self.last_seen_time - self.first_seen_time)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.current_bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.current_bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return max(1.0, self.current_bbox[2] - self.current_bbox[0])

    @property
    def height(self) -> float:
        return max(1.0, self.current_bbox[3] - self.current_bbox[1])

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def update(
        self,
        detection: DetectionResult,
        frame_idx: int,
        timestamp: float,
        loiter_radius_px: float = 45.0
    ):
        """Updates entity state with a new frame detection"""
        dt = timestamp - self.last_seen_time
        if dt <= 0:
            dt = 1.0 / 30.0 # default fallback ~33ms
            
        self.last_seen_frame = frame_idx
        self.last_seen_time = timestamp
        self.current_bbox = detection.bbox
        self.current_confidence = detection.confidence
        self.current_keypoints = detection.keypoints

        curr_pt = detection.bottom_center
        prev_pt = (self.history[-1][0], self.history[-1][1]) if len(self.history) > 0 else curr_pt
        
        # Instantaneous displacement & speed
        dx = curr_pt[0] - prev_pt[0]
        dy = curr_pt[1] - prev_pt[1]
        dist = math.hypot(dx, dy)
        instant_speed = dist / dt
        
        # Exponential smoothing for speed (alpha = 0.3)
        self.smoothed_speed = 0.3 * instant_speed + 0.7 * self.smoothed_speed
        self.acceleration = (instant_speed - self.speed) / dt
        self.speed = instant_speed

        # Heading angle
        if dist > 3.0:
            angle = math.degrees(math.atan2(dy, dx))
            # Heading variance
            diff = abs(angle - self.direction_angle)
            if diff > 180:
                diff = 360 - diff
            self.heading_variation = 0.25 * diff + 0.75 * self.heading_variation
            self.direction_angle = angle

        # Append to history
        self.history.append((curr_pt[0], curr_pt[1], timestamp))

        # Stationary / Loitering tracking
        dist_from_ref = math.hypot(curr_pt[0] - self.reference_stationary_pt[0],
                                   curr_pt[1] - self.reference_stationary_pt[1])
        if dist_from_ref <= loiter_radius_px:
            # Still within stationary radius
            if self.stationary_start_time is None:
                self.stationary_start_time = timestamp
            self.stationary_duration = timestamp - self.stationary_start_time
        else:
            # Moved out of stationary radius - reset stationary reference point
            self.reference_stationary_pt = curr_pt
            self.stationary_start_time = timestamp
            self.stationary_duration = 0.0

    def update_zones(self, zone_names: List[str], dt: float):
        """Updates dwell times in active zones"""
        self.current_zones = zone_names
        for z in zone_names:
            self.zone_dwell_times[z] = self.zone_dwell_times.get(z, 0.0) + dt

class TrackManager:
    """Maintains active and historical entities across the entire video session"""
    def __init__(self, max_missing_frames: int = 30):
        self.entities: Dict[int, TrackedEntity] = {}
        self.next_unassigned_id: int = 1000
        self.max_missing_frames = max_missing_frames

    def update_tracks(
        self,
        detections: List[DetectionResult],
        frame_idx: int,
        timestamp: float,
        loiter_radius_px: float = 45.0
    ) -> List[TrackedEntity]:
        """
        Associates detections with tracked entities.
        Uses tracker assigned ID, or falls back to distance association.
        """
        active_ids = set()
        matched_entities: List[TrackedEntity] = []

        for det in detections:
            tid = det.track_id
            if tid is None:
                # Find nearest existing track without an update
                best_id = None
                best_dist = float('inf')
                for eid, entity in self.entities.items():
                    if eid not in active_ids and (frame_idx - entity.last_seen_frame) < self.max_missing_frames:
                        d = math.hypot(det.bottom_center[0] - entity.bottom_center[0],
                                       det.bottom_center[1] - entity.bottom_center[1])
                        if d < best_dist and d < 120.0:
                            best_dist = d
                            best_id = eid
                if best_id is not None:
                    tid = best_id
                else:
                    tid = self.next_unassigned_id
                    self.next_unassigned_id += 1

            active_ids.add(tid)
            if tid in self.entities:
                self.entities[tid].update(det, frame_idx, timestamp, loiter_radius_px)
            else:
                self.entities[tid] = TrackedEntity(
                    track_id=tid,
                    initial_detection=det,
                    frame_idx=frame_idx,
                    timestamp=timestamp
                )
            matched_entities.append(self.entities[tid])

        return matched_entities

    def get_active_entities(self, current_frame: int, tolerance_frames: int = 5) -> List[TrackedEntity]:
        """Returns entities seen in the last few frames"""
        return [
            e for e in self.entities.values()
            if (current_frame - e.last_seen_frame) <= tolerance_frames
        ]
