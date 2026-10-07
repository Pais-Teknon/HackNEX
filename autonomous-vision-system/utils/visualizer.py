"""
Advanced Visualizer and HUD Overlay Engine
Renders high-clarity computer vision overlays:
- Semi-transparent zone polygons
- Trajectory breadcrumbs and motion vectors
- Entity identification, action badges, and dynamic severity bounding boxes
- Pose keypoints and skeletal linkages
- Real-time incident alert banners and system telemetry
"""

from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.tracker import TrackedEntity
from scenarios.scenario_base import Scenario, Zone, ZoneType
from core.event_engine import VisionEvent
from config import COLORS

# Skeleton connection pairs for COCO 17 keypoints
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Facial
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Upper body / Arms
    (5, 11), (6, 12), (11, 12),               # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
]

class VisionVisualizer:
    """Renders overlays onto OpenCV BGR image frames"""
    def __init__(
        self,
        show_zones: bool = True,
        show_trajectories: bool = True,
        show_skeleton: bool = True,
        show_telemetry: bool = True,
        show_alert_banner: bool = True
    ):
        self.show_zones = show_zones
        self.show_trajectories = show_trajectories
        self.show_skeleton = show_skeleton
        self.show_telemetry = show_telemetry
        self.show_alert_banner = show_alert_banner

    def render_zones(self, frame: np.ndarray, scenario: Scenario) -> np.ndarray:
        """Draws semi-transparent polygon zones with crisp borders and labels"""
        if not self.show_zones or not scenario.zones:
            return frame

        overlay = frame.copy()
        alpha = 0.22 # Transparency

        for zone in scenario.zones:
            pts = zone.polygon.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], zone.color)
            cv2.polylines(frame, [pts], isClosed=True, color=zone.color, thickness=2, lineType=cv2.LINE_AA)
            
            # Label near top centroid of zone
            moments = cv2.moments(zone.polygon)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = zone.polygon[0]

            label = f"[{zone.zone_type}] {zone.name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (cx - tw // 2 - 4, cy - th - 6), (cx + tw // 2 + 4, cy + 4), (20, 20, 20), -1)
            cv2.putText(frame, label, (cx - tw // 2, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, zone.color, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def render_trajectories(self, frame: np.ndarray, entities: List[TrackedEntity]):
        """Draws fading motion breadcrumb trails for each tracked entity"""
        if not self.show_trajectories:
            return

        for entity in entities:
            pts = list(entity.history)
            if len(pts) < 2:
                continue

            color = COLORS[entity.severity]
            n_pts = len(pts)
            for i in range(1, n_pts):
                pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))
                # Increasing thickness for recent points
                thickness = max(1, int(1 + 3 * (i / n_pts)))
                cv2.line(frame, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    def render_skeleton(self, frame: np.ndarray, keypoints: Optional[np.ndarray], color: Tuple[int, int, int]):
        """Draws 17-point pose skeleton"""
        if not self.show_skeleton or keypoints is None or len(keypoints) < 17:
            return

        for p1_idx, p2_idx in SKELETON_PAIRS:
            p1, p2 = keypoints[p1_idx], keypoints[p2_idx]
            if p1[2] > 0.35 and p2[2] > 0.35:
                pt1 = (int(p1[0]), int(p1[1]))
                pt2 = (int(p2[0]), int(p2[1]))
                cv2.line(frame, pt1, pt2, color, 2, lineType=cv2.LINE_AA)

        for kp in keypoints:
            if kp[2] > 0.35:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    def render_entity(self, frame: np.ndarray, entity: TrackedEntity):
        """Renders bounding box, badge, action name, and metrics for an entity"""
        x1, y1, x2, y2 = [int(v) for v in entity.current_bbox]
        severity_color = COLORS[entity.severity]
        
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), severity_color, 2, lineType=cv2.LINE_AA)

        # Corner accents for polished look
        corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), severity_color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), severity_color, 4)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), severity_color, 4)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), severity_color, 4)

        # Skeleton overlay
        self.render_skeleton(frame, entity.current_keypoints, severity_color)

        # Action and ID Badge Header
        id_text = f"Person #{entity.track_id}"
        action_map = {
            "SITTING": "SITTING",
            "STANDING": "STANDING",
            "WALKING": "WALKING",
            "RUNNING": "RUNNING",
            "CROUCHING": "CROUCHING",
            "RAISING_HANDS": "GESTURING / WAVING",
            "FALLEN": "FALL DETECTED!"
        }
        action_display = action_map.get(entity.current_action, entity.current_action)
        if entity.current_action == "FALLEN":
            action_display = "FALL DETECTED!"
        elif entity.is_anomalous:
            action_display = f"{action_display} [{entity.severity}]"

        # Context-aware metric string
        if entity.current_action == "RAISING_HANDS":
            metric_str = "Hands Raised"
        elif entity.current_action == "SITTING":
            metric_str = f"Seated ({entity.stationary_duration:.0f}s)" if entity.stationary_duration > 2 else "Seated"
        elif entity.stationary_duration > 1.5:
            metric_str = f"Stationary ({entity.stationary_duration:.1f}s)"
        else:
            metric_str = f"{entity.smoothed_speed:.0f} px/s"

        badge_line1 = f"{id_text} | {action_display}"
        badge_line2 = f"State: {metric_str} | Anom: {int(entity.anomaly_score * 100)}%"

        # Draw background badge above bbox
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.44
        thick = 1
        (w1, h1), _ = cv2.getTextSize(badge_line1, font, scale, thick)
        (w2, h2), _ = cv2.getTextSize(badge_line2, font, scale, thick)
        badge_w = max(w1, w2) + 12
        badge_h = h1 + h2 + 14

        badge_y1 = max(0, y1 - badge_h)
        badge_x1 = max(0, x1)
        badge_x2 = min(frame.shape[1], badge_x1 + badge_w)
        badge_y2 = badge_y1 + badge_h

        # Badge box
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (25, 25, 25), -1)
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), severity_color, 1)

        # Text lines
        cv2.putText(frame, badge_line1, (badge_x1 + 6, badge_y1 + h1 + 3), font, scale, severity_color, thick, cv2.LINE_AA)
        cv2.putText(frame, badge_line2, (badge_x1 + 6, badge_y1 + h1 + h2 + 8), font, scale, (210, 210, 210), thick, cv2.LINE_AA)

    def render_alert_banner(self, frame: np.ndarray, active_events: List[VisionEvent]):
        """Displays prominent top warning/critical banner if incident is active"""
        if not self.show_alert_banner or not active_events:
            return

        crit_events = [e for e in active_events if e.severity == "CRITICAL"]
        warn_events = [e for e in active_events if e.severity == "WARNING"]
        
        target_event = crit_events[0] if crit_events else (warn_events[0] if warn_events else None)
        if not target_event:
            return

        color = COLORS[target_event.severity]
        h, w = frame.shape[:2]
        banner_h = 42

        # Semi-transparent dark banner
        banner_overlay = frame.copy()
        cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), (15, 15, 20), -1)
        cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)

        # Solid color accent line
        cv2.line(frame, (0, banner_h), (w, banner_h), color, 3)

        # Alert text
        prefix = "CRITICAL ALERT" if target_event.severity == "CRITICAL" else "SECURITY WARNING"
        msg = f"[{prefix}] {target_event.summary} (At {target_event.timestamp_str})"
        cv2.putText(frame, msg, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def render_telemetry(self, frame: np.ndarray, fps: float, active_count: int, anomaly_count: int, scenario_name: str):
        """Displays top-right/bottom telemetry metrics"""
        if not self.show_telemetry:
            return
            
        h, w = frame.shape[:2]
        hud_box_w = 260
        hud_box_h = 75
        x1 = w - hud_box_w - 15
        y1 = 15

        # Background box
        sub = frame[y1:y1 + hud_box_h, x1:x1 + hud_box_w]
        dark = np.zeros_like(sub)
        cv2.addWeighted(dark, 0.7, sub, 0.3, 0, sub)
        frame[y1:y1 + hud_box_h, x1:x1 + hud_box_w] = sub
        cv2.rectangle(frame, (x1, y1), (x1 + hud_box_w, y1 + hud_box_h), (80, 80, 80), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"SCENARIO: {scenario_name[:20]}", (x1 + 10, y1 + 18), font, 0.40, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"PROCESSING FPS: {fps:.1f}", (x1 + 10, y1 + 36), font, 0.42, (0, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"ACTIVE ENTITIES: {active_count}", (x1 + 10, y1 + 54), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        
        anom_color = (60, 200, 70) if anomaly_count == 0 else (50, 50, 240)
        cv2.putText(frame, f"ANOMALIES ACTIVE: {anomaly_count}", (x1 + 10, y1 + 70), font, 0.42, anom_color, 1, cv2.LINE_AA)

    def draw_frame(
        self,
        raw_frame: np.ndarray,
        entities: List[TrackedEntity],
        scenario: Scenario,
        active_events: List[VisionEvent],
        fps: float = 30.0
    ) -> np.ndarray:
        """Composes all visual layers onto a clean frame"""
        annotated = raw_frame.copy()
        
        # 1. Zone polygons
        annotated = self.render_zones(annotated, scenario)
        
        # 2. Historical trajectories
        self.render_trajectories(annotated, entities)
        
        # 3. Entity boxes and badges
        for entity in entities:
            self.render_entity(annotated, entity)

        # 4. Top alert banner
        self.render_alert_banner(annotated, active_events)

        # 5. Telemetry
        anomaly_count = sum(1 for e in entities if e.is_anomalous)
        self.render_telemetry(annotated, fps, len(entities), anomaly_count, scenario.name)

        return annotated
