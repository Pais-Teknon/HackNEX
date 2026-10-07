"""
Action Recognition Module (Fine-Grained Human Behavior Understanding)
Fuses 17 COCO skeletal keypoints, 2D geometric body constraints, and temporal kinematics:
- SITTING (Seated at desk / chair, head upright above shoulders)
- STANDING (Upright posture, extended legs, stationary)
- WALKING (Steady locomotion speed 25-115 px/s)
- RUNNING (High-velocity locomotion > 115 px/s)
- CROUCHING (Knees bent / hips lowered or bending downwards)
- RAISING_HANDS (Wrists raised above shoulder level)
- FALLEN (Genuine horizontal body orientation on the floor)
- ERRATIC_MOTION (Rapid direction oscillation / panic motion)
"""

from typing import Tuple, Optional, Dict, Any, List
import math
import numpy as np
from .tracker import TrackedEntity

try:
    from ..config import BehaviorThresholds
except (ImportError, ValueError):
    from config import BehaviorThresholds

class ActionType:
    STANDING = "STANDING"
    SITTING = "SITTING"
    WALKING = "WALKING"
    RUNNING = "RUNNING"
    CROUCHING = "CROUCHING"
    RAISING_HANDS = "RAISING_HANDS"
    FALLEN = "FALLEN"
    ERRATIC_MOTION = "ERRATIC_MOTION"

class ActionRecognizer:
    """Accurately classifies human actions using multi-stage pose and kinematic fusion"""
    def __init__(self, thresholds: Optional[BehaviorThresholds] = None):
        self.thresh = thresholds or BehaviorThresholds()

    def analyze_pose_geometry(
        self,
        keypoints: Optional[np.ndarray],
        bbox: Tuple[float, float, float, float]
    ) -> Dict[str, Any]:
        """
        Analyzes 17 COCO keypoints:
        0: nose, 1: l_eye, 2: r_eye, 3: l_ear, 4: r_ear
        5: l_shoulder, 6: r_shoulder, 7: l_elbow, 8: r_elbow
        9: l_wrist, 10: r_wrist, 11: l_hip, 12: r_hip
        13: l_knee, 14: r_knee, 15: l_ankle, 16: r_ankle
        """
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        aspect = w / h

        feat = {
            "head_above_shoulders": False,
            "head_to_shoulder_dy": 0.0,
            "torso_angle_deg": 90.0,
            "is_horizontal_spine": False,
            "hands_raised": False,
            "is_crouched": False,
            "lower_body_visible": False,
            "aspect_ratio": aspect,
            "has_face": False,
            "has_torso": False,
            "keypoints_detected": False
        }

        if keypoints is None or len(keypoints) < 17:
            if aspect > 1.35 and h < 120:
                feat["is_horizontal_spine"] = True
            return feat

        # 1. Identify Head Position
        nose = keypoints[0]
        l_eye, r_eye = keypoints[1], keypoints[2]
        l_ear, r_ear = keypoints[3], keypoints[4]
        
        has_face = (nose[2] > 0.30) or (l_eye[2] > 0.30) or (r_eye[2] > 0.30)
        feat["has_face"] = has_face
        
        if has_face:
            # Best estimate of head (x, y)
            head_pts = [p for p in [nose, l_eye, r_eye, l_ear, r_ear] if p[2] > 0.30]
            head_x = float(np.mean([p[0] for p in head_pts]))
            head_y = float(np.mean([p[1] for p in head_pts]))
        else:
            head_x = (bbox[0] + bbox[2]) / 2.0
            head_y = float(bbox[1]) + 0.15 * h

        # 2. Identify Shoulders
        l_sh, r_sh = keypoints[5], keypoints[6]
        has_l_sh = l_sh[2] > 0.25
        has_r_sh = r_sh[2] > 0.25
        has_shoulders = has_l_sh or has_r_sh

        if has_shoulders:
            feat["has_torso"] = True
            feat["keypoints_detected"] = True
            if has_l_sh and has_r_sh:
                sh_x = float((l_sh[0] + r_sh[0]) / 2.0)
                sh_y = float((l_sh[1] + r_sh[1]) / 2.0)
            elif has_l_sh:
                sh_x, sh_y = float(l_sh[0]), float(l_sh[1])
            else:
                sh_x, sh_y = float(r_sh[0]), float(r_sh[1])
        else:
            # Estimate shoulder level anatomically below head
            sh_x = head_x
            sh_y = head_y + max(25.0, 0.28 * h)

        # 3. Head-to-Shoulder Alignment
        dy_head = sh_y - head_y # Positive when head is vertically above shoulders
        feat["head_to_shoulder_dy"] = dy_head
        if dy_head > 10.0:
            feat["head_above_shoulders"] = True

        # 4. Hand / Arm Gestures
        l_wr, r_wr = keypoints[9], keypoints[10]
        if (l_wr[2] > 0.30 and l_wr[1] < (sh_y - 12.0)) or (r_wr[2] > 0.30 and r_wr[1] < (sh_y - 12.0)):
            feat["hands_raised"] = True

        # 5. Hips and Spine Angle
        l_hip, r_hip = keypoints[11], keypoints[12]
        has_l_hip = l_hip[2] > 0.25
        has_r_hip = r_hip[2] > 0.25
        if has_l_hip or has_r_hip:
            feat["keypoints_detected"] = True
            if has_l_hip and has_r_hip:
                hip_x = float((l_hip[0] + r_hip[0]) / 2.0)
                hip_y = float((l_hip[1] + r_hip[1]) / 2.0)
            elif has_l_hip:
                hip_x, hip_y = float(l_hip[0]), float(l_hip[1])
            else:
                hip_x, hip_y = float(r_hip[0]), float(r_hip[1])

            dx = hip_x - sh_x
            dy = hip_y - sh_y
            angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))
            feat["torso_angle_deg"] = angle

            # Fall: spine is strictly horizontal AND head is not vertically above shoulders
            if angle < 28.0 and not feat["head_above_shoulders"]:
                feat["is_horizontal_spine"] = True

            # 6. Lower Body / Legs Check (Higher confidence required to avoid hallucinated standing legs)
            l_knee, r_knee = keypoints[13], keypoints[14]
            l_ank, r_ank = keypoints[15], keypoints[16]
            has_legs = (l_knee[2] > 0.40) or (r_knee[2] > 0.40) or (l_ank[2] > 0.40) or (r_ank[2] > 0.40)
            if has_legs:
                feat["lower_body_visible"] = True
                torso_len = max(20.0, math.hypot(dx, dy))
                
                # Check knees & ankles to estimate leg extension
                best_ank_y = None
                if l_ank[2] > 0.40 and r_ank[2] > 0.40:
                    best_ank_y = max(l_ank[1], r_ank[1])
                elif l_ank[2] > 0.40:
                    best_ank_y = l_ank[1]
                elif r_ank[2] > 0.40:
                    best_ank_y = r_ank[1]
                else:
                    best_ank_y = hip_y + 1.2 * torso_len  # Fallback

                leg_len = abs(best_ank_y - hip_y)
                compression = leg_len / torso_len
                feat["leg_compression_ratio"] = compression
                
                # Crouch: compressed leg length or forward bend
                if compression < 0.65 and angle > 45.0:
                    feat["is_crouched"] = True

        return feat

    def classify_action(self, entity: TrackedEntity) -> Tuple[str, float]:
        """
        Determines the exact human action using pose keypoints, geometry, and kinematics.
        Returns (action_string, confidence).
        """
        speed = entity.smoothed_speed
        aspect = entity.aspect_ratio
        feat = self.analyze_pose_geometry(entity.current_keypoints, entity.current_bbox)

        # 1. GESTURING / RAISING HANDS (Hands near or above head/shoulder level)
        if feat["hands_raised"]:
            return ActionType.RAISING_HANDS, 0.90

        # 2. FALL DETECTION (Strict criteria: horizontal spine, head NOT upright)
        if feat["is_horizontal_spine"] and not feat["head_above_shoulders"]:
            return ActionType.FALLEN, 0.95

        # 3. CROUCHING / BENDING
        if feat["is_crouched"]:
            return ActionType.CROUCHING, 0.85

        # 4. KINEMATIC LOCOMOTION (Running vs Walking)
        if speed >= self.thresh.running_min_speed:
            return ActionType.RUNNING, 0.92
        elif speed >= self.thresh.standing_max_speed:
            return ActionType.WALKING, 0.88

        # 5. STATIONARY POSTURE: SITTING vs STANDING
        # If head is above shoulders (upright):
        if feat["head_above_shoulders"] or feat["has_face"]:
            # If the bounding box is square-ish/wide (typical for desk work)
            if aspect > 0.70:
                return ActionType.SITTING, 0.88
                
            # If lower body is definitely not visible (upper body only shot)
            if not feat["lower_body_visible"]:
                return ActionType.SITTING, 0.88
                
            # If full body is visible, but legs are bent/compressed relative to torso (sitting in a chair)
            compression = feat.get("leg_compression_ratio", 1.0)
            if feat["lower_body_visible"] and compression < 0.90:
                return ActionType.SITTING, 0.88

            return ActionType.STANDING, 0.90

        # Aspect ratio based fallback when keypoints are missing or messy
        if aspect > 0.70:
            return ActionType.SITTING, 0.78

        return ActionType.STANDING, 0.80

    def update_entity_action(self, entity: TrackedEntity):
        """Classifies action and applies temporal smoothing"""
        raw_action, conf = self.classify_action(entity)
        entity.action_history.append(raw_action)

        # Fast response for gesture raising hands
        if raw_action == ActionType.RAISING_HANDS:
            entity.current_action = ActionType.RAISING_HANDS
            entity.action_confidence = conf
            return

        # Fall incident confirmation requires at least 3 frames of horizontal posture
        recent = list(entity.action_history)[-5:]
        if recent.count(ActionType.FALLEN) >= 3:
            entity.current_action = ActionType.FALLEN
            entity.action_confidence = 0.95
            return

        # Majority vote across recent 5 frames to eliminate single-frame flickering
        counts: Dict[str, int] = {}
        for a in recent:
            if a != ActionType.FALLEN:
                counts[a] = counts.get(a, 0) + 1
        
        if counts:
            best_action = max(counts.items(), key=lambda x: x[1])[0]
            entity.current_action = best_action
            entity.action_confidence = conf
        else:
            entity.current_action = raw_action
            entity.action_confidence = conf
