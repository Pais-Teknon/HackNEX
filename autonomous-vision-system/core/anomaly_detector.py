"""
Behavior Anomaly Detection Engine
Evaluates entity action, spatial position relative to zones, stationary dwell times,
and kinematics against scenario rules to compute anomaly scores and surface flags.
"""

from typing import List, Dict, Tuple, Optional
from .tracker import TrackedEntity
from .action_recognizer import ActionType

try:
    from ..scenarios.scenario_base import Scenario, ZoneType
    from ..config import BehaviorThresholds
except (ImportError, ValueError):
    from scenarios.scenario_base import Scenario, ZoneType
    from config import BehaviorThresholds

class AnomalyType:
    LOITERING = "LOITERING"
    RESTRICTED_INTRUSION = "RESTRICTED_INTRUSION"
    FALL_INCIDENT = "FALL_INCIDENT"
    UNSAFE_SPEED = "UNSAFE_SPEED"
    ERRATIC_MOTION = "ERRATIC_MOTION"
    OVERSTAY = "OVERSTAY"

class AnomalyFlag:
    """Detailed anomaly explanation satisfying the requirement: say WHO, WHAT, WHEN, and WHY"""
    def __init__(
        self,
        anomaly_type: str,
        severity: str, # "WARNING" or "CRITICAL"
        explanation: str,
        score: float,
        metric_value: float,
        threshold_value: float,
        zone_name: Optional[str] = None
    ):
        self.anomaly_type = anomaly_type
        self.severity = severity
        self.explanation = explanation
        self.score = score
        self.metric_value = metric_value
        self.threshold_value = threshold_value
        self.zone_name = zone_name

class AnomalyDetector:
    """Evaluates behavior patterns against scenario rules"""
    def __init__(self, scenario: Scenario, thresholds: Optional[BehaviorThresholds] = None):
        self.scenario = scenario
        self.thresh = thresholds or BehaviorThresholds()

    def evaluate_entity(self, entity: TrackedEntity, current_timestamp: float) -> List[AnomalyFlag]:
        """
        Evaluates a single entity's behavior and returns any detected anomalies.
        Updates entity's anomaly score, severity, and active anomaly list.
        """
        flags: List[AnomalyFlag] = []
        feet_pos = entity.bottom_center
        
        # 1. Zone Containment Check
        containing_zones = self.scenario.check_zone_containment(feet_pos)
        zone_names = [z.name for z in containing_zones]
        dt = current_timestamp - entity.last_seen_time
        if dt <= 0:
            dt = 1.0 / 30.0
        entity.update_zones(zone_names, dt)

        # 2. Check Restricted Zone Intrusion (CRITICAL)
        for zone in containing_zones:
            if zone.zone_type == ZoneType.RESTRICTED:
                dwell = entity.zone_dwell_times.get(zone.name, 0.0)
                if dwell >= zone.max_dwell_time:
                    flags.append(AnomalyFlag(
                        anomaly_type=AnomalyType.RESTRICTED_INTRUSION,
                        severity="CRITICAL",
                        explanation=f"Intrusion into restricted area '{zone.name}' (Dwell: {dwell:.1f}s)",
                        score=0.95,
                        metric_value=dwell,
                        threshold_value=zone.max_dwell_time,
                        zone_name=zone.name
                    ))

        # 3. Fall Detection Check (CRITICAL)
        if entity.current_action == ActionType.FALLEN and self.scenario.alert_on_fall:
            flags.append(AnomalyFlag(
                anomaly_type=AnomalyType.FALL_INCIDENT,
                severity="CRITICAL",
                explanation=f"Worker/Person fallen on floor (Torso horizontal, motionless)",
                score=0.98,
                metric_value=entity.aspect_ratio,
                threshold_value=self.thresh.fall_aspect_ratio_threshold
            ))

        # 4. Stationary Loitering / Inactivity Check (WARNING / CRITICAL)
        # Directly addresses the prompt: 'A person standing still in one spot for 10 minutes is unusual'
        is_in_monitored_zone = any(z.zone_type in (ZoneType.MONITORED_LOITER, ZoneType.RESTRICTED) for z in containing_zones)
        loiter_limit = self.scenario.max_loiter_seconds
        
        # Check if inside a designated monitored loitering zone with a custom threshold
        for zone in containing_zones:
            if zone.zone_type == ZoneType.MONITORED_LOITER:
                loiter_limit = min(loiter_limit, zone.max_dwell_time)

        # In designated zones, enforce loiter_limit; in general open areas, allow normal presence
        effective_limit = loiter_limit if is_in_monitored_zone else (loiter_limit * 2.5)
                
        if entity.stationary_duration >= effective_limit and entity.current_action != ActionType.FALLEN:
            # Normal seated presence in general areas is compliant
            if entity.current_action == ActionType.SITTING and not is_in_monitored_zone:
                pass
            else:
                severity = "CRITICAL" if entity.stationary_duration > (effective_limit * 2.0) else "WARNING"
                score = min(0.90, 0.50 + 0.40 * (entity.stationary_duration / (effective_limit * 2.0)))
                flags.append(AnomalyFlag(
                    anomaly_type=AnomalyType.LOITERING,
                    severity=severity,
                    explanation=f"Abnormal stationary dwell: Inactive in one spot for {entity.stationary_duration:.1f}s (Limit: {effective_limit:.1f}s)",
                    score=score,
                    metric_value=entity.stationary_duration,
                    threshold_value=effective_limit,
                    zone_name=zone_names[0] if zone_names else "General Area"
                ))

        # 5. Unsafe Running Check (WARNING)
        if entity.current_action == ActionType.RUNNING and self.scenario.alert_on_running:
            flags.append(AnomalyFlag(
                anomaly_type=AnomalyType.UNSAFE_SPEED,
                severity="WARNING",
                explanation=f"Running in restricted corridor (Speed: {entity.smoothed_speed:.1f} px/s, Limit: {self.scenario.max_safe_speed:.1f} px/s)",
                score=0.75,
                metric_value=entity.smoothed_speed,
                threshold_value=self.scenario.max_safe_speed
            ))

        # 6. Erratic / Panic Motion Check (WARNING)
        if entity.current_action == ActionType.ERRATIC_MOTION:
            flags.append(AnomalyFlag(
                anomaly_type=AnomalyType.ERRATIC_MOTION,
                severity="WARNING",
                explanation=f"Erratic direction changes detected ({entity.heading_variation:.1f}° variation)",
                score=0.65,
                metric_value=entity.heading_variation,
                threshold_value=self.thresh.erratic_turn_angle_deg
            ))

        # Update entity state
        if flags:
            entity.is_anomalous = True
            entity.anomaly_score = max(f.score for f in flags)
            entity.active_anomalies = [f.anomaly_type for f in flags]
            entity.severity = "CRITICAL" if any(f.severity == "CRITICAL" for f in flags) else "WARNING"
        else:
            entity.is_anomalous = False
            entity.anomaly_score = max(0.0, entity.anomaly_score * 0.8) # Decay
            entity.active_anomalies = []
            entity.severity = "NORMAL"

        return flags
