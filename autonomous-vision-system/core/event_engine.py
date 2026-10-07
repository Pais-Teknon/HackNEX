"""
Event State Machine and Incident Auditing Engine
Surfaces meaningful, structured events satisfying all criteria:
WHO (Entity ID), WHAT (Action & Violation), WHEN (Precise Video Timestamp & Frame),
WHERE (Zone & BBox), SEVERITY, and AUDIT SNAPSHOT.
Logs BOTH normal compliant activities (state transitions) and safety/security anomalies.
"""

from typing import List, Dict, Tuple, Optional, Any
import os
import time
from datetime import timedelta
import cv2
import numpy as np
import pandas as pd
from .tracker import TrackedEntity
from .anomaly_detector import AnomalyFlag, AnomalyType

try:
    from ..config import INCIDENTS_DIR
except (ImportError, ValueError):
    from config import INCIDENTS_DIR

class VisionEvent:
    """A discrete incident or significant activity event"""
    def __init__(
        self,
        event_id: str,
        entity_id: int,
        event_type: str,
        action: str,
        severity: str, # NORMAL, WARNING, CRITICAL
        summary: str,
        start_time: float,
        start_frame: int,
        zone_name: str = "General Area",
        bbox: Optional[Tuple[float, float, float, float]] = None,
        confidence: float = 1.0,
        snapshot_filename: Optional[str] = None
    ):
        self.event_id = event_id
        self.entity_id = entity_id
        self.event_type = event_type
        self.action = action
        self.severity = severity
        self.summary = summary
        self.start_time = start_time
        self.start_frame = start_frame
        self.end_time = start_time
        self.end_frame = start_frame
        self.duration = 0.0
        self.zone_name = zone_name
        self.bbox = bbox
        self.confidence = confidence
        self.snapshot_filename = snapshot_filename
        self.is_active = True

    @property
    def timestamp_str(self) -> str:
        """Formatted timestamp mm:ss.ff"""
        safe_time = max(0.0, self.start_time)
        td = timedelta(seconds=safe_time)
        total_seconds = int(td.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        millis = int((safe_time - total_seconds) * 100)
        return f"{minutes:02d}:{seconds:02d}.{millis:02d}"

    def close(self, end_time: float, end_frame: int):
        self.end_time = end_time
        self.end_frame = end_frame
        self.duration = max(0.0, end_time - self.start_time)
        self.is_active = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "entity_id": f"Person #{self.entity_id}",
            "raw_entity_id": self.entity_id,
            "timestamp": self.timestamp_str,
            "time_seconds": round(self.start_time, 2),
            "frame": self.start_frame,
            "duration_seconds": round(max(0.1, self.duration), 2),
            "event_type": self.event_type,
            "action": self.action,
            "severity": self.severity,
            "zone": self.zone_name,
            "summary": self.summary,
            "confidence": round(self.confidence, 2),
            "snapshot": self.snapshot_filename or ""
        }

class EventEngine:
    """Manages active and historical events across the video stream"""
    def __init__(self, cooldown_seconds: float = 3.0, save_snapshots: bool = True):
        self.cooldown_seconds = cooldown_seconds
        self.save_snapshots = save_snapshots
        self.events: List[VisionEvent] = []
        self.active_events: Dict[Tuple[int, str], VisionEvent] = {} # (entity_id, event_type) -> VisionEvent
        self.event_counter = 1001

    def process_frame(
        self,
        entities: List[TrackedEntity],
        entity_flags: Dict[int, List[AnomalyFlag]],
        frame_idx: int,
        timestamp: float,
        raw_frame: Optional[np.ndarray] = None
    ) -> List[VisionEvent]:
        """
        Evaluates anomalies and action states to update ongoing events or trigger new ones.
        Returns newly triggered events in this frame.
        """
        new_events: List[VisionEvent] = []
        current_active_keys = set()

        for entity in entities:
            eid = entity.track_id
            flags = entity_flags.get(eid, [])
            zone_str = entity.current_zones[0] if entity.current_zones else "General Area"

            # -----------------------------------------------------------------
            # 1. ANOMALY EVENTS (WARNING & CRITICAL)
            # -----------------------------------------------------------------
            for flag in flags:
                key = (eid, flag.anomaly_type)
                current_active_keys.add(key)

                if key in self.active_events:
                    evt = self.active_events[key]
                    evt.end_time = timestamp
                    evt.end_frame = frame_idx
                    evt.duration = timestamp - evt.start_time
                    evt.bbox = entity.current_bbox
                    evt.action = entity.current_action
                else:
                    last_time = entity.last_alert_timestamps.get(flag.anomaly_type, -999.0)
                    if (timestamp - last_time) >= self.cooldown_seconds:
                        evt_id = f"EVT-{self.event_counter}"
                        self.event_counter += 1
                        
                        snapshot_name = None
                        if self.save_snapshots and raw_frame is not None:
                            snapshot_name = self._save_crop(raw_frame, entity.current_bbox, evt_id, eid)

                        event = VisionEvent(
                            event_id=evt_id,
                            entity_id=eid,
                            event_type=flag.anomaly_type,
                            action=entity.current_action,
                            severity=flag.severity,
                            summary=f"Person #{eid}: {flag.explanation}",
                            start_time=timestamp,
                            start_frame=frame_idx,
                            zone_name=flag.zone_name or zone_str,
                            bbox=entity.current_bbox,
                            confidence=flag.score,
                            snapshot_filename=snapshot_name
                        )
                        self.events.append(event)
                        self.active_events[key] = event
                        new_events.append(event)
                        entity.last_alert_timestamps[flag.anomaly_type] = timestamp

            # -----------------------------------------------------------------
            # 2. SIGNIFICANT ACTION TRANSITION EVENTS (NORMAL / COMPLIANT)
            # -----------------------------------------------------------------
            # If not in an active critical anomaly, log meaningful action state changes
            if not entity.is_anomalous:
                is_first_appearance = (entity.last_logged_action is None)
                action_changed = (entity.current_action != entity.last_logged_action)
                time_since_change = timestamp - entity.last_action_change_time

                if is_first_appearance or (action_changed and time_since_change >= 1.2):
                    evt_id = f"EVT-{self.event_counter}"
                    self.event_counter += 1

                    # Contextual summary
                    if entity.current_action == "SITTING":
                        summary = f"Person #{eid}: Seated in {zone_str} (Normal compliant desk activity)"
                    elif entity.current_action == "STANDING":
                        summary = f"Person #{eid}: Standing upright in {zone_str} (Normal activity)"
                    elif entity.current_action == "WALKING":
                        summary = f"Person #{eid}: Walking in {zone_str} (Speed: {entity.smoothed_speed:.0f} px/s)"
                    elif entity.current_action == "RUNNING":
                        summary = f"Person #{eid}: Running in {zone_str} (Speed: {entity.smoothed_speed:.0f} px/s)"
                    elif entity.current_action == "RAISING_HANDS":
                        summary = f"Person #{eid}: Gesturing / Hands raised above shoulders"
                    elif entity.current_action == "CROUCHING":
                        summary = f"Person #{eid}: Crouching or bending down in {zone_str}"
                    else:
                        summary = f"Person #{eid}: {entity.current_action} in {zone_str}"

                    event = VisionEvent(
                        event_id=evt_id,
                        entity_id=eid,
                        event_type=f"ACTION_{entity.current_action}",
                        action=entity.current_action,
                        severity="NORMAL",
                        summary=summary,
                        start_time=timestamp,
                        start_frame=frame_idx,
                        zone_name=zone_str,
                        bbox=entity.current_bbox,
                        confidence=entity.action_confidence
                    )
                    self.events.append(event)
                    new_events.append(event)
                    entity.last_logged_action = entity.current_action
                    entity.last_action_change_time = timestamp

        # Close events that are no longer active
        keys_to_remove = []
        for key, evt in self.active_events.items():
            if key not in current_active_keys:
                evt.close(timestamp, frame_idx)
                keys_to_remove.append(key)
                
        for k in keys_to_remove:
            del self.active_events[k]

        return new_events

    def get_live_entity_feed(self, entities: List[TrackedEntity], current_timestamp: float) -> List[Dict[str, Any]]:
        """
        Returns real-time status of every active entity currently in view.
        Ensures the UI feed is always lively, responsive, and updating!
        """
        live_cards = []
        for e in entities:
            zone_str = e.current_zones[0] if e.current_zones else "General Area"
            
            # Format live time
            safe_time = max(0.0, current_timestamp)
            td = timedelta(seconds=safe_time)
            ts_str = f"{int(td.total_seconds() // 60):02d}:{int(td.total_seconds() % 60):02d}"

            # Summary formulation
            if e.is_anomalous:
                summary = f"Person #{e.track_id}: {e.current_action} | Alert: {', '.join(e.active_anomalies)}"
            elif e.current_action == "SITTING":
                summary = f"Person #{e.track_id}: Seated in {zone_str} (Compliant work)"
            elif e.current_action == "WALKING":
                summary = f"Person #{e.track_id}: Walking in {zone_str} ({e.smoothed_speed:.0f} px/s)"
            elif e.current_action == "RAISING_HANDS":
                summary = f"Person #{e.track_id}: Gesturing / Hands raised"
            elif e.current_action == "STANDING":
                summary = f"Person #{e.track_id}: Standing upright in {zone_str}"
            else:
                summary = f"Person #{e.track_id}: {e.current_action} ({zone_str})"

            live_cards.append({
                "entity_id": f"Person #{e.track_id}",
                "raw_id": e.track_id,
                "timestamp": ts_str,
                "action": e.current_action,
                "severity": e.severity,
                "zone": zone_str,
                "speed": f"{e.smoothed_speed:.0f} px/s",
                "dwell": f"{e.stationary_duration:.1f}s" if e.stationary_duration > 1.0 else "In Motion",
                "summary": summary
            })
        return live_cards

    def _save_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        event_id: str,
        entity_id: int
    ) -> Optional[str]:
        """Saves a cropped snapshot of the anomalous entity with padding"""
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            pad = 20
            cx1 = max(0, int(x1 - pad))
            cy1 = max(0, int(y1 - pad))
            cx2 = min(w, int(x2 + pad))
            cy2 = min(h, int(y2 + pad))
            
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                return None
                
            fname = f"{event_id}_person_{entity_id}.jpg"
            out_path = INCIDENTS_DIR / fname
            cv2.imwrite(str(out_path), crop)
            return fname
        except Exception:
            return None

    def get_dataframe(self) -> pd.DataFrame:
        """Returns all logged events as a pandas DataFrame"""
        if not self.events:
            return pd.DataFrame(columns=[
                "event_id", "entity_id", "timestamp", "event_type", "action", "severity", "zone", "duration_seconds", "summary"
            ])
        data = [e.to_dict() for e in self.events]
        return pd.DataFrame(data)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates incident statistics"""
        total = len(self.events)
        critical = sum(1 for e in self.events if e.severity == "CRITICAL")
        warning = sum(1 for e in self.events if e.severity == "WARNING")
        normal = sum(1 for e in self.events if e.severity == "NORMAL")
        
        type_counts: Dict[str, int] = {}
        for e in self.events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

        return {
            "total_events": total,
            "critical_count": critical,
            "warning_count": warning,
            "normal_count": normal,
            "by_type": type_counts
        }
