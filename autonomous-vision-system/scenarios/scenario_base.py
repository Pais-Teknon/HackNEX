"""
Scenario Definitions and Zone Management
Supports defining custom geometric zones (Polygons / Rectangles) and rules for behavior analysis.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

class ZoneType:
    SAFE = "SAFE"
    RESTRICTED = "RESTRICTED"
    MONITORED_LOITER = "MONITORED_LOITER"
    HAZARD = "HAZARD"

class Zone:
    """Represents a spatial region with specific rules"""
    def __init__(
        self,
        name: str,
        zone_type: str,
        polygon: List[Tuple[int, int]],
        max_dwell_time: float = 0.0, # 0 = zero tolerance (instant alert if restricted)
        description: str = "",
        color: Optional[Tuple[int, int, int]] = None
    ):
        self.name = name
        self.zone_type = zone_type
        self.polygon = np.array(polygon, dtype=np.int32)
        self.max_dwell_time = max_dwell_time
        self.description = description
        
        # Color mapping (BGR)
        if color is not None:
            self.color = color
        elif zone_type == ZoneType.RESTRICTED:
            self.color = (50, 50, 220) # Red
        elif zone_type == ZoneType.MONITORED_LOITER:
            self.color = (40, 180, 240) # Orange/Amber
        elif zone_type == ZoneType.HAZARD:
            self.color = (30, 100, 240) # Dark Red/Orange
        else:
            self.color = (60, 200, 70) # Green

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if point (x, y) is inside the zone polygon"""
        res = cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False)
        return res >= 0

class Scenario:
    """Base class for vision understanding scenarios"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.zones: List[Zone] = []
        self.allowed_actions: List[str] = ["WALKING", "STANDING"]
        self.prohibited_actions: List[str] = ["RUNNING", "FALLEN", "LOITERING"]
        self.max_loiter_seconds: float = 8.0
        self.max_safe_speed: float = 120.0
        self.alert_on_fall: bool = True
        self.alert_on_running: bool = True

    def add_zone(self, zone: Zone):
        self.zones.append(zone)

    def check_zone_containment(self, point: Tuple[float, float]) -> List[Zone]:
        """Returns list of zones containing this point"""
        return [z for z in self.zones if z.contains_point(point)]
