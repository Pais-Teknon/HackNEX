"""
Campus Perimeter Security Scenario
Monitors unauthorized after-hours perimeter access, fence lingering/loitering,
and running/panic incidents across campus zones.
"""

from .scenario_base import Scenario, Zone, ZoneType

def create_campus_scenario(frame_width: int = 1280, frame_height: int = 720) -> Scenario:
    scenario = Scenario(
        name="Campus Perimeter Security",
        description="Tracks student and staff flow, detects after-hours fence intrusion, and surfaces suspicious loitering."
    )
    
    w_scale = frame_width / 1280.0
    h_scale = frame_height / 720.0
    
    # 1. Perimeter Fence Restricted Area
    fence_poly = [
        (int(850 * w_scale), int(50 * h_scale)),
        (int(1260 * w_scale), int(50 * h_scale)),
        (int(1260 * w_scale), int(450 * h_scale)),
        (int(850 * w_scale), int(450 * h_scale)),
    ]
    scenario.add_zone(Zone(
        name="North Perimeter Fence (Restricted)",
        zone_type=ZoneType.RESTRICTED,
        polygon=fence_poly,
        max_dwell_time=2.0,
        description="Restricted boundary fence. Immediate alert upon presence."
    ))

    # 2. Main Plaza / Common Pathway
    plaza_poly = [
        (int(50 * w_scale), int(150 * h_scale)),
        (int(600 * w_scale), int(150 * h_scale)),
        (int(600 * w_scale), int(680 * h_scale)),
        (int(50 * w_scale), int(680 * h_scale)),
    ]
    scenario.add_zone(Zone(
        name="Central Courtyard Walkway",
        zone_type=ZoneType.SAFE,
        polygon=plaza_poly,
        description="Public campus passage."
    ))

    scenario.max_loiter_seconds = 10.0
    scenario.max_safe_speed = 130.0
    scenario.alert_on_fall = True
    scenario.alert_on_running = True
    
    return scenario
