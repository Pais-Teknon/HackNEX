"""
Workplace & Warehouse Safety Scenario
Focuses on worker safety, forklift/machinery danger zones, running restrictions,
slip/fall detection, and prolonged loitering/inactivity detection.
"""

from .scenario_base import Scenario, Zone, ZoneType

def create_warehouse_scenario(frame_width: int = 1280, frame_height: int = 720) -> Scenario:
    scenario = Scenario(
        name="Warehouse & Industrial Safety",
        description="Monitors worker locomotion, machinery hazard zones, fall incidents, and prolonged stationary dwell time."
    )
    
    # Scale coordinates to frame resolution
    w_scale = frame_width / 1280.0
    h_scale = frame_height / 720.0
    
    # 1. Machinery & Forklift Operation Zone (RESTRICTED - Unauthorized personnel must not enter)
    danger_poly = [
        (int(750 * w_scale), int(120 * h_scale)),
        (int(1220 * w_scale), int(120 * h_scale)),
        (int(1220 * w_scale), int(580 * h_scale)),
        (int(750 * w_scale), int(580 * h_scale)),
    ]
    scenario.add_zone(Zone(
        name="Forklift & Heavy Machinery Zone",
        zone_type=ZoneType.RESTRICTED,
        polygon=danger_poly,
        max_dwell_time=1.0,
        description="High risk automated machinery operating area. Immediate alert on worker entry."
    ))

    # 2. Main Corridor / Loading Bay Loitering Watch Zone (MONITORED_LOITER)
    # A person walking through is normal; standing still for too long blocks emergency access
    corridor_poly = [
        (int(60 * w_scale), int(200 * h_scale)),
        (int(450 * w_scale), int(200 * h_scale)),
        (int(450 * w_scale), int(680 * h_scale)),
        (int(60 * w_scale), int(680 * h_scale)),
    ]
    scenario.add_zone(Zone(
        name="Emergency Egress Corridor",
        zone_type=ZoneType.MONITORED_LOITER,
        polygon=corridor_poly,
        max_dwell_time=6.0, # 6s stationary triggers alert
        description="Corridor must be kept clear. Stationary presence / loitering triggers warning."
    ))

    # Scenario parameters
    scenario.max_loiter_seconds = 7.0
    scenario.max_safe_speed = 110.0
    scenario.alert_on_fall = True
    scenario.alert_on_running = True
    
    return scenario
