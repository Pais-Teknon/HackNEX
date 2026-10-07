"""
Retail Store & Checkout Analytics Scenario
Monitors shopping flow, cashier desk security, unattended aisles, and prolonged loitering near valuable counters.
"""

from .scenario_base import Scenario, Zone, ZoneType

def create_retail_scenario(frame_width: int = 1280, frame_height: int = 720) -> Scenario:
    scenario = Scenario(
        name="Retail Store & Loss Prevention",
        description="Analyzes shopper movement, detects checkout queue dwell time, and flags unauthorized access behind counters."
    )
    
    w_scale = frame_width / 1280.0
    h_scale = frame_height / 720.0
    
    # 1. Cash Counter / Staff Only Zone
    counter_poly = [
        (int(800 * w_scale), int(100 * h_scale)),
        (int(1250 * w_scale), int(100 * h_scale)),
        (int(1250 * w_scale), int(450 * h_scale)),
        (int(800 * w_scale), int(450 * h_scale)),
    ]
    scenario.add_zone(Zone(
        name="Cash Register & Safe Vault",
        zone_type=ZoneType.RESTRICTED,
        polygon=counter_poly,
        max_dwell_time=1.0,
        description="Employee-only checkout counter and register vault."
    ))

    # 2. High-Value Merchandise Display
    display_poly = [
        (int(100 * w_scale), int(200 * h_scale)),
        (int(500 * w_scale), int(200 * h_scale)),
        (int(500 * w_scale), int(650 * h_scale)),
        (int(100 * w_scale), int(650 * h_scale)),
    ]
    scenario.add_zone(Zone(
        name="High-Value Electronics Display",
        zone_type=ZoneType.MONITORED_LOITER,
        polygon=display_poly,
        max_dwell_time=8.0,
        description="Monitored for prolonged hovering without active engagement."
    ))

    scenario.max_loiter_seconds = 8.0
    scenario.max_safe_speed = 90.0
    scenario.alert_on_fall = True
    scenario.alert_on_running = True
    
    return scenario
