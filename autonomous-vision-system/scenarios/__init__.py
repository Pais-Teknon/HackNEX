from .scenario_base import Scenario, Zone, ZoneType
from .warehouse_safety import create_warehouse_scenario
from .campus_security import create_campus_scenario
from .retail_store import create_retail_scenario

SCENARIOS = {
    "warehouse": ("Warehouse & Industrial Safety", create_warehouse_scenario),
    "campus": ("Campus Perimeter Security", create_campus_scenario),
    "retail": ("Retail Store & Loss Prevention", create_retail_scenario),
}

def get_scenario(name: str, width: int = 1280, height: int = 720) -> Scenario:
    key = name.lower().replace(" ", "_")
    for k, (display_name, factory) in SCENARIOS.items():
        if k in key or key in k:
            return factory(width, height)
    # Default to warehouse
    return create_warehouse_scenario(width, height)

__all__ = [
    "Scenario",
    "Zone",
    "ZoneType",
    "get_scenario",
    "SCENARIOS",
    "create_warehouse_scenario",
    "create_campus_scenario",
    "create_retail_scenario",
]
