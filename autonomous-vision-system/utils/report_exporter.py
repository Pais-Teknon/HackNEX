"""
Incident and Behavior Audit Report Exporter
Generates CSV, JSON, and professional Markdown incident audit reports.
"""

from typing import List, Dict, Any, Optional
import os
import json
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from ..core.event_engine import EventEngine, VisionEvent
    from ..scenarios.scenario_base import Scenario
except (ImportError, ValueError):
    from core.event_engine import EventEngine, VisionEvent
    from scenarios.scenario_base import Scenario

class ReportExporter:
    """Exports structured audit logs and incident summaries"""
    def __init__(self, event_engine: EventEngine, scenario: Scenario):
        self.engine = event_engine
        self.scenario = scenario

    def export_csv(self, output_path: str) -> str:
        """Exports all detected incidents to CSV"""
        df = self.engine.get_dataframe()
        df.to_csv(output_path, index=False)
        return output_path

    def export_json(self, output_path: str) -> str:
        """Exports all incidents and metadata to JSON"""
        stats = self.engine.get_summary_stats()
        events_data = [e.to_dict() for e in self.engine.events]
        payload = {
            "scenario": self.scenario.name,
            "scenario_description": self.scenario.description,
            "statistics": stats,
            "events": events_data
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        return output_path

    def export_markdown_report(self, output_path: str) -> str:
        """Generates a comprehensive audit summary in Markdown"""
        stats = self.engine.get_summary_stats()
        df = self.engine.get_dataframe()

        md = []
        md.append(f"# Autonomous Vision & Behaviour Understanding — Incident Audit Report")
        md.append(f"**Project**: HNX26PSI07 | **Scenario**: {self.scenario.name}\n")
        md.append(f"**Description**: {self.scenario.description}\n")
        md.append(f"---\n")

        # Executive Summary
        md.append(f"## 1. Executive Summary")
        md.append(f"- **Total Surfaced Incidents**: {stats['total_events']}")
        md.append(f"- **Critical Severity (Life Safety / Breach)**: {stats['critical_count']}")
        md.append(f"- **Warning Severity (Protocol / Velocity)**: {stats['warning_count']}")
        md.append(f"- **Normal / Informational Events**: {stats['normal_count']}\n")

        # Breakdown by Anomaly Type
        md.append(f"## 2. Incidents by Anomaly Type")
        if stats['by_type']:
            for k, v in stats['by_type'].items():
                md.append(f"- **{k}**: {v} event(s)")
        else:
            md.append(f"*No abnormal behavior detected.*")
        md.append("")

        # Detailed Incident Timeline Log (Satisfying WHO, WHAT, WHEN, WHERE)
        md.append(f"## 3. Incident Chronology Log (Who, What, When, Where)")
        if not df.empty:
            md.append("| Event ID | Entity | Timestamp | Event Type | Action | Severity | Zone | Duration |")
            md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for _, row in df.iterrows():
                md.append(f"| `{row['event_id']}` | **{row['entity_id']}** | `{row['timestamp']}` | `{row['event_type']}` | {row['action']} | **{row['severity']}** | {row['zone']} | {row['duration_seconds']}s |")
        else:
            md.append(f"*No events logged.*")
        md.append("")

        # Recommendations
        md.append(f"## 4. Automated Safety Protocol Recommendations")
        if stats['critical_count'] > 0:
            md.append(f"> [!CAUTION]\n> **Immediate Attention Required**: Critical incidents detected (e.g. fallen workers or restricted zone breaches). Dispatch site safety officers immediately.")
        if stats['warning_count'] > 0:
            md.append(f"> [!WARNING]\n> **Policy Violations**: Loitering in egress corridors or excessive walking speed detected. Review walkway guidelines with shift supervisors.")
        if stats['total_events'] == 0:
            md.append(f"> [!NOTE]\n> Normal operations maintained. All tracked individuals remained compliant with zone policies.")

        content = "\n".join(md)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path
