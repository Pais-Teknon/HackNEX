# Autonomous Vision & Behaviour Understanding — Incident Audit Report
**Project**: HNX26PSI07 | **Scenario**: Campus Perimeter Security

**Description**: Tracks student and staff flow, detects after-hours fence intrusion, and surfaces suspicious loitering.

---

## 1. Executive Summary
- **Total Surfaced Incidents**: 2
- **Critical Severity (Life Safety / Breach)**: 1
- **Warning Severity (Protocol / Velocity)**: 1
- **Normal / Informational Events**: 0

## 2. Incidents by Anomaly Type
- **FALL_INCIDENT**: 1 event(s)
- **UNSAFE_SPEED**: 1 event(s)

## 3. Incident Chronology Log (Who, What, When, Where)
| Event ID | Entity | Timestamp | Event Type | Action | Severity | Zone | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EVT-1001` | **Person #2** | `00:01.75` | `FALL_INCIDENT` | FALLEN | **CRITICAL** | General Area | 1.08s |
| `EVT-1002` | **Person #2** | `00:03.00` | `UNSAFE_SPEED` | RUNNING | **WARNING** | General Area | 0.25s |

## 4. Automated Safety Protocol Recommendations
> [!CAUTION]
> **Immediate Attention Required**: Critical incidents detected (e.g. fallen workers or restricted zone breaches). Dispatch site safety officers immediately.
> [!WARNING]
> **Policy Violations**: Loitering in egress corridors or excessive walking speed detected. Review walkway guidelines with shift supervisors.