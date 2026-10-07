# Autonomous Vision & Behaviour Understanding — Incident Audit Report
**Project**: HNX26PSI07 | **Scenario**: Warehouse & Industrial Safety

**Description**: Monitors worker locomotion, machinery hazard zones, fall incidents, and prolonged stationary dwell time.

---

## 1. Executive Summary
- **Total Surfaced Incidents**: 2
- **Critical Severity (Life Safety / Breach)**: 2
- **Warning Severity (Protocol / Velocity)**: 0
- **Normal / Informational Events**: 0

## 2. Incidents by Anomaly Type
- **FALL_INCIDENT**: 2 event(s)

## 3. Incident Chronology Log (Who, What, When, Where)
| Event ID | Entity | Timestamp | Event Type | Action | Severity | Zone | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EVT-1001` | **Person #1** | `-1:59.00` | `FALL_INCIDENT` | FALLEN | **CRITICAL** | General Area | -3.0s |
| `EVT-1002` | **Person #2** | `-1:59.00` | `FALL_INCIDENT` | FALLEN | **CRITICAL** | Emergency Egress Corridor | -3.0s |

## 4. Automated Safety Protocol Recommendations
> [!CAUTION]
> **Immediate Attention Required**: Critical incidents detected (e.g. fallen workers or restricted zone breaches). Dispatch site safety officers immediately.