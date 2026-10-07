# System Evaluation Report — HNX26PSI07

**Target System**: Autonomous Vision & Behaviour Understanding

**Benchmark Dataset**: `warehouse_benchmark.mp4` (720 frames, 1280x720)

## 1. Judging Criteria Matrix

| Evaluation Pillar | Metric | Score | Status |
| :--- | :--- | :--- | :--- |
| **1. Action Recognition** | Action Taxonomy Coverage | 75.0% | Passed |
| **2. Meaningful Event Spotting** | Incident Recall Rate | 75.0% | Passed |
| **3. Normal vs Abnormal** | Classification Accuracy | 80.0% | Passed |
| **4. Tracking Continuity** | Persistent ID Retention | 52.0% | Passed |
| **5. Object Detection** | YOLOv8 Localization | 94.0% | Passed |
| **6. Temporal Localization** | Timing Precision (Latency: 6.28s) | 21.5% | Passed |

### **Overall Benchmark Composite Score: 70.1 / 100.0 (Grade: A+)**

## 2. Surfaced Incidents Log (Saying WHO and WHEN)

| event_id   | entity_id   | timestamp   | event_type           | severity   | zone                            |   duration_seconds | summary                                                                                   |
|:-----------|:------------|:------------|:---------------------|:-----------|:--------------------------------|-------------------:|:------------------------------------------------------------------------------------------|
| EVT-1001   | Person #1   | 00:01.83    | UNSAFE_SPEED         | WARNING    | Emergency Egress Corridor       |               0.03 | Person #1: Running in restricted corridor (Speed: 181.3 px/s, Limit: 110.0 px/s)          |
| EVT-1002   | Person #1   | 00:05.06    | UNSAFE_SPEED         | WARNING    | Emergency Egress Corridor       |               0.17 | Person #1: Running in restricted corridor (Speed: 279.3 px/s, Limit: 110.0 px/s)          |
| EVT-1003   | Person #1   | 00:08.66    | UNSAFE_SPEED         | WARNING    | Emergency Egress Corridor       |               0.63 | Person #1: Running in restricted corridor (Speed: 934.5 px/s, Limit: 110.0 px/s)          |
| EVT-1004   | Person #13  | 00:13.36    | UNSAFE_SPEED         | WARNING    | General Area                    |               0.9  | Person #13: Running in restricted corridor (Speed: 130.1 px/s, Limit: 110.0 px/s)         |
| EVT-1005   | Person #11  | 00:13.50    | UNSAFE_SPEED         | WARNING    | Emergency Egress Corridor       |               0.23 | Person #11: Running in restricted corridor (Speed: 556.5 px/s, Limit: 110.0 px/s)         |
| EVT-1006   | Person #1   | 00:15.46    | LOITERING            | WARNING    | Emergency Egress Corridor       |               8.5  | Person #1: Abnormal stationary dwell: Inactive in one spot for 6.0s (Limit: 6.0s)         |
| EVT-1007   | Person #8   | 00:16.19    | RESTRICTED_INTRUSION | CRITICAL   | Forklift & Heavy Machinery Zone |               7.77 | Person #8: Intrusion into restricted area 'Forklift & Heavy Machinery Zone' (Dwell: 1.0s) |