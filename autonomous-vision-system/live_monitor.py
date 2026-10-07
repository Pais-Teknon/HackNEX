"""
Live Webcam Monitor for Autonomous Vision & Behaviour Understanding
(HNX26PSI07)
Real-time, zero-latency desktop monitoring using DirectShow on Windows.
Hotkeys:
    'q' or ESC : Exit monitoring and export incident report
    'z'        : Toggle zone overlay
    'p'        : Toggle 17-keypoint skeletal pose
    't'        : Toggle trajectory breadcrumbs
    's'        : Save manual incident screenshot
    'r'        : Reset tracker & incident counters
    '1', '2'   : Switch between Warehouse and Campus scenarios
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_CONFIG, BehaviorThresholds, INCIDENTS_DIR
from core.detector import VisionDetector
from core.tracker import TrackManager
from core.action_recognizer import ActionRecognizer
from core.anomaly_detector import AnomalyDetector
from core.event_engine import EventEngine
from scenarios import get_scenario
from utils.visualizer import VisionVisualizer
from utils.report_exporter import ReportExporter

def get_available_cameras(max_probe: int = 3) -> list:
    """Probes camera indices on Windows using DirectShow"""
    available = []
    for idx in range(max_probe):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((idx, f"Camera {idx} ({w}x{h})"))
            cap.release()
    return available

def run_live_webcam_monitor(
    camera_index: int = 0,
    scenario_name: str = "warehouse",
    conf_thresh: float = 0.35,
    loiter_limit: float = 6.0,
    width: int = 1280,
    height: int = 720
):
    print("==================================================================")
    print(" HNX26PSI07: Live Autonomous Vision & Behaviour Monitor")
    print("==================================================================")
    print(f"[*] Probing camera index {camera_index} via DirectShow (Windows)...")

    # Open camera with DirectShow backend on Windows
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        print(f"[!] Warning: Could not open Camera {camera_index} with DirectShow.")
        print(f"[*] Scanning for other available cameras...")
        cams = get_available_cameras()
        if cams:
            print(f"[+] Found {len(cams)} working camera(s):")
            for c_idx, c_label in cams:
                print(f"    - Index {c_idx}: {c_label}")
            print(f"[*] Trying to open Camera {cams[0][0]} instead...")
            cap = cv2.VideoCapture(cams[0][0], backend)
            camera_index = cams[0][0]
        else:
            print("[!] Error: No accessible webcam found. Ensure camera is plugged in and not in use by Zoom/Teams/Browser.")
            return

    # Attempt to request resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 120:
        fps = 30.0

    print(f"[+] Live Camera Connected! Device: {camera_index} | Resolution: {actual_w}x{actual_h}")
    print(f"[*] Scenario: {scenario_name} (Loiter limit: {loiter_limit}s)")
    print(f"[*] Loading YOLOv8-Pose model...")

    scenario = get_scenario(scenario_name, actual_w, actual_h)
    scenario.max_loiter_seconds = loiter_limit

    detector = VisionDetector(
        model_path=DEFAULT_CONFIG.pose_path,
        use_pose=True,
        conf_thresh=conf_thresh
    )
    thresholds = BehaviorThresholds(loitering_time_seconds=loiter_limit)
    tracker = TrackManager()
    action_recognizer = ActionRecognizer(thresholds=thresholds)
    anomaly_detector = AnomalyDetector(scenario=scenario, thresholds=thresholds)
    event_engine = EventEngine(cooldown_seconds=3.0, save_snapshots=True)
    visualizer = VisionVisualizer()

    window_name = f"HNX26PSI07 - Autonomous Vision & Live Monitor [Camera {camera_index}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, actual_w), min(720, actual_h))

    print("\n------------------------------------------------------------------")
    print(" LIVE MONITOR RUNNING. Click the video window for controls:")
    print("   [q] or [ESC] : Stop and save audit report")
    print("   [z]          : Toggle zone display")
    print("   [p]          : Toggle skeletal pose")
    print("   [t]          : Toggle trajectory trails")
    print("   [s]          : Save incident snapshot")
    print("   [1]          : Switch to Warehouse Safety scenario")
    print("   [2]          : Switch to Campus Security scenario")
    print("------------------------------------------------------------------\n")

    frame_idx = 0
    start_time = time.time()
    t_prev = time.time()
    recent_fps = fps

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to grab frame from camera.")
                time.sleep(0.05)
                continue

            t_now = time.time()
            dt = max(0.001, t_now - t_prev)
            t_prev = t_now
            recent_fps = 0.1 * (1.0 / dt) + 0.9 * recent_fps
            timestamp = t_now - start_time
            frame_idx += 1

            # 1. Detection and Tracking
            detections = detector.detect_and_track(frame, persist=True)

            # 2. Update Tracks
            entities = tracker.update_tracks(
                detections=detections,
                frame_idx=frame_idx,
                timestamp=timestamp,
                loiter_radius_px=thresholds.loitering_radius_pixels
            )

            # 3. Action Recognition
            for entity in entities:
                action_recognizer.update_entity_action(entity)

            # 4. Anomaly Evaluation
            entity_flags = {}
            for entity in entities:
                flags = anomaly_detector.evaluate_entity(entity, timestamp)
                if flags:
                    entity_flags[entity.track_id] = flags

            # 5. Event Engine (WHO & WHEN)
            new_events = event_engine.process_frame(
                entities=entities,
                entity_flags=entity_flags,
                frame_idx=frame_idx,
                timestamp=timestamp,
                raw_frame=frame
            )

            for evt in new_events:
                sev_tag = "CRITICAL" if evt.severity == "CRITICAL" else "WARNING"
                print(f"[{evt.timestamp_str}] 🚨 [{sev_tag}] {evt.summary} | Zone: {evt.zone_name}")

            # 6. Render HUD Overlays
            active_events = list(event_engine.active_events.values())
            annotated = visualizer.draw_frame(
                raw_frame=frame,
                entities=entities,
                scenario=scenario,
                active_events=active_events,
                fps=recent_fps
            )

            # Render hotkey help bar at bottom of frame
            h_bar, w_bar = annotated.shape[:2]
            cv2.rectangle(annotated, (0, h_bar - 24), (w_bar, h_bar), (20, 20, 20), -1)
            help_txt = "[Q] Quit & Export  |  [Z] Toggle Zones  |  [P] Toggle Pose  |  [S] Snapshot  |  [1/2] Switch Scenario"
            cv2.putText(annotated, help_txt, (15, h_bar - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(window_name, annotated)

            # Keyboard handler
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 'q' or ESC
                break
            elif key == ord('z'):
                visualizer.show_zones = not visualizer.show_zones
                print(f"[*] Zones overlay: {'ON' if visualizer.show_zones else 'OFF'}")
            elif key == ord('p'):
                visualizer.show_skeleton = not visualizer.show_skeleton
                print(f"[*] Skeleton pose: {'ON' if visualizer.show_skeleton else 'OFF'}")
            elif key == ord('t'):
                visualizer.show_trajectories = not visualizer.show_trajectories
                print(f"[*] Trajectories: {'ON' if visualizer.show_trajectories else 'OFF'}")
            elif key == ord('s'):
                snap_path = INCIDENTS_DIR / f"manual_snap_{int(time.time())}.jpg"
                cv2.imwrite(str(snap_path), annotated)
                print(f"[+] Saved snapshot to: {snap_path}")
            elif key == ord('1'):
                scenario = get_scenario("warehouse", actual_w, actual_h)
                anomaly_detector.scenario = scenario
                print(f"[*] Switched to Scenario: {scenario.name}")
            elif key == ord('2'):
                scenario = get_scenario("campus", actual_w, actual_h)
                anomaly_detector.scenario = scenario
                print(f"[*] Switched to Scenario: {scenario.name}")
            elif key == ord('r'):
                tracker = TrackManager()
                print("[*] Tracking states reset.")

    except KeyboardInterrupt:
        print("\n[!] Monitoring interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Generate incident audit report
    print("\n------------------------------------------------------------------")
    print("[*] Live Session Ended. Generating Incident Audit Log...")
    exporter = ReportExporter(event_engine, scenario)
    t_stamp = int(time.time())
    csv_file = INCIDENTS_DIR / f"live_session_audit_{t_stamp}.csv"
    md_file = INCIDENTS_DIR / f"live_session_audit_{t_stamp}.md"

    exporter.export_csv(str(csv_file))
    exporter.export_markdown_report(str(md_file))

    stats = event_engine.get_summary_stats()
    print(f"[+] Total Incidents Logged: {stats['total_events']} ({stats['critical_count']} Critical, {stats['warning_count']} Warnings)")
    print(f"[+] CSV Incident Log:       {csv_file}")
    print(f"[+] Markdown Audit Report:  {md_file}")
    print("==================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Webcam Behavior Understanding Monitor")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--scenario", type=str, default="warehouse", choices=["warehouse", "campus", "retail"], help="Scenario preset")
    parser.add_argument("--loiter", type=float, default=6.0, help="Loitering threshold in seconds")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    run_live_webcam_monitor(
        camera_index=args.camera,
        scenario_name=args.scenario,
        conf_thresh=args.conf,
        loiter_limit=args.loiter
    )
