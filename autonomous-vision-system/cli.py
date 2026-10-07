"""
Command Line Interface for Autonomous Vision & Behaviour Understanding
Usage:
    python cli.py --video data/sample_videos/warehouse_benchmark.mp4 --scenario warehouse --output output.mp4
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure root directory is on sys.path
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

def process_video(
    video_source: str,
    scenario_name: str = "warehouse",
    output_path: str = None,
    use_pose: bool = True,
    conf_thresh: float = 0.35,
    loiter_limit: float = None,
    max_frames: int = None,
    headless: bool = True
):
    print(f"============================================================")
    print(f" HNX26PSI07: Autonomous Vision & Behaviour Understanding")
    print(f"============================================================")
    print(f"[*] Input Source: {video_source}")
    print(f"[*] Selected Scenario: {scenario_name}")
    print(f"[*] Pose Model: {'Enabled (17 Skeletal Keypoints)' if use_pose else 'Disabled (BBox only)'}")

    # Initialize video capture
    is_webcam = video_source.isdigit()
    source_val = int(video_source) if is_webcam else video_source
    if is_webcam:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(source_val, backend)
    else:
        cap = cv2.VideoCapture(source_val)

    if not cap.isOpened():
        print(f"[!] Error: Could not open video source {video_source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps > 120:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
    print(f"[*] Video Dimensions: {width}x{height} @ {fps:.1f} FPS (Total Frames: {total_frames})")

    # Initialize scenario
    scenario = get_scenario(scenario_name, width, height)
    if loiter_limit is not None:
        scenario.max_loiter_seconds = loiter_limit
    print(f"[*] Active Scenario: '{scenario.name}' with {len(scenario.zones)} configured zones")

    # Initialize models
    model_path = DEFAULT_CONFIG.pose_path if use_pose else DEFAULT_CONFIG.detector_path
    print(f"[*] Loading model from {model_path} ...")
    detector = VisionDetector(model_path=model_path, use_pose=use_pose, conf_thresh=conf_thresh)
    
    thresholds = BehaviorThresholds()
    if loiter_limit is not None:
        thresholds.loitering_time_seconds = loiter_limit

    tracker = TrackManager()
    action_recognizer = ActionRecognizer(thresholds=thresholds)
    anomaly_detector = AnomalyDetector(scenario=scenario, thresholds=thresholds)
    event_engine = EventEngine(cooldown_seconds=3.0, save_snapshots=True)
    visualizer = VisionVisualizer()

    # Video Writer if output is specified
    writer = None
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))
        print(f"[*] Writing annotated video to: {out_p}")

    frame_idx = 0
    start_time = time.time()
    last_print_time = start_time

    print(f"[*] Beginning vision analysis stream...")
    print(f"------------------------------------------------------------")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = (time.time() - start_time) if is_webcam else (frame_idx / fps)
            frame_idx += 1

            # 1. Detection and Tracking
            detections = detector.detect_and_track(frame, persist=True)

            # 2. Entity Management
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

            # 5. Event Engine & Incident Logging
            new_events = event_engine.process_frame(
                entities=entities,
                entity_flags=entity_flags,
                frame_idx=frame_idx,
                timestamp=timestamp,
                raw_frame=frame
            )

            # Surface new events to console in real-time
            for evt in new_events:
                sev_icon = "🚨 CRITICAL" if evt.severity == "CRITICAL" else "⚠️ WARNING"
                print(f"[{evt.timestamp_str}] {sev_icon} | {evt.summary} | Zone: {evt.zone_name}")

            # 6. Visualization & Rendering
            active_events = list(event_engine.active_events.values())
            annotated_frame = visualizer.draw_frame(
                raw_frame=frame,
                entities=entities,
                scenario=scenario,
                active_events=active_events,
                fps=fps
            )

            if writer:
                writer.write(annotated_frame)

            # Progress logging every 3 seconds
            curr_time = time.time()
            if curr_time - last_print_time >= 3.0:
                elapsed = curr_time - start_time
                proc_fps = frame_idx / max(0.001, elapsed)
                pct = f"({(frame_idx / total_frames * 100):.1f}%)" if total_frames > 0 else ""
                print(f"[*] Processing: Frame {frame_idx}/{total_frames} {pct} | Speed: {proc_fps:.1f} FPS | Active Tracks: {len(entities)}")
                last_print_time = curr_time

            if max_frames and frame_idx >= max_frames:
                print(f"[*] Reached max_frames limit ({max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\n[!] Processing interrupted by user.")
    finally:
        cap.release()
        if writer:
            writer.release()

    total_time = time.time() - start_time
    avg_fps = frame_idx / max(0.001, total_time)
    print(f"------------------------------------------------------------")
    print(f"[*] Analysis Complete in {total_time:.2f}s ({avg_fps:.1f} avg FPS, {frame_idx} frames processed)")

    # Generate incident report
    exporter = ReportExporter(event_engine, scenario)
    stats = event_engine.get_summary_stats()
    print(f"[*] Incident Audit Summary:")
    print(f"    - Total Events Logged: {stats['total_events']}")
    print(f"    - Critical Incidents:  {stats['critical_count']}")
    print(f"    - Warnings:            {stats['warning_count']}")

    report_dir = INCIDENTS_DIR
    csv_file = report_dir / f"audit_report_{int(time.time())}.csv"
    md_file = report_dir / f"audit_report_{int(time.time())}.md"
    json_file = report_dir / f"audit_report_{int(time.time())}.json"
    
    exporter.export_csv(str(csv_file))
    exporter.export_markdown_report(str(md_file))
    exporter.export_json(str(json_file))

    print(f"[+] Audit CSV:  {csv_file}")
    print(f"[+] Audit MD:   {md_file}")
    print(f"[+] Audit JSON: {json_file}")
    print(f"============================================================")

def main():
    parser = argparse.ArgumentParser(description="Autonomous Vision & Behaviour Understanding System")
    parser.add_argument("--video", type=str, default="data/sample_videos/warehouse_benchmark.mp4", help="Path to video or camera index")
    parser.add_argument("--scenario", type=str, default="warehouse", choices=["warehouse", "campus", "retail"], help="Target scenario preset")
    parser.add_argument("--output", type=str, default="output_annotated.mp4", help="Path to save annotated output video")
    parser.add_argument("--no-pose", action="store_true", help="Disable pose estimation (bbox only)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--loiter-limit", type=float, default=None, help="Loitering threshold in seconds")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to process")
    args = parser.parse_args()

    process_video(
        video_source=args.video,
        scenario_name=args.scenario,
        output_path=args.output,
        use_pose=not args.no_pose,
        conf_thresh=args.conf,
        loiter_limit=args.loiter_limit,
        max_frames=args.max_frames,
        headless=True
    )

if __name__ == "__main__":
    main()
