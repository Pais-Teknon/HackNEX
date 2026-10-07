"""
Streamlit Web Application — Autonomous Vision & Behaviour Understanding
Project: HNX26PSI07
Computer Vision · Action Recognition · Object Tracking · Behaviour Analysis
"""

import os
import sys
import time
import tempfile
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Setup sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_CONFIG, BehaviorThresholds, SAMPLE_VIDEOS_DIR, INCIDENTS_DIR, HEX_COLORS
from core.detector import VisionDetector
from core.tracker import TrackManager
from core.action_recognizer import ActionRecognizer, ActionType
from core.anomaly_detector import AnomalyDetector
from core.event_engine import EventEngine
from core.enhancer import VisionEnhancer
from scenarios import get_scenario, SCENARIOS
from utils.visualizer import VisionVisualizer
from utils.report_exporter import ReportExporter
from utils.video_generator import generate_warehouse_benchmark_video

# Streamlit Page Setup
st.set_page_config(
    page_title="Autonomous Vision & Behaviour Understanding",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Modern Theme)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #a0aec0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e2430;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #3498db;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .badge-critical {
        background-color: #e74c3c;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-warning {
        background-color: #f39c12;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-normal {
        background-color: #2ecc71;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Control Center")
st.sidebar.markdown("**HNX26PSI07 Vision Engine**")

# Scenario Selection
scenario_keys = list(SCENARIOS.keys())
scenario_choice = st.sidebar.selectbox(
    "Choose Monitoring Scenario",
    options=scenario_keys,
    format_func=lambda k: SCENARIOS[k][0],
    index=0
)

# Video Input Selection
st.sidebar.markdown("---")
st.sidebar.subheader("📹 Video Source")

# Check for existing sample videos
sample_videos = list(SAMPLE_VIDEOS_DIR.glob("*.mp4"))
sample_names = [v.name for v in sample_videos]

source_mode = st.sidebar.radio(
    "Source Mode",
    options=[
        "Built-in Benchmark & Samples",
        "Upload Video File",
        "Live Hardware Webcam (DirectShow)",
        "📱 IP / Phone Camera (WiFi Stream)",
        "Browser Camera (Direct Snapshot & Test)",
    ],
    index=0,
)

video_path_to_use = None
is_live_stream = False  # True for webcam / IP cam / RTSP (use wall-clock timestamps)

if source_mode == "Built-in Benchmark & Samples":
    if not sample_names:
        st.sidebar.info("Generating warehouse benchmark video...")
        generate_warehouse_benchmark_video()
        sample_videos = list(SAMPLE_VIDEOS_DIR.glob("*.mp4"))
        sample_names = [v.name for v in sample_videos]

    selected_sample = st.sidebar.selectbox("Select Scenario Video", options=sample_names, index=0)
    video_path_to_use = str(SAMPLE_VIDEOS_DIR / selected_sample)

elif source_mode == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI Video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path_to_use = tfile.name
    else:
        st.sidebar.info("Please upload a video file to begin.")

elif source_mode == "Live Hardware Webcam (DirectShow)":
    def get_available_cameras(max_probe=3):
        available = []
        for idx in range(max_probe):
            try:
                backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
                cap_test = cv2.VideoCapture(idx, backend)
                if cap_test.isOpened():
                    ret, _ = cap_test.read()
                    if ret:
                        w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        available.append((idx, f"Camera {idx} ({w}x{h})"))
                    cap_test.release()
            except Exception:
                pass
        return available

    cams = get_available_cameras()
    if cams:
        cam_options = [c[0] for c in cams]
        cam_format = lambda idx: next((c[1] for c in cams if c[0] == idx), f"Camera {idx}")
        camera_idx = st.sidebar.selectbox("Detected Webcams", options=cam_options, format_func=cam_format, index=0)
        st.sidebar.success(f"Connected: {cam_format(camera_idx)}")
    else:
        st.sidebar.warning("No camera automatically detected via DirectShow.")
        camera_idx = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0)

    video_path_to_use = str(camera_idx)
    is_live_stream = True
    st.sidebar.caption("💡 Close other apps (Teams, Zoom, OBS) using the webcam.")

elif source_mode == "📱 IP / Phone Camera (WiFi Stream)":
    st.sidebar.markdown(
        "Connect your **phone camera**, **IP camera**, **RTSP stream**, "
        "or use a **public test stream** below."
    )

    # --- Quick Preset Streams ---
    PRESET_STREAMS = {
        "— Enter Custom URL —": "",
        "🔴 Public: Times Square NYC (EarthCam)": "https://videos3.earthcam.com/fecnetwork/9974.flv/chunklist_w1702498498.m3u8",
        "🔴 Public: Abbey Road London (EarthCam)": "https://videos3.earthcam.com/fecnetwork/AbbeyRoadHD1.flv/chunklist_w136498498.m3u8",
        "🔵 Test: Big Buck Bunny (HTTP MP4)": "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
        "🔵 Test: Sintel Trailer (MP4)": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4",
        "🔵 Test: People Walking (MP4)": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4",
    }

    preset_choice = st.sidebar.selectbox(
        "🎯 Quick Connect (Presets & Test Streams)",
        options=list(PRESET_STREAMS.keys()),
        index=0,
    )

    if preset_choice == "— Enter Custom URL —":
        ip_url = st.sidebar.text_input(
            "📡 Camera Stream URL",
            value="",
            placeholder="http://192.168.1.5:8080/video",
        )
    else:
        ip_url = PRESET_STREAMS[preset_choice]
        st.sidebar.code(ip_url, language=None)

    # URL format helper for custom URLs
    if preset_choice == "— Enter Custom URL —":
        ip_url_format = st.sidebar.selectbox(
            "URL Format Helper",
            options=[
                "Custom URL (use as typed)",
                "IP Webcam (Android) — auto-append /video",
                "IP Webcam (Android) — auto-append /videofeed",
                "DroidCam — auto-append /video",
                "RTSP Generic",
            ],
            index=0,
        )

        with st.sidebar.expander("📖 Setup Instructions", expanded=False):
            st.markdown(
                "**Android (IP Webcam):**\n"
                "1. Install **IP Webcam** from Play Store\n"
                "2. Open → tap **Start Server**\n"
                "3. Paste URL: `http://<IP>:8080/video`\n\n"
                "**Android (DroidCam):**\n"
                "1. Install **DroidCam** from Play Store\n"
                "2. Open → note IP shown\n"
                "3. Paste URL: `http://<IP>:4747/video`\n\n"
                "**iPhone (EpocCam / DroidCam):**\n"
                "1. Install app, start streaming\n"
                "2. Paste the URL shown in app\n\n"
                "**IP/CCTV Camera (RTSP):**\n"
                "- `rtsp://user:pass@<IP>:554/stream`\n\n"
                "**Any MJPEG HTTP stream URL also works.**"
            )
    else:
        ip_url_format = "Custom URL (use as typed)"

    if ip_url:
        url = ip_url.strip()
        if ip_url_format.startswith("IP Webcam") and "/video" in ip_url_format and not url.endswith("/video") and not url.endswith("/videofeed"):
            if "videofeed" in ip_url_format:
                url = url.rstrip("/") + "/videofeed"
            else:
                url = url.rstrip("/") + "/video"
        elif ip_url_format.startswith("DroidCam") and not url.endswith("/video"):
            url = url.rstrip("/") + "/video"

        video_path_to_use = url
        is_live_stream = True
        st.sidebar.success(f"📡 Ready: `{url}`")
    else:
        st.sidebar.warning("Select a preset or enter a stream URL to begin.")

# Threshold Tuning
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Behavior Thresholds")
loiter_time_threshold = st.sidebar.slider("Loitering Alert Limit (seconds)", min_value=3.0, max_value=30.0, value=7.0, step=1.0)
speed_limit_threshold = st.sidebar.slider("Unsafe Running Speed (px/s)", min_value=60.0, max_value=250.0, value=125.0, step=5.0)
conf_threshold = st.sidebar.slider("Detection Confidence", min_value=0.20, max_value=0.80, value=0.35, step=0.05)

# Advanced Processing
st.sidebar.markdown("---")
st.sidebar.subheader("🌙 Advanced Processing")
enable_night_vision = st.sidebar.checkbox("Enable Night Vision (Low Light)", value=False)
nv_green_tint = st.sidebar.checkbox("Apply Green Night-Vision Tint", value=False)
nv_thermal = st.sidebar.checkbox("Apply Heat Radar (Thermal)", value=False)

# Visual Overlays Toggles
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Visual HUD Overlays")
show_zones = st.sidebar.checkbox("Show Zone Boundaries", value=True)
show_trajectories = st.sidebar.checkbox("Show Motion Trajectories", value=True)
show_skeleton = st.sidebar.checkbox("Show Skeletal Pose (17 Keypoints)", value=True)
show_banner = st.sidebar.checkbox("Show Real-time Top Alert Banner", value=True)

# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Autonomous Vision & Behaviour Understanding</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Computer Vision · Action Recognition · Object Tracking · Behaviour Analysis (HNX26PSI07)</div>', unsafe_allow_html=True)

# Scenario Description Callout
scenario_meta = SCENARIOS[scenario_choice]
st.info(f"**Active Scenario**: {scenario_meta[0]} — Real-time tracking, pose kinematics, spatial zone compliance, and temporal anomaly detection.")

# Metrics Row
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
metric_entities = m_col1.empty()
metric_normal = m_col2.empty()
metric_anomalies = m_col3.empty()
metric_fps = m_col4.empty()

metric_entities.metric("Tracked Entities", "0")
metric_normal.metric("Normal Behavior", "0", delta="Compliant")
metric_anomalies.metric("Active Violations", "0", delta="Safe", delta_color="normal")
metric_fps.metric("Inference Speed", "0.0 FPS")

st.markdown("---")

if source_mode == "Browser Camera (Direct Snapshot & Test)":
    st.subheader("📷 Browser Camera Live Inspection")
    st.caption("Capture a live frame directly through your browser webcam to test action recognition, 17-point pose skeleton, and zone violations.")
    
    col_cam, col_cam_feed = st.columns([1.5, 1.0])
    with col_cam:
        cam_picture = st.camera_input("Capture Camera Frame")
        if cam_picture is not None:
            bytes_data = cam_picture.getvalue()
            raw_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            h, w = raw_bgr.shape[:2]

            scenario = get_scenario(scenario_choice, w, h)
            detector = VisionDetector(model_path=DEFAULT_CONFIG.pose_path, use_pose=True, conf_thresh=conf_threshold)
            tracker = TrackManager()
            action_recognizer = ActionRecognizer()
            anomaly_detector = AnomalyDetector(scenario=scenario)
            event_engine = EventEngine()
            visualizer = VisionVisualizer(
                show_zones=show_zones,
                show_skeleton=show_skeleton,
                show_trajectories=show_trajectories,
                show_alert_banner=show_banner
            )
            
            if enable_night_vision:
                enhancer = VisionEnhancer(mode='clahe', tint_green=nv_green_tint, thermal=nv_thermal)
                raw_bgr = enhancer.enhance(raw_bgr)

            detections = detector.detect_and_track(raw_bgr, persist=False)
            entities = tracker.update_tracks(detections, 1, 0.0)
            for entity in entities:
                action_recognizer.update_entity_action(entity)
            entity_flags = {}
            for entity in entities:
                flags = anomaly_detector.evaluate_entity(entity, 0.0)
                if flags:
                    entity_flags[entity.track_id] = flags
            event_engine.process_frame(entities, entity_flags, 1, 0.0, raw_bgr)

            annotated = visualizer.draw_frame(raw_bgr, entities, scenario, list(event_engine.active_events.values()), fps=30.0)
            rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(rgb_annotated, caption="Autonomous Vision Analysis Result", width="stretch")

            with col_cam_feed:
                st.subheader("🚨 Incident & Behavior Analysis")
                if entities:
                    for e in entities:
                        st.markdown(f"**Entity {e.track_id}**: `{e.current_action}` ({e.severity})")
                
                df_evts = event_engine.get_dataframe()
                if not df_evts.empty:
                    st.dataframe(df_evts[["entity_id", "severity", "action", "summary"]], width="stretch", hide_index=True)
                else:
                    st.success("✅ Compliant: No violations or anomalies detected in captured frame.")

    start_processing = False
else:
    # Main Content Layout (Video Stream on Left, Live Event Feed on Right)
    col_video, col_feed = st.columns([1.5, 1.0])

    with col_video:
        st.subheader("📺 Live Vision Analytics Stream")
        video_placeholder = st.empty()
        run_btn_col1, run_btn_col2 = st.columns(2)
        start_processing = run_btn_col1.button("▶️ Start Autonomous Analysis", width="stretch", type="primary")
        stop_processing = run_btn_col2.button("⏹️ Stop Stream", width="stretch")

    with col_feed:
        st.subheader("🚨 Incident Event Feed (Who & When)")
        st.caption("Satisfies Rule: Flags point directly to entity ID, timestamp, action & reason.")
        feed_placeholder = st.empty()

# -----------------------------------------------------------------------------
# PROCESSING LOOP
# -----------------------------------------------------------------------------
if start_processing and video_path_to_use:
    is_webcam_idx = video_path_to_use.isdigit()
    is_network_stream = video_path_to_use.startswith(("http://", "https://", "rtsp://", "rtsps://"))

    # Open the video source
    use_urllib_fallback = False
    snapshot_url = ""
    
    if is_webcam_idx:
        source_val = int(video_path_to_use)
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(source_val, backend)
    elif is_network_stream:
        # For IP cameras / RTSP: use FFMPEG backend with buffering tweaks
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000000|rtsp_transport;tcp"
        cap = cv2.VideoCapture(video_path_to_use, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            # Retry once with default backend
            cap = cv2.VideoCapture(video_path_to_use)
            
        # If OpenCV fails on an HTTP stream (common for IP Webcam MJPEG on Windows), use urllib snapshot fallback
        if not cap.isOpened() and video_path_to_use.startswith("http"):
            use_urllib_fallback = True
            if video_path_to_use.endswith("/video"):
                snapshot_url = video_path_to_use.replace("/video", "/shot.jpg")
            elif video_path_to_use.endswith("/videofeed"):
                snapshot_url = video_path_to_use.replace("/videofeed", "/shot.jpg")
            else:
                snapshot_url = video_path_to_use
    else:
        cap = cv2.VideoCapture(video_path_to_use)

    # Validate stream is working (either via OpenCV or urllib)
    stream_ok = False
    if use_urllib_fallback:
        try:
            req = urllib.request.urlopen(snapshot_url, timeout=3.0)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            test_img = cv2.imdecode(arr, -1)
            if test_img is not None:
                stream_ok = True
                width = test_img.shape[1]
                height = test_img.shape[0]
                fps = 10.0 # Approximate for snapshot fetching
        except Exception:
            stream_ok = False
    else:
        stream_ok = cap.isOpened()
        if stream_ok:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0 or fps > 120:
                fps = 30.0

    if not stream_ok:
        if is_network_stream:
            st.error(
                f"❌ Failed to connect to network camera: `{video_path_to_use}`\n\n"
                "**Troubleshooting:**\n"
                "- Ensure your phone/IP camera app is running and streaming\n"
                "- Verify both devices are on the **same WiFi network**\n"
                "- Check the URL is correct (try opening it in a browser)\n"
                "- For IP Webcam (Android): URL should be `http://<IP>:8080/video`\n"
                "- For DroidCam: URL should be `http://<IP>:4747/video`\n"
                "- Firewall may be blocking the connection"
            )
        else:
            st.error(f"Failed to open video source: {video_path_to_use}. If using webcam, please ensure no other app (Teams, Zoom, browser) has locked the camera.")
    else:
        if is_network_stream:
            if use_urllib_fallback:
                st.success(f"📡 Connected to network camera via snapshot fallback: `{snapshot_url}`")
            else:
                st.success(f"📡 Connected to network camera: `{video_path_to_use}`")

        # Scenario & Pipeline Instantiation
        scenario = get_scenario(scenario_choice, width, height)
        scenario.max_loiter_seconds = loiter_time_threshold
        scenario.max_safe_speed = speed_limit_threshold

        thresholds = BehaviorThresholds(
            loitering_time_seconds=loiter_time_threshold,
            running_min_speed=speed_limit_threshold
        )

        detector = VisionDetector(
            model_path=DEFAULT_CONFIG.pose_path,
            use_pose=True,
            conf_thresh=conf_threshold
        )
        tracker = TrackManager()
        action_recognizer = ActionRecognizer(thresholds=thresholds)
        anomaly_detector = AnomalyDetector(scenario=scenario, thresholds=thresholds)
        event_engine = EventEngine(cooldown_seconds=3.0, save_snapshots=True)
        visualizer = VisionVisualizer(
            show_zones=show_zones,
            show_trajectories=show_trajectories,
            show_skeleton=show_skeleton,
            show_alert_banner=show_banner
        )

        frame_idx = 0
        loop_start_time = time.time()
        recent_fps = 0.0
        consecutive_failures = 0  # Track network stream read failures

        # Process frames
        while stream_ok:
            if use_urllib_fallback:
                try:
                    req = urllib.request.urlopen(snapshot_url, timeout=3.0)
                    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                    frame = cv2.imdecode(arr, -1)
                    ret = frame is not None
                except Exception:
                    ret = False
            else:
                ret, frame = cap.read()
                
            if not ret:
                if is_live_stream and consecutive_failures < 30:
                    # Network streams can drop frames; retry
                    consecutive_failures += 1
                    time.sleep(0.05)
                    continue
                break
            if stop_processing:
                break
            consecutive_failures = 0

            timestamp = (time.time() - loop_start_time) if is_live_stream else (frame_idx / fps)
            frame_idx += 1
            t0 = time.time()
            
            if enable_night_vision:
                enhancer = VisionEnhancer(mode='clahe', tint_green=nv_green_tint, thermal=nv_thermal)
                frame = enhancer.enhance(frame)

            # 1. Detection and Tracking
            detections = detector.detect_and_track(frame, persist=True)

            # 2. Track Management
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

            # 5. Event Engine
            event_engine.process_frame(
                entities=entities,
                entity_flags=entity_flags,
                frame_idx=frame_idx,
                timestamp=timestamp,
                raw_frame=frame
            )

            # 6. Visualization Overlay
            active_events = list(event_engine.active_events.values())
            annotated = visualizer.draw_frame(
                raw_frame=frame,
                entities=entities,
                scenario=scenario,
                active_events=active_events,
                fps=recent_fps
            )

            # Update FPS calculation
            t1 = time.time()
            frame_time = t1 - t0
            recent_fps = 0.2 * (1.0 / max(0.001, frame_time)) + 0.8 * recent_fps

            # Update Live Metrics
            n_entities = len(entities)
            n_anomalies = sum(1 for e in entities if e.is_anomalous)
            n_normal = n_entities - n_anomalies

            metric_entities.metric("Tracked Entities", str(n_entities))
            metric_normal.metric("Normal Behavior", str(n_normal))
            metric_anomalies.metric(
                "Active Violations",
                str(n_anomalies),
                delta=f"{n_anomalies} Alert(s)" if n_anomalies > 0 else "Safe",
                delta_color="inverse" if n_anomalies > 0 else "normal"
            )
            metric_fps.metric("Inference Speed", f"{recent_fps:.1f} FPS")

            # Stream Video Frame to Streamlit
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", width="stretch")

            # Update Event Feed on the Right
            with feed_placeholder.container():
                # --- LIVE ENTITY STATUS (always shows every visible person) ---
                live_cards = event_engine.get_live_entity_feed(entities, timestamp)
                if live_cards:
                    st.markdown("**🟢 Live Entity Status**")
                    for card in live_cards:
                        sev = card["severity"]
                        if sev == "CRITICAL":
                            icon, color = "🔴", "red"
                        elif sev == "WARNING":
                            icon, color = "🟡", "orange"
                        else:
                            icon, color = "🟢", "green"
                        st.markdown(
                            f"{icon} **{card['entity_id']}** — `{card['action']}` | "
                            f"Severity: **:{color}[{sev}]** | {card['zone']}  \n"
                            f"_{card['summary']}_  \n"
                            f"⏱ {card['timestamp']} | Speed: {card['speed']} | Dwell: {card['dwell']}"
                        )
                    st.markdown("---")
                else:
                    st.info("No entities detected in frame.")

                # --- EVENT HISTORY LOG (anomalies + action transitions) ---
                df_events = event_engine.get_dataframe()
                if not df_events.empty:
                    st.markdown("**📋 Recent Event History**")
                    display_df = df_events.tail(15)[["timestamp", "entity_id", "severity", "action", "summary"]].iloc[::-1]
                    st.dataframe(
                        display_df,
                        column_config={
                            "timestamp": st.column_config.TextColumn("When"),
                            "entity_id": st.column_config.TextColumn("Who"),
                            "severity": st.column_config.TextColumn("Severity"),
                            "action": st.column_config.TextColumn("Action"),
                            "summary": st.column_config.TextColumn("Event Summary"),
                        },
                        width="stretch",
                        hide_index=True
                    )

        cap.release()
        st.success("✅ Video stream analysis completed!")

        # ---------------------------------------------------------------------
        # AUDIT REPORT & DEEP-DIVE INSPECTION
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Post-Analysis Incident Audit & Intelligence")
        
        tab_log, tab_inspector, tab_export = st.tabs(["📋 Complete Event Log", "🔍 Entity Deep Dive", "📥 Export Reports"])
        
        with tab_log:
            df_all = event_engine.get_dataframe()
            if not df_all.empty:
                st.dataframe(df_all, width="stretch")
            else:
                st.write("No incidents occurred during this monitoring session.")

        with tab_inspector:
            if tracker.entities:
                entity_options = list(tracker.entities.keys())
                sel_id = st.selectbox("Select Tracked Entity ID", options=entity_options, format_func=lambda x: f"Person #{x}")
                ent = tracker.entities[sel_id]
                
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Total Tracked Duration:** {ent.total_active_time:.1f}s")
                c2.write(f"**Current Action:** {ent.current_action}")
                c3.write(f"**Status:** {ent.severity}")

                # Trajectory Points
                pts = list(ent.history)
                if pts:
                    df_traj = pd.DataFrame(pts, columns=["x", "y", "time"])
                    st.line_chart(df_traj.set_index("time")["x"])
            else:
                st.info("No tracked entities recorded.")

        with tab_export:
            exporter = ReportExporter(event_engine, scenario)
            t_now = int(time.time())
            csv_path = INCIDENTS_DIR / f"report_{t_now}.csv"
            md_path = INCIDENTS_DIR / f"report_{t_now}.md"
            json_path = INCIDENTS_DIR / f"report_{t_now}.json"

            exporter.export_csv(str(csv_path))
            exporter.export_markdown_report(str(md_path))
            exporter.export_json(str(json_path))

            with open(csv_path, "r") as f:
                csv_data = f.read()
            with open(md_path, "r", encoding="utf-8") as f:
                md_data = f.read()
            with open(json_path, "r") as f:
                json_data = f.read()

            c1, c2, c3 = st.columns(3)
            c1.download_button("📥 Download CSV Incident Log", data=csv_data, file_name=f"incidents_{t_now}.csv", mime="text/csv")
            c2.download_button("📥 Download Markdown Audit Report", data=md_data, file_name=f"audit_report_{t_now}.md", mime="text/markdown")
            c3.download_button("📥 Download JSON Full Audit Trace", data=json_data, file_name=f"audit_trace_{t_now}.json", mime="application/json")
