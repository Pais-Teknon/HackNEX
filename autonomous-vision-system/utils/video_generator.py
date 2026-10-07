"""
Synthetic & Benchmark Video Generator with Photorealistic Neural Activations
Extracts real human textures and composites them into calibrated scenarios:
- Person #1 (Normal): Walking safely along walkway
- Person #2 (Abnormal - Loitering): Prolonged stationary dwell in emergency corridor (> 8s)
- Person #3 (Abnormal - Unsafe Speed): High-speed running across warehouse walkway
- Person #4 (Abnormal - Intrusion): Unauthorized entry into Forklift Machinery Zone
- Person #5 (Abnormal - Fall Incident): Worker collapses horizontally onto floor
"""

from typing import List, Dict, Tuple, Optional
import os
import math
import json
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import SAMPLE_VIDEOS_DIR

def draw_warehouse_background(width: int = 1280, height: int = 720) -> np.ndarray:
    """Renders a clean industrial warehouse environment"""
    frame = np.full((height, width, 3), (225, 228, 232), dtype=np.uint8)

    # Floor concrete tiles with grid perspective
    for y in range(int(height * 0.25), height, 60):
        cv2.line(frame, (0, y), (width, y), (200, 205, 210), 1)

    # Wall / ceiling boundary
    wall_h = int(height * 0.22)
    cv2.rectangle(frame, (0, 0), (width, wall_h), (180, 185, 192), -1)
    cv2.line(frame, (0, wall_h), (width, wall_h), (140, 145, 150), 3)

    # Warehouse storage racks on top wall
    for rx in range(40, width - 100, 220):
        cv2.rectangle(frame, (rx, 30), (rx + 180, wall_h - 10), (90, 100, 115), -1)
        # Shelving slots
        cv2.rectangle(frame, (rx + 10, 45), (rx + 80, wall_h - 20), (50, 130, 200), -1)
        cv2.rectangle(frame, (rx + 95, 45), (rx + 165, wall_h - 20), (50, 180, 80), -1)

    # Yellow/Black safety caution stripes along machinery zone boundary
    caution_x = 740
    for cy in range(wall_h + 10, height - 30, 30):
        pts = np.array([[caution_x, cy], [caution_x + 15, cy + 15], [caution_x + 15, cy + 25], [caution_x, cy + 10]], np.int32)
        cv2.fillPoly(frame, [pts], (30, 210, 240)) # Yellow
        cv2.line(frame, (caution_x, wall_h + 10), (caution_x, height - 30), (30, 30, 30), 2)

    # Overhead warehouse high-bay lamp glow
    for lx in range(200, width, 300):
        cv2.circle(frame, (lx, 25), 18, (255, 255, 255), -1)
        cv2.circle(frame, (lx, 25), 35, (200, 240, 255), 1)

    # Environmental signage
    cv2.putText(frame, "BAY A - SAFE WALKWAY", (80, wall_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 110, 120), 2)
    cv2.putText(frame, "HAZARD: AUTOMATED FORKLIFTS", (caution_x + 30, wall_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 200), 2)

    return frame

def extract_person_sprites() -> List[np.ndarray]:
    """Extracts photographic human patches from campus_corridor.mp4"""
    src_video = SAMPLE_VIDEOS_DIR / "campus_corridor.mp4"
    sprites = []
    
    if src_video.exists():
        from ultralytics import YOLO
        m = YOLO(str(ROOT_DIR / "models" / "yolov8n.pt"))
        cap = cv2.VideoCapture(str(src_video))
        
        # Sample frames with clear people
        for fid in [40, 200, 360]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret:
                continue
            res = m.predict(frame, conf=0.25, classes=[0], verbose=False)
            boxes = res[0].boxes.xyxy.cpu().numpy()
            for b in boxes:
                x1, y1, x2, y2 = [int(v) for v in b]
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] > 60 and crop.shape[1] > 25:
                    sprites.append(crop)
        cap.release()

    # Fallback to high-contrast realistic silhouette if video not found
    if not sprites:
        blank = np.zeros((180, 70, 3), dtype=np.uint8)
        cv2.circle(blank, (35, 30), 20, (180, 150, 120), -1)
        cv2.rectangle(blank, (15, 55), (55, 120), (50, 150, 220), -1)
        cv2.line(blank, (25, 120), (25, 175), (40, 40, 50), 10)
        cv2.line(blank, (45, 120), (45, 175), (40, 40, 50), 10)
        sprites.append(blank)

    return sprites

def composite_sprite(
    target_frame: np.ndarray,
    sprite: np.ndarray,
    pos: Tuple[int, int],
    scale: float = 1.0,
    rotate_deg: float = 0.0
):
    """Blends sprite onto target frame at given position with optional rotation"""
    h_bg, w_bg = target_frame.shape[:2]
    
    # Scale
    if scale != 1.0:
        new_w = max(10, int(sprite.shape[1] * scale))
        new_h = max(20, int(sprite.shape[0] * scale))
        sprite = cv2.resize(sprite, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Rotate if needed
    if abs(rotate_deg) > 5.0:
        if abs(rotate_deg - 90.0) < 10.0:
            sprite = cv2.rotate(sprite, cv2.ROTATE_90_CLOCKWISE)
        elif abs(rotate_deg + 90.0) < 10.0:
            sprite = cv2.rotate(sprite, cv2.ROTATE_90_COUNTERCLOCKWISE)

    sh, sw = sprite.shape[:2]
    x, y = pos
    
    # Bottom center alignment
    x1 = int(x - sw // 2)
    y1 = int(y - sh)
    x2 = x1 + sw
    y2 = y1 + sh

    # Clip to frame
    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = sw - max(0, x2 - w_bg)
    src_y2 = sh - max(0, y2 - h_bg)

    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(w_bg, x2)
    dst_y2 = min(h_bg, y2)

    if (dst_x2 > dst_x1) and (dst_y2 > dst_y1) and (src_x2 > src_x1) and (src_y2 > src_y1):
        patch = sprite[src_y1:src_y2, src_x1:src_x2]
        # Soft shadow under feet
        cv2.ellipse(target_frame, (int(x), int(y)), (int(sw * 0.45), 8), 0, 0, 360, (50, 50, 60), -1)
        # Direct composite
        target_frame[dst_y1:dst_y2, dst_x1:dst_x2] = patch

def generate_warehouse_benchmark_video(
    output_filename: str = "warehouse_benchmark.mp4",
    duration_seconds: int = 24,
    fps: int = 30,
    width: int = 1280,
    height: int = 720
) -> Tuple[str, str]:
    """Generates benchmark video with photorealistic sprites and ground truth metadata"""
    out_video_path = SAMPLE_VIDEOS_DIR / output_filename
    out_meta_path = SAMPLE_VIDEOS_DIR / f"{Path(output_filename).stem}_ground_truth.json"

    sprites = extract_person_sprites()
    sp_walker = sprites[0 % len(sprites)]
    sp_loiter = sprites[1 % len(sprites)]
    sp_sprinter = sprites[2 % len(sprites)]
    sp_trespass = sprites[3 % len(sprites)]
    sp_fall = sprites[4 % len(sprites)]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    total_frames = duration_seconds * fps
    bg_frame = draw_warehouse_background(width, height)

    ground_truth_events = [
        {
            "entity_id": 1,
            "role": "Normal Walker",
            "behavior": "NORMAL",
            "action": "WALKING",
            "start_time": 0.0,
            "end_time": 22.0
        },
        {
            "entity_id": 2,
            "role": "Loitering Worker",
            "behavior": "ABNORMAL",
            "event_type": "LOITERING",
            "start_time": 7.0,
            "end_time": 24.0,
            "description": "Stationary dwell > 7s in emergency exit corridor"
        },
        {
            "entity_id": 3,
            "role": "Speeding Sprinter",
            "behavior": "ABNORMAL",
            "event_type": "UNSAFE_SPEED",
            "start_time": 8.0,
            "end_time": 15.0,
            "description": "Running in warehouse walkway exceeding safe speed limit"
        },
        {
            "entity_id": 4,
            "role": "Zone Trespasser",
            "behavior": "ABNORMAL",
            "event_type": "RESTRICTED_INTRUSION",
            "start_time": 12.0,
            "end_time": 24.0,
            "description": "Unauthorized worker entry into Forklift Machinery Zone"
        },
        {
            "entity_id": 5,
            "role": "Fall Casualty",
            "behavior": "ABNORMAL",
            "event_type": "FALL_INCIDENT",
            "start_time": 14.0,
            "end_time": 24.0,
            "description": "Worker slipped and collapsed horizontally onto floor"
        }
    ]

    for frame_idx in range(total_frames):
        t = frame_idx / float(fps)
        frame = bg_frame.copy()

        # Entity 1: Normal Walker (starts x=80, moves to x=600 at t=22s)
        if t <= 22.0:
            x1 = int(80 + (t / 22.0) * 520)
            y1 = int(360 + math.sin(t * 1.5) * 5)
            composite_sprite(frame, sp_walker, (x1, y1), scale=1.0)

        # Entity 2: Loitering Worker (walks into corridor by t=3s, then stays still at (220, 520))
        if t <= 3.0:
            x2 = int(100 + (t / 3.0) * 120)
            y2 = int(520)
            composite_sprite(frame, sp_loiter, (x2, y2), scale=1.05)
        else:
            # Stationary loitering in corridor
            composite_sprite(frame, sp_loiter, (220, 520), scale=1.05)

        # Entity 3: Sprinter (runs across screen from t=7s to t=14s)
        if 7.0 <= t <= 14.5:
            rel_t = (t - 7.0) / 7.5
            x3 = int(40 + rel_t * 800)
            y3 = 640
            composite_sprite(frame, sp_sprinter, (x3, y3), scale=1.1)

        # Entity 4: Zone Trespasser (walks from x=580 into x=960 forklift zone)
        if 8.0 <= t <= 24.0:
            rel_t = (t - 8.0) / 16.0
            x4 = int(580 + rel_t * 380)
            y4 = int(320 + rel_t * 100)
            composite_sprite(frame, sp_trespass, (x4, y4), scale=0.95)

        # Entity 5: Fall Casualty (walks from t=5s to t=13.5s, then collapses horizontally)
        if 5.0 <= t:
            if t < 13.5:
                rel_t = (t - 5.0) / 8.5
                x5 = int(320 + rel_t * 160)
                y5 = 480
                composite_sprite(frame, sp_fall, (x5, y5), scale=1.0)
            else:
                # Fallen flat horizontally
                composite_sprite(frame, sp_fall, (480, 510), scale=1.0, rotate_deg=90.0)

        writer.write(frame)

    writer.release()

    metadata = {
        "video_name": output_filename,
        "duration_seconds": duration_seconds,
        "fps": fps,
        "resolution": [width, height],
        "events": ground_truth_events
    }
    with open(out_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return str(out_video_path), str(out_meta_path)

if __name__ == "__main__":
    generate_warehouse_benchmark_video()
