"""
Configuration settings for Autonomous Vision & Behaviour Understanding System
(HNX26PSI07)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"
INCIDENTS_DIR = DATA_DIR / "incidents"

# Create directories if they don't exist
SAMPLE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
DEFAULT_DETECTOR_MODEL = str(MODELS_DIR / "yolov8n.pt")
DEFAULT_POSE_MODEL = str(MODELS_DIR / "yolov8n-pose.pt")

# Display / Visualizer Colors (BGR format for OpenCV)
COLORS = {
    "NORMAL": (60, 200, 70),       # Green
    "WARNING": (30, 170, 240),      # Amber/Yellow
    "CRITICAL": (50, 50, 240),      # Red
    "INFO": (220, 180, 50),         # Cyan
    "ZONE_SAFE": (80, 180, 80),     # Soft Green
    "ZONE_RESTRICTED": (50, 50, 220), # Soft Red
    "ZONE_WATCH": (40, 180, 220),   # Soft Yellow
    "TRACK_PATH": (200, 200, 200),  # Light Gray
    "TEXT": (255, 255, 255),        # White
    "BOX_BG": (30, 30, 30),         # Dark Gray
}

# Hex colors for web dashboard
HEX_COLORS = {
    "NORMAL": "#2ecc71",
    "WARNING": "#f39c12",
    "CRITICAL": "#e74c3c",
    "INFO": "#3498db",
    "MUTED": "#95a5a6",
    "CARD_BG": "#1e222d",
}

@dataclass
class BehaviorThresholds:
    """Thresholds for action recognition and anomaly scoring"""
    # Velocity thresholds (in pixels/second, scaled to video resolution)
    standing_max_speed: float = 25.0
    walking_max_speed: float = 120.0
    running_min_speed: float = 130.0
    erratic_turn_angle_deg: float = 110.0 # Direction reversal threshold
    
    # Temporal thresholds (in seconds)
    loitering_time_seconds: float = 8.0   # Default alert after 8s stationary in watch zone
    loitering_radius_pixels: float = 45.0 # Max radius movement to still count as loitering
    fall_confidence_frames: int = 5      # Consecutive frames to confirm a fall
    zone_dwell_warning_seconds: float = 3.0 # Quick trigger for entering restricted zone
    
    # Posture aspect ratio & keypoint thresholds
    fall_aspect_ratio_threshold: float = 0.85 # Width / Height > 0.85 indicates horizontal lying
    crouch_height_compression_ratio: float = 0.65
    
    # Event cooldown (seconds between repeated logs of same anomaly for same person)
    event_cooldown_seconds: float = 4.0

@dataclass
class SystemConfig:
    """Overall system configuration"""
    model_type: str = "pose" # "pose" or "box"
    detector_path: str = DEFAULT_DETECTOR_MODEL
    pose_path: str = DEFAULT_POSE_MODEL
    confidence_threshold: float = 0.40
    iou_threshold: float = 0.50
    device: str = "cpu"
    tracker_type: str = "bytetrack"
    trail_length: int = 40
    thresholds: BehaviorThresholds = field(default_factory=BehaviorThresholds)

DEFAULT_CONFIG = SystemConfig()
