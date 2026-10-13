"""Centralized Configuration for Tello Vision Autopilot.

Depth map obstacle avoidance & Person silhouette tracking.
"""

from pathlib import Path
import os


def get_safe_path(path: Path) -> str:
    """Return an ASCII/8.3 safe path for OpenCV/ONNX C++ APIs on Windows with non-ASCII usernames."""
    if os.name == "nt":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(500)
            parent_dir = str(path.parent.resolve())
            if ctypes.windll.kernel32.GetShortPathNameW(parent_dir, buf, 500):
                return os.path.join(buf.value, path.name)
        except Exception:
            pass
    return str(path.resolve())


# Project base paths
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
DA2_CHECKPOINT = CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"
YOLO_MODEL_PATH = BASE_DIR / "yolov8n.pt"

# Depth Map Obstacle Avoidance (disparity values 0..255, where 255 = nearest)
DEPTH_OBSTACLE_THRESH = 150      # If center depth > 150, obstacle ahead -> steer away
DEPTH_CRITICAL_BRAKE = 195       # Point-blank obstacle -> halt immediately
DEPTH_SCAN_Y_MAX_PCT = 70        # Cut bottom 30% of frame to completely ignore floor

# Flight Speeds (-100..100)
EXPLORE_FORWARD_SPEED = 20       # Forward cruise speed when path is clear
EXPLORE_AVOID_YAW_SPEED = 28     # Reactive turning speed to steer away from obstacles
MANUAL_SPEED = 35

# Person Silhouette Tracking
PERSON_TARGET_AREA = 0.12        # Target silhouette area ratio (approx 1.5 - 2 meters distance)
PERSON_FOLLOW_SPEED = 18         # Speed moving towards person
PERSON_TRACK_YAW_LIMIT = 35      # Max yaw turning rate when tracking person

# Altitude limits (ToF sensor in cm)
TARGET_ALTITUDE_CM = 70
MIN_ALTITUDE_CM = 45
MAX_ALTITUDE_CM = 110

# AI Models resolution
DEPTH_INPUT_SIZE = 252
DEPTH_EMA_ALPHA = 0.80

WINDOW_TITLE = "Tello Vision: Depth Obstacle Avoidance & Silhouette Following"
