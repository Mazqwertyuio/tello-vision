"""Global Configuration for Tello Vision Autopilot.

Contains centralized parameters for LiDAR, Navigation, Follower, Route Tracking, and Safety Shield.
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
YUNET_MODEL = BASE_DIR / "face_detection_yunet_2023mar.onnx"

# Visual LiDAR settings: Thin scan line strictly in the UPPER THIRD of the frame
LIDAR_FOV_DEG = 82.6                 # Tello camera horizontal Field of View
LIDAR_NUM_RAYS = 90                  # Resolution of LiDAR radial sweep
LIDAR_SCAN_LINE_Y_PCT = 33.0         # Upper third horizontal line (eye-level horizon)
LIDAR_SCAN_LINE_HALF_THICK_PCT = 1.5 # Thin band (e.g. 31.5% - 34.5% height)

# Safety & Clearance thresholds (in meters)
LIDAR_BRAKE_DIST_M = 0.85            # Critical boundary -> stop/avoid collision
LIDAR_CAUTION_DIST_M = 1.35          # Slowdown boundary
LIDAR_MAX_RANGE_M = 4.5              # Maximum range in meters
MAZE_CLEAR_DIST_M = 1.20             # Threshold: if center distance > 1.20m, path is OPEN

# Speeds & Maze exploration parameters
MANUAL_SPEED = 35
EXPLORE_FORWARD_SPEED = 20           # Forward speed when path is clear
MAZE_TURN_YAW_SPEED = 32             # Yaw speed for reliable ~10-15° turn pulses
MAZE_STEP_PULSE_DURATION = 0.42      # Pulse duration to rotate ~10-15 degrees
MAZE_CHECK_PAUSE_DURATION = 0.25     # Settle time between turn steps to scan depth

# Person Follower parameters
FOLLOW_TARGET_AREA = 0.05            # Target face area ratio in frame
FOLLOW_SPEED_GAIN = 60               # Responsiveness
FOLLOW_MAX_SPEED = 30
FOLLOW_TRACK_YAW_LIMIT = 40
FOLLOW_SEARCH_YAW_SPEED = 24
FOLLOW_AREA_FILTER_ALPHA = 0.75

# Altitude control (ToF sensor in cm)
TARGET_ALTITUDE_CM = 70              # Ideal eye-level cruising height
MIN_ALTITUDE_CM = 45                 # Too close to floor -> ascend
MAX_ALTITUDE_CM = 110                # Too close to ceiling -> descend

# Neural network settings
DEPTH_INPUT_SIZE = 252               # Real-time DA2 resolution
DEPTH_EMA_ALPHA = 0.80               # Temporal smoothing
FACE_DETECTION_SCALE = 0.5           # Downscale factor for face detection
FACE_CONFIDENCE_THRESH = 0.65

# GUI Settings
WINDOW_TITLE = "Tello Vision: Thin-Line LiDAR, Maze Explorer & Route Tracker"
HUD_RADAR_SIZE = 240
