"""Direct Depth Map Obstacle Analyzer for Autonomous Avoidance.

Analyzes 3 spatial sectors (Left, Center Corridor, Right) of the depth map:
- Ignores floor plane by cropping out the bottom 30% of the image.
- Computes obstacle proximity levels [0..255] where higher values mean closer objects.
- Evaluates whether the central flight path is open.
- Determines the clearest escape direction (Left vs Right).
"""

from typing import Dict, Any, Optional
import cv2
import numpy as np

from config import (
    DEPTH_OBSTACLE_THRESH,
    DEPTH_CRITICAL_BRAKE,
    DEPTH_SCAN_Y_MAX_PCT,
)


def disp_to_metric(disp_val: float) -> float:
    """Approximate distance in meters from uint8 disparity value."""
    disp_clamped = np.clip(disp_val, 12.0, 248.0)
    return float(0.40 + 3.8 * ((248.0 - disp_clamped) / 236.0) ** 1.32)


class VisualLiDARScanner:
    """Monocular Depth Map Obstacle Analyzer."""

    def __init__(
        self,
        obstacle_thresh: float = DEPTH_OBSTACLE_THRESH,
        brake_thresh: float = DEPTH_CRITICAL_BRAKE,
    ):
        self.obstacle_thresh = obstacle_thresh
        self.brake_thresh = brake_thresh
        self.smooth_dangers = None  # (left, center, right)

    def reset(self):
        self.smooth_dangers = None

    def scan(self, norm_depth_uint8: np.ndarray) -> Dict[str, Any]:
        """Analyze Left, Center, and Right sectors directly from depth map.

        Args:
            norm_depth_uint8: uint8 disparity [0..255] (255=obstacle closest, 0=farthest).

        Returns:
            Dictionary containing sector danger levels, clearance status, and escape vector.
        """
        dh, dw = norm_depth_uint8.shape[:2]

        # 1. Height Crop: Eye-level flight horizon (ignores floor plane and ceiling)
        y1 = int(dh * 0.15)
        y2 = int(dh * (DEPTH_SCAN_Y_MAX_PCT / 100.0))
        depth_roi = norm_depth_uint8[y1:y2, :]

        # 2. Slice 3 horizontal sectors
        w_left_end = int(dw * 0.35)
        w_right_start = int(dw * 0.65)

        roi_left = depth_roi[:, :w_left_end]
        roi_center = depth_roi[:, w_left_end:w_right_start]
        roi_right = depth_roi[:, w_right_start:]

        # 3. 85th percentile: conservative obstacle catch
        raw_l = float(np.percentile(roi_left, 85))
        raw_c = float(np.percentile(roi_center, 85))
        raw_r = float(np.percentile(roi_right, 85))

        # Temporal smoothing with fast-attack on sudden close obstacles
        if self.smooth_dangers is None:
            self.smooth_dangers = np.array([raw_l, raw_c, raw_r], dtype=np.float32)
        else:
            raw_arr = np.array([raw_l, raw_c, raw_r], dtype=np.float32)
            # Fast attack (0.85) when obstacle gets closer (higher disp), smooth release (0.40)
            alpha = np.where(raw_arr > self.smooth_dangers, 0.85, 0.40)
            self.smooth_dangers = alpha * raw_arr + (1.0 - alpha) * self.smooth_dangers

        danger_l, danger_c, danger_r = self.smooth_dangers

        # Convert to approximate meters
        dist_l = disp_to_metric(danger_l)
        dist_c = disp_to_metric(danger_c)
        dist_r = disp_to_metric(danger_r)

        # Clearance state
        is_path_clear = bool(danger_c < self.obstacle_thresh)

        # Safety level
        if danger_c >= self.brake_thresh:
            safety_level = "EMERGENCY_BRAKE"
        elif danger_c >= self.obstacle_thresh:
            safety_level = "CAUTION"
        else:
            safety_level = "SAFE"

        # Clearest direction if center blocked
        steer_dir = "LEFT" if danger_l < danger_r else "RIGHT"

        return {
            "danger_l": float(danger_l),
            "danger_c": float(danger_c),
            "danger_r": float(danger_r),
            "dist_l_m": dist_l,
            "dist_c_m": dist_c,
            "dist_r_m": dist_r,
            "is_path_clear": is_path_clear,
            "safety_level": safety_level,
            "steer_dir": steer_dir,
            "roi_y": (y1, y2),
            "sectors": {
                "left": {"min_m": dist_l},
                "center": {"min_m": dist_c},
                "right": {"min_m": dist_r},
            },
            "closest": {"dist_m": min(dist_l, dist_c, dist_r)},
        }
