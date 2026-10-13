"""Thin-Line Visual LiDAR Scanner in the Upper Third of the Frame.

Extracts a horizontal LiDAR scan line strictly located at the upper third
of the depth image (eye-level horizon):
- Completely immune to floor carpet/rugs, ground reflections, and ceilings.
- Directly determines whether the forward corridor is clear or blocked.
- Provides Left, Center, and Right sector distances.
- Supplies line Y-coordinates for visual laser HUD overlay.
"""

import math
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

from config import (
    LIDAR_FOV_DEG,
    LIDAR_NUM_RAYS,
    LIDAR_SCAN_LINE_Y_PCT,
    LIDAR_SCAN_LINE_HALF_THICK_PCT,
    LIDAR_BRAKE_DIST_M,
    LIDAR_CAUTION_DIST_M,
    LIDAR_MAX_RANGE_M,
    MAZE_CLEAR_DIST_M,
)


class VisualLiDARScanner:
    """Thin-line horizontal LiDAR scanner across the camera horizon."""

    def __init__(
        self,
        fov_deg: float = LIDAR_FOV_DEG,
        num_rays: int = LIDAR_NUM_RAYS,
        brake_dist_m: float = LIDAR_BRAKE_DIST_M,
        caution_dist_m: float = LIDAR_CAUTION_DIST_M,
        clear_dist_m: float = MAZE_CLEAR_DIST_M,
        depth_scale_factor: float = 3.8,
    ):
        self.fov_deg = fov_deg
        self.fov_rad = math.radians(fov_deg)
        self.num_rays = num_rays
        self.brake_dist_m = brake_dist_m
        self.caution_dist_m = caution_dist_m
        self.clear_dist_m = clear_dist_m
        self.depth_scale_factor = depth_scale_factor

        self.smoothed_dists = None
        self.alpha_smooth = 0.70

    def reset(self):
        """Reset temporal smoothing."""
        self.smoothed_dists = None

    def scan(self, norm_depth_uint8: np.ndarray) -> Dict[str, Any]:
        """Scan horizontal LiDAR line strictly along the upper third of the depth image.

        Args:
            norm_depth_uint8: uint8 disparity [0..255] (255=closest, 0=farthest).

        Returns:
            Dictionary containing:
                rays: list of dicts {angle_deg, dist_m, x_m, y_m}
                sectors: {left, center, right} with min_m and mean_m
                closest: {dist_m, angle_deg, x_m, y_m}
                is_path_clear: bool (True if forward corridor > clear_dist_m)
                recommended_heading_deg: angle towards deepest open passage
                safety_level: 'SAFE', 'CAUTION', or 'EMERGENCY_BRAKE'
                scan_line_y: integer Y pixel index in the frame for HUD drawing
        """
        dh, dw = norm_depth_uint8.shape[:2]

        # 1. THIN SCAN LINE in the UPPER THIRD (~33% height)
        y_center = int(dh * (LIDAR_SCAN_LINE_Y_PCT / 100.0))
        half_t = max(2, int(dh * (LIDAR_SCAN_LINE_HALF_THICK_PCT / 100.0)))
        y1 = max(0, y_center - half_t)
        y2 = min(dh, y_center + half_t)

        depth_line = norm_depth_uint8[y1:y2, :]

        # Max / 75th percentile disparity across the thin strip: reliable obstacle catch
        col_disp = np.percentile(depth_line, 75, axis=0)

        # Smooth horizontally with 1D Gaussian kernel to suppress single-pixel sensor noise
        ksize = max(5, int(dw * 0.05))
        if ksize % 2 == 0:
            ksize += 1
        col_disp_smooth = cv2.GaussianBlur(
            col_disp.astype(np.float32), (ksize, 1), 0
        ).ravel()

        # 2. Sample radial rays across the FOV
        step_col = max(1, dw // self.num_rays)
        sample_cols = np.arange(0, dw, step_col)
        raw_dists = []
        ray_angles_rad = []

        for col_idx in sample_cols:
            disp_val = float(col_disp_smooth[col_idx])
            disp_clamped = np.clip(disp_val, 12.0, 248.0)
            # Metric curve: [0.40m .. 4.5m]
            dist_m = 0.40 + self.depth_scale_factor * (
                (248.0 - disp_clamped) / 236.0
            ) ** 1.32
            raw_dists.append(dist_m)

            # Ray angle relative to camera axis [-fov/2 .. +fov/2]
            ang = (col_idx / float(dw) - 0.5) * self.fov_rad
            ray_angles_rad.append(ang)

        raw_dists = np.array(raw_dists, dtype=np.float32)

        # Temporal smoothing: Asymmetric filter (Fast attack on new obstacles, smooth release)
        if self.smoothed_dists is None or len(self.smoothed_dists) != len(raw_dists):
            self.smoothed_dists = raw_dists.copy()
        else:
            alpha = np.where(raw_dists < self.smoothed_dists, 0.90, 0.50)
            self.smoothed_dists = alpha * raw_dists + (1.0 - alpha) * self.smoothed_dists

        # 3. Analyze Sectors (Left, Center, Right)
        rays = []
        left_dists = []
        center_dists = []
        right_dists = []

        center_cone_rad = math.radians(14.0)

        for ang, d_m in zip(ray_angles_rad, self.smoothed_dists):
            x_m = float(d_m * math.sin(ang))
            y_m = float(d_m * math.cos(ang))
            ang_deg = math.degrees(ang)

            rays.append({
                "angle_deg": ang_deg,
                "dist_m": float(d_m),
                "x_m": x_m,
                "y_m": y_m,
            })

            if ang < -center_cone_rad:
                left_dists.append(d_m)
            elif ang > center_cone_rad:
                right_dists.append(d_m)
            else:
                center_dists.append(d_m)

        def sector_stats(dist_list):
            if not dist_list:
                return {"min_m": LIDAR_MAX_RANGE_M, "mean_m": LIDAR_MAX_RANGE_M}
            arr = np.array(dist_list)
            return {"min_m": float(np.min(arr)), "mean_m": float(np.mean(arr))}

        sectors = {
            "left": sector_stats(left_dists),
            "center": sector_stats(center_dists),
            "right": sector_stats(right_dists),
        }

        # Closest obstacle along the thin line
        closest_idx = int(np.argmin(self.smoothed_dists))
        closest = {
            "dist_m": float(self.smoothed_dists[closest_idx]),
            "angle_deg": float(math.degrees(ray_angles_rad[closest_idx])),
            "x_m": float(rays[closest_idx]["x_m"]),
            "y_m": float(rays[closest_idx]["y_m"]),
        }

        # Recommended heading vector (deepest passage)
        w_size = 5
        padded = np.pad(self.smoothed_dists, (w_size // 2, w_size // 2), mode="edge")
        weights = np.ones(w_size) / w_size
        conv = np.convolve(padded, weights, mode="valid")
        best_ray_idx = int(np.argmax(conv))
        rec_heading_deg = float(math.degrees(ray_angles_rad[best_ray_idx]))

        # Core Check: IS FORWARD CORRIDOR CLEAR?
        c_min = sectors["center"]["min_m"]
        is_path_clear = bool(c_min >= self.clear_dist_m and closest["dist_m"] >= self.brake_dist_m)

        # Safety status
        if c_min < self.brake_dist_m or closest["dist_m"] < 0.65:
            safety_level = "EMERGENCY_BRAKE"
        elif c_min < self.caution_dist_m or not is_path_clear:
            safety_level = "CAUTION"
        else:
            safety_level = "SAFE"

        return {
            "rays": rays,
            "sectors": sectors,
            "closest": closest,
            "is_path_clear": is_path_clear,
            "recommended_heading_deg": rec_heading_deg,
            "safety_level": safety_level,
            "scan_line_y": y_center,
        }
