"""Inertial Odometry & Flight Route Tracker Module.

Tracks drone trajectory in 3D (X, Y, Z, Yaw) using dead reckoning from Tello IMU:
- Integrates velocities (vx, vy, vz) and compass heading (yaw).
- Records flight path breadcrumbs.
- Computes total distance traveled.
- Generates a top-down 2D minimap of the route for the HUD.
- Exports flight trajectory to CSV for replay and analysis.
"""

import math
import csv
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np


class InertialRouteTracker:
    """Inertial dead-reckoning trajectory tracker."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset trajectory to start origin (0, 0, 0)."""
        self.pos_x = 0.0          # meters in world frame
        self.pos_y = 0.0          # meters in world frame
        self.pos_z = 0.0          # meters (altitude from ToF)
        self.current_yaw = 0.0    # degrees
        self.total_distance_m = 0.0
        self.last_update_time = None
        self.path_points: List[Tuple[float, float, float, float, float]] = [] # (x, y, z, yaw, timestamp)
        self.last_breadcrumb_time = 0.0

    def update(
        self,
        telemetry: Dict[str, Any],
        rc_command: Optional[Tuple[int, int, int, int]] = None,
        now: Optional[float] = None,
    ):
        """Integrate motion step from IMU telemetry and velocities."""
        if now is None:
            now = time.monotonic()

        if self.last_update_time is None:
            self.last_update_time = now
            self.current_yaw = float(telemetry.get("yaw", 0.0) or 0.0)
            self.pos_z = float(telemetry.get("tof_cm", 65)) / 100.0
            self.path_points.append((0.0, 0.0, self.pos_z, self.current_yaw, now))
            return

        dt = now - self.last_update_time
        self.last_update_time = now

        # Bound dt against frame freezes
        if dt <= 0.0 or dt > 1.0:
            return

        self.current_yaw = float(telemetry.get("yaw", 0.0) or 0.0)
        self.pos_z = float(telemetry.get("tof_cm", 65)) / 100.0

        # Read IMU speed in m/s
        vx_local = float(telemetry.get("vx", 0.0) or 0.0) # lateral right
        vy_local = float(telemetry.get("vy", 0.0) or 0.0) # forward

        # Dead reckoning fallback from RC commands if IMU speed reads zero while airborne
        if abs(vx_local) < 0.01 and abs(vy_local) < 0.01 and rc_command and telemetry.get("airborne"):
            lr, fb, _, _ = rc_command
            # Tello speed calibration: speed 20 ~= 0.35 m/s
            vx_local = (lr / 100.0) * 1.5
            vy_local = (fb / 100.0) * 1.5

        # Transform local drone speeds into World Frame coordinates
        yaw_rad = math.radians(self.current_yaw)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        # Forward (vy) moves along yaw heading, lateral (vx) moves perpendicular
        dx_world = (vy_local * sin_y + vx_local * cos_y) * dt
        dy_world = (vy_local * cos_y - vx_local * sin_y) * dt

        self.pos_x += dx_world
        self.pos_y += dy_world

        step_dist = math.hypot(dx_world, dy_world)
        self.total_distance_m += step_dist

        # Record point every 0.15s or 10cm movement
        if (now - self.last_breadcrumb_time >= 0.15) or (step_dist > 0.10):
            self.path_points.append((self.pos_x, self.pos_y, self.pos_z, self.current_yaw, now))
            self.last_breadcrumb_time = now

    def get_position(self) -> Tuple[float, float, float, float]:
        """Return current world position (x_m, y_m, z_m, yaw_deg)."""
        return self.pos_x, self.pos_y, self.pos_z, self.current_yaw

    def render_minimap(self, size: int = 190, scale_px_per_m: float = 24.0) -> np.ndarray:
        """Generate top-down 2D route map image."""
        map_img = np.zeros((size, size, 3), dtype=np.uint8)
        map_img[:] = (18, 22, 28)

        # Grid circles / range rings
        cx = size // 2
        cy = size // 2
        for r_m in [1.0, 2.0, 3.0, 4.0]:
            r_px = int(r_m * scale_px_per_m)
            if r_px < size // 2:
                cv2.circle(map_img, (cx, cy), r_px, (35, 45, 55), 1)

        # Draw coordinate axes
        cv2.line(map_img, (cx, 0), (cx, size), (40, 50, 60), 1)
        cv2.line(map_img, (0, cy), (size, cy), (40, 50, 60), 1)

        # Center on drone or start point
        # Draw recorded trajectory
        if len(self.path_points) > 1:
            pts = []
            for px, py, _, _, _ in self.path_points:
                screen_x = int(cx + px * scale_px_per_m)
                screen_y = int(cy - py * scale_px_per_m) # up is forward (y)
                pts.append((screen_x, screen_y))

            for i in range(1, len(pts)):
                cv2.line(map_img, pts[i - 1], pts[i], (0, 220, 255), 2)

        # Start Origin (Green circle)
        cv2.circle(map_img, (cx, cy), 4, (0, 255, 0), -1)

        # Current Drone Position & Heading (Red/Orange Arrow)
        cur_x = int(cx + self.pos_x * scale_px_per_m)
        cur_y = int(cy - self.pos_y * scale_px_per_m)
        yaw_rad = math.radians(self.current_yaw)
        tip_x = int(cur_x + 12 * math.sin(yaw_rad))
        tip_y = int(cur_y - 12 * math.cos(yaw_rad))

        cv2.circle(map_img, (cur_x, cur_y), 4, (0, 165, 255), -1)
        cv2.arrowedLine(map_img, (cur_x, cur_y), (tip_x, tip_y), (0, 80, 255), 2, tipLength=0.4)

        # Info overlays
        cv2.putText(
            map_img, f"PATH: {self.total_distance_m:.2f}m", (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1
        )
        cv2.putText(
            map_img, f"X:{self.pos_x:+.1f} Y:{self.pos_y:+.1f}m", (6, size - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1
        )

        return map_img

    def save_to_csv(self, file_path: Path):
        """Save path coordinates to CSV file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "pos_x_m", "pos_y_m", "pos_z_m", "yaw_deg", "timestamp"])
            for idx, (px, py, pz, yaw, ts) in enumerate(self.path_points):
                writer.writerow([idx, f"{px:.3f}", f"{py:.3f}", f"{pz:.3f}", f"{yaw:.1f}", f"{ts:.3f}"])
