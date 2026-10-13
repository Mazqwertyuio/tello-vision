"""Aerospace HUD: Thin-Line LiDAR Laser, Inertial Route Map & Radar Overlay.

Draws:
  - Upper-third Thin Laser LiDAR scan line directly on the video feed.
  - Interactive clickable mode buttons (Takeoff/Land, Explore Mode, Follow Mode, View Toggle).
  - High-resolution Polar LiDAR Radar widget or 2D Inertial Route Minimap.
  - Target tracking boxes.
  - Telemetry bar (Battery, Height/ToF, Compass, Distance Traveled, Shield status).
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

from config import (
    LIDAR_BRAKE_DIST_M,
    LIDAR_CAUTION_DIST_M,
    LIDAR_MAX_RANGE_M,
    HUD_RADAR_SIZE,
)


class HUDRenderer:
    """Renders graphics, thin LiDAR laser line, route minimap, and HUD widgets."""

    def __init__(self):
        # Interactive button bounding boxes (x1, y1, x2, y2)
        self.buttons = {
            "takeoff": (10, 15, 110, 55),
            "land": (120, 15, 210, 55),
            "explore": (220, 15, 345, 55),
            "follow": (355, 15, 475, 55),
            "view": (485, 15, 605, 55),
        }
        self.active_view = "RADAR"  # "RADAR", "MAP", "OFF"

    def check_button_click(self, x: int, y: int) -> Optional[str]:
        """Check if mouse click falls inside any UI button."""
        for name, (bx1, by1, bx2, by2) in self.buttons.items():
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                return name
        return None

    def draw_hud(
        self,
        frame_bgr: np.ndarray,
        telemetry: Dict[str, Any],
        active_mode: str,
        shield_state: str,
        scan_res: Optional[Dict[str, Any]],
        faces: List[Tuple[int, int, int, int, float]],
        target_face: Optional[Tuple[int, int, int, int, float]],
        status_msg: str,
        depth_ms: float = 0.0,
        minimap_img: Optional[np.ndarray] = None,
        total_dist_m: float = 0.0,
    ) -> np.ndarray:
        """Render complete HUD overlay on camera frame."""
        canvas = frame_bgr.copy()
        h, w = canvas.shape[:2]

        # 1. DRAW THIN LIDAR LASER LINE IN UPPER THIRD (~33% height)
        if scan_res is not None:
            self._draw_thin_lidar_line(canvas, scan_res)

        # 2. Draw Target Faces
        for f in faces:
            fx, fy, fw, fh = f[:4]
            conf = f[4] if len(f) > 4 else 0.8
            is_target = target_face is not None and (fx, fy, fw, fh) == target_face[:4]

            if is_target:
                color = (0, 255, 0)
                label = f"Target {conf:.2f}"
            else:
                color = (180, 180, 180)
                label = f"Person {conf:.2f}"

            cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), color, 2)
            cv2.putText(
                canvas, label, (fx, max(20, fy - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # 3. Draw Bottom-Right Widget (LiDAR Radar or Inertial Route Map)
        if self.active_view == "RADAR" and scan_res is not None:
            self._draw_lidar_radar(canvas, scan_res)
        elif self.active_view == "MAP" and minimap_img is not None:
            self._draw_minimap(canvas, minimap_img)

        # 4. Draw Top Control Buttons
        self._draw_buttons(canvas, telemetry.get("airborne", False), active_mode)

        # 5. Draw Telemetry Dashboard
        self._draw_telemetry_bar(
            canvas, telemetry, active_mode, shield_state, status_msg, depth_ms, total_dist_m
        )

        return canvas

    def _draw_thin_lidar_line(self, canvas: np.ndarray, scan_res: Dict[str, Any]):
        """Render glowing laser line strictly along the upper third horizon."""
        h, w = canvas.shape[:2]
        scan_y = scan_res.get("scan_line_y", int(h * 0.33))
        is_clear = scan_res.get("is_path_clear", False)
        c_dist = scan_res["sectors"]["center"]["min_m"]

        # Color: bright neon green if clear, neon red/orange if blocked
        line_color = (0, 255, 120) if is_clear else (0, 70, 255)

        # Thin laser horizontal beam
        cv2.line(canvas, (0, scan_y), (w, scan_y), line_color, 2)

        # Center flight corridor tick marks [-14° .. +14°]
        cx1 = int(w * 0.38)
        cx2 = int(w * 0.62)
        cv2.line(canvas, (cx1, scan_y - 8), (cx1, scan_y + 8), (255, 255, 255), 2)
        cv2.line(canvas, (cx2, scan_y - 8), (cx2, scan_y + 8), (255, 255, 255), 2)

        # Status text above laser line
        tag = f"LIDAR BEAM [UPPER 1/3]: {c_dist:.2f}m [{'CLEAR' if is_clear else 'BLOCKED'}]"
        cv2.putText(
            canvas, tag, (cx1 - 20, max(25, scan_y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 2
        )

    def _draw_buttons(self, canvas: np.ndarray, airborne: bool, active_mode: str):
        """Render top interactive buttons with active states."""
        # TAKEOFF
        self._draw_single_btn(
            canvas, self.buttons["takeoff"], "TAKEOFF",
            active=airborne, active_color=(0, 200, 0), idle_color=(40, 120, 40)
        )
        # LAND
        self._draw_single_btn(
            canvas, self.buttons["land"], "LAND",
            active=not airborne, active_color=(0, 0, 200), idle_color=(50, 50, 140)
        )
        # EXPLORE
        self._draw_single_btn(
            canvas, self.buttons["explore"], "MAZE [E]",
            active=(active_mode == "EXPLORE"), active_color=(220, 140, 0), idle_color=(70, 70, 80)
        )
        # FOLLOW
        self._draw_single_btn(
            canvas, self.buttons["follow"], "FOLLOW [F]",
            active=(active_mode == "FOLLOW"), active_color=(0, 200, 200), idle_color=(70, 70, 80)
        )
        # VIEW TOGGLE
        lbl = f"VIEW: {self.active_view}"
        self._draw_single_btn(
            canvas, self.buttons["view"], lbl,
            active=(self.active_view != "OFF"), active_color=(160, 60, 200), idle_color=(60, 60, 70)
        )

    def _draw_single_btn(
        self,
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        text: str,
        active: bool,
        active_color: Tuple[int, int, int],
        idle_color: Tuple[int, int, int],
    ):
        x1, y1, x2, y2 = rect
        bg = active_color if active else idle_color
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        border_color = (255, 255, 255) if active else (150, 150, 150)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 2 if active else 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.40
        thick = 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 + th) // 2
        cv2.putText(canvas, text, (tx, ty), font, scale, (255, 255, 255), thick)

    def _draw_minimap(self, canvas: np.ndarray, minimap_img: np.ndarray):
        """Overlay top-down inertial route minimap in bottom-right corner."""
        h, w = canvas.shape[:2]
        mh, mw = minimap_img.shape[:2]
        rx2 = w - 15
        ry2 = h - 15
        rx1 = rx2 - mw
        ry1 = ry2 - mh

        if rx1 >= 0 and ry1 >= 0:
            canvas[ry1:ry2, rx1:rx2] = minimap_img
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 200, 255), 1)

    def _draw_lidar_radar(self, canvas: np.ndarray, scan_res: Dict[str, Any]):
        """Render circular polar LiDAR HUD in bottom-right corner."""
        h, w = canvas.shape[:2]
        size = HUD_RADAR_SIZE
        rx2 = w - 15
        ry2 = h - 15
        rx1 = rx2 - size
        ry1 = ry2 - size

        sub = canvas[ry1:ry2, rx1:rx2]
        overlay = sub.copy()
        cv2.rectangle(overlay, (0, 0), (size, size), (15, 20, 25), -1)
        cv2.addWeighted(overlay, 0.85, sub, 0.15, 0, sub)
        canvas[ry1:ry2, rx1:rx2] = sub
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 220, 220), 1)

        cx = rx1 + size // 2
        cy = ry2 - 20
        max_r = size - 35
        scale_m_to_px = max_r / LIDAR_MAX_RANGE_M

        for ring_dist in [1.0, 2.0, 3.0, 4.0]:
            r_px = int(ring_dist * scale_m_to_px)
            if r_px < max_r:
                cv2.circle(canvas, (cx, cy), r_px, (45, 60, 65), 1)

        brake_r = int(LIDAR_BRAKE_DIST_M * scale_m_to_px)
        cv2.circle(canvas, (cx, cy), brake_r, (0, 0, 180), 1)

        rays = scan_res.get("rays", [])
        for r in rays:
            dist = r["dist_m"]
            ang = math.radians(r["angle_deg"])
            r_px = min(max_r, int(dist * scale_m_to_px))
            color = (0, 0, 255) if dist < LIDAR_BRAKE_DIST_M else (
                (0, 220, 255) if dist < LIDAR_CAUTION_DIST_M else (0, 220, 0)
            )
            px = int(cx + r_px * math.sin(ang))
            py = int(cy - r_px * math.cos(ang))
            cv2.line(canvas, (cx, cy), (px, py), color, 1)

        pts = np.array([[cx, cy - 8], [cx - 6, cy + 5], [cx + 6, cy + 5]], np.int32)
        cv2.fillPoly(canvas, [pts], (255, 255, 255))

        rec_deg = scan_res.get("recommended_heading_deg", 0.0)
        rec_rad = math.radians(rec_deg)
        rec_px = int(cx + (max_r * 0.85) * math.sin(rec_rad))
        rec_py = int(cy - (max_r * 0.85) * math.cos(rec_rad))
        cv2.arrowedLine(canvas, (cx, cy), (rec_px, rec_py), (0, 255, 100), 2, tipLength=0.25)

    def _draw_telemetry_bar(
        self,
        canvas: np.ndarray,
        telemetry: Dict[str, Any],
        active_mode: str,
        shield_state: str,
        status_msg: str,
        depth_ms: float,
        total_dist_m: float,
    ):
        """Draw bottom status bar with battery, mode, distance, and shield metrics."""
        h, w = canvas.shape[:2]
        bar_h = 55
        cv2.rectangle(canvas, (0, h - bar_h), (w - HUD_RADAR_SIZE - 20, h), (20, 20, 25), -1)
        cv2.line(canvas, (0, h - bar_h), (w - HUD_RADAR_SIZE - 20, h - bar_h), (80, 80, 90), 1)

        batt = telemetry.get("battery", 100)
        tof = telemetry.get("tof_cm", 65)
        yaw = telemetry.get("yaw", 0.0)

        m_color = (0, 200, 0) if active_mode == "MANUAL" else (
            (255, 160, 0) if active_mode == "EXPLORE" else (0, 220, 220)
        )
        cv2.putText(
            canvas, f"MODE: {active_mode}", (12, h - 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, m_color, 2
        )

        s_color = (0, 255, 0) if shield_state == "SAFE" else (
            (0, 220, 255) if shield_state == "CAUTION" else (0, 0, 255)
        )
        cv2.putText(
            canvas, f"SHIELD: {shield_state}", (170, h - 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, s_color, 2
        )

        cv2.putText(
            canvas, f"BAT:{batt}% | TOF:{tof}cm | YAW:{yaw:+.0f}° | DIST:{total_dist_m:.2f}m | DA2:{depth_ms:.0f}ms",
            (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1
        )

        if status_msg:
            cv2.putText(
                canvas, f"> {status_msg}", (360, h - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1
            )
