"""Clean HUD Interface for Depth Obstacle Avoidance & Silhouette Tracking.

Renders:
  - Bounding box for detected human silhouettes.
  - 3-sector depth clearance indicators [LEFT | CENTER | RIGHT].
  - Interactive touch buttons (Takeoff, Land, Explore, Follow).
  - Telemetry bar (Battery, Height/ToF, Compass, Status).
"""

from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np


class HUDRenderer:
    """Renders visual overlay on camera feed."""

    def __init__(self):
        self.buttons = {
            "takeoff": (10, 15, 110, 55),
            "land": (120, 15, 210, 55),
            "explore": (220, 15, 345, 55),
            "follow": (355, 15, 475, 55),
        }

    def check_button_click(self, x: int, y: int) -> Optional[str]:
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
        persons: List[Tuple[int, int, int, int, float]],
        target_person: Optional[Tuple[int, int, int, int, float]],
        status_msg: str,
        depth_ms: float = 0.0,
        minimap_img: Optional[np.ndarray] = None,
        total_dist_m: float = 0.0,
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        h, w = canvas.shape[:2]

        # 1. Draw Human Silhouette BBoxes
        for p in persons:
            px, py, pw, ph = p[:4]
            conf = p[4] if len(p) > 4 else 0.8
            is_target = target_person is not None and (px, py, pw, ph) == target_person[:4]

            color = (0, 255, 0) if is_target else (180, 180, 180)
            thick = 3 if is_target else 1
            label = f"TARGET PERSON {conf:.2f}" if is_target else f"PERSON {conf:.2f}"

            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), color, thick)
            cv2.putText(
                canvas, label, (px, max(22, py - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

        # 2. Draw 3 Depth Clearance Badges at Top Center
        if scan_res is not None:
            self._draw_depth_badges(canvas, scan_res)

        # 3. Draw Top Control Buttons
        self._draw_buttons(canvas, telemetry.get("airborne", False), active_mode)

        # 4. Draw Bottom Telemetry Bar
        self._draw_telemetry_bar(canvas, telemetry, active_mode, shield_state, status_msg, depth_ms)

        return canvas

    def _draw_depth_badges(self, canvas: np.ndarray, scan_res: Dict[str, Any]):
        """Draw 3 visual clearance blocks for Left, Center, and Right."""
        h, w = canvas.shape[:2]
        dl = scan_res.get("dist_l_m", 3.0)
        dc = scan_res.get("dist_c_m", 3.0)
        dr = scan_res.get("dist_r_m", 3.0)
        is_clear = scan_res.get("is_path_clear", True)

        cx = w // 2
        by = 35

        def get_col(d_m):
            if d_m < 0.85:
                return (0, 0, 230)      # Red: Danger
            elif d_m < 1.35:
                return (0, 200, 255)    # Yellow: Caution
            return (0, 220, 50)         # Green: Safe

        # Center Badge
        col_c = get_col(dc)
        cv2.rectangle(canvas, (cx - 70, by - 20), (cx + 70, by + 12), (20, 20, 25), -1)
        cv2.rectangle(canvas, (cx - 70, by - 20), (cx + 70, by + 12), col_c, 2)
        cv2.putText(
            canvas, f"CENTER: {dc:.2f}m", (cx - 60, by + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )

        # Left Badge
        col_l = get_col(dl)
        cv2.rectangle(canvas, (cx - 190, by - 20), (cx - 80, by + 12), (20, 20, 25), -1)
        cv2.rectangle(canvas, (cx - 190, by - 20), (cx - 80, by + 12), col_l, 1)
        cv2.putText(
            canvas, f"L: {dl:.2f}m", (cx - 175, by + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1
        )

        # Right Badge
        col_r = get_col(dr)
        cv2.rectangle(canvas, (cx + 80, by - 20), (cx + 190, by + 12), (20, 20, 25), -1)
        cv2.rectangle(canvas, (cx + 80, by - 20), (cx + 190, by + 12), col_r, 1)
        cv2.putText(
            canvas, f"R: {dr:.2f}m", (cx + 95, by + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1
        )

    def _draw_buttons(self, canvas: np.ndarray, airborne: bool, active_mode: str):
        # TAKEOFF
        self._draw_btn(canvas, self.buttons["takeoff"], "TAKEOFF", airborne, (0, 200, 0), (40, 110, 40))
        # LAND
        self._draw_btn(canvas, self.buttons["land"], "LAND", not airborne, (0, 0, 200), (50, 50, 130))
        # EXPLORE
        self._draw_btn(canvas, self.buttons["explore"], "EXPLORE [E]", active_mode == "EXPLORE", (220, 140, 0), (70, 70, 80))
        # FOLLOW
        self._draw_btn(canvas, self.buttons["follow"], "FOLLOW [F]", active_mode == "FOLLOW", (0, 200, 200), (70, 70, 80))

    def _draw_btn(self, canvas, rect, text, active, act_col, idle_col):
        x1, y1, x2, y2 = rect
        bg = act_col if active else idle_col
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        border = (255, 255, 255) if active else (140, 140, 140)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 2 if active else 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.40, 1)
        cv2.putText(canvas, text, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2), font, 0.40, (255, 255, 255), 1)

    def _draw_telemetry_bar(self, canvas, telemetry, active_mode, shield_state, status_msg, depth_ms):
        h, w = canvas.shape[:2]
        bar_h = 45
        cv2.rectangle(canvas, (0, h - bar_h), (w, h), (15, 15, 20), -1)
        cv2.line(canvas, (0, h - bar_h), (w, h - bar_h), (60, 60, 70), 1)

        batt = telemetry.get("battery", 100)
        tof = telemetry.get("tof_cm", 65)
        yaw = telemetry.get("yaw", 0.0)

        # Mode
        m_col = (0, 200, 0) if active_mode == "MANUAL" else ((255, 160, 0) if active_mode == "EXPLORE" else (0, 220, 220))
        cv2.putText(canvas, f"MODE: {active_mode}", (12, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, m_col, 2)

        # Status text
        cv2.putText(canvas, f"STATUS: {status_msg}", (160, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1)

        # Stats
        cv2.putText(canvas, f"BAT: {batt}% | TOF: {tof}cm | YAW: {yaw:+.0f}° | DEPTH: {depth_ms:.0f}ms", (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
