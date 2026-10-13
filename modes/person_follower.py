"""Person Follower Flight Mode with LiDAR Forward Passage Clearance.

Tracks human face/target using optical guidance while strictly verifying
that the forward passage along the upper-third LiDAR line is open:
  - If target lost: rotates smoothly to search.
  - If target acquired: centers yaw and altitude.
  - If following: advances toward person ONLY IF path clearance is confirmed!
  - If obstacle blocks the flight path: halts forward movement and warns pilot.
"""

import time
from typing import Tuple, List, Optional, Dict, Any

from config import (
    FOLLOW_TARGET_AREA,
    FOLLOW_SPEED_GAIN,
    FOLLOW_MAX_SPEED,
    FOLLOW_TRACK_YAW_LIMIT,
    FOLLOW_SEARCH_YAW_SPEED,
    FOLLOW_AREA_FILTER_ALPHA,
    LIDAR_BRAKE_DIST_M,
)


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


class PersonFollower:
    """Follows a person only when the forward passage is clear of obstacles."""

    def __init__(self):
        self.phase = "SEARCH"  # "SEARCH", "CENTER", "FOLLOW"
        self.last_target = None
        self.last_seen_at = 0.0
        self.last_x_error = 0.0
        self.centered_since = None
        self.center_start_time = None
        self.filtered_area = None
        self.hit_count = 0

    def reset(self):
        """Reset tracking state."""
        self.phase = "SEARCH"
        self.last_target = None
        self.last_seen_at = 0.0
        self.last_x_error = 0.0
        self.centered_since = None
        self.center_start_time = None
        self.filtered_area = None
        self.hit_count = 0

    def update(
        self,
        target_face: Optional[Tuple[int, int, int, int, float]],
        frame_shape: Tuple[int, int],
        scan_res: Optional[Dict[str, Any]] = None,
        now: Optional[float] = None,
    ) -> Tuple[Tuple[int, int, int, int], str]:
        """Compute flight commands to track and follow target.

        Args:
            target_face: (x, y, w, h, conf) or None
            frame_shape: (height, width)
            scan_res: thin-line LiDAR scan output
            now: monotonic timestamp

        Returns:
            rc_cmd: (left_right, fwd_back, up_down, yaw)
            status_text: explanation of current state
        """
        if now is None:
            now = time.monotonic()

        h, w = frame_shape[:2]

        is_clear = True
        c_dist = 4.0
        if scan_res is not None:
            is_clear = scan_res.get("is_path_clear", False)
            c_dist = scan_res["sectors"]["center"]["min_m"]

        if target_face is not None:
            x, y, fw, fh = target_face[:4]
            fc_x = x + fw / 2.0
            fc_y = y + fh / 2.0
            x_error = fc_x / float(w) - 0.5  # [-0.5 .. +0.5]
            y_error = fc_y / float(h) - 0.5  # [-0.5 .. +0.5]
            area = (fw * fh) / float(w * h)

            if self.filtered_area is None:
                self.filtered_area = area
            else:
                self.filtered_area = (
                    FOLLOW_AREA_FILTER_ALPHA * area
                    + (1.0 - FOLLOW_AREA_FILTER_ALPHA) * self.filtered_area
                )

            self.last_seen_at = now
            self.last_x_error = x_error
            self.last_target = target_face

            # Transition SEARCH -> CENTER
            if self.phase == "SEARCH":
                self.hit_count += 1
                if self.hit_count >= 1:
                    self.phase = "CENTER"
                    self.centered_since = None
                    self.center_start_time = now

            # PHASE: CENTER
            if self.phase == "CENTER":
                yaw = int(clamp(x_error * 50.0, -FOLLOW_TRACK_YAW_LIMIT, FOLLOW_TRACK_YAW_LIMIT))
                ud = int(clamp(-y_error * 45.0, -24.0, 24.0)) if abs(y_error) >= 0.06 else 0
                if 0 < abs(ud) < 8:
                    ud = 8 if ud > 0 else -8

                is_centered = abs(x_error) < 0.08 and abs(y_error) < 0.10
                self.centered_since = (self.centered_since or now) if is_centered else None
                held_long = self.centered_since is not None and (now - self.centered_since >= 0.25)
                timeout = (now - self.center_start_time) >= 0.80

                if held_long or timeout:
                    self.phase = "FOLLOW"

                return (0, 0, ud, yaw), f"CENTERING: x_err={x_error:+.2f}, y_err={y_error:+.2f}"

            # PHASE: FOLLOW
            elif self.phase == "FOLLOW":
                lateral = int(clamp(x_error * 36.0, -22.0, 22.0))
                size_ratio = (max(self.filtered_area, 1e-4) / FOLLOW_TARGET_AREA) ** 0.5
                distance_error = 1.0 - size_ratio  # positive: too far, negative: too close

                raw_fwd = int(clamp(distance_error * FOLLOW_SPEED_GAIN, -FOLLOW_MAX_SPEED, FOLLOW_MAX_SPEED))
                if abs(distance_error) < 0.05:
                    raw_fwd = 0
                elif raw_fwd == 0:
                    raw_fwd = 7 if distance_error > 0 else -7

                yaw = int(clamp(x_error * 48.0, -FOLLOW_TRACK_YAW_LIMIT, FOLLOW_TRACK_YAW_LIMIT))
                ud = int(clamp(-y_error * 42.0, -22.0, 22.0)) if abs(y_error) >= 0.06 else 0
                if 0 < abs(ud) < 7:
                    ud = 7 if ud > 0 else -7

                # FORWARD PASSAGE CLEARANCE CHECK:
                # Follow forward ONLY IF path along the thin LiDAR line is open!
                # Backward flight is strictly forbidden (rear is blind).
                if raw_fwd > 0 and not is_clear:
                    safe_fwd = 0
                    status_text = f"FOLLOW: Passage BLOCKED ({c_dist:.2f}m)! Halting forward motion"
                elif c_dist < LIDAR_BRAKE_DIST_M:
                    safe_fwd = 0
                    status_text = f"FOLLOW: Target/Obstacle close ({c_dist:.2f}m)! Hovering safely in place"
                elif raw_fwd < 0:
                    # Target approached drone: hover in place without blind retreat
                    safe_fwd = 0
                    status_text = f"FOLLOW: Target close. Hovering in place"
                else:
                    safe_fwd = raw_fwd
                    status_text = f"FOLLOW: Passage open ({c_dist:.2f}m). Tracking target (fwd={safe_fwd})"

                return (lateral, safe_fwd, ud, yaw), status_text

        # Target NOT visible in current frame
        else:
            time_since_seen = now - self.last_seen_at
            if self.phase == "SEARCH":
                self.hit_count = 0
                return (0, 0, 0, FOLLOW_SEARCH_YAW_SPEED), "SEARCH: Scanning for person"
            elif time_since_seen <= 0.25:
                return (0, 0, 0, 0), "TRACK: Occlusion, hovering"
            elif time_since_seen <= 2.2:
                steer_dir = 1 if self.last_x_error >= 0 else -1
                return (0, 0, 0, steer_dir * 22), "TRACK: Turning toward last known target"
            else:
                self.phase = "SEARCH"
                self.hit_count = 0
                return (0, 0, 0, FOLLOW_SEARCH_YAW_SPEED), "SEARCH: Person lost, scanning"
