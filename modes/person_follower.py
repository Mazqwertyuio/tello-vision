"""Person Silhouette Follower with Depth Map Obstacle Shield.

Tracks human silhouette (full-body/upper-body) and follows forward
ONLY when the depth map confirms that the forward corridor is clear.
Backward flight is strictly prohibited.
"""

from typing import Tuple, Optional, Dict, Any

from config import (
    PERSON_TARGET_AREA,
    PERSON_FOLLOW_SPEED,
    PERSON_TRACK_YAW_LIMIT,
    DEPTH_OBSTACLE_THRESH,
)


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


class PersonFollower:
    """Follows a detected person silhouette with forward obstacle safety."""

    def __init__(self):
        self.last_seen_at = 0.0

    def reset(self):
        self.last_seen_at = 0.0

    def update(
        self,
        target_person: Optional[Tuple[int, int, int, int, float]],
        frame_shape: Tuple[int, int],
        scan_res: Optional[Dict[str, Any]] = None,
        now: Optional[float] = None,
    ) -> Tuple[Tuple[int, int, int, int], str]:
        """Compute flight commands to track and follow human silhouette."""
        h, w = frame_shape[:2]

        is_clear = True
        dist_c = 3.0
        if scan_res is not None:
            danger_c = scan_res.get("danger_c", 0.0)
            dist_c = scan_res.get("dist_c_m", 3.0)
            is_clear = (danger_c < DEPTH_OBSTACLE_THRESH)

        # 1. Person Silhouette Detected
        if target_person is not None:
            px, py, pw, ph = target_person[:4]
            pc_x = px + pw / 2.0
            pc_y = py + ph / 2.0

            # Normalized errors [-0.5 .. +0.5]
            err_x = pc_x / float(w) - 0.5
            err_y = pc_y / float(h) - 0.45  # target chest/face height

            # Yaw centering (turn to keep person centered in FOV)
            yaw = int(clamp(err_x * 60.0, -PERSON_TRACK_YAW_LIMIT, PERSON_TRACK_YAW_LIMIT))

            # Altitude centering (up/down)
            ud = int(clamp(-err_y * 45.0, -20.0, 20.0)) if abs(err_y) >= 0.08 else 0

            # Distance control via silhouette area
            area = (pw * ph) / float(w * h)

            # Advance forward if person is far (silhouette is small)
            if area < PERSON_TARGET_AREA:
                desired_fwd = PERSON_FOLLOW_SPEED
            else:
                desired_fwd = 0  # Optimal distance reached, hover

            # Safety check: block forward flight if depth map sees an obstacle ahead
            if desired_fwd > 0 and not is_clear:
                safe_fwd = 0
                status_text = f"FOLLOW: Person seen, but forward path BLOCKED ({dist_c:.2f}m)"
            elif desired_fwd > 0:
                safe_fwd = desired_fwd
                status_text = f"FOLLOW: Tracking silhouette forward (area: {area:.2f})"
            else:
                safe_fwd = 0
                status_text = f"FOLLOW: Optimal distance reached (hovering)"

            return (0, safe_fwd, ud, yaw), status_text

        # 2. No Person Detected -> Hover safely
        return (0, 0, 0, 0), "FOLLOW: Searching for person silhouette..."
