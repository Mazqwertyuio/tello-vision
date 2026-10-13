"""Collision Prevention Shield for Depth Map Autopilot.

Ensures the drone never flies forward into walls, barriers, or objects.
Backward movement is strictly prohibited in autonomous modes (rear is blind).
"""

from typing import Tuple, Dict, Any, Optional

from config import DEPTH_OBSTACLE_THRESH, DEPTH_CRITICAL_BRAKE


class CollisionShield:
    """Filters flight commands against depth map obstacle proximity."""

    def __init__(
        self,
        obstacle_thresh: float = DEPTH_OBSTACLE_THRESH,
        brake_thresh: float = DEPTH_CRITICAL_BRAKE,
    ):
        self.obstacle_thresh = obstacle_thresh
        self.brake_thresh = brake_thresh

    def filter_command(
        self,
        raw_cmd: Tuple[int, int, int, int],
        scan_res: Optional[Dict[str, Any]],
        flank_left_m: float = 4.0,
        flank_right_m: float = 4.0,
        allow_reverse: bool = False,
    ) -> Tuple[Tuple[int, int, int, int], str]:
        """Verify command against depth map danger levels."""
        lr, fb, ud, yaw = raw_cmd

        if scan_res is None:
            return raw_cmd, "NO_DEPTH_DATA"

        danger_c = scan_res.get("danger_c", 0.0)
        is_clear = scan_res.get("is_path_clear", True)

        # 1. EMERGENCY STOP (Point-blank obstacle ahead)
        if danger_c >= self.brake_thresh:
            safe_fb = 0 if fb > 0 else (fb if allow_reverse else 0)
            return (0, int(safe_fb), int(ud), int(yaw)), "EMERGENCY_BRAKE"

        # 2. CAUTION (Obstacle blocking forward path)
        if not is_clear or danger_c >= self.obstacle_thresh:
            safe_fb = 0 if fb > 0 else (fb if allow_reverse else 0)
            return (0, int(safe_fb), int(ud), int(yaw)), "CAUTION"

        # 3. CLEAR PATH
        safe_fb = fb if (fb > 0 or allow_reverse) else 0
        return (int(lr), int(safe_fb), int(ud), int(yaw)), "SAFE"
