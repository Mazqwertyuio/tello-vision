"""Autonomous Collision Prevention Shield.

Monitors proposed RC commands against real-time LiDAR clearance:
- Prevents collisions with walls, furniture, and persons in front.
- Clamps forward velocity under CAUTION.
- Imposes EMERGENCY BRAKE / REVERSE when distance drops below safety buffer.
- Performs equidistant centering between lateral obstacles.
"""

from typing import Tuple, Dict, Any, Optional

from config import (
    LIDAR_BRAKE_DIST_M,
    LIDAR_CAUTION_DIST_M,
)


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


class CollisionShield:
    """Interceps and filters drone RC commands through LiDAR safety field."""

    def __init__(
        self,
        brake_dist_m: float = LIDAR_BRAKE_DIST_M,
        caution_dist_m: float = LIDAR_CAUTION_DIST_M,
    ):
        self.brake_dist_m = brake_dist_m
        self.caution_dist_m = caution_dist_m

    def filter_command(
        self,
        raw_cmd: Tuple[int, int, int, int],
        scan_res: Optional[Dict[str, Any]],
        flank_left_m: float = 4.0,
        flank_right_m: float = 4.0,
        allow_reverse: bool = False,
    ) -> Tuple[Tuple[int, int, int, int], str]:
        """Filter (lr, fb, ud, yaw) through safety envelope.

        Returns:
            safe_cmd: (lr, fb, ud, yaw)
            shield_state: 'SAFE', 'CAUTION_SLOWDOWN', 'EMERGENCY_STOP', or 'COLLISION_AVOID'
        """
        lr, fb, ud, yaw = raw_cmd

        if scan_res is None:
            return raw_cmd, "NO_LIDAR_DATA"

        c_min = scan_res["sectors"]["center"]["min_m"]
        l_min = scan_res["sectors"]["left"]["min_m"]
        r_min = scan_res["sectors"]["right"]["min_m"]
        closest_m = scan_res["closest"]["dist_m"]

        d_left = min(l_min, flank_left_m)
        d_right = min(r_min, flank_right_m)

        # 1. EMERGENCY BRAKE: obstacle dead-ahead
        if c_min < self.brake_dist_m or closest_m < 0.65:
            # Block forward flight completely (no blind reverse, rear is blind!)
            safe_fb = 0 if fb > 0 else (fb if allow_reverse else 0)

            # Suppress lateral twitching when rotating in place
            safe_lr = lr
            if abs(yaw) < 5 and fb > 0:
                if d_left < 0.70:
                    safe_lr = 10
                elif d_right < 0.70:
                    safe_lr = -10
            else:
                # In-place turn: keep lateral strictly stable (0)
                safe_lr = 0

            return (int(safe_lr), int(safe_fb), int(ud), int(yaw)), "EMERGENCY_BRAKE"

        # 2. CAUTION: obstacle approaching
        if c_min < self.caution_dist_m or closest_m < self.brake_dist_m:
            safe_fb = min(fb, 10) if fb > 0 else fb

            safe_lr = lr
            # Only apply centering when actively flying straight forward
            if abs(yaw) < 5 and fb > 0:
                if d_left < 1.6 and d_right < 1.6:
                    diff = d_right - d_left  # positive -> left is closer
                    if abs(diff) > 0.10:
                        safe_lr = int(clamp(diff * 14.0, -12, 12))
                elif d_left < 0.85:
                    safe_lr = 10
                elif d_right < 0.85:
                    safe_lr = -10
            else:
                # During rotation / hover, eliminate lateral twitching
                safe_lr = 0

            return (int(safe_lr), int(safe_fb), int(ud), int(yaw)), "CAUTION"

        # 3. SAFE
        safe_lr = lr
        if abs(yaw) < 5 and fb > 0 and d_left < 1.5 and d_right < 1.5:
            diff = d_right - d_left
            if abs(diff) > 0.12:
                safe_lr = int(clamp(safe_lr + diff * 8.0, -10, 10))

        return (int(safe_lr), int(fb), int(ud), int(yaw)), "SAFE"
