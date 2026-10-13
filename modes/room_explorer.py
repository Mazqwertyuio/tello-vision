"""Autonomous Room Exploration via Real-Time Depth Map Obstacle Avoidance.

Reactive Flight Logic:
  - If Center is CLEAR -> Fly FORWARD directly.
  - If Center is BLOCKED -> STOP forward flight and ROTATE in-place
    towards the clearer side (Left or Right) until the center opens up.
  - Backward flight (reverse) is STRICTLY PROHIBITED (rear is blind).
"""

from typing import Tuple, Dict, Any, Optional

from config import (
    EXPLORE_FORWARD_SPEED,
    EXPLORE_AVOID_YAW_SPEED,
    DEPTH_OBSTACLE_THRESH,
    TARGET_ALTITUDE_CM,
    MIN_ALTITUDE_CM,
    MAX_ALTITUDE_CM,
)


class RoomExplorer:
    """Reactive room explorer driven by 3-sector depth map."""

    def __init__(self, polar_memory=None):
        self.phase = "FORWARD"
        self.turn_dir = 1  # 1 = Right, -1 = Left

    def reset(self, current_yaw: float = 0.0):
        self.phase = "FORWARD"
        self.turn_dir = 1

    def update(
        self,
        scan_res: Optional[Dict[str, Any]],
        telemetry: Dict[str, Any],
        now: Optional[float] = None,
    ) -> Tuple[Tuple[int, int, int, int], str]:
        """Compute flight commands for depth-guided room exploration."""
        tof_h = telemetry.get("tof_cm", TARGET_ALTITUDE_CM) or TARGET_ALTITUDE_CM

        # Altitude hold
        ud = 0
        if tof_h < MIN_ALTITUDE_CM:
            ud = 18   # ascend away from floor
        elif tof_h > MAX_ALTITUDE_CM:
            ud = -14  # descend away from ceiling

        if scan_res is None:
            return (0, 0, ud, 0), "EXPLORE: Waiting for depth map"

        danger_c = scan_res.get("danger_c", 0.0)
        danger_l = scan_res.get("danger_l", 0.0)
        danger_r = scan_res.get("danger_r", 0.0)
        dist_c = scan_res.get("dist_c_m", 3.0)
        is_clear = scan_res.get("is_path_clear", True)

        # 1. FORWARD: Center path is clear
        if is_clear and danger_c < DEPTH_OBSTACLE_THRESH:
            self.phase = "FORWARD"

            # Subtle yaw bias if one flank is closer than the other
            yaw_bias = 0
            if danger_l > danger_r + 25:
                yaw_bias = 12   # steer slightly right
            elif danger_r > danger_l + 25:
                yaw_bias = -12  # steer slightly left

            return (0, EXPLORE_FORWARD_SPEED, ud, yaw_bias), f"FORWARD: Path clear ({dist_c:.2f}m)"

        # 2. AVOID: Center is blocked -> Turn in-place towards clearer side
        self.phase = "AVOID"

        # Choose turn direction based on which side is deeper/clearer
        if danger_l < danger_r:
            self.turn_dir = -1  # turn Left
            dir_text = "LEFT"
        else:
            self.turn_dir = 1   # turn Right
            dir_text = "RIGHT"

        yaw_cmd = int(self.turn_dir * EXPLORE_AVOID_YAW_SPEED)

        # Strictly forward=0, lateral=0 (no blind movements)
        return (0, 0, ud, yaw_cmd), f"AVOID: Center blocked ({dist_c:.2f}m) -> Steering {dir_text}"
