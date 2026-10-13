"""Maze Room Explorer Mode: Forward Flight & Locked-Direction Turn on Obstacle.

CRITICAL SAFETY RULE:
  Camera faces forward only. The rear is completely blind.
  Backward flight (reverse, fb < 0) is STRICTLY FORBIDDEN during room exploration!
  Drone only flies FORWARD when clear, and STOPS & ROTATES in-place when blocked.

Algorithm:
  1. Inspect thin LiDAR line in the upper third of depth map.
  2. If NO obstacle (path is clear) -> Fly FORWARD directly.
  3. If OBSTACLE encountered -> STOP in place (fb=0) and turn in discrete ~10-15°
     steps in a locked direction until an opening appears.
  4. As soon as an opening is detected -> Instantly resume FORWARD flight!
"""

import time
import math
from typing import Tuple, Dict, Any, Optional

from config import (
    EXPLORE_FORWARD_SPEED,
    MAZE_TURN_YAW_SPEED,
    MAZE_STEP_PULSE_DURATION,
    MAZE_CHECK_PAUSE_DURATION,
    MAZE_CLEAR_DIST_M,
    TARGET_ALTITUDE_CM,
    MIN_ALTITUDE_CM,
    MAX_ALTITUDE_CM,
)


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


class RoomExplorer:
    """Reactive maze exploration: STRICTLY forward flight and in-place turns (no reverse)."""

    def __init__(self, polar_memory=None):
        self.phase = "FORWARD"  # "FORWARD", "STEP_TURN"
        self.turn_direction = 1  # +1 = Right, -1 = Left
        self.turn_locked = False
        self.turn_step_start_time = 0.0
        self.steps_count = 0

    def reset(self, current_yaw: float = 0.0):
        """Reset exploration mode."""
        self.phase = "FORWARD"
        self.turn_direction = 1
        self.turn_locked = False
        self.turn_step_start_time = 0.0
        self.steps_count = 0

    def update(
        self,
        scan_res: Optional[Dict[str, Any]],
        telemetry: Dict[str, Any],
        now: Optional[float] = None,
    ) -> Tuple[Tuple[int, int, int, int], str]:
        """Compute flight commands for maze room exploration.

        NOTE: Backward movement (fb < 0) is completely prohibited.

        Returns:
            rc_cmd: (left_right, fwd_back, up_down, yaw) where fwd_back >= 0 always!
            status_text: explanation of current action
        """
        if now is None:
            now = time.monotonic()

        tof_h = telemetry.get("tof_cm", TARGET_ALTITUDE_CM) or TARGET_ALTITUDE_CM

        # Altitude control: hold eye-level height ~65-75 cm
        ud = 0
        if tof_h < MIN_ALTITUDE_CM:
            ud = 18   # ascend away from floor
        elif tof_h > MAX_ALTITUDE_CM:
            ud = -14  # descend away from ceiling

        if scan_res is None:
            return (0, 0, ud, 0), "EXPLORE: Waiting for LiDAR data"

        is_clear = scan_res.get("is_path_clear", False)
        c_min = scan_res["sectors"]["center"]["min_m"]
        l_min = scan_res["sectors"]["left"]["min_m"]
        r_min = scan_res["sectors"]["right"]["min_m"]
        rec_heading = scan_res.get("recommended_heading_deg", 0.0)

        # -------------------------------------------------------------
        # STATE 1: FORWARD (Path is CLEAR along the upper-third line)
        # -------------------------------------------------------------
        if is_clear:
            self.phase = "FORWARD"
            self.turn_locked = False
            self.steps_count = 0

            # Subtle centering ONLY during forward flight down corridors
            lateral = 0
            if l_min < 1.6 and r_min < 1.6:
                diff_m = r_min - l_min  # positive: left is closer
                if abs(diff_m) > 0.12:
                    lateral = int(clamp(diff_m * 12.0, -10, 10))

            return (lateral, EXPLORE_FORWARD_SPEED, ud, 0), f"FORWARD: Path clear ({c_min:.2f}m). Flying straight"

        # -------------------------------------------------------------
        # STATE 2: STEP_TURN (Stop & turn in-place by ~10-15° steps; NO REVERSE)
        # -------------------------------------------------------------
        if not is_clear:
            if not self.turn_locked:
                # Lock turn direction on first detection of obstacle
                self.phase = "STEP_TURN"
                if abs(l_min - r_min) > 0.15:
                    self.turn_direction = -1 if l_min > r_min else 1
                elif abs(rec_heading) > 5.0:
                    self.turn_direction = 1 if rec_heading > 0 else -1
                else:
                    self.turn_direction = 1  # default right turn

                self.turn_locked = True
                self.turn_step_start_time = now
                self.steps_count = 1
                dir_str = "RIGHT" if self.turn_direction > 0 else "LEFT"
                print(f"[MAZE] Obstacle ahead ({c_min:.2f}m). Stop & turn in-place: {dir_str} (NO REVERSE)")

            elapsed_step = now - self.turn_step_start_time
            total_cycle = MAZE_STEP_PULSE_DURATION + MAZE_CHECK_PAUSE_DURATION
            dir_str = "RIGHT" if self.turn_direction > 0 else "LEFT"

            # Sub-phase A: Active rotation pulse (~0.42s = ~12-15°) strictly in place (fb=0, lr=0)
            if elapsed_step < MAZE_STEP_PULSE_DURATION:
                yaw_cmd = int(self.turn_direction * MAZE_TURN_YAW_SPEED)
                return (0, 0, ud, yaw_cmd), f"MAZE TURN: In-place {dir_str} Step #{self.steps_count} (obs: {c_min:.2f}m)"

            # Sub-phase B: Brief settle pause (~0.25s) to read depth map cleanly
            elif elapsed_step < total_cycle:
                return (0, 0, ud, 0), f"MAZE SCAN: Checking gap (Center: {c_min:.2f}m)"

            # Sub-phase C: Step cycle finished without opening -> start next step in SAME direction
            else:
                self.turn_step_start_time = now
                self.steps_count += 1
                yaw_cmd = int(self.turn_direction * MAZE_TURN_YAW_SPEED)
                return (0, 0, ud, yaw_cmd), f"MAZE TURN: Next {dir_str} step #{self.steps_count}"
