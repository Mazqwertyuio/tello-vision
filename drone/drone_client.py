"""DJI Tello connection, command abstraction, and telemetry interface.

Strictly encapsulates socket communication, video streaming, and flight commands.
No navigation or AI logic lives here.
"""

import time
from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    from djitellopy import Tello
    HAS_TELLO = True
except ImportError:
    HAS_TELLO = False


class DroneClient:
    """Manages connection, RC commands, video capture, and state for DJI Tello."""

    def __init__(self, command_timeout: float = 0.08):
        if not HAS_TELLO:
            raise ImportError("djitellopy is required. Install via 'pip install djitellopy'.")

        self.tello = Tello()
        self.command_timeout = command_timeout
        self.frame_reader = None
        self.airborne = False
        self.connected = False
        self.last_battery_check = 0.0
        self.cached_battery = 100
        self.last_cmd = (0, 0, 0, 0)
        self.last_cmd_time = 0.0

    def connect(self) -> bool:
        """Connect to drone, start video stream, and fetch initial battery."""
        print("[DRONE] Connecting to DJI Tello...")
        try:
            self.tello.connect()
            self.tello.TIME_BTW_RC_CONTROL_COMMANDS = self.command_timeout
            self.connected = True

            try:
                self.cached_battery = self.tello.get_battery()
            except Exception:
                self.cached_battery = 100
            print(f"[DRONE] Connected successfully! Battery: {self.cached_battery}%")

            print("[DRONE] Enabling video stream...")
            self.tello.streamon()
            self.frame_reader = self.tello.get_frame_read()
            print("[DRONE] Video stream active.")
            return True
        except Exception as exc:
            print(f"[DRONE ERROR] Failed to connect to Tello: {exc}")
            self.connected = False
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Fetch current BGR frame from camera."""
        if not self.connected or self.frame_reader is None:
            return None
        frame = self.frame_reader.frame
        if frame is None or frame.size == 0:
            return None
        return frame

    def takeoff(self) -> bool:
        """Execute takeoff command."""
        if not self.connected:
            return False
        try:
            print("[DRONE] Initiating takeoff...")
            self.tello.takeoff()
            self.airborne = True
            return True
        except Exception as exc:
            print(f"[DRONE ERROR] Takeoff failed: {exc}")
            return False

    def land(self) -> bool:
        """Execute land command."""
        if not self.connected:
            return False
        try:
            print("[DRONE] Initiating landing...")
            self.send_rc(0, 0, 0, 0)
            self.tello.land()
            self.airborne = False
            return True
        except Exception as exc:
            print(f"[DRONE ERROR] Land failed: {exc}")
            self.airborne = False
            return False

    def emergency(self):
        """Emergency engine cut."""
        if not self.connected:
            return
        try:
            self.tello.emergency()
            self.airborne = False
        except Exception as exc:
            print(f"[DRONE ERROR] Emergency stop failed: {exc}")

    def send_rc(self, left_right: int, fwd_back: int, up_down: int, yaw: int):
        """Send RC velocity commands [-100..100] with rate limiting."""
        if not self.connected or not self.airborne:
            return

        cmd = (int(left_right), int(fwd_back), int(up_down), int(yaw))
        now = time.monotonic()

        # Deduplicate identical stop commands if recently sent
        if cmd == (0, 0, 0, 0) and self.last_cmd == (0, 0, 0, 0) and (now - self.last_cmd_time < 0.15):
            return

        try:
            self.tello.send_rc_control(*cmd)
            self.last_cmd = cmd
            self.last_cmd_time = now
        except Exception as exc:
            print(f"[DRONE ERROR] Failed to send RC control: {exc}")

    def get_telemetry(self) -> Dict[str, Any]:
        """Read flight telemetry (battery, height, IMU, velocities)."""
        if not self.connected:
            return {
                "battery": 0, "tof_cm": 65, "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                "vx": 0.0, "vy": 0.0, "vz": 0.0, "airborne": False
            }

        now = time.monotonic()
        if now - self.last_battery_check > 5.0:
            try:
                self.cached_battery = self.tello.get_battery()
            except Exception:
                pass
            self.last_battery_check = now

        # Read full state dict if available
        state = {}
        try:
            state = self.tello.get_current_state() or {}
        except Exception:
            state = {}

        tof = state.get("tof")
        if tof is None or tof <= 0:
            try:
                tof = self.tello.get_distance_tof()
            except Exception:
                tof = None

        yaw = float(state.get("yaw", 0) or 0)
        pitch = float(state.get("pitch", 0) or 0)
        roll = float(state.get("roll", 0) or 0)

        # Velocities: vgx/vgy/vgz are in dm/s or cm/s in Tello state
        vx = float(state.get("vgx", 0) or 0) / 100.0  # converted to m/s
        vy = float(state.get("vgy", 0) or 0) / 100.0  # converted to m/s
        vz = float(state.get("vgz", 0) or 0) / 100.0

        if yaw == 0.0 and pitch == 0.0 and roll == 0.0:
            try:
                yaw = float(self.tello.get_yaw())
                pitch = float(self.tello.get_pitch())
                roll = float(self.tello.get_roll())
            except Exception:
                pass

        return {
            "battery": self.cached_battery,
            "tof_cm": int(tof) if (tof is not None and tof > 0) else 65,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "airborne": self.airborne,
        }

    def disconnect(self):
        """Gracefully land drone and stop stream."""
        if not self.connected:
            return
        print("[DRONE] Disconnecting...")
        if self.airborne:
            try:
                self.land()
            except Exception:
                pass
        try:
            self.tello.streamoff()
        except Exception:
            pass
        self.connected = False
        print("[DRONE] Disconnected.")
