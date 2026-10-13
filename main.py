"""Tello Vision Autopilot: Depth Obstacle Avoidance & Person Silhouette Tracking.

Streamlined, reactive, robust:
  - Depth Map Obstacle Avoidance (Depth Anything V2)
  - Real-Time Human Silhouette Following (YOLOv8n on CUDA)
  - Mode 1: Free Room Exploration (Explore)
  - Mode 2: Person Silhouette Following (Follow)
"""

import time
import cv2
import numpy as np

from config import WINDOW_TITLE, MANUAL_SPEED
from drone import DroneClient
from ai import DepthEstimator, PersonDetector
from lidar import VisualLiDARScanner, CollisionShield
from modes import RoomExplorer, PersonFollower
from ui import HUDRenderer


def main():
    # 1. Initialize Drone
    drone = DroneClient()
    connected = drone.connect()
    if not connected:
        print("[WARNING] Running in simulation/offline mode (Drone not connected).")

    # 2. Initialize AI Models
    print("\n[AI] Initializing Depth Anything V2 & YOLO Silhouette Detector...")
    try:
        depth_estimator = DepthEstimator()
    except Exception as exc:
        print(f"[ERROR] Failed to load Depth Anything V2: {exc}")
        return

    person_detector = PersonDetector()

    # 3. Initialize Obstacle Analyzer & Safety Shield
    depth_analyzer = VisualLiDARScanner()
    collision_shield = CollisionShield()

    # 4. Initialize Flight Modes
    explorer = RoomExplorer()
    follower = PersonFollower()

    # 5. Initialize HUD UI
    hud = HUDRenderer()
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    # State variables
    active_mode = "MANUAL"  # "MANUAL", "EXPLORE", "FOLLOW"
    status_msg = "Ready. [T] Takeoff | [E] Explore (Avoid Obstacles) | [F] Follow Silhouette"
    manual_rc = (0, 0, 0, 0)
    manual_active_until = 0.0

    # Mouse Click Callback for UI Buttons
    def on_mouse_click(event, x, y, flags, param):
        nonlocal active_mode, status_msg
        if event == cv2.EVENT_LBUTTONDOWN:
            btn = hud.check_button_click(x, y)
            if btn == "takeoff":
                drone.takeoff()
                status_msg = "Takeoff command sent."
            elif btn == "land":
                drone.land()
                active_mode = "MANUAL"
                status_msg = "Landing. Mode set to MANUAL."
            elif btn == "explore":
                if active_mode == "EXPLORE":
                    active_mode = "MANUAL"
                    status_msg = "Explore Mode stopped. MANUAL."
                else:
                    active_mode = "EXPLORE"
                    explorer.reset()
                    status_msg = "Mode 1: Room Exploration (Depth Avoidance) Active."
            elif btn == "follow":
                if active_mode == "FOLLOW":
                    active_mode = "MANUAL"
                    status_msg = "Follow Mode stopped. MANUAL."
                else:
                    active_mode = "FOLLOW"
                    follower.reset()
                    status_msg = "Mode 2: Person Silhouette Following Active."

    cv2.setMouseCallback(WINDOW_TITLE, on_mouse_click)

    print("\n================ CONTROLS ================")
    print("  [T]         : Takeoff")
    print("  [L] / Space : Land")
    print("  [E]         : Toggle Mode 1 (Explore & Depth Obstacle Avoidance)")
    print("  [F]         : Toggle Mode 2 (Follow Person Silhouette)")
    print("  [W/S]       : Forward / Backward (Manual)")
    print("  [A/D]       : Left / Right (Manual)")
    print("  [Up/Down]   : Ascend / Descend (Manual)")
    print("  [Left/Right]: Yaw Rotate (Manual)")
    print("  [Esc] / [Q] : Emergency Stop & Quit")
    print("==========================================\n")

    try:
        while True:
            now = time.monotonic()
            frame = drone.get_frame()

            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame, "Waiting for video stream...", (140, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
                )
                cv2.imshow(WINDOW_TITLE, frame)
                if cv2.waitKey(30) & 0xFF in (27, ord("q")):
                    break
                continue

            telemetry = drone.get_telemetry()

            # ---------------------------------------------------------
            # 1. Depth Map Inference & 3-Sector Obstacle Analysis
            # ---------------------------------------------------------
            norm_depth, depth_colormap, depth_ms = depth_estimator.estimate(frame)
            scan_res = depth_analyzer.scan(norm_depth)

            # ---------------------------------------------------------
            # 2. Human Silhouette Detection (YOLOv8n on GPU)
            # ---------------------------------------------------------
            persons = person_detector.detect(frame)
            target_person = person_detector.select_target(persons, frame.shape)

            # ---------------------------------------------------------
            # 3. Autonomous Flight Modes
            # ---------------------------------------------------------
            raw_rc = (0, 0, 0, 0)
            mode_status = ""

            if now < manual_active_until:
                # Pilot manual control
                raw_rc = manual_rc
                mode_status = "Manual pilot control active"
            elif active_mode == "EXPLORE":
                # Mode 1: Depth map reactive obstacle avoidance
                raw_rc, mode_status = explorer.update(scan_res, telemetry, now)
                status_msg = mode_status
            elif active_mode == "FOLLOW":
                # Mode 2: Person silhouette following with forward depth protection
                raw_rc, mode_status = follower.update(target_person, frame.shape, scan_res, now)
                status_msg = mode_status
            else:
                raw_rc = (0, 0, 0, 0)

            # ---------------------------------------------------------
            # 4. Collision Safety Shield
            # ---------------------------------------------------------
            is_manual = (now < manual_active_until)
            safe_rc, shield_state = collision_shield.filter_command(
                raw_rc, scan_res, allow_reverse=is_manual
            )

            # Send safe filtered command to drone
            drone.send_rc(*safe_rc)

            # ---------------------------------------------------------
            # 5. Render HUD Display
            # ---------------------------------------------------------
            hud_frame = hud.draw_hud(
                frame,
                telemetry,
                active_mode,
                shield_state,
                scan_res,
                persons,
                target_person,
                status_msg,
                depth_ms,
            )

            cv2.imshow(WINDOW_TITLE, hud_frame)

            # ---------------------------------------------------------
            # 6. Keyboard Handling
            # ---------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # Esc or Q
                print("[SYSTEM] Exiting...")
                break

            elif key == ord("t"):  # Takeoff
                drone.takeoff()
                status_msg = "Takeoff command sent."

            elif key in (ord("l"), ord(" ")):  # Land
                drone.land()
                active_mode = "MANUAL"
                status_msg = "Landing initiated."

            elif key == ord("e"):  # Toggle Explore
                if active_mode == "EXPLORE":
                    active_mode = "MANUAL"
                    status_msg = "Explore Mode stopped. MANUAL."
                else:
                    active_mode = "EXPLORE"
                    explorer.reset()
                    status_msg = "Mode 1: Explore (Depth Avoidance) Active."

            elif key == ord("f"):  # Toggle Follow
                if active_mode == "FOLLOW":
                    active_mode = "MANUAL"
                    status_msg = "Follow Mode stopped. MANUAL."
                else:
                    active_mode = "FOLLOW"
                    follower.reset()
                    status_msg = "Mode 2: Person Silhouette Following Active."

            # Manual Flight Keys
            elif key == ord("w"):
                manual_rc = (0, MANUAL_SPEED, 0, 0)
                manual_active_until = now + 0.25
            elif key == ord("s"):
                manual_rc = (0, -MANUAL_SPEED, 0, 0)
                manual_active_until = now + 0.25
            elif key == ord("a"):
                manual_rc = (-MANUAL_SPEED, 0, 0, 0)
                manual_active_until = now + 0.25
            elif key == ord("d"):
                manual_rc = (MANUAL_SPEED, 0, 0, 0)
                manual_active_until = now + 0.25

    finally:
        print("\n[SYSTEM] Shutting down...")
        drone.disconnect()
        cv2.destroyAllWindows()
        print("[SYSTEM] Done.")


if __name__ == "__main__":
    main()
