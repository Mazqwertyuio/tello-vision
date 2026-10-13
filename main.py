"""Tello Vision Autopilot: Thin-Line LiDAR, Maze Explorer & Inertial Route Tracker.

Main entry point:
  - Hardware connection & Telemetry (DroneClient)
  - Inertial dead-reckoning route tracker (InertialRouteTracker)
  - Upper-third thin-line LiDAR depth synthesis (DepthEstimator, VisualLiDARScanner)
  - Human target tracking (PersonDetector)
  - Autonomous Collision Prevention Shield (CollisionShield)
  - Mode 1: Reactive Maze Room Explorer (RoomExplorer) - Forward flight & ~10° turn on obstacle
  - Mode 2: Human Target Follower (PersonFollower) - Follows only when passage is clear
  - Laser Beam HUD, Polar Radar & Route Minimap (HUDRenderer)
"""

import time
from pathlib import Path
from datetime import datetime
import cv2
import winsound
import numpy as np

from config import WINDOW_TITLE, MANUAL_SPEED
from drone import DroneClient, InertialRouteTracker
from ai import DepthEstimator, PersonDetector
from lidar import VisualLiDARScanner, SpatialObstacleMemory, CollisionShield
from modes import RoomExplorer, PersonFollower
from ui import HUDRenderer


def main():
    # 1. Initialize Drone hardware & Inertial Route Tracker
    drone = DroneClient()
    route_tracker = InertialRouteTracker()
    connected = drone.connect()
    if not connected:
        print("[WARNING] Running in simulation/offline mode (Drone not connected).")

    # 2. Initialize AI Modules
    print("\n[AI] Initializing Neural Networks...")
    try:
        depth_estimator = DepthEstimator()
    except Exception as exc:
        print(f"[ERROR] Failed to load Depth Anything V2: {exc}")
        return

    person_detector = PersonDetector()

    # 3. Initialize Thin-Line LiDAR & Safety Modules
    lidar_scanner = VisualLiDARScanner()
    polar_memory = SpatialObstacleMemory()
    collision_shield = CollisionShield()

    # 4. Initialize Flight Modes (Maze Explorer & Follower)
    explorer = RoomExplorer()
    follower = PersonFollower()

    # 5. Initialize UI
    hud = HUDRenderer()
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    # State variables
    active_mode = "MANUAL"  # "MANUAL", "EXPLORE", "FOLLOW"
    status_msg = "Ready. Press [T] to Takeoff, [E] for Maze Explore, [F] for Follow."
    manual_rc = (0, 0, 0, 0)
    manual_active_until = 0.0

    # Mouse Click Callback for HUD Buttons
    def on_mouse_click(event, x, y, flags, param):
        nonlocal active_mode, status_msg
        if event == cv2.EVENT_LBUTTONDOWN:
            btn = hud.check_button_click(x, y)
            if btn == "takeoff":
                drone.takeoff()
                route_tracker.reset()
                status_msg = "Takeoff command sent. Route tracker started."
            elif btn == "land":
                drone.land()
                active_mode = "MANUAL"
                status_msg = "Landing initiated. Mode reset to MANUAL."
            elif btn == "explore":
                if active_mode == "EXPLORE":
                    active_mode = "MANUAL"
                    status_msg = "Maze Mode stopped. Switched to MANUAL."
                else:
                    active_mode = "EXPLORE"
                    explorer.reset(drone.get_telemetry().get("yaw", 0.0))
                    status_msg = "Mode 1: Maze Room Explorer active (Fly forward / 10° turn on obstacle)."
                    try:
                        winsound.Beep(1200, 100)
                    except Exception:
                        pass
            elif btn == "follow":
                if active_mode == "FOLLOW":
                    active_mode = "MANUAL"
                    status_msg = "Follow Mode stopped. Switched to MANUAL."
                else:
                    active_mode = "FOLLOW"
                    follower.reset()
                    status_msg = "Mode 2: Person Follower active (Forward only when passage is clear)."
                    try:
                        winsound.Beep(1400, 100)
                    except Exception:
                        pass
            elif btn == "view":
                # Cycle view: RADAR -> MAP -> OFF
                if hud.active_view == "RADAR":
                    hud.active_view = "MAP"
                elif hud.active_view == "MAP":
                    hud.active_view = "OFF"
                else:
                    hud.active_view = "RADAR"

    cv2.setMouseCallback(WINDOW_TITLE, on_mouse_click)

    print("\n================ CONTROLS ================")
    print("  [T]         : Takeoff & start route tracking")
    print("  [L] / Space : Land")
    print("  [E]         : Toggle Mode 1 (Maze Room Exploration)")
    print("  [F]         : Toggle Mode 2 (Person Follower)")
    print("  [V] / [M]   : Toggle View (LiDAR Radar / Inertial Route Map / Off)")
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

            # If no frame yet, show blank canvas
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
            curr_yaw = telemetry.get("yaw", 0.0)

            # ---------------------------------------------------------
            # 1. Neural Depth & Thin-Line LiDAR Scan (Upper Third Horizon)
            # ---------------------------------------------------------
            norm_depth, depth_colormap, depth_ms = depth_estimator.estimate(frame)
            scan_res = lidar_scanner.scan(norm_depth)

            # ---------------------------------------------------------
            # 2. Human Detection (YuNet / Haar)
            # ---------------------------------------------------------
            faces = person_detector.detect(frame)
            target_face = person_detector.select_target(faces, frame.shape)

            # ---------------------------------------------------------
            # 3. Flight Mode State Machine
            # ---------------------------------------------------------
            raw_rc = (0, 0, 0, 0)
            mode_status = ""

            if now < manual_active_until:
                raw_rc = manual_rc
                mode_status = "Manual pilot control active"
            elif active_mode == "EXPLORE":
                raw_rc, mode_status = explorer.update(scan_res, telemetry, now)
                status_msg = mode_status
            elif active_mode == "FOLLOW":
                raw_rc, mode_status = follower.update(target_face, frame.shape, scan_res, now)
                status_msg = mode_status
            else:
                raw_rc = (0, 0, 0, 0)

            # ---------------------------------------------------------
            # 4. Collision Shield (LiDAR Obstacle Protection)
            # ---------------------------------------------------------
            is_manual = (now < manual_active_until)
            safe_rc, shield_state = collision_shield.filter_command(
                raw_rc, scan_res, flank_left_m=mem_l, flank_right_m=mem_r,
                allow_reverse=is_manual
            )

            # Send safe filtered command to drone
            drone.send_rc(*safe_rc)

            # ---------------------------------------------------------
            # 5. Inertial Route Tracker Update
            # ---------------------------------------------------------
            route_tracker.update(telemetry, rc_command=safe_rc, now=now)
            minimap = route_tracker.render_minimap(size=190)

            # ---------------------------------------------------------
            # 6. Render HUD with Laser Beam, Telemetry & Minimap
            # ---------------------------------------------------------
            hud_frame = hud.draw_hud(
                frame,
                telemetry,
                active_mode,
                shield_state,
                scan_res,
                faces,
                target_face,
                status_msg,
                depth_ms,
                minimap_img=minimap,
                total_dist_m=route_tracker.total_distance_m,
            )

            cv2.imshow(WINDOW_TITLE, hud_frame)

            # ---------------------------------------------------------
            # 7. Keyboard Handling
            # ---------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # Esc or Q
                print("[SYSTEM] Exiting flight controller...")
                break

            elif key == ord("t"):  # Takeoff
                drone.takeoff()
                route_tracker.reset()
                status_msg = "Takeoff command sent. Route tracker started."

            elif key in (ord("l"), ord(" ")):  # Land
                drone.land()
                active_mode = "MANUAL"
                status_msg = "Landing initiated."

            elif key == ord("e"):  # Toggle Explore
                if active_mode == "EXPLORE":
                    active_mode = "MANUAL"
                    status_msg = "Maze Mode stopped. Switched to MANUAL."
                else:
                    active_mode = "EXPLORE"
                    explorer.reset(curr_yaw)
                    status_msg = "Mode 1: Maze Room Explorer active."
                    try:
                        winsound.Beep(1200, 100)
                    except Exception:
                        pass

            elif key == ord("f"):  # Toggle Follow
                if active_mode == "FOLLOW":
                    active_mode = "MANUAL"
                    status_msg = "Follow Mode stopped. Switched to MANUAL."
                else:
                    active_mode = "FOLLOW"
                    follower.reset()
                    status_msg = "Mode 2: Person Follower active."
                    try:
                        winsound.Beep(1400, 100)
                    except Exception:
                        pass

            elif key in (ord("v"), ord("m")):  # Cycle View
                if hud.active_view == "RADAR":
                    hud.active_view = "MAP"
                elif hud.active_view == "MAP":
                    hud.active_view = "OFF"
                else:
                    hud.active_view = "RADAR"

            # Manual Flight Keys (W, A, S, D)
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
        # Save route log on teardown
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        route_file = Path("debug") / f"route_{run_stamp}.csv"
        try:
            route_tracker.save_to_csv(route_file)
            print(f"[SYSTEM] Flight route saved to {route_file}")
        except Exception:
            pass

        print("\n[SYSTEM] Shutting down...")
        drone.disconnect()
        cv2.destroyAllWindows()
        print("[SYSTEM] Done.")


if __name__ == "__main__":
    main()
