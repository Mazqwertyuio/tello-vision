"""Real-Time Interactive Flight Viewer & Trajectory Visualizer with Depth Anything V2 (DA2).

Runs live in real-time with an interactive OpenCV window:
- Real-time DA2 depth estimation (~18-20 ms on CUDA).
- Live 3D AR green flight path ribbon projected directly onto the camera feed.
- Live Top-Down Radar Mini-Map (BEV) comparing Old vs New Planned routes.
- Live RC Command HUD & Room Geometry interpretations.
- Interactive calibration of the Flight Window (size, position, floor cutoff):
    * Trackbars: Y-Top, Floor Cutoff Y, Width, X-Center.
    * Hotkeys: W/S (Up/Down), A/D (Left/Right), [ / ] (Width), U/J (Floor Cutoff).
    * P: Save calibration to disk (flight_window_config.json).
- Playback controls: SPACE (Pause), Arrows (Step), Trackbar (Scrub), R (Rewind).
"""

import argparse
from pathlib import Path
import sys
import time
import cv2
import numpy as np

from da2_depth import DA2DepthEstimator
from room_geometry_da2 import RoomGeometryEvaluator
from trajectory_planner import FlightTrajectoryPlanner, render_bev_minimap, draw_ar_perspective_path
from test_da2_flight_offline import find_latest_flight_dir, load_telemetry, render_evaluation_hud
from room_mapper import RoomOccupancyMapper
from flight_window_config import (
    load_flight_window_config, save_flight_window_config, get_pixel_bounds, DEFAULT_CONFIG
)


def main():
    parser = argparse.ArgumentParser(description="Real-Time DA2 Flight Navigator & Trajectory Viewer")
    parser.add_argument("flight", nargs="?", default="latest", help="Flight folder name in debug/ or full path")
    parser.add_argument("--fps", type=float, default=None, help="Playback speed FPS (default: native video FPS)")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO semantic door tracker")
    args = parser.parse_args()

    base_debug = Path(__file__).resolve().parent / "debug"

    if args.flight == "latest":
        flight_dir = find_latest_flight_dir(base_debug)
    else:
        cand = Path(args.flight)
        if cand.exists():
            flight_dir = cand.resolve()
        else:
            flight_dir = (base_debug / args.flight).resolve()

    if not flight_dir or not flight_dir.exists():
        print(f"Error: Flight folder '{args.flight}' not found in {base_debug}")
        return

    # Video & telemetry
    video_path = flight_dir / "flight_raw.mp4"
    if not video_path.exists():
        video_path = flight_dir / "flight_airborne_clean.mp4"

    if not video_path.exists():
        print(f"Error: No video found in {flight_dir}")
        return

    csv_path = flight_dir / "flight_telemetry.csv"
    if not csv_path.exists():
        csv_path = flight_dir / "commands.csv"

    print("=" * 75)
    print("TELLO REAL-TIME DA2 NAVIGATION & INTERACTIVE WINDOW CALIBRATOR")
    print(f"Directory:    {flight_dir.name}")
    print(f"Video file:   {video_path.name}")
    print(f"Telemetry:    {csv_path.name if csv_path.exists() else 'None'}")
    print("Controls:")
    print("  SPACE       - Pause / Resume real-time playback")
    print("  LEFT/RIGHT  - Step 1 frame backward / forward")
    print("  R           - Rewind to start")
    print("Flight Window Calibration Controls:")
    print("  W / S       - Move Flight Window Up / Down")
    print("  A / D       - Move Flight Window Left / Right")
    print("  [ / ]       - Decrease / Increase Corridor Width")
    print("  U / J       - Move Floor Cutoff Line Up (Cut More) / Down")
    print("  P           - Save calibration to flight_window_config.json")
    print("  O           - Reset calibration to defaults")
    print("  Q / ESC     - Quit viewer")
    print("=" * 75)

    # Models initialization
    print("Loading Depth Anything V2 on CUDA...")
    da2 = DA2DepthEstimator(input_size=252)

    # Window configuration
    win_cfg = load_flight_window_config()
    da2.window_cfg = win_cfg

    yolo_model = None
    if not args.no_yolo:
        try:
            from ultralytics import YOLO
            yolo_p = Path("yolov8s-worldv2.pt")
            if yolo_p.exists():
                print("Loading YOLO-World v2 for semantic doorways...")
                yolo_model = YOLO(str(yolo_p))
                yolo_model.set_classes(["door", "door frame", "doorway", "person", "wall"])
                print("YOLO-World ready!")
        except Exception as e:
            print(f"YOLO skipped ({e})")

    evaluator = RoomGeometryEvaluator(yolo_detector=yolo_model)
    planner = FlightTrajectoryPlanner()
    planner.window_cfg = win_cfg
    mapper = RoomOccupancyMapper()
    telemetry = load_telemetry(csv_path)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_fps = args.fps or native_fps
    delay_ms = max(1, int(1000.0 / target_fps))

    window_name = f"Tello Vision Real-Time: {flight_dir.name} [DA2 & Interactive Window Calibrator]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    current_frame_idx = 0
    is_paused = False
    save_msg_until = 0.0

    # Trackbar callbacks
    def sync_config_to_models():
        da2.window_cfg = win_cfg
        planner.window_cfg = win_cfg

    def on_trackbar_timeline(val):
        nonlocal current_frame_idx
        current_frame_idx = val
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        da2.reset()
        mapper.reset()

    def on_trackbar_ytop(val):
        win_cfg["y_min_pct"] = min(val, win_cfg["y_max_pct"] - 4)
        sync_config_to_models()

    def on_trackbar_ybottom(val):
        win_cfg["y_max_pct"] = max(val, win_cfg["y_min_pct"] + 4)
        sync_config_to_models()

    def on_trackbar_width(val):
        win_cfg["width_pct"] = max(6, min(90, val))
        sync_config_to_models()

    def on_trackbar_xcenter(val):
        win_cfg["x_center_pct"] = max(10, min(90, val))
        sync_config_to_models()

    # Create interactive Trackbars
    cv2.createTrackbar("Timeline", window_name, 0, max(1, total_frames - 1), on_trackbar_timeline)
    cv2.createTrackbar("Y-Top (%)", window_name, win_cfg["y_min_pct"], 100, on_trackbar_ytop)
    cv2.createTrackbar("Floor Cutoff Y (%)", window_name, win_cfg["y_max_pct"], 100, on_trackbar_ybottom)
    cv2.createTrackbar("Corridor Width (%)", window_name, win_cfg["width_pct"], 100, on_trackbar_width)
    cv2.createTrackbar("X-Center (%)", window_name, win_cfg["x_center_pct"], 100, on_trackbar_xcenter)

    def sync_trackbars_ui():
        try:
            cv2.setTrackbarPos("Y-Top (%)", window_name, win_cfg["y_min_pct"])
            cv2.setTrackbarPos("Floor Cutoff Y (%)", window_name, win_cfg["y_max_pct"])
            cv2.setTrackbarPos("Corridor Width (%)", window_name, win_cfg["width_pct"])
            cv2.setTrackbarPos("X-Center (%)", window_name, win_cfg["x_center_pct"])
        except Exception:
            pass
        sync_config_to_models()

    while True:
        t_loop0 = time.perf_counter()

        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame_idx = 0
                da2.reset()
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                try:
                    cv2.setTrackbarPos("Timeline", window_name, current_frame_idx)
                except Exception:
                    pass
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

        if frame is None:
            break

        # Live DA2 Depth Inference
        norm_depth, depth_colormap, sectors, center_max, infer_ms = da2.predict(frame)

        # Live Room Geometry & Command Planning
        eval_res = evaluator.evaluate(frame, norm_depth, sectors, center_max)

        telem_row = telemetry.get(current_frame_idx + 1, {})

        # Render combined HUD: Camera (with AR Corridor + BEV Radar) | DA2 Depth Map + Room Map
        combined_view = render_evaluation_hud(
            frame, norm_depth, depth_colormap, eval_res, telem_row,
            current_frame_idx + 1, total_frames, target_fps, infer_ms, planner=planner, mapper=mapper
        )

        # Calibration banner overlay in top-center
        c_banner = f"CALIBRATION: Y=[{win_cfg['y_min_pct']}%..{win_cfg['y_max_pct']}%] X_Center={win_cfg['x_center_pct']}% Width={win_cfg['width_pct']}% | [P]: Save to Disk"
        if time.time() < save_msg_until:
            c_banner = ">>> CALIBRATION SAVED TO DISK (flight_window_config.json)! <<<"
            b_col = (0, 255, 0)
        else:
            b_col = (0, 220, 255)

        cv2.rectangle(combined_view, (combined_view.shape[1] // 2 - 380, 5), (combined_view.shape[1] // 2 + 380, 30), (15, 15, 15), -1)
        cv2.putText(combined_view, c_banner, (combined_view.shape[1] // 2 - 370, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, b_col, 1)

        # Scale down if too wide for 1080p desktop monitors
        ch, cw = combined_view.shape[:2]
        if cw > 1550:
            scale = 1550.0 / cw
            combined_view = cv2.resize(combined_view, (int(cw * scale), int(ch * scale)))

        cv2.imshow(window_name, combined_view)

        # Compute remaining wait time to maintain real-time FPS
        elapsed_loop_ms = (time.perf_counter() - t_loop0) * 1000.0
        wait_time = 0 if is_paused else max(1, int(delay_ms - elapsed_loop_ms))

        key = cv2.waitKeyEx(wait_time)

        if key in (ord("q"), 27):  # ESC or Q: Quit
            break
        elif key == ord(" "):  # Space: Pause/Resume
            is_paused = not is_paused
        elif key in (2424832, 65361):  # Left Arrow: Step back
            is_paused = True
            current_frame_idx = max(0, current_frame_idx - 1)
            cv2.setTrackbarPos("Timeline", window_name, current_frame_idx)
            da2.reset()
        elif key in (2555904, 65363):  # Right Arrow: Step forward
            is_paused = True
            current_frame_idx = min(total_frames - 1, current_frame_idx + 1)
            cv2.setTrackbarPos("Timeline", window_name, current_frame_idx)
            da2.reset()
        elif key == ord("r"):  # R: Rewind
            current_frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cv2.setTrackbarPos("Timeline", window_name, 0)
            da2.reset()
            mapper.reset()
            is_paused = False

        # --- Interactive Window Calibration Keys ---
        elif key in (ord("w"), ord("W")):  # W: Move window Up
            step = min(win_cfg["y_min_pct"], 2)
            win_cfg["y_min_pct"] -= step
            win_cfg["y_max_pct"] -= step
            sync_trackbars_ui()
        elif key in (ord("s"), ord("S")):  # S: Move window Down
            step = min(100 - win_cfg["y_max_pct"], 2)
            win_cfg["y_min_pct"] += step
            win_cfg["y_max_pct"] += step
            sync_trackbars_ui()
        elif key in (ord("a"), ord("A")):  # A: Move window Left
            win_cfg["x_center_pct"] = max(10, win_cfg["x_center_pct"] - 2)
            sync_trackbars_ui()
        elif key in (ord("d"), ord("D")):  # D: Move window Right
            win_cfg["x_center_pct"] = min(90, win_cfg["x_center_pct"] + 2)
            sync_trackbars_ui()
        elif key in (ord("["), ord("{")):  # [: Narrow Corridor Width
            win_cfg["width_pct"] = max(6, win_cfg["width_pct"] - 2)
            sync_trackbars_ui()
        elif key in (ord("]"), ord("}")):  # ]: Widen Corridor Width
            win_cfg["width_pct"] = min(90, win_cfg["width_pct"] + 2)
            sync_trackbars_ui()
        elif key in (ord("u"), ord("U")):  # U: Raise Floor Cutoff Line (Cut more floor)
            win_cfg["y_max_pct"] = max(win_cfg["y_min_pct"] + 4, win_cfg["y_max_pct"] - 2)
            sync_trackbars_ui()
        elif key in (ord("j"), ord("J")):  # J: Lower Floor Cutoff Line
            win_cfg["y_max_pct"] = min(98, win_cfg["y_max_pct"] + 2)
            sync_trackbars_ui()
        elif key in (ord("p"), ord("P")):  # P: Save Calibration to Disk
            save_flight_window_config(win_cfg)
            save_msg_until = time.time() + 2.5
        elif key in (ord("o"), ord("O")):  # O: Reset Calibration to Defaults
            win_cfg = DEFAULT_CONFIG.copy()
            sync_trackbars_ui()
            save_flight_window_config(win_cfg)
            save_msg_until = time.time() + 2.0

    cap.release()
    cv2.destroyAllWindows()
    print("Real-time viewer closed.")


if __name__ == "__main__":
    main()
