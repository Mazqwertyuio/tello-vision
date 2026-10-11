"""Offline flight replay and obstacle detection test harness.

Replays recorded Tello flight video (flight_raw.mp4) synchronized with
telemetry and RC commands (flight_telemetry.csv) with real-time Monocular
Depth Estimation (96x96 / 128x96) and collision avoidance simulation.
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from door_navigator import (
    DoorwayDetector, draw_door_hud,
    analyze_forward_corridor, draw_corridor_hud
)

try:
    from da2_depth import DA2DepthEstimator
    HAS_DA2 = True
except ImportError:
    HAS_DA2 = False


class MonoDepthEstimator:
    """Lightweight Monocular Depth Estimation on CUDA/CPU (supports 96x96 and 128x96)
    with real-time temporal flicker reduction and smooth range normalization.
    """

    def __init__(self, input_size=(128, 96), temporal_alpha=0.75):
        self.input_size = input_size  # (width, height)
        self.temporal_alpha = temporal_alpha  # 0.70-0.75: fast response (~15ms) without flicker
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading MonoDepth (MiDaS Small) on {self.device} at {self.input_size[0]}x{self.input_size[1]}...")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(self.device)
        self.model.eval()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Real-time temporal stabilization state
        self.smooth_depth = None
        self.running_min = None
        self.running_max = None
        self.smooth_sectors = None
        print("MonoDepth initialized with real-time temporal stabilizer!")

    def reset(self):
        """Reset temporal state when video is scrubbed or jumped."""
        self.smooth_depth = None
        self.running_min = None
        self.running_max = None
        self.smooth_sectors = None

    def set_resolution(self, width, height):
        self.input_size = (width, height)
        self.reset()
        print(f"Depth input resolution set to: {width}x{height}")

    def predict_crop(self, crop_bgr):
        """Stateless depth for a single image crop (e.g. a detected door + surrounding wall).

        Does NOT touch the temporal filter of the main stream. The crop gets its own
        normalization, so the depth contrast between door interior and wall is maximized.
        Returns float32 disparity map in [0, 1] (1 = closest, 0 = farthest).
        """
        ch, cw = crop_bgr.shape[:2]
        aspect = cw / max(1, ch)
        if aspect < 0.8:
            size = (128, 192)
        elif aspect > 1.25:
            size = (192, 128)
        else:
            size = (160, 160)
        small = cv2.resize(crop_bgr, size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            disp = self.model(tensor).squeeze().cpu().numpy().astype(np.float32)
        lo = float(np.percentile(disp, 2))
        hi = float(np.percentile(disp, 98))
        disp = np.clip((disp - lo) / max(hi - lo, 1e-4), 0.0, 1.0)
        return cv2.resize(disp, (cw, ch), interpolation=cv2.INTER_LINEAR)

    def predict(self, frame_bgr):
        """Predict depth map, apply temporal smoothing to eliminate flicker,
        and calculate rock-steady sector obstacle proximities in real time.
        """
        t0 = time.perf_counter()
        small = cv2.resize(frame_bgr, self.input_size)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            raw_depth = self.model(tensor).squeeze().cpu().numpy()

        infer_ms = (time.perf_counter() - t0) * 1000

        # 1. Temporal IIR / EMA filter on 2D depth map (removes frame-to-frame pixel noise)
        if self.smooth_depth is None or self.smooth_depth.shape != raw_depth.shape:
            self.smooth_depth = raw_depth.copy()
        else:
            self.smooth_depth = self.temporal_alpha * raw_depth + (1.0 - self.temporal_alpha) * self.smooth_depth

        # 2. Smooth running dynamic range (eliminates global brightness flashing / pumping)
        curr_min = float(np.percentile(self.smooth_depth, 2))
        curr_max = float(np.percentile(self.smooth_depth, 98))
        if self.running_min is None:
            self.running_min = curr_min
            self.running_max = curr_max
        else:
            self.running_min = 0.20 * curr_min + 0.80 * self.running_min
            self.running_max = 0.20 * curr_max + 0.80 * self.running_max

        scale = max(self.running_max - self.running_min, 1e-4)
        norm_depth = np.clip((self.smooth_depth - self.running_min) / scale * 255.0, 0, 255).astype(np.uint8)

        # 3. Spatial edge smoothing (3x3 Gaussian removes low-res quantization blockiness)
        norm_depth = cv2.GaussianBlur(norm_depth, (3, 3), 0)
        color_map = cv2.applyColorMap(norm_depth, cv2.COLORMAP_INFERNO)

        # 4. 3 Sectors: Left, Center, Right (middle 50% vertical zone)
        # Note: Central sector is the most optically reliable and aligns with forward flight corridor.
        h, w = norm_depth.shape
        mid_h0 = int(h * 0.25)
        mid_h1 = int(h * 0.75)
        sw = w // 3

        l_raw = float(norm_depth[mid_h0:mid_h1, 0:sw].mean())
        center_zone = norm_depth[mid_h0:mid_h1, sw:2 * sw]
        c_raw = float(center_zone.mean())
        r_raw = float(norm_depth[mid_h0:mid_h1, 2 * sw:].mean())
        curr_sectors = np.array([l_raw, c_raw, r_raw], dtype=np.float32)

        # 5. Sector hazard smoothing (prevents motor chatter on single-frame glitches)
        if self.smooth_sectors is None:
            self.smooth_sectors = curr_sectors
        else:
            self.smooth_sectors = 0.70 * curr_sectors + 0.30 * self.smooth_sectors

        left_val, center_val, right_val = self.smooth_sectors
        center_max = float(np.percentile(center_zone, 85))

        return norm_depth, color_map, (float(left_val), float(center_val), float(right_val)), float(center_max), infer_ms


def find_latest_flight_dir(base_debug_dir: Path) -> Path:
    """Find the most recent flight directory containing recordings."""
    if not base_debug_dir.exists():
        return None
    candidates = []
    for p in base_debug_dir.iterdir():
        if p.is_dir() and ((p / "flight_raw.mp4").exists() or (p / "flight_slideshow.mp4").exists()):
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def load_telemetry(csv_path: Path) -> dict:
    """Load telemetry indexed by frame_idx."""
    telemetry = {}
    if not csv_path.exists():
        return telemetry
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row.get("frame_idx", 0))
                telemetry[idx] = row
            except ValueError:
                continue
    return telemetry


def draw_hud(frame, telem_row, frame_idx, total_frames, is_paused, depth_info=None, door_info=None):
    """Render telemetry, face boxes, doorway tracking, and collision risk status."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top banner background
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
    # Bottom banner background
    cv2.rectangle(overlay, (0, h - 60), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Frame & Time status
    status_text = "PAUSED" if is_paused else "PLAYING"
    status_color = (0, 200, 255) if is_paused else (0, 255, 0)
    cv2.putText(frame, f"REPLAY [{status_text}]", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if telem_row:
        elapsed = telem_row.get("elapsed_s", "0.0")
        state = telem_row.get("state", "UNKNOWN")
        phase = telem_row.get("phase", "IDLE")
        battery = telem_row.get("battery", "?")

        cv2.putText(frame, f"Time: {elapsed}s | Battery: {battery}%", (15, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        phase_color = (0, 255, 0) if phase in ("CENTER", "FOLLOW") else (0, 180, 255)
        cv2.putText(frame, f"Mode: {state} | Phase: {phase}", (270, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2)

        # Draw RC commands
        lr = int(telem_row.get("lr", 0) or 0)
        fb = int(telem_row.get("fb", 0) or 0)
        ud = int(telem_row.get("ud", 0) or 0)
        yaw = int(telem_row.get("yaw", 0) or 0)

        # Check collision safety and room geometry intervention
        safe_action = ""
        if door_info:
            d_status = door_info.get("status", "WALL_OR_CLOSED")
            is_pass = door_info.get("passable", False)
            if is_pass and d_status in ["OPEN_DOORWAY", "OPEN_CORRIDOR"]:
                safe_action = f" [OPEN PASSAGE DETECTED: ALIGN & ENTER]"
                rc_color = (0, 255, 0)
            elif d_status == "CLOSED_DOOR":
                safe_action = f" [CLOSED DOOR AHEAD - STOP]"
                rc_color = (0, 0, 255)
            elif d_status == "DOORWAY_BLOCKED":
                safe_action = f" [DOOR BLOCKED BY OBSTACLE - STOP]"
                rc_color = (0, 140, 255)
            else:
                if depth_info:
                    _, (l_val, c_val, r_val), center_max, _ = depth_info
                    if c_val >= 160 or center_max >= 200:
                        safe_action = f" [CLOSE WALL/OBSTACLE - BRAKE (C={c_val:.0f})]"
                        rc_color = (0, 0, 255)
                    elif c_val >= 90:
                        safe_action = f" [WALL AHEAD - NO DOORWAY (C={c_val:.0f})]"
                        rc_color = (0, 180, 255)
                    else:
                        safe_action = f" [ENCLOSED ROOM - SCANNING (C={c_val:.0f})]"
                        rc_color = (200, 200, 200)
                else:
                    safe_action = " [WALL/NO PASSAGE]"
                    rc_color = (200, 200, 200)
        elif depth_info:
            _, (l_val, c_val, r_val), center_max, _ = depth_info
            if c_val >= 160:
                safe_action = f" [CLOSE WALL - BRAKE (C={c_val:.0f})]"
                rc_color = (0, 0, 255)
            else:
                safe_action = f" [WALL AHEAD (C={c_val:.0f})]"
                rc_color = (0, 200, 255)
        else:
            rc_color = (255, 255, 0)

        rc_str = f"RC -> Strafe: {lr:+d} | Fwd/Back: {fb:+d} | Up/Dn: {ud:+d} | Yaw: {yaw:+d}{safe_action}"
        cv2.putText(frame, rc_str, (15, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, rc_color, 1)

        # Door passage autopilot vector display
        if door_info and door_info.get("found"):
            d_status = door_info.get("status", "")
            dx = door_info.get("center_x_norm", 0.0)
            dy = door_info.get("center_y_norm", 0.0)
            is_pass = door_info.get("passable", False)
            if is_pass:
                d_color = (0, 255, 0)
                sim_yaw = int(dx * 55)
                sim_vert = int(-dy * 45)
                sim_fwd = 18 if abs(dx) < 0.15 else 0
                door_str = f"DOORWAY: {d_status} -> Autopilot (Yaw:{sim_yaw:+d}, Up:{sim_vert:+d}, Fwd:{sim_fwd:+d})"
            else:
                d_color = (0, 0, 255) if d_status == "CLOSED_DOOR" else (0, 165, 255)
                door_str = f"DOORWAY: {d_status} [HOLD POSITION - DO NOT ENTER]"
            cv2.putText(frame, door_str, (15, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.46, d_color, 2)

        # Draw recorded face bounding box
        fx = telem_row.get("face_x")
        fy = telem_row.get("face_y")
        fw = telem_row.get("face_w")
        fh = telem_row.get("face_h")
        fconf = telem_row.get("face_conf", "")
        if fx and fy and fw and fh:
            try:
                x, y, bw, bh = int(fx), int(fy), int(fw), int(fh)
                color = (0, 255, 0) if phase in ("CENTER", "FOLLOW") else (0, 200, 255)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                label = f"Target {fconf}" if fconf else "Target"
                cv2.putText(frame, label, (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            except ValueError:
                pass

    # Instructions in top right
    cv2.putText(frame, "N: Door Mode | D: Depth | C: Invert R/B | Space: Pause | Left/Right: Step",
                (w - 620, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


def draw_depth_sectors(depth_vis, sectors, center_max, infer_ms, res_str):
    """Draw sector proximity bars and indicators over the depth map."""
    h, w = depth_vis.shape[:2]
    l_val, c_val, r_val = sectors

    # Draw vertical sector dividing lines
    sw = w // 3
    cv2.line(depth_vis, (sw, 0), (sw, h), (100, 100, 100), 1)
    cv2.line(depth_vis, (2 * sw, 0), (2 * sw, h), (100, 100, 100), 1)

    # Highlight center flight corridor (most accurate depth zone)
    c_color = (0, 0, 255) if c_val >= 180 else ((0, 200, 255) if c_val >= 140 else (0, 255, 0))
    cv2.rectangle(depth_vis, (sw + 2, 2), (2 * sw - 2, h - 2), c_color, 2)

    # Sector labels and proximities
    def get_color(v):
        if v >= 180:
            return (0, 0, 255)  # Red (Wall danger!)
        elif v >= 140:
            return (0, 200, 255)  # Orange/Yellow (Caution)
        return (0, 255, 0)  # Green (Safe)

    cv2.putText(depth_vis, f"LEFT: {l_val:.0f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, get_color(l_val), 2)
    cv2.putText(depth_vis, f"CENTER: {c_val:.0f} (max={center_max:.0f})", (sw + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, get_color(c_val), 2)
    cv2.putText(depth_vis, f"RIGHT: {r_val:.0f}", (2 * sw + 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, get_color(r_val), 2)

    cv2.putText(depth_vis, f"Depth AI: {res_str} ({infer_ms:.1f}ms)", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    base_debug = Path(__file__).resolve().parent / "debug"

    # Allow passing custom directory as argument
    if len(sys.argv) > 1:
        flight_dir = Path(sys.argv[1]).resolve()
    else:
        flight_dir = find_latest_flight_dir(base_debug)

    if not flight_dir:
        print("No flight recording found yet!")
        return

    video_path = flight_dir / "flight_raw.mp4"
    if (flight_dir / "flight_slideshow.mp4").exists():
        test_cap = cv2.VideoCapture(str(video_path))
        if int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT)) < 10:
            video_path = flight_dir / "flight_slideshow.mp4"
        test_cap.release()

    csv_path = flight_dir / "flight_telemetry.csv"
    if not csv_path.exists():
        csv_path = flight_dir / "commands.csv"

    print("=" * 65)
    print("TELLO OFFLINE FLIGHT REPLAY & DEPTH TEST HARNESS")
    print(f"Directory:  {flight_dir}")
    print(f"Video file: {video_path}")
    print(f"Telemetry:  {csv_path}")
    print("Controls:")
    print("  SPACE       - Pause / Resume playback")
    print("  T           - Toggle 30cm Table Corridor Navigation overlay on/off")
    print("  N           - Toggle Doorway Passage Navigation overlay on/off")
    print("  D           - Toggle Monocular Depth Estimation on/off")
    print("  C           - Toggle Color Inversion (R/B Swap) for legacy videos")
    print("  1           - Switch Depth resolution to 96x96 (ultra-fast)")
    print("  2           - Switch Depth resolution to 128x96 (4:3 aspect)")
    print("  3           - Switch Depth model to Depth Anything V2 (DA2)")
    print("  LEFT ARROW  - Step 1 frame backward")
    print("  RIGHT ARROW - Step 1 frame forward")
    print("  R           - Rewind to beginning")
    print("  Q / ESC     - Exit replay")
    print("=" * 65)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1, int(1000.0 / fps))
    telemetry = load_telemetry(csv_path)

    window_name = "Tello Flight Replay with Monocular Depth"
    cv2.namedWindow(window_name)

    is_paused = False
    depth_enabled = True  # Enable depth estimation by default for testing!
    use_da2 = "--da2" in sys.argv
    if use_da2 and HAS_DA2:
        print("Initializing with Depth Anything V2 (DA2)...")
        depth_res = (252, 252)
        depth_estimator = DA2DepthEstimator(input_size=252)
    else:
        depth_res = (128, 96)
        depth_estimator = MonoDepthEstimator(input_size=depth_res)
    current_frame_idx = 0
    swap_rb = False  # Toggle with 'C' to fix color inversion in legacy recordings
    door_detector = DoorwayDetector(depth_estimator=depth_estimator, use_yolo=True)
    corridor_mode = True  # Enable 30cm table corridor navigation overlay by default
    door_mode = False     # Toggle with 'N'

    def on_trackbar(val):
        nonlocal current_frame_idx
        current_frame_idx = val
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        if depth_estimator:
            depth_estimator.reset()
        if door_detector:
            door_detector.reset_votes()

    cv2.createTrackbar("Frame", window_name, 0, max(1, total_frames - 1), on_trackbar)

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                is_paused = True
                current_frame_idx = total_frames - 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, frame = cap.read()
            else:
                current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                cv2.setTrackbarPos("Frame", window_name, current_frame_idx)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

        if frame is None:
            break

        # Fix color inversion if requested (for videos recorded before color fix)
        if swap_rb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        display_frame = frame.copy()
        telem_row = telemetry.get(current_frame_idx, {})

        depth_info = None
        door_info = None
        corridor_info = None
        norm_depth = None
        if depth_enabled:
            norm_depth, depth_colormap, sectors, center_max, infer_ms = depth_estimator.predict(frame)
            depth_info = (depth_colormap, sectors, center_max, infer_ms)
            if corridor_mode:
                corridor_info = analyze_forward_corridor(norm_depth, frame_bgr=frame)
            if door_mode:
                door_info = door_detector.analyze(frame, norm_depth)

        if corridor_mode and corridor_info is not None:
            draw_corridor_hud(display_frame, corridor_info, current_tof=35, target_height=35)
        elif door_mode and door_info is not None and norm_depth is not None:
            draw_door_hud(display_frame, door_info, norm_depth.shape[:2])

        draw_hud(display_frame, telem_row, current_frame_idx, total_frames, is_paused, depth_info, door_info)
        if swap_rb:
            cv2.putText(display_frame, "[R/B SWAP ACTIVE]", (display_frame.shape[1] - 210, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        if depth_enabled and depth_info is not None:
            depth_colormap, sectors, center_max, infer_ms = depth_info
            depth_vis = cv2.resize(depth_colormap, (display_frame.shape[1], display_frame.shape[0]))
            res_label = f"{depth_res[0]}x{depth_res[1]}"
            draw_depth_sectors(depth_vis, sectors, center_max, infer_ms, res_label)
            # Side-by-side view (Camera Feed | Depth Map)
            combined_view = np.hstack([display_frame, depth_vis])
            # Scale down if too wide for typical 1080p display
            h_c, w_c = combined_view.shape[:2]
            if w_c > 1400:
                scale = 1400.0 / w_c
                combined_view = cv2.resize(combined_view, (int(w_c * scale), int(h_c * scale)))
            cv2.imshow(window_name, combined_view)
        else:
            cv2.imshow(window_name, display_frame)

        wait_time = 0 if is_paused else delay_ms
        key = cv2.waitKeyEx(wait_time)

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            is_paused = not is_paused
        elif key in (ord("t"), ord("T")):
            corridor_mode = not corridor_mode
            print(f"Table corridor navigation overlay (30cm): {'ENABLED' if corridor_mode else 'DISABLED'}")
        elif key in (ord("n"), ord("N")):
            door_mode = not door_mode
            print(f"Doorway passage navigation overlay: {'ENABLED' if door_mode else 'DISABLED'}")
        elif key in (ord("c"), ord("C")):
            swap_rb = not swap_rb
            print(f"Color R/B inversion swap: {'ENABLED (Fixed colors for legacy recordings)' if swap_rb else 'DISABLED'}")
            if depth_estimator:
                depth_estimator.reset()
        elif key in (ord("d"), ord("D")):
            depth_enabled = not depth_enabled
            print(f"Depth map display: {'ENABLED' if depth_enabled else 'DISABLED'}")
        elif key == ord("1"):
            depth_res = (96, 96)
            if isinstance(depth_estimator, MonoDepthEstimator):
                depth_estimator.set_resolution(96, 96)
            else:
                depth_estimator = MonoDepthEstimator(input_size=depth_res)
                if door_detector:
                    door_detector.depth_estimator = depth_estimator
        elif key == ord("2"):
            depth_res = (128, 96)
            if isinstance(depth_estimator, MonoDepthEstimator):
                depth_estimator.set_resolution(128, 96)
            else:
                depth_estimator = MonoDepthEstimator(input_size=depth_res)
                if door_detector:
                    door_detector.depth_estimator = depth_estimator
        elif key == ord("3"):
            if HAS_DA2:
                print("Switching depth engine to Depth Anything V2 (DA2)...")
                depth_estimator = DA2DepthEstimator(input_size=252)
                depth_res = (252, 252)
                if door_detector:
                    door_detector.depth_estimator = depth_estimator
                print("Active Depth Engine: Depth Anything V2 (DA2)")
            else:
                print("DA2 Depth Estimator not available.")
        elif key in (2424832, 65361, ord("a")):
            is_paused = True
            current_frame_idx = max(0, current_frame_idx - 1)
            cv2.setTrackbarPos("Frame", window_name, current_frame_idx)
            if depth_estimator:
                depth_estimator.reset()
        elif key in (2555904, 65363, ord("f")):
            is_paused = True
            current_frame_idx = min(total_frames - 1, current_frame_idx + 1)
            cv2.setTrackbarPos("Frame", window_name, current_frame_idx)
            if depth_estimator:
                depth_estimator.reset()
        elif key == ord("r"):
            current_frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cv2.setTrackbarPos("Frame", window_name, 0)
            if depth_estimator:
                depth_estimator.reset()
            is_paused = False

    if door_detector and getattr(door_detector, "semantic_tracker", None):
        door_detector.semantic_tracker.stop()

    cap.release()
    cv2.destroyAllWindows()
    print("Replay closed.")


if __name__ == "__main__":
    main()
