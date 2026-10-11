"""DJI Tello face detection, automatic approach, and manual keyboard control."""

import csv
import os
import time
from datetime import datetime
from pathlib import Path
import winsound

import cv2
import numpy as np
import torch
from djitellopy import Tello
from door_navigator import (
    DoorwayDetector, draw_door_hud,
    analyze_forward_corridor, draw_corridor_hud
)


MANUAL_SPEED = 35
KEY_TIMEOUT = 0.25
SEARCH_YAW_SPEED = 28
TRACK_YAW_LIMIT = 40
AUTO_DURATION = 65.0
FACE_CONFIRM_FRAMES = 1
CENTER_HOLD_SECONDS = 0.25
CENTER_PHASE_TIMEOUT = 0.8
FOLLOW_TARGET_AREA = 0.05
FOLLOW_SPEED_GAIN = 65
FOLLOW_MAX_SPEED = 35
AREA_FILTER_ALPHA = 0.75
DEBUG_LOG_INTERVAL = 0.2
DEBUG_SCREENSHOT_INTERVAL = 1.0
DETECTION_INTERVAL = 0.06
DETECTION_SCALE = 0.5
DEPTH_INTERVAL = 0.08
DEPTH_RESOLUTION = (128, 96)
WINDOW_NAME = "Tello Vision & Door Navigator"
TAKEOFF_BUTTON = (10, 45, 125, 85)
LAND_BUTTON = (135, 45, 245, 85)
CORRIDOR_BUTTON = (255, 45, 430, 85)
DOOR_BUTTON = (440, 45, 575, 85)
SCAN_YAW_SPEED = 20
SCAN_ROTATION_TIME = 9.0  # Duration for full ~360 degree room survey (seconds)
EXPLORE_BURST_TIME = 2.5  # Duration to translate into open room space (~1 meter)
DOOR_ALIGN_YAW_LIMIT = 26
DOOR_APPROACH_SPEED = 20
DOOR_PASS_SPEED = 25
DOOR_PASS_DURATION = 3.2  # 3.2s at speed 25 covers ~1.6m through the door aperture
DOOR_CONFIRM_FRAMES = 3
CORRIDOR_TARGET_HEIGHT = 35   # Target altitude in cm (low corridor beneath/between tables)
CORRIDOR_FORWARD_SPEED = 18
CORRIDOR_CAUTION_SPEED = 10
UP_KEYS = {2490368, 65362, 0x260000}
DOWN_KEYS = {2621440, 65364, 0x280000}


def clamp(value, low, high):
    return max(low, min(high, value))


def get_safe_path(path: Path) -> str:
    """Return an ASCII/8.3 safe path for OpenCV C++ APIs on Windows with non-ASCII usernames."""
    if os.name == "nt":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(500)
            parent_dir = str(path.parent.resolve())
            if ctypes.windll.kernel32.GetShortPathNameW(parent_dir, buf, 500):
                return os.path.join(buf.value, path.name)
        except Exception:
            pass
    return str(path.resolve())


def load_cascade(filename):
    path = Path(cv2.data.haarcascades) / filename
    data = path.read_bytes().decode("utf-8")
    storage = cv2.FileStorage(data, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    detector = cv2.CascadeClassifier()
    detector.read(storage.getFirstTopLevelNode())
    storage.release()
    if detector.empty():
        raise RuntimeError("Could not load face cascade: " + filename)
    return detector


def load_face_detectors():
    detectors = [load_cascade("haarcascade_frontalface_default.xml")]
    # The alternate frontal cascade helps with varied lighting and face angles.
    try:
        detectors.append(load_cascade("haarcascade_frontalface_alt2.xml"))
    except (OSError, cv2.error, RuntimeError):
        pass
    return detectors


class FaceDetector:
    """Robust face detector: uses OpenCV YuNet DNN when model is present,
    with hardened Haar cascades as fallback.
    """

    def __init__(self, model_filename="face_detection_yunet_2023mar.onnx"):
        self.yunet = None
        self.haars = []
        self.name = "Haar Cascade"

        model_path = Path(__file__).resolve().parent / model_filename
        if model_path.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                safe_model_str = get_safe_path(model_path)
                # YuNet: score_threshold=0.60 effectively eliminates false positives
                self.yunet = cv2.FaceDetectorYN.create(
                    model=safe_model_str,
                    config="",
                    input_size=(480, 360),
                    score_threshold=0.60,
                    nms_threshold=0.30,
                    top_k=5000,
                )
                self.name = "YuNet (DNN)"
                print(f"Loaded deep-learning face detector: YuNet ({model_path.name})")
            except Exception as exc:
                print(f"Failed to initialize YuNet, falling back to Haar: {exc}")
                self.yunet = None

        if self.yunet is None:
            self.haars = load_face_detectors()
            print("Loaded fallback Haar cascades (tuned against false positives).")

    def detect(self, small_frame, full_w, full_h):
        """Detect faces on a downscaled frame and return scaled boxes.
        Returns list of (x, y, w, h, score).
        """
        sh, sw = small_frame.shape[:2]
        scale_x = full_w / sw
        scale_y = full_h / sh

        if self.yunet is not None:
            self.yunet.setInputSize((sw, sw) if sw == sh else (sw, sh))
            _, detections = self.yunet.detect(small_frame)
            if detections is None:
                return []
            results = []
            for d in detections:
                x, y, w, h = d[:4]
                conf = float(d[-1])
                fw = int(w * scale_x)
                fh = int(h * scale_y)
                # Filter out tiny noise (< 32px on full image)
                if fw >= 32 and fh >= 32:
                    results.append((int(x * scale_x), int(y * scale_y), fw, fh, conf))
            return results

        # Fallback Haar cascade:
        # 1. Blur slightly to eliminate H.264 video compression blocking/noise.
        # 2. Avoid aggressive CLAHE which generates false facial edges.
        # 3. Use minNeighbors=7 and minSize=(30, 30) on small image (60x60 on full).
        # 4. Use detectMultiScale3 to filter by detection confidence weights.
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        found = []
        try:
            rects, _, weights = self.haars[0].detectMultiScale3(
                gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels=True,
            )
            for (x, y, w, h), weight in zip(rects, weights):
                if float(weight) >= 3.5:
                    found.append((x, y, w, h, float(weight)))
        except Exception:
            rects = self.haars[0].detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for x, y, w, h in rects:
                found.append((x, y, w, h, 1.0))

        if not found and len(self.haars) > 1:
            try:
                rects, _, weights = self.haars[1].detectMultiScale3(
                    gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels=True,
                )
                for (x, y, w, h), weight in zip(rects, weights):
                    if float(weight) >= 3.5:
                        found.append((x, y, w, h, float(weight)))
            except Exception:
                pass

        results = []
        for x, y, w, h, score in sorted(found, key=lambda b: b[2] * b[3], reverse=True):
            fw = int(w * scale_x)
            fh = int(h * scale_y)
            fx = int(x * scale_x)
            fy = int(y * scale_y)
            # Aspect ratio sanity check (faces are roughly square / slight rectangle)
            if 0.65 <= (fw / max(1, fh)) <= 1.45:
                results.append((fx, fy, fw, fh, score))
        return results


def select_target_face(faces, last_face):
    """Select the best face box, prioritizing continuity with the tracked target."""
    if not faces:
        return None
    if last_face is None:
        return max(faces, key=lambda b: b[2] * b[3])

    lx, ly, lw, lh = last_face[:4]
    last_center_x = lx + lw / 2
    last_center_y = ly + lh / 2
    last_area = lw * lh

    def match_score(candidate):
        cx = candidate[0] + candidate[2] / 2
        cy = candidate[1] + candidate[3] / 2
        c_area = candidate[2] * candidate[3]
        dist = ((cx - last_center_x) ** 2 + (cy - last_center_y) ** 2) ** 0.5
        size_ratio = min(c_area, last_area) / max(c_area, last_area, 1)
        return size_ratio / (dist + 40.0)

    return max(faces, key=match_score)


class MonoDepthEstimator:
    """Lightweight Monocular Depth Estimation on CUDA/CPU (supports 96x96 and 128x96)
    with real-time temporal flicker reduction and smooth range normalization.
    """

    def __init__(self, input_size=(128, 96), temporal_alpha=0.75):
        self.input_size = input_size  # (width, height)
        self.temporal_alpha = temporal_alpha
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
        """Reset temporal state."""
        self.smooth_depth = None
        self.running_min = None
        self.running_max = None
        self.smooth_sectors = None

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


def inside_button(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def on_mouse(event, x, y, _flags, requested_action):
    if event == cv2.EVENT_LBUTTONUP:
        if inside_button(x, y, TAKEOFF_BUTTON):
            requested_action["takeoff"] = True
        elif inside_button(x, y, LAND_BUTTON):
            requested_action["land"] = True
        elif inside_button(x, y, CORRIDOR_BUTTON):
            requested_action["corridor_nav"] = True
        elif inside_button(x, y, DOOR_BUTTON):
            requested_action["door_nav"] = True


def draw_button(frame, rect, label, enabled):
    x1, y1, x2, y2 = rect
    color = (0, 150, 0) if enabled else (85, 85, 85)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.putText(frame, label, (x1 + 8, y1 + 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2)


def main():
    drone = Tello()
    airborne = False
    requested_action = {"takeoff": False, "land": False, "corridor_nav": False, "door_nav": False}
    manual_movement = 0
    last_arrow_time = 0.0
    auto_active = False
    auto_phase = "IDLE"
    centered_since = None
    center_phase_started_at = None
    last_face_seen_at = 0.0
    last_face_x_error = 0.0
    face_hit_count = 0
    last_tracked_face = None
    detector = FaceDetector()
    depth_estimator = MonoDepthEstimator(input_size=DEPTH_RESOLUTION)
    door_detector = DoorwayDetector(depth_estimator=depth_estimator, use_yolo=True)
    door_nav_active = False
    door_phase = "IDLE"  # IDLE, SCAN_ROOM, EXPLORE_MOVE, VERIFY, ALIGN, APPROACH, TRAVERSE, CLEARED
    scan_started_at = 0.0
    explore_started_at = 0.0
    best_scan_depth = 999.0
    door_verify_started_at = None
    door_lost_since = None
    scan_ignore_until = 0.0
    door_confirm_count = 0
    door_aligned_since = None
    door_traverse_started_at = None
    door_cleared_at = None
    door_overlay_enabled = True
    latest_door_info = None
    corridor_nav_active = False
    corridor_overlay_enabled = True
    latest_corridor_info = None
    current_tof = 35
    last_tof_at = 0.0
    current_tello_state = {}
    last_depth_at = 0.0
    latest_depth_map = None
    latest_depth_colormap = None
    latest_sectors = (0.0, 0.0, 0.0)
    latest_center_max = 0.0
    depth_ms = 0.0

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = Path(__file__).resolve().parent / "debug" / run_stamp
    debug_dir.mkdir(parents=True, exist_ok=True)
    command_log_file = (debug_dir / "flight_telemetry.csv").open("w", newline="", encoding="utf-8")
    command_log = csv.writer(command_log_file)
    command_log.writerow(("frame_idx", "timestamp", "elapsed_s", "state", "phase", "faces_count",
                          "face_x", "face_y", "face_w", "face_h", "face_conf", "lr", "fb", "ud", "yaw", "battery",
                          "depth_left", "depth_center", "depth_right", "depth_center_max", "depth_ms",
                          "door_status", "door_phase", "door_passable", "door_cx", "door_cy", "door_step",
                          "imu_pitch", "imu_roll", "imu_yaw", "speed_vx", "speed_vy", "speed_vz", "tof_cm", "baro_cm"))
    legacy_log_file = (debug_dir / "commands.csv").open("w", newline="", encoding="utf-8")
    legacy_log = csv.writer(legacy_log_file)
    legacy_log.writerow(("timestamp", "elapsed_s", "phase", "faces", "face_x", "face_y",
                         "face_w", "face_h", "face_area", "filtered_area", "lr", "fb", "ud", "yaw",
                         "depth_left", "depth_center", "depth_right"))
    video_writer = None
    video_path = debug_dir / "flight_raw.mp4"
    airborne_video_writer = None
    airborne_video_path = debug_dir / "flight_airborne_clean.mp4"
    depth_video_writer = None
    depth_video_path = debug_dir / "flight_depth.mp4"
    manual_rec_writer = None
    manual_rec_idx = 0
    is_manual_recording = False
    frame_idx = 0
    flight_started_at = None
    battery_level = 100
    last_battery_check_at = 0.0
    auto_started_at = None
    filtered_face_area = None
    last_debug_log_at = 0.0
    last_screenshot_at = 0.0
    last_detection_at = 0.0
    cached_faces = []
    detection_ms = 0.0

    try:
        print("Connecting to DJI Tello...")
        drone.connect()
        drone.TIME_BTW_RC_CONTROL_COMMANDS = 0.08
        try:
            battery_level = drone.get_battery()
        except Exception:
            battery_level = 100
        print(f"Connected. Battery: {battery_level}%")
        drone.streamon()
        frame_read = drone.get_frame_read()
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse, requested_action)
        print("Controls:")
        print("  a / Button     - Low-Altitude (30cm) Table Corridor Autopilot (between tables)")
        print("  e              - Doorway Navigation Autopilot")
        print("  t / l          - Takeoff / Land")
        print("  f              - Legacy Face Follow mode")
        print("  n              - Toggle HUD overlay")
        print("  r              - Record clean DA3 clip")
        print("  SPACE          - Emergency Stop / Manual Hover")
        print("  q              - Quit")
        print(f"Recording video, depth maps, and telemetry to: {debug_dir}")

        while True:
            frame_raw_rgb = frame_read.frame
            if frame_raw_rgb is None or frame_raw_rgb.shape[1] < 600 or frame_raw_rgb.shape[0] < 450:
                time.sleep(0.01)
                continue

            # CRITICAL COLOR INVERSION FIX:
            # djitellopy BackgroundFrameRead decodes video frames to RGB (via PyAV / PIL to_image).
            # OpenCV (imshow, VideoWriter, cv2 DNN, Haar) operates in standard BGR order.
            # Convert RGB -> BGR immediately so:
            # 1. Video playback and display have 100% natural colors (no inverted/blue skin).
            # 2. Face detector receives correct color channels.
            # 3. Depth neural network receives true RGB (after its internal BGR->RGB conversion),
            #    matching its ImageNet pretraining weights and eliminating inverted-color hallucinations.
            frame = cv2.cvtColor(frame_raw_rgb, cv2.COLOR_RGB2BGR)
            raw_frame = frame.copy()
            height, width = frame.shape[:2]
            now = time.monotonic()
            if flight_started_at is None:
                flight_started_at = now

            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
                depth_video_writer = cv2.VideoWriter(str(depth_video_path), fourcc, 30.0, (width, height))
                print(f"Video recorder initialized: {video_path} ({width}x{height} @ 30fps)")
                print(f"Depth video recorder initialized: {depth_video_path} ({width}x{height} @ 30fps)")

            # 1. Full clean raw video (entire session)
            video_writer.write(raw_frame)
            frame_idx += 1

            # 2. Clean airborne video (only active flight, zero table/floor idle frames)
            if airborne:
                if airborne_video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    airborne_video_writer = cv2.VideoWriter(str(airborne_video_path), fourcc, 30.0, (width, height))
                    print(f"Airborne clean flight recorder started: {airborne_video_path}")
                airborne_video_writer.write(raw_frame)

            # 3. Manual clean clip for Depth Anything 3 (started/stopped with 'r')
            if is_manual_recording and manual_rec_writer is not None:
                manual_rec_writer.write(raw_frame)

            # Depth estimation, corridor analysis and room/doorway analysis
            if now - last_depth_at >= DEPTH_INTERVAL:
                try:
                    (latest_depth_map, latest_depth_colormap,
                     latest_sectors, latest_center_max, depth_ms) = depth_estimator.predict(frame)
                    latest_corridor_info = analyze_forward_corridor(latest_depth_map, frame_bgr=frame)
                    latest_door_info = door_detector.analyze(frame, latest_depth_map)
                except Exception as exc:
                    pass
                last_depth_at = now

            # Query Tello onboard state sensors (IMU Gyro, ToF, Optical Flow velocities)
            if now - last_tof_at >= 0.05:
                try:
                    current_tello_state = drone.get_current_state() or {}
                    tof_val = current_tello_state.get("tof")
                    if tof_val and 5 < tof_val < 300:
                        current_tof = tof_val
                    elif not tof_val:
                        tof_val = drone.get_distance_tof()
                        if tof_val and 5 < tof_val < 300:
                            current_tof = tof_val
                except Exception:
                    pass
                last_tof_at = now

            # Record synchronized depth colormap frame to depth video
            if depth_video_writer is not None:
                if latest_depth_colormap is not None:
                    depth_full = cv2.resize(latest_depth_colormap, (width, height), interpolation=cv2.INTER_NEAREST)
                    depth_video_writer.write(depth_full)
                else:
                    depth_video_writer.write(raw_frame)

            if now - last_battery_check_at >= 4.0:
                try:
                    battery_level = drone.get_battery()
                except Exception:
                    pass
                last_battery_check_at = now
            fresh_detection = False
            if now - last_detection_at >= DETECTION_INTERVAL:
                detection_started = time.perf_counter()
                small = cv2.resize(frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE,
                                   interpolation=cv2.INTER_AREA)
                cached_faces = detector.detect(small, width, height)
                detection_ms = (time.perf_counter() - detection_started) * 1000
                last_detection_at = now
                fresh_detection = True

            faces = cached_faces
            if now - last_face_seen_at > 1.5:
                last_tracked_face = None
            face = select_target_face(faces, last_tracked_face)
            if face is not None:
                last_tracked_face = face

            face_bbox = face[:4] if face is not None else (None, None, None, None)
            area = 0.0
            rc_command = (0, 0, 0, 0)

            for item in faces:
                x, y, w, h = item[:4]
                conf = item[4] if len(item) > 4 else None
                is_target = (face is not None and (x, y, w, h) == face[:4])
                if is_target and auto_phase in ("CENTER", "FOLLOW"):
                    color = (0, 255, 0)
                    label = f"Target {conf:.2f}" if conf is not None else "Target"
                elif is_target:
                    color = (0, 220, 0)
                    label = f"Face {conf:.2f}" if conf is not None else "Face"
                else:
                    color = (180, 180, 180)
                    label = f"Face {conf:.2f}" if conf is not None else "Face"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if auto_active:
                if face is not None:
                    x, y, face_w, face_h = face[:4]
                    face_center_x = x + face_w / 2
                    face_center_y = y + face_h / 2
                    x_error = face_center_x / width - 0.5
                    y_error = face_center_y / height - 0.5
                    area = face_w * face_h / (width * height)
                    filtered_face_area = area if filtered_face_area is None else (
                        AREA_FILTER_ALPHA * area + (1 - AREA_FILTER_ALPHA) * filtered_face_area
                    )
                    last_face_seen_at = now
                    last_face_x_error = x_error

                    if auto_phase == "SEARCH":
                        if fresh_detection:
                            face_hit_count += 1
                        if face_hit_count >= FACE_CONFIRM_FRAMES:
                            auto_phase = "CENTER"
                            centered_since = None
                            center_phase_started_at = now
                            winsound.Beep(1000, 140)
                            print("Face confirmed; centering phase started.")

                    if auto_phase == "CENTER":
                        yaw = int(clamp(x_error * 50, -TRACK_YAW_LIMIT, TRACK_YAW_LIMIT))
                        vertical = int(clamp(-y_error * 45, -24, 24)) if abs(y_error) >= 0.06 else 0
                        if 0 < abs(vertical) < 8:
                            vertical = 8 if vertical > 0 else -8
                        rc_command = (0, 0, vertical, yaw)
                        centered = abs(x_error) < 0.08 and abs(y_error) < 0.10
                        centered_since = (centered_since or now) if centered else None
                        held_centered = centered_since is not None and now - centered_since >= CENTER_HOLD_SECONDS
                        centered_timed_out = now - center_phase_started_at >= CENTER_PHASE_TIMEOUT
                        if held_centered or centered_timed_out:
                            auto_phase = "FOLLOW"
                            winsound.Beep(1200, 140)
                            print("Centering complete; follow phase started.")
                    elif auto_phase == "FOLLOW":
                        # Keep the face at a steady apparent size while moving with it.
                        left_right = int(clamp(x_error * 38, -25, 25))
                        size_ratio = (max(filtered_face_area, 0.0001) / FOLLOW_TARGET_AREA) ** 0.5
                        distance_error = 1.0 - size_ratio
                        forward_back = int(clamp(distance_error * FOLLOW_SPEED_GAIN,
                                                 -FOLLOW_MAX_SPEED, FOLLOW_MAX_SPEED))
                        if abs(distance_error) < 0.05:
                            forward_back = 0
                        elif forward_back == 0:
                            forward_back = 7 if distance_error > 0 else -7
                        yaw = int(clamp(x_error * 48, -TRACK_YAW_LIMIT, TRACK_YAW_LIMIT))
                        vertical = int(clamp(-y_error * 42, -22, 22)) if abs(y_error) >= 0.06 else 0
                        if 0 < abs(vertical) < 7:
                            vertical = 7 if vertical > 0 else -7
                        rc_command = (left_right, forward_back, vertical, yaw)
                else:
                    if auto_phase == "SEARCH":
                        if fresh_detection:
                            face_hit_count = 0
                        rc_command = (0, 0, 0, SEARCH_YAW_SPEED)
                    elif now - last_face_seen_at <= 0.25:
                        rc_command = (0, 0, 0, 0)
                    elif now - last_face_seen_at <= 2.5:
                        direction = 1 if last_face_x_error >= 0 else -1
                        rc_command = (0, 0, 0, direction * 24)
                    else:
                        auto_phase = "SEARCH"
                        face_hit_count = 0
                        centered_since = None
                        center_phase_started_at = None
                        rc_command = (0, 0, 0, SEARCH_YAW_SPEED)

                if auto_active and now - auto_started_at >= AUTO_DURATION:
                    rc_command = (0, 0, 0, 0)
                    print("65-second automatic mode finished. Landing.")
                    drone.land()
                    airborne = False
                    auto_active = False
                    auto_phase = "IDLE"
                    manual_movement = 0

            # Low-Altitude Forward Corridor Navigation between Tables (~30-35cm)
            if corridor_nav_active and airborne:
                ci = latest_corridor_info or {}

                # 1. Active Altitude Control Loop: target 30-35 cm above floor
                # Tello takes off to ~100 cm; autopilot smoothly brings it down between tables
                tof_h = current_tof if current_tof is not None else 35

                if tof_h > 46:
                    ud = -18  # descending from high takeoff altitude
                    altitude_locked = False
                elif tof_h > 38:
                    ud = -14  # gentle descent trim
                    altitude_locked = True
                elif tof_h < 26:
                    ud = +15  # lift up away from carpet/floor
                    altitude_locked = True
                else:
                    ud = 0    # locked perfectly at 26-38 cm!
                    altitude_locked = True

                # 2. Forward Course Guidance (focus strictly on forward corridor ahead!)
                if not altitude_locked:
                    # While still descending from 100cm down to ~35cm, hold forward speed at 0
                    rc_command = (0, 0, ud, 0)
                else:
                    c_state = ci.get("state", "CLEAR_AISLE")
                    fwd_cmd = ci.get("forward", CORRIDOR_FORWARD_SPEED)
                    lr_cmd = ci.get("lr", 0)
                    yaw_cmd = ci.get("yaw", 0)

                    # Obstacle dead-ahead escape:
                    if c_state == "BLOCKED_AHEAD":
                        fwd_cmd = 0
                        dl = ci.get("dl", 100)
                        dr = ci.get("dr", 100)
                        # Turn towards the clearer side (min obstacle depth)
                        yaw_cmd = -22 if dl < dr else +22
                        lr_cmd = -8 if dl < dr else +8

                    rc_command = (lr_cmd, fwd_cmd, ud, yaw_cmd)

            # Autonomous Door Navigation State Machine
            if door_nav_active and airborne:
                d_l, d_c, d_r = latest_sectors

                di = latest_door_info or {}
                door_seen = bool(di.get("door_seen"))
                d_status = di.get("status", "NO_DOOR")
                if door_seen:
                    door_lost_since = None
                else:
                    door_lost_since = door_lost_since or now
                lost_for = 0.0 if door_lost_since is None else now - door_lost_since

                if door_phase == "SCAN_ROOM":
                    # 1. Rotate slowly to scan 360 surrounding; track clearest open direction
                    rc_command = (0, 0, 0, SCAN_YAW_SPEED)
                    if d_c < best_scan_depth:
                        best_scan_depth = d_c

                    if door_seen and now >= scan_ignore_until:
                        door_phase = "VERIFY"
                        door_verify_started_at = now
                        door_detector.reset_votes()
                        rc_command = (0, 0, 0, 0)
                        try:
                            winsound.Beep(1000, 100)
                        except Exception:
                            pass
                        print("\n[DOOR NAV] Door detected -> hovering to build its depth map (VERIFY).")
                    elif now - scan_started_at >= SCAN_ROTATION_TIME:
                        # Full 360 scan done, no open door found -> translate to explore room!
                        door_phase = "EXPLORE_MOVE"
                        explore_started_at = now
                        rc_command = (0, 0, 0, 0)
                        print("\n[DOOR NAV] 360 scan complete. No open door from current spot -> EXPLORE_MOVE: translating into open space.")
                        try:
                            winsound.Beep(800, 120)
                        except Exception:
                            pass

                elif door_phase == "EXPLORE_MOVE":
                    # 1b. Active room translation: move into open space to find new vantage point / door
                    if door_seen and now >= scan_ignore_until:
                        door_phase = "VERIFY"
                        door_verify_started_at = now
                        door_detector.reset_votes()
                        rc_command = (0, 0, 0, 0)
                        print("\n[DOOR NAV] Door spotted during translation -> VERIFY.")
                    else:
                        forward = 16
                        lateral = 0
                        # Nudge away from side obstacles
                        if d_l > 165 and d_r < 125:
                            lateral = 8   # move slightly right
                        elif d_r > 165 and d_l < 125:
                            lateral = -8  # move slightly left

                        # Obstacle / Wall proximity safety shield
                        if d_c >= 165 or latest_center_max >= 200:
                            print(f"[SAFETY SHIELD] Front barrier approaching (d_c={d_c:.0f}). Stopping translation.")
                            door_phase = "SCAN_ROOM"
                            scan_started_at = now
                            best_scan_depth = 999.0
                            rc_command = (0, 0, 0, 0)
                        else:
                            # Dynamic yaw steering toward the deepest open room space
                            yaw_steer = int(clamp((d_l - d_r) * 0.25, -20, 20))
                            rc_command = (lateral, forward, 0, yaw_steer)
                            if now - explore_started_at >= EXPLORE_BURST_TIME:
                                print("\n[DOOR NAV] Reached new vantage point in room. Beginning new 360 scan.")
                                door_phase = "SCAN_ROOM"
                                scan_started_at = now
                                best_scan_depth = 999.0
                                rc_command = (0, 0, 0, 0)

                elif door_phase == "VERIFY":
                    # 2. Hover, keep the door centred so wall is visible on both sides,
                    #    and let the door-crop depth verdicts vote.
                    ex = di.get("center_x_norm", 0.0) if door_seen else 0.0
                    yaw = int(clamp(ex * 30, -15, 15)) if abs(ex) > 0.05 else 0
                    rc_command = (0, 0, 0, yaw)

                    if d_status == "OPEN_DOORWAY" and di.get("passable"):
                        door_phase = "ALIGN"
                        door_aligned_since = None
                        try:
                            winsound.Beep(1200, 150)
                        except Exception:
                            pass
                        print(f"\n[DOOR NAV] Door depth confirms OPEN (votes {di.get('votes')}). -> ALIGN")
                    elif d_status in ("CLOSED_DOOR", "DOORWAY_BLOCKED", "PARTIAL_OPEN"):
                        print(f"\n[DOOR NAV] Door rejected: {d_status} (votes {di.get('votes')}). Continue scanning.")
                        try:
                            winsound.Beep(500, 200)
                        except Exception:
                            pass
                        door_phase = "SCAN_ROOM"
                        scan_started_at = now
                        scan_ignore_until = now + 4.0  # rotate past this door
                    elif lost_for > 1.0 or now - door_verify_started_at > 4.0:
                        print("\n[DOOR NAV] Could not verify door (lost/uncertain). Continue scanning.")
                        door_phase = "SCAN_ROOM"
                        scan_started_at = now
                        scan_ignore_until = now + 3.0

                elif door_phase == "ALIGN":
                    # 3. Centre on the OPEN GAP (not the whole door box)
                    if door_seen and d_status == "OPEN_DOORWAY":
                        ex = di["center_x_norm"]
                        ey = di["center_y_norm"]
                        yaw = int(clamp(ex * 45, -DOOR_ALIGN_YAW_LIMIT, DOOR_ALIGN_YAW_LIMIT))
                        ud = int(clamp(-ey * 35, -18, 18))
                        # Gentle forward creep so drone doesn't freeze in mid-air
                        forward = 12 if abs(ex) < 0.25 else 0
                        rc_command = (0, forward, ud, yaw)
                        # Yaw aligned enough to advance into approach
                        if abs(ex) < 0.16:
                            door_aligned_since = door_aligned_since or now
                            if now - door_aligned_since >= 0.3:
                                door_phase = "APPROACH"
                                try:
                                    winsound.Beep(1500, 150)
                                except Exception:
                                    pass
                                print("\n[DOOR NAV] Aligned on open gap -> APPROACH")
                        else:
                            door_aligned_since = None
                    elif d_status in ("CLOSED_DOOR", "DOORWAY_BLOCKED", "PARTIAL_OPEN"):
                        rc_command = (0, 0, 0, 0)
                        door_phase = "SCAN_ROOM"
                        scan_started_at = now
                        scan_ignore_until = now + 4.0
                        print(f"\n[DOOR NAV] Door became {d_status} during ALIGN. Abort -> SCAN_ROOM")
                    else:
                        rc_command = (0, 0, 0, 0)  # hover through short dropouts
                        if lost_for > 1.0:
                            door_phase = "SCAN_ROOM"
                            scan_started_at = now
                            print("\n[DOOR NAV] Lost door during ALIGN -> SCAN_ROOM")

                elif door_phase == "APPROACH":
                    # 4. Active forward approach while tracking the open gap
                    if door_seen and d_status == "OPEN_DOORWAY":
                        ex = di["center_x_norm"]
                        ey = di["center_y_norm"]
                        db = di.get("door_box") or di.get("box") or (0, 0, 0, 0)
                        gap_b = di.get("gap_box") or db
                        yaw = int(clamp(ex * 45, -20, 20))
                        ud = int(clamp(-ey * 35, -16, 16))
                        forward = DOOR_APPROACH_SPEED
                        if d_c >= 175 or latest_center_max >= 215:
                            forward = 0
                            print("[SAFETY SHIELD] Obstacle / person ahead - holding forward speed at 0")
                        rc_command = (0, forward, ud, yaw)

                        # Threshold trigger: reached the doorway boundary
                        # 1. Height fills the view (bh >= 80% screen) OR
                        # 2. Open gap itself is wide (gw >= 40% screen) OR
                        # 3. Door fills >= 65% width and >= 70% height
                        at_threshold = (
                            db[3] >= int(height * 0.82) or
                            gap_b[2] >= int(width * 0.42) or
                            (db[2] >= int(width * 0.65) and db[3] >= int(height * 0.70))
                        )
                        if at_threshold:
                            door_phase = "TRAVERSE"
                            door_traverse_started_at = now
                            try:
                                winsound.Beep(1800, 200)
                            except Exception:
                                pass
                            print(f"\n[DOOR NAV] Doorway threshold reached (box {db[2]}x{db[3]}px, gap {gap_b[2]}px) -> TRAVERSE thrust!")
                    elif d_status in ("CLOSED_DOOR", "DOORWAY_BLOCKED", "PARTIAL_OPEN"):
                        rc_command = (0, 0, 0, 0)
                        door_phase = "SCAN_ROOM"
                        scan_started_at = now
                        scan_ignore_until = now + 4.0
                        print(f"\n[DOOR NAV] Door became {d_status} during APPROACH. STOP -> SCAN_ROOM")
                    elif lost_for < 0.6:
                        # Smooth inertial coasting during brief single-frame detector dropouts
                        rc_command = (0, 14, 0, 0)
                    else:
                        rc_command = (0, 0, 0, 0)  # hover, never fly blind
                        if lost_for > 1.2:
                            door_phase = "SCAN_ROOM"
                            scan_started_at = now
                            print("\n[DOOR NAV] Lost door during APPROACH -> hover, SCAN_ROOM")

                elif door_phase == "TRAVERSE":
                    # 5. Punch through the doorway aperture into the next room
                    rc_command = (0, DOOR_PASS_SPEED, 0, 0)
                    if now - door_traverse_started_at >= DOOR_PASS_DURATION:
                        door_phase = "CLEARED"
                        door_cleared_at = now
                        rc_command = (0, 0, 0, 0)
                        try:
                            winsound.Beep(2200, 400)
                        except Exception:
                            pass
                        print("\n[DOOR NAV] DOORWAY CLEARED! Successfully passed through door into new room.")

                elif door_phase == "CLEARED":
                    # 5. Safe hover in the new space
                    rc_command = (0, 0, 0, 0)
                    if now - door_cleared_at >= 4.0:
                        print("\n[DOOR NAV] Mission complete. Landing safely.")
                        drone.send_rc_control(0, 0, 0, 0)
                        drone.land()
                        airborne = False
                        door_nav_active = False
                        door_phase = "IDLE"

            if airborne and not auto_active and not door_nav_active and not corridor_nav_active:
                if manual_movement and now - last_arrow_time > KEY_TIMEOUT:
                    manual_movement = 0
                rc_command = (0, manual_movement * MANUAL_SPEED, 0, 0)
            if airborne:
                drone.send_rc_control(*rc_command)

            state = "AIRBORNE" if airborne else "LANDED"
            if corridor_nav_active:
                c_st = latest_corridor_info.get("state", "HOLD") if latest_corridor_info else "INIT"
                tof_str = f"{current_tof}cm" if current_tof is not None else "??"
                state += f" | CORRIDOR 30cm: {c_st} [ToF={tof_str}]"
            elif door_nav_active:
                state += " | AUTOPILOT: " + door_phase
            elif auto_active:
                state += " | FACE: " + auto_phase
            else:
                state += " | MANUAL"

            # Draw Navigation HUD overlays
            if corridor_nav_active or (corridor_overlay_enabled and not door_nav_active):
                if latest_corridor_info is not None:
                    draw_corridor_hud(frame, latest_corridor_info, current_tof, CORRIDOR_TARGET_HEIGHT)
            elif door_overlay_enabled and latest_door_info is not None:
                draw_door_hud(frame, latest_door_info, DEPTH_RESOLUTION)

            cv2.putText(frame, state, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 180, 255), 2)
            cv2.putText(frame, f"RC lr/fb/ud/yaw: {rc_command}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Detection: {detection_ms:.0f} ms / {DETECTION_INTERVAL:.2f}s",
                        (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Detector: {detector.name}", (width - 240, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(frame, f"BAT: {battery_level}%", (width - 240, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if battery_level > 20 else (0, 0, 255), 1)
            rec_blink = int(now * 2) % 2 == 0
            rec_color = (0, 0, 255) if rec_blink else (0, 0, 150)
            cv2.circle(frame, (width - 150, 70), 5, rec_color, -1)
            cv2.putText(frame, f"REC #{frame_idx}", (width - 138, 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            if is_manual_recording:
                mrec_blink = int(now * 3) % 2 == 0
                if mrec_blink:
                    cv2.circle(frame, (width - 340, 70), 6, (0, 0, 255), -1)
                    cv2.putText(frame, f"DA3 CLIP #{manual_rec_idx} (r)", (width - 325, 74),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "r: record DA3 clip", (width - 340, 74),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

            # Log frame-accurate telemetry matching raw video frame_idx
            d_l, d_c, d_r = latest_sectors
            target_box = face[:4] if face is not None else ("", "", "", "")
            target_conf = f"{face[4]:.2f}" if face is not None and len(face) > 4 else ""
            door_st = latest_door_info.get("status", "NONE") if latest_door_info else "NONE"
            door_pass = latest_door_info.get("passable", False) if latest_door_info else False
            door_cx = round(latest_door_info.get("center_x_norm", 0.0), 3) if latest_door_info else 0.0
            door_cy = round(latest_door_info.get("center_y_norm", 0.0), 3) if latest_door_info else 0.0
            door_st_diff = round(latest_door_info.get("depth_step", 0.0), 1) if latest_door_info else 0.0

            command_log.writerow((
                frame_idx,
                datetime.now().isoformat(timespec="milliseconds"),
                round(now - (flight_started_at or now), 3),
                state,
                auto_phase if auto_active else door_phase,
                len(faces),
                *target_box,
                target_conf,
                *rc_command,
                battery_level,
                round(d_l, 1),
                round(d_c, 1),
                round(d_r, 1),
                round(latest_center_max, 1),
                round(depth_ms, 1),
                door_st,
                door_phase,
                door_pass,
                door_cx,
                door_cy,
                door_st_diff,
                current_tello_state.get("pitch", 0),
                current_tello_state.get("roll", 0),
                current_tello_state.get("yaw", 0),
                current_tello_state.get("vgx", 0),
                current_tello_state.get("vgy", 0),
                current_tello_state.get("vgz", 0),
                current_tello_state.get("tof", current_tof or 0),
                round(float(current_tello_state.get("baro", 0.0) or 0.0), 1),
            ))
            command_log_file.flush()

            if airborne and now - last_debug_log_at >= DEBUG_LOG_INTERVAL:
                legacy_log.writerow((
                    datetime.now().isoformat(timespec="milliseconds"),
                    round(now - (auto_started_at or now), 3), auto_phase, len(faces),
                    *face_bbox, round(area, 6),
                    round(filtered_face_area or 0.0, 6), *rc_command,
                    round(d_l, 1), round(d_c, 1), round(d_r, 1),
                ))
                legacy_log_file.flush()
                last_debug_log_at = now
            if airborne and auto_active and now - last_screenshot_at >= DEBUG_SCREENSHOT_INTERVAL:
                screenshot_path = debug_dir / f"frame_{now - auto_started_at:.1f}s.jpg"
                encoded_ok, encoded_frame = cv2.imencode(".jpg", frame)
                if encoded_ok:
                    screenshot_path.write_bytes(encoded_frame.tobytes())
                last_screenshot_at = now

            # Render Depth Picture-in-Picture (PiP) and Central Corridor
            if latest_depth_colormap is not None:
                pip_w, pip_h = 200, 150
                pip_x = width - pip_w - 15
                pip_y = height - pip_h - 15

                pip_img = cv2.resize(latest_depth_colormap, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)
                psw = pip_w // 3
                cv2.line(pip_img, (psw, 0), (psw, pip_h), (80, 80, 80), 1)
                cv2.line(pip_img, (2 * psw, 0), (2 * psw, pip_h), (80, 80, 80), 1)

                c_color = (0, 0, 255) if d_c >= 180 else ((0, 200, 255) if d_c >= 140 else (0, 255, 0))
                # Highlight center flight corridor (most accurate depth zone)
                cv2.rectangle(pip_img, (psw + 2, 2), (2 * psw - 2, pip_h - 2), c_color, 2)
                cv2.rectangle(pip_img, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 1)
                cv2.putText(pip_img, "DEPTH (CENTER)", (8, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
                cv2.putText(pip_img, f"C:{d_c:.0f}", (psw + 10, pip_h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c_color, 2)

                frame[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = pip_img

                depth_hud = f"Depth L/C/R: {d_l:.0f} / {d_c:.0f} / {d_r:.0f} (max={latest_center_max:.0f}) | {depth_ms:.0f}ms [DEPTH ACTIVE]"
                cv2.putText(frame, depth_hud, (10, height - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)

            draw_button(frame, TAKEOFF_BUTTON, "Takeoff (t)", not airborne)
            draw_button(frame, LAND_BUTTON, "Land (l)", airborne)
            draw_button(frame, CORRIDOR_BUTTON, "Corridor 30cm (a)", corridor_nav_active)
            draw_button(frame, DOOR_BUTTON, "Door Nav (e)", door_nav_active)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKeyEx(1)
            if key == ord("q"):
                break
            elif key in (ord("r"), ord("R")):
                if not is_manual_recording:
                    manual_rec_idx += 1
                    manual_clip_path = debug_dir / f"da3_clip_{manual_rec_idx:02d}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    manual_rec_writer = cv2.VideoWriter(str(manual_clip_path), fourcc, 30.0, (width, height))
                    is_manual_recording = True
                    print(f"\n[DA3] Started recording clean clip #{manual_rec_idx}: {manual_clip_path.name}")
                    try:
                        winsound.Beep(1200, 120)
                    except Exception:
                        pass
                else:
                    is_manual_recording = False
                    if manual_rec_writer is not None:
                        manual_rec_writer.release()
                        manual_rec_writer = None
                    print(f"\n[DA3] Finished recording clean clip #{manual_rec_idx}.")
                    try:
                        winsound.Beep(800, 120)
                    except Exception:
                        pass
            takeoff_requested = requested_action["takeoff"] or key == ord("t")
            land_requested = requested_action["land"] or key == ord("l")
            corridor_nav_requested = requested_action["corridor_nav"] or key in (ord("a"), ord("A"))
            door_nav_requested = requested_action["door_nav"] or key in (ord("e"), ord("E"))
            requested_action["takeoff"] = requested_action["land"] = False
            requested_action["corridor_nav"] = requested_action["door_nav"] = False

            if takeoff_requested and not airborne:
                drone.takeoff()
                airborne = True
                manual_movement = 0
            elif land_requested and airborne:
                drone.send_rc_control(0, 0, 0, 0)
                drone.land()
                airborne = False
                auto_active = False
                auto_phase = "IDLE"
                door_nav_active = False
                door_phase = "IDLE"
                corridor_nav_active = False
                manual_movement = 0
            elif corridor_nav_requested:
                if not airborne:
                    print("Taking off for Table Corridor Navigation (~30-35cm)...")
                    drone.takeoff()
                    airborne = True
                    time.sleep(0.5)
                auto_active = False
                door_nav_active = False
                door_phase = "IDLE"
                corridor_nav_active = not corridor_nav_active
                if corridor_nav_active:
                    try:
                        winsound.Beep(1200, 150)
                    except Exception:
                        pass
                    print(f"\n[CORRIDOR NAV] Low-Altitude 30cm Table Corridor Autopilot ACTIVATED! (Target Height={CORRIDOR_TARGET_HEIGHT}cm)")
                else:
                    drone.send_rc_control(0, 0, 0, 0)
                    print("\n[CORRIDOR NAV] Autopilot DEACTIVATED. Hovering in manual control.")
            elif door_nav_requested:
                if not airborne:
                    print("Taking off for Autonomous Room Exploration & Doorway Autopilot...")
                    drone.takeoff()
                    airborne = True
                    time.sleep(0.5)
                auto_active = False
                corridor_nav_active = False
                door_nav_active = not door_nav_active
                if door_nav_active:
                    door_phase = "SCAN_ROOM"
                    scan_started_at = now
                    best_scan_depth = 999.0
                    door_confirm_count = 0
                    door_aligned_since = None
                    try:
                        winsound.Beep(1000, 150)
                    except Exception:
                        pass
                    print("\n[DOOR NAV] Autonomous Room Exploration & Door Navigation ACTIVATED!")
                else:
                    door_phase = "IDLE"
                    drone.send_rc_control(0, 0, 0, 0)
                    print("\n[DOOR NAV] Autonomous mode DEACTIVATED. Hovering in manual control.")
            elif key in (ord("f"), ord("F")):
                if not airborne:
                    drone.takeoff()
                    airborne = True
                    time.sleep(0.5)
                door_nav_active = False
                corridor_nav_active = False
                door_phase = "IDLE"
                auto_active = not auto_active
                if auto_active:
                    manual_movement = 0
                    face_hit_count = 0
                    filtered_face_area = None
                    last_tracked_face = None
                    auto_phase = "SEARCH"
                    centered_since = None
                    center_phase_started_at = None
                    auto_started_at = time.monotonic()
                    print("\n[FACE FOLLOW] Activated. Searching for face...")
                else:
                    auto_phase = "IDLE"
                    drone.send_rc_control(0, 0, 0, 0)
                    print("\n[FACE FOLLOW] Deactivated.")
            elif key in UP_KEYS and airborne:
                manual_movement = 1
                last_arrow_time = time.monotonic()
                door_nav_active = False
                corridor_nav_active = False
                auto_active = False
            elif key in DOWN_KEYS and airborne:
                manual_movement = -1
                last_arrow_time = time.monotonic()
                door_nav_active = False
                corridor_nav_active = False
                auto_active = False
            elif key in (ord("c"), ord("C")):
                corridor_overlay_enabled = not corridor_overlay_enabled
                print(f"Corridor HUD overlay: {'ENABLED' if corridor_overlay_enabled else 'DISABLED'}")
            elif key in (ord("n"), ord("N")):
                door_overlay_enabled = not door_overlay_enabled
                print(f"Door HUD overlay: {'ENABLED' if door_overlay_enabled else 'DISABLED'}")
            elif key == ord(" ") and airborne:
                manual_movement = 0
                auto_active = False
                auto_phase = "IDLE"
                door_nav_active = False
                door_phase = "IDLE"
                corridor_nav_active = False
                drone.send_rc_control(0, 0, 0, 0)

    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as exc:
        print(f"Error: {exc}")
    finally:
        if door_detector and getattr(door_detector, "semantic_tracker", None):
            door_detector.semantic_tracker.stop()
        try:
            drone.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass
        if airborne:
            try:
                time.sleep(0.3)
                drone.land()
                airborne = False
            except Exception:
                # Avoid repeated land attempts from end() and __del__ if the
                # socket has already failed; Tello will land on link timeout.
                drone.is_flying = False
        try:
            drone.end()
        except Exception:
            pass
        if video_writer is not None:
            video_writer.release()
            print(f"Full session clean video saved: {video_path} ({frame_idx} frames)")
        if airborne_video_writer is not None:
            airborne_video_writer.release()
            print(f"Airborne clean flight video saved: {airborne_video_path}")
        if manual_rec_writer is not None:
            manual_rec_writer.release()
            print(f"Manual clean DA3 clip saved: {debug_dir / f'da3_clip_{manual_rec_idx:02d}.mp4'}")
        if depth_video_writer is not None:
            depth_video_writer.release()
            print(f"Depth video saved: {depth_video_path} ({frame_idx} frames)")
        command_log_file.close()
        legacy_log_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
