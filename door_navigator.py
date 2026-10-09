"""Door-first navigation: detect a door, then verify it with its OWN depth map.

Pipeline (per new YOLO detection):
  1. SemanticDoorTracker (YOLO-World, background thread) finds a door box.
  2. DoorDepthVerifier crops the door + surrounding wall from the SAME frame the
     detection was made on, and runs a dedicated depth pass on that crop.
  3. Inside the crop it compares the door interior with the wall next to the door:
       - interior much farther than the wall  -> OPEN   (passage behind the door)
       - interior in the same plane as wall   -> CLOSED (door leaf / flat surface)
     Column-by-column analysis finds the actually open gap (e.g. one leaf of a
     double door), and the steering target is the centre of that gap.
  4. Temporal voting over several detections: flight is allowed only when the
     majority of recent verdicts for the same door say OPEN.
"""

import threading
import time
from collections import deque

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


# Prompt set chosen empirically on our flight recordings (8/8 frames with a door found,
# versus 4/8 for "door"+"person" only). Context classes (wall, window) help YOLO-World
# separate doors from flat surfaces.
YOLO_CLASSES = ["door", "door frame", "person", "wall", "window"]
DOOR_CLASSES = {"door", "door frame"}
OBSTACLE_CLASSES = {"person"}

DOOR_CONF_MIN = 0.08          # YOLO-World door scores are low; depth check filters the rest
DETECTION_MAX_AGE = 0.6       # seconds; older detections are ignored

# Depth verification thresholds (disparity normalized to 0..1 inside the crop, 1 = near)
OPEN_DELTA = 0.20             # column counts as "open" if it is this much farther than the wall
CLOSED_DELTA = 0.10           # column counts as "same plane as wall" within this tolerance
OPEN_GAP_MIN_FRAC = 0.40      # open gap must cover >= 40% of door width to be passable
PARTIAL_GAP_MIN_FRAC = 0.15   # 15-40% => door ajar, too narrow to fly
CLOSED_COLS_FRAC = 0.60       # >= 60% of columns in the wall plane => closed
PERSON_DOOR_OVERLAP = 0.20    # person covering >= 20% of door box => blocked / unreliable depth

VOTE_WINDOW = 5
VOTES_TO_CONFIRM = 3


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / float(aw * ah + bw * bh - inter)


def _longest_true_run(mask):
    best_start, best_len, start = 0, 0, None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = None
    if start is not None and len(mask) - start > best_len:
        best_start, best_len = start, len(mask) - start
    return best_start, best_len


class SemanticDoorTracker:
    """Background YOLO-World detector. Keeps the frame each result was computed on,
    so depth verification uses exactly the same image as the box (no motion skew)."""

    def __init__(self, model_path="yolov8s-worldv2.pt"):
        self.enabled = False
        self.model = None
        self.lock = threading.Lock()
        self.pending_frame = None
        self.snapshot = None  # dict(frame, items, t, seq)
        self.seq = 0
        self.is_running = False
        self.thread = None

        if not HAS_YOLO:
            print("SemanticDoorTracker: ultralytics not installed - door detection disabled.")
            return
        try:
            print("Loading YOLO-World for door detection...")
            self.model = YOLO(model_path)
            self.model.set_classes(YOLO_CLASSES)
            self.enabled = True
            self.is_running = True
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()
            print("SemanticDoorTracker: background thread active.")
        except Exception as exc:
            print(f"SemanticDoorTracker: failed to load YOLO ({exc}) - door detection disabled.")

    def update_frame(self, frame):
        if self.enabled:
            with self.lock:
                self.pending_frame = frame.copy()

    def get_snapshot(self):
        with self.lock:
            return self.snapshot

    def _worker_loop(self):
        while self.is_running:
            with self.lock:
                frame = self.pending_frame
                self.pending_frame = None
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                res = self.model(frame, imgsz=480, conf=0.05, verbose=False)[0]
                items = []
                for b in res.boxes:
                    name = self.model.names[int(b.cls[0])]
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    items.append({"cls": name, "conf": conf, "box": (x1, y1, x2 - x1, y2 - y1)})
                self.seq += 1
                with self.lock:
                    self.snapshot = {"frame": frame, "items": items, "t": time.monotonic(), "seq": self.seq}
            except Exception:
                time.sleep(0.05)

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)


class DoorDepthVerifier:
    """Builds a dedicated depth map of one door (plus surrounding wall) and decides
    whether there is free space behind it."""

    def __init__(self, depth_fn):
        self.depth_fn = depth_fn  # crop_bgr -> float disparity [0..1], 1 = near

    def verify(self, frame, box):
        fh, fw = frame.shape[:2]
        x, y, w, h = box
        mx = int(w * 0.35)
        my = int(h * 0.10)
        cx1, cy1 = max(0, x - mx), max(0, y - my)
        cx2, cy2 = min(fw, x + w + mx), min(fh, y + h + my)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            return {"verdict": "UNCERTAIN", "reason": "door too small"}

        disp = self.depth_fn(crop)

        # Door box in crop coordinates
        dx1, dy1 = x - cx1, y - cy1
        dx2, dy2 = dx1 + w, dy1 + h
        # Rows: upper/middle part of the door (floor is near even behind an open door)
        r1, r2 = dy1 + int(h * 0.15), dy1 + int(h * 0.70)
        pad = max(2, int(w * 0.04))

        # Wall reference: strips just outside the door frame on both sides
        strips = []
        if dx1 - pad > 3:
            strips.append(disp[r1:r2, max(0, dx1 - mx):dx1 - pad])
        if dx2 + pad < disp.shape[1] - 3:
            strips.append(disp[r1:r2, dx2 + pad:min(disp.shape[1], dx2 + mx)])
        strips = [s for s in strips if s.size > 0]
        if not strips:
            return {"verdict": "UNCERTAIN", "reason": "no wall visible next to door", "disp": disp,
                    "crop_origin": (cx1, cy1)}
        wall_ref = float(np.median(np.concatenate([s.ravel() for s in strips])))

        # Interior columns (skip the frame/jamb itself)
        ix1 = dx1 + int(w * 0.08)
        ix2 = dx2 - int(w * 0.08)
        interior = disp[r1:r2, ix1:ix2]
        if interior.shape[1] < 6:
            return {"verdict": "UNCERTAIN", "reason": "interior too narrow", "disp": disp,
                    "crop_origin": (cx1, cy1)}
        col_med = np.median(interior, axis=0)
        col_med = np.convolve(col_med, np.ones(3) / 3.0, mode="same")

        open_cols = col_med < (wall_ref - OPEN_DELTA)
        plane_cols = np.abs(col_med - wall_ref) < CLOSED_DELTA
        gap_start, gap_len = _longest_true_run(open_cols)
        n = len(col_med)
        gap_frac = gap_len / n
        plane_frac = float(plane_cols.mean())
        contrast = float(wall_ref - col_med[gap_start:gap_start + gap_len].mean()) if gap_len else 0.0

        if gap_frac >= OPEN_GAP_MIN_FRAC and contrast >= 0.25:
            verdict = "OPEN"
        elif gap_frac >= PARTIAL_GAP_MIN_FRAC:
            verdict = "PARTIAL"
        elif plane_frac >= CLOSED_COLS_FRAC:
            verdict = "CLOSED"
        else:
            verdict = "UNCERTAIN"

        # Gap box in full-frame coordinates (steering target = centre of the open gap)
        gx = cx1 + ix1 + gap_start
        gap_box = (gx, y, max(1, gap_len), h) if gap_len else None

        return {
            "verdict": verdict,
            "wall_ref": wall_ref,
            "contrast": contrast,
            "gap_frac": gap_frac,
            "plane_frac": plane_frac,
            "gap_box": gap_box,
            "disp": disp,
            "crop_origin": (cx1, cy1),
            "reason": f"gap={gap_frac:.0%} contrast={contrast:.2f} plane={plane_frac:.0%}",
        }


class DoorwayDetector:
    """Door-first detector with per-door depth verification and temporal voting."""

    def __init__(self, depth_estimator=None, use_yolo=True):
        self.depth_estimator = depth_estimator
        self.semantic_tracker = SemanticDoorTracker() if use_yolo else None
        self.votes = deque(maxlen=VOTE_WINDOW)
        self.tracked_box = None
        self.last_seq = -1
        self.last_check = None
        self.last_door = None

    def _depth_fn(self, crop):
        return self.depth_estimator.predict_crop(crop)

    def analyze_depth_corridor(self, norm_depth, frame=None):
        """Backward-compatible wrapper."""
        if frame is None:
            dh, dw = norm_depth.shape[:2]
            frame = np.zeros((dh * 4, dw * 4, 3), dtype=np.uint8)
        return self.analyze(frame, norm_depth)

    def _no_door(self, message, frame_shape):
        return {
            "found": False, "door_seen": False, "status": "NO_DOOR", "passable": False,
            "box": None, "door_box": None, "center_x_norm": 0.0, "center_y_norm": 0.0,
            "depth_step": 0.0, "inner_depth": 0.0, "votes": "".join(self.votes),
            "door_depth_vis": None, "message": message,
        }

    def analyze(self, frame, norm_depth=None):
        fh, fw = frame.shape[:2]
        if self.semantic_tracker is None or not self.semantic_tracker.enabled:
            return self._no_door("YOLO unavailable - door detection disabled", frame.shape)
        if self.depth_estimator is None:
            return self._no_door("No depth estimator attached", frame.shape)

        self.semantic_tracker.update_frame(frame)
        snap = self.semantic_tracker.get_snapshot()
        if snap is None or time.monotonic() - snap["t"] > DETECTION_MAX_AGE:
            return self._no_door("Searching for door...", frame.shape)

        # Run verification once per new detection
        if snap["seq"] != self.last_seq:
            self.last_seq = snap["seq"]
            doors = [d for d in snap["items"] if d["cls"] in DOOR_CLASSES and d["conf"] >= DOOR_CONF_MIN]
            persons = [d for d in snap["items"] if d["cls"] in OBSTACLE_CLASSES and d["conf"] >= 0.30]
            # Plausible door shape: taller than wide
            doors = [d for d in doors if d["box"][3] >= 1.1 * d["box"][2] and d["box"][2] >= 25]

            if not doors:
                self.last_door = None
                self.last_check = None
            else:
                # Prefer the door we already track; otherwise the largest/most confident one
                if self.tracked_box is not None:
                    door = max(doors, key=lambda d: _iou(d["box"], self.tracked_box) * 2
                               + d["conf"] + d["box"][2] * d["box"][3] / (fw * fh))
                    if _iou(door["box"], self.tracked_box) < 0.3:
                        self.votes.clear()
                else:
                    door = max(doors, key=lambda d: d["conf"] + d["box"][2] * d["box"][3] / (fw * fh))
                self.tracked_box = door["box"]

                check = DoorDepthVerifier(self._depth_fn).verify(snap["frame"], door["box"])

                # Person in front of / inside the door: depth of the crop is dominated by
                # the person, so the door cannot be judged -> treat as blocked.
                dx, dy, dw, dh = door["box"]
                for p in persons:
                    px, py, pw, ph = p["box"]
                    ox = max(0, min(dx + dw, px + pw) - max(dx, px))
                    oy = max(0, min(dy + dh, py + ph) - max(dy, py))
                    if ox * oy > PERSON_DOOR_OVERLAP * dw * dh:
                        check["verdict"] = "BLOCKED"
                        check["reason"] = check.get("reason", "") + " | person at door"
                        break

                # Person standing in the open gap => blocked
                if check["verdict"] in ("OPEN", "PARTIAL") and check.get("gap_box"):
                    gx, gy, gw, gh = check["gap_box"]
                    for p in persons:
                        px, py, pw, ph = p["box"]
                        ox = max(0, min(gx + gw, px + pw) - max(gx, px))
                        oy = max(0, min(gy + gh, py + ph) - max(gy, py))
                        if ox * oy > 0.25 * gw * gh:
                            check["verdict"] = "BLOCKED"
                            check["reason"] += " | person in gap"
                            break

                # UNCERTAIN (e.g. too close, no wall visible) does not change the vote
                if check["verdict"] != "UNCERTAIN":
                    self.votes.append(check["verdict"][0])  # O / P / C / B
                self.last_door = door
                self.last_check = check

        if self.last_door is None:
            return self._no_door("No door in view", frame.shape)

        door = self.last_door
        check = self.last_check or {}
        n_open = self.votes.count("O")
        n_closed = self.votes.count("C")
        n_block = self.votes.count("B")
        n_part = self.votes.count("P")

        if n_open >= VOTES_TO_CONFIRM and (not self.votes or self.votes[-1] in ("O",)):
            status, passable = "OPEN_DOORWAY", True
        elif n_block >= 2:
            status, passable = "DOORWAY_BLOCKED", False
        elif n_closed >= VOTES_TO_CONFIRM:
            status, passable = "CLOSED_DOOR", False
        elif n_part >= VOTES_TO_CONFIRM:
            status, passable = "PARTIAL_OPEN", False
        else:
            status, passable = "VERIFYING", False

        target = check.get("gap_box") if status == "OPEN_DOORWAY" and check.get("gap_box") else door["box"]
        tx, ty, tw, th = target
        cx = tx + tw / 2.0
        cy = ty + th * 0.45  # aim slightly above mid-height of the door

        # Visualization of the door's own depth map
        vis = None
        if check.get("disp") is not None:
            vis = cv2.applyColorMap((check["disp"] * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

        return {
            "found": True,
            "door_seen": True,
            "status": status,
            "passable": passable,
            "box": target,
            "door_box": door["box"],
            "door_conf": door["conf"],
            "center_x_norm": float(cx / fw - 0.5),
            "center_y_norm": float(cy / fh - 0.5),
            "depth_step": float(check.get("contrast", 0.0)) * 100.0,
            "inner_depth": float(check.get("wall_ref", 0.0)),
            "gap_frac": float(check.get("gap_frac", 0.0)),
            "votes": "".join(self.votes),
            "last_verdict": check.get("verdict", "UNCERTAIN"),
            "door_depth_vis": vis,
            "message": f"{status} [{''.join(self.votes)}] {check.get('reason', '')}",
        }

    def reset_votes(self):
        self.votes.clear()
        self.tracked_box = None


STATUS_STYLE = {
    "OPEN_DOORWAY": ((0, 255, 0), "OPEN - SAFE TO PASS"),
    "CLOSED_DOOR": ((0, 0, 255), "CLOSED DOOR"),
    "DOORWAY_BLOCKED": ((0, 140, 255), "BLOCKED (PERSON)"),
    "PARTIAL_OPEN": ((0, 215, 255), "AJAR - GAP TOO NARROW"),
    "VERIFYING": ((255, 200, 0), "VERIFYING DEPTH..."),
}


def draw_door_hud(frame, door_res, depth_shape=None):
    """Draw door box, open gap, verdict, vote history and the door's own depth map."""
    fh, fw = frame.shape[:2]
    if not door_res or not door_res.get("found"):
        msg = door_res.get("message", "Searching for door...") if door_res else "Searching for door..."
        cv2.rectangle(frame, (15, fh - 60), (520, fh - 30), (60, 60, 60), -1)
        cv2.putText(frame, f"DOOR: {msg}", (22, fh - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return

    status = door_res["status"]
    color, tag = STATUS_STYLE.get(status, ((200, 200, 200), status))

    # Door box (thin)
    x, y, w, h = door_res["door_box"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Open gap / target (thick) + crosshair
    tx, ty, tw, th = door_res["box"]
    if status == "OPEN_DOORWAY":
        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 3)
    tcx = int(tx + tw / 2)
    tcy = int(ty + th * 0.45)
    cv2.drawMarker(frame, (tcx, tcy), color, cv2.MARKER_CROSS, 26, 2)
    if door_res["passable"]:
        cv2.line(frame, (fw // 2, fh // 2), (tcx, tcy), (0, 255, 0), 2)

    label = f"{tag} | votes [{door_res.get('votes', '')}] | conf {door_res.get('door_conf', 0):.2f}"
    cv2.rectangle(frame, (x, max(0, y - 24)), (x + min(fw - x, 9 * len(label)), max(22, y)), (20, 20, 20), -1)
    cv2.putText(frame, label, (x + 4, max(16, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    info = (f"gap {door_res.get('gap_frac', 0):.0%}  step {door_res.get('depth_step', 0):.0f}  "
            f"err X={door_res['center_x_norm']:+.2f} Y={door_res['center_y_norm']:+.2f}")
    cv2.putText(frame, info, (x, min(fh - 8, y + h + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Door depth map PiP (left side, under the buttons)
    vis = door_res.get("door_depth_vis")
    if vis is not None:
        ph = 180
        pw = max(40, int(vis.shape[1] * ph / max(1, vis.shape[0])))
        pw = min(pw, 240)
        pip = cv2.resize(vis, (pw, ph))
        cv2.rectangle(pip, (0, 0), (pw - 1, ph - 1), color, 2)
        cv2.putText(pip, "DOOR DEPTH", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        px, py = 10, 95
        if py + ph < fh and px + pw < fw:
            frame[py:py + ph, px:px + pw] = pip


def analyze_forward_corridor(norm_depth, frame_bgr=None):
    """Analyze ONLY the forward flight corridor directly ahead of the drone.

    Includes Option 1: Low-Texture / Monochrome Wall Guard.
    When facing a featureless flat surface (white/plain wall) where monocular
    depth estimation fails/collapses, detects low gradient variance and immediately
    triggers safety brake and turn away.
    """
    h, w = norm_depth.shape[:2]

    # Vertical corridor: middle 50% (eye level at 30 cm altitude)
    y1 = int(h * 0.25)
    y2 = int(h * 0.75)

    # 3 Forward Sectors: Left [10%-38%], Center [38%-62%], Right [62%-90%]
    x_l1, x_l2 = int(w * 0.10), int(w * 0.38)
    x_c1, x_c2 = int(w * 0.38), int(w * 0.62)
    x_r1, x_r2 = int(w * 0.62), int(w * 0.90)

    roi_l = norm_depth[y1:y2, x_l1:x_l2]
    roi_c = norm_depth[y1:y2, x_c1:x_c2]
    roi_r = norm_depth[y1:y2, x_r1:x_r2]

    dl = float(np.mean(roi_l)) if roi_l.size else 0.0
    dc = float(np.mean(roi_c)) if roi_c.size else 0.0
    dr = float(np.mean(roi_r)) if roi_r.size else 0.0
    dc_max = float(np.percentile(roi_c, 85)) if roi_c.size else 0.0

    # 1. Option 1: Texture & Monochrome Wall Guard (analyzes camera RGB/grayscale)
    is_flat_wall = False
    tex_grad = 99.0
    tex_lap = 99.0
    edge_density = 99.0

    if frame_bgr is not None:
        fh, fw = frame_bgr.shape[:2]
        fy1, fy2 = int(fh * 0.25), int(fh * 0.75)
        fx_c1, fx_c2 = int(fw * 0.38), int(fw * 0.62)
        center_rgb = frame_bgr[fy1:fy2, fx_c1:fx_c2]
        gray_c = cv2.cvtColor(center_rgb, cv2.COLOR_BGR2GRAY)

        tex_lap = float(cv2.Laplacian(gray_c, cv2.CV_64F).var())
        sobel_x = cv2.Sobel(gray_c, cv2.CV_32F, 1, 0)
        sobel_y = cv2.Sobel(gray_c, cv2.CV_32F, 0, 1)
        tex_grad = float(np.mean(np.abs(sobel_x) + np.abs(sobel_y)))
        canny_edges = cv2.Canny(gray_c, 30, 80)
        edge_density = float((canny_edges > 0).mean() * 100.0)

        # Monotone wall signature: zero textures, edges < 1.8%, Sobel gradient < 10.5
        if tex_grad < 10.5 and tex_lap < 22.0 and edge_density < 1.8:
            is_flat_wall = True

    # Lateral bias: positive means left table is closer -> steer right (+lr, +yaw)
    # negative means right table is closer -> steer left (-lr, -yaw)
    bias = dl - dr

    # Proportional corridor centering between table rows
    if abs(bias) < 14:
        lr = 0
        yaw = 0
    else:
        # Scale to gentle correction commands (max speed 18)
        lr = int(np.clip(bias * 0.35, -18, 18))
        yaw = int(np.clip(bias * 0.40, -20, 20))

    # Flight control logic
    if is_flat_wall:
        # Emergency brake for monochrome wall directly in front
        forward = 0
        state = "FLAT_WALL_BRAKE"
        # Determine escape direction by comparing side texture richness
        if frame_bgr is not None:
            fx_l1, fx_l2 = int(fw * 0.10), int(fw * 0.38)
            fx_r1, fx_r2 = int(fw * 0.62), int(fw * 0.90)
            gray_l = cv2.cvtColor(frame_bgr[fy1:fy2, fx_l1:fx_l2], cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_bgr[fy1:fy2, fx_r1:fx_r2], cv2.COLOR_BGR2GRAY)
            grad_l = float(np.mean(np.abs(cv2.Sobel(gray_l, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray_l, cv2.CV_32F, 0, 1))))
            grad_r = float(np.mean(np.abs(cv2.Sobel(gray_r, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray_r, cv2.CV_32F, 0, 1))))
            if grad_l > grad_r:
                lr = -10
                yaw = -24
            else:
                lr = +10
                yaw = +24
        else:
            yaw = -22 if dl < dr else +22
            lr = -8 if dl < dr else +8
    elif dc >= 165 or dc_max >= 195:
        # Table edge or dead end directly in front of drone
        forward = 0
        state = "BLOCKED_AHEAD"
    elif dc >= 135:
        # Approaching table/obstacle, slow crawl
        forward = 10
        state = "CAUTION"
    else:
        # Clear corridor between tables
        forward = 18
        state = "CLEAR_AISLE"

    return {
        "dl": dl, "dc": dc, "dr": dr, "dc_max": dc_max,
        "bias": bias, "lr": lr, "forward": forward, "yaw": yaw,
        "state": state,
        "is_flat_wall": is_flat_wall,
        "tex_grad": tex_grad,
        "tex_lap": tex_lap,
        "edge_density": edge_density,
        "boxes": ((x_l1, y1, x_l2 - x_l1, y2 - y1),
                  (x_c1, y1, x_c2 - x_c1, y2 - y1),
                  (x_r1, y1, x_r2 - x_r1, y2 - y1)),
    }


def draw_corridor_hud(frame, corridor_info, current_tof=None, target_height=35):
    """Render Forward Flight Corridor HUD on camera frame for low-altitude table navigation."""
    fh, fw = frame.shape[:2]
    if not corridor_info:
        return

    dl = corridor_info["dl"]
    dc = corridor_info["dc"]
    dr = corridor_info["dr"]
    bias = corridor_info["bias"]
    lr = corridor_info["lr"]
    fwd = corridor_info["forward"]
    yaw = corridor_info["yaw"]
    state = corridor_info["state"]
    box_l, box_c, box_r = corridor_info["boxes"]
    is_flat_wall = corridor_info.get("is_flat_wall", False)
    tex_grad = corridor_info.get("tex_grad", 0.0)
    tex_lap = corridor_info.get("tex_lap", 0.0)

    # Scale 128x96 depth box coords to frame coords
    dw, dh = 128, 96
    sx = fw / float(dw)
    sy = fh / float(dh)

    def scale_box(b):
        bx, by, bw, bh = b
        return int(bx * sx), int(by * sy), int(bw * sx), int(bh * sy)

    fl_x, fl_y, fl_w, fl_h = scale_box(box_l)
    fc_x, fc_y, fc_w, fc_h = scale_box(box_c)
    fr_x, fr_y, fr_w, fr_h = scale_box(box_r)

    # Colors
    if state == "FLAT_WALL_BRAKE":
        c_color = (0, 0, 255)
    elif state == "CLEAR_AISLE":
        c_color = (0, 255, 0)
    elif state == "CAUTION":
        c_color = (0, 200, 255)
    else:
        c_color = (0, 0, 255)

    l_color = (0, 0, 255) if dl >= 150 else ((0, 200, 255) if dl >= 120 else (200, 150, 0))
    r_color = (0, 0, 255) if dr >= 150 else ((0, 200, 255) if dr >= 120 else (0, 150, 200))

    # Sector boxes
    cv2.rectangle(frame, (fl_x, fl_y), (fl_x + fl_w, fl_y + fl_h), l_color, 1)
    cv2.rectangle(frame, (fc_x, fc_y), (fc_x + fc_w, fc_y + fc_h), c_color, 3 if is_flat_wall else 2)
    cv2.rectangle(frame, (fr_x, fr_y), (fr_x + fr_w, fr_y + fr_h), r_color, 1)

    if is_flat_wall:
        cv2.putText(frame, "MONOCHROME WALL", (fc_x + 4, fc_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Center flight crosshair
    cx = fc_x + fc_w // 2
    cy = fc_y + fc_h // 2
    cv2.drawMarker(frame, (cx, cy), c_color, cv2.MARKER_CROSS, 20, 2)

    # Lateral nudge arrow if centering
    if lr != 0:
        arrow_x = cx + int(lr * 4)
        cv2.arrowedLine(frame, (cx, cy), (arrow_x, cy), (0, 255, 255), 2, tipLength=0.3)

    # HUD Banner (bottom)
    banner_y = fh - 75
    cv2.rectangle(frame, (10, banner_y), (fw - 10, fh - 10), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, banner_y), (fw - 10, fh - 10), (100, 100, 100), 1)

    # Line 1: Depth Sectors and State
    if is_flat_wall:
        txt1 = f"WALL GUARD [MONOCHROME WALL]: Tex {tex_grad:.1f} | Lap {tex_lap:.1f} -> BRAKE & ESCAPE"
    else:
        txt1 = f"CORRIDOR [30cm AISLE]: L:{dl:.0f} | C:{dc:.0f} | R:{dr:.0f}  [{state}]"
    cv2.putText(frame, txt1, (20, banner_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, c_color, 2)

    # Line 2: ToF altitude + RC corrections
    tof_txt = f"ALT: {current_tof}cm (target {target_height}cm)" if current_tof is not None else "ALT: ~35cm"
    txt2 = f"{tof_txt} | Bias:{bias:+.0f} -> lr:{lr} fwd:{fwd} yaw:{yaw}"
    cv2.putText(frame, txt2, (20, banner_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
