"""Indoor Room Geometry & Navigation Command Generator based on Depth Anything V2 (DA2).

Analyzes 3D room structure, aperture discontinuities (doors / corridors),
monochrome wall hazards, and computes real-time drone flight commands.
"""

import cv2
import numpy as np

# Thresholds for DA2 disparity (0..255; higher = closer to drone)
WALL_NEAR_THRESHOLD = 165       # Center sector disparity indicating close wall
WALL_MAX_THRESHOLD = 195        # Center max peak disparity indicating hazard
CAUTION_THRESHOLD = 135         # Slow crawl threshold

OPEN_DOOR_MIN_CONTRAST = 25     # Min disparity drop inside door vs flanking walls
OPEN_DOOR_MAX_INNER = 100       # Max disparity inside door opening (must be distant)
PASSAGE_MIN_WIDTH_FRAC = 0.10   # Aperture must be >= 10% of frame width
PASSAGE_MAX_WIDTH_FRAC = 0.70   # Aperture must be <= 70% of frame width


def analyze_monochrome_wall(frame_bgr):
    """Detect flat, featureless surfaces where optical flow / vision can degrade."""
    fh, fw = frame_bgr.shape[:2]
    fy1, fy2 = int(fh * 0.18), int(fh * 0.54)  # Eye-level corridor window (cuts off floor)
    fx_c1, fx_c2 = int(fw * 0.38), int(fw * 0.62)
    center_roi = frame_bgr[fy1:fy2, fx_c1:fx_c2]
    gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad_mean = float(np.mean(np.abs(sobel_x) + np.abs(sobel_y)))
    canny = cv2.Canny(gray, 30, 80)
    edge_density = float((canny > 0).mean() * 100.0)

    is_flat = grad_mean < 10.5 and lap_var < 22.0 and edge_density < 1.8
    return is_flat, grad_mean, lap_var, edge_density


def detect_geometric_passages(norm_depth):
    """Detect passage apertures from depth step discontinuities (jambs).

    Returns list of candidate passages: (x1, x2, inner_depth, depth_step, center_x)
    """
    dh, dw = norm_depth.shape[:2]
    mid_y0, mid_y1 = int(dh * 0.18), int(dh * 0.54)  # Eye-level (floor excluded)
    mid_band = norm_depth[mid_y0:mid_y1, :]

    col_profile = mid_band.mean(axis=0)
    col_smooth = cv2.GaussianBlur(col_profile.reshape(1, -1), (1, 5), 1.0)[0]
    grad_x = np.diff(col_smooth)

    # Disparity drop: left wall ends and recedes into depth (negative gradient)
    left_jambs = [x for x in range(5, dw - 20) if grad_x[x] < -2.8]
    # Disparity rise: space ends and rises into right wall (positive gradient)
    right_jambs = [x for x in range(20, dw - 5) if grad_x[x] > 2.8]

    passages = []
    if left_jambs and right_jambs:
        for lj in left_jambs:
            for rj in right_jambs:
                w_pix = rj - lj
                if int(dw * PASSAGE_MIN_WIDTH_FRAC) <= w_pix <= int(dw * PASSAGE_MAX_WIDTH_FRAC):
                    inside_val = float(col_smooth[lj:rj].mean())
                    outside_l = float(col_smooth[max(0, lj - 8):lj].mean())
                    outside_r = float(col_smooth[rj:min(dw, rj + 8)].mean())
                    step = min(outside_l, outside_r) - inside_val
                    if step > 18:
                        cx = (lj + rj) / 2.0
                        passages.append({
                            "x1": lj,
                            "x2": rj,
                            "width": w_pix,
                            "inner_depth": inside_val,
                            "depth_step": step,
                            "cx_norm": float(cx / dw - 0.5),
                        })

    if passages:
        # Sort by contrast step descending
        passages.sort(key=lambda p: p["depth_step"], reverse=True)
    return passages


class RoomGeometryEvaluator:
    """Evaluates 3D room geometry and generates flight commands using DA2."""

    def __init__(self, yolo_detector=None):
        self.yolo_detector = yolo_detector

    def evaluate(self, frame_bgr, norm_depth, sectors, center_max):
        """Analyze full frame room geometry and calculate steering commands.

        Returns:
            dict containing geometry state, door status, navigation state,
            and suggested RC flight commands (lr, fb, ud, yaw).
        """
        fh, fw = frame_bgr.shape[:2]
        dh, dw = norm_depth.shape[:2]
        dl, dc, dr = sectors

        # 1. Monochrome wall inspection
        is_flat, grad_mean, lap_var, edge_density = analyze_monochrome_wall(frame_bgr)

        # 2. Geometric apertures
        passages = detect_geometric_passages(norm_depth)
        best_passage = passages[0] if passages else None

        # 3. Semantic Door Analysis (if YOLO available)
        door_info = {"status": "NO_DOOR", "box": None, "passable": False, "conf": 0.0}
        if self.yolo_detector is not None:
            door_info = self._evaluate_yolo_doors(frame_bgr, norm_depth)

        # 4. Corridor lateral bias (Left vs Right)
        bias = dl - dr

        # 5. Decide Navigation State and RC Commands
        state = "CLEAR_AISLE"
        lr = 0
        fb = 0
        ud = 0
        yaw = 0

        # Priority 1: Semantic Doorway in sight
        if door_info.get("status") == "OPEN_DOORWAY" and door_info.get("passable"):
            state = "OPEN_DOORWAY"
            cx_n = door_info.get("center_x_norm", 0.0)
            cy_n = door_info.get("center_y_norm", 0.0)

            # Center towards doorway aperture
            yaw = int(np.clip(cx_n * 60, -25, 25))
            lr = int(np.clip(cx_n * 40, -16, 16))
            ud = int(np.clip(-cy_n * 45, -20, 20))

            # Move forward when roughly aligned
            if abs(cx_n) < 0.16:
                fb = 20
            else:
                fb = 8

        elif door_info.get("status") in ("CLOSED_DOOR", "DOORWAY_BLOCKED"):
            state = door_info["status"]
            fb = 0
            lr = 0
            ud = 0
            yaw = 0

        # Priority 2: Monochrome Wall Guard
        elif is_flat and dc > 90:
            state = "FLAT_WALL_BRAKE"
            fb = 0
            ud = 0
            # Escape turn towards side with lower disparity (more open space)
            if dl < dr:
                lr = -12
                yaw = -24
            else:
                lr = +12
                yaw = +24

        # Priority 3: Geometric Passage (open doorway/corridor without YOLO frame)
        elif best_passage is not None and dc < 150:
            state = "OPEN_CORRIDOR"
            p_cx_n = best_passage["cx_norm"]
            yaw = int(np.clip(p_cx_n * 50, -20, 20))
            lr = int(np.clip(p_cx_n * 35, -15, 15))
            fb = 18 if abs(p_cx_n) < 0.20 else 10

        # Priority 4: Direct Obstacle or End of Corridor
        elif dc >= WALL_NEAR_THRESHOLD or center_max >= WALL_MAX_THRESHOLD:
            state = "BLOCKED_AHEAD"
            fb = 0
            # Turn away from closer obstacle
            if abs(bias) > 10:
                lr = int(np.clip(bias * 0.40, -18, 18))
                yaw = int(np.clip(bias * 0.50, -22, 22))
            else:
                lr = 0
                yaw = 0

        # Priority 5: Approaching Obstacle (Caution)
        elif dc >= CAUTION_THRESHOLD:
            state = "CAUTION"
            fb = 10
            if abs(bias) >= 12:
                lr = int(np.clip(bias * 0.35, -16, 16))
                yaw = int(np.clip(bias * 0.40, -18, 18))

        # Priority 6: Clear corridor
        else:
            state = "CLEAR_AISLE"
            fb = 18
            if abs(bias) >= 14:
                lr = int(np.clip(bias * 0.35, -18, 18))
                yaw = int(np.clip(bias * 0.40, -20, 20))

        return {
            "state": state,
            "commands": {"lr": lr, "fb": fb, "ud": ud, "yaw": yaw},
            "sectors": (dl, dc, dr),
            "center_max": center_max,
            "bias": bias,
            "door_info": door_info,
            "passages": passages,
            "is_flat_wall": is_flat,
            "tex_metrics": {"grad": grad_mean, "lap": lap_var, "edge": edge_density},
        }

    def _evaluate_yolo_doors(self, frame_bgr, norm_depth):
        """Run YOLO door detector and verify opening using DA2 depth."""
        fh, fw = frame_bgr.shape[:2]
        dh, dw = norm_depth.shape[:2]
        sx = fw / float(dw)
        sy = fh / float(dh)

        try:
            results = self.yolo_detector(frame_bgr, conf=0.08, verbose=False)[0]
        except Exception:
            return {"status": "NO_DOOR", "box": None, "passable": False, "conf": 0.0}

        doors = []
        persons = []
        for b in results.boxes:
            cls_name = self.yolo_detector.names[int(b.cls[0])]
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            box = (x1, y1, x2 - x1, y2 - y1)
            if cls_name in ["door", "door frame", "doorway"]:
                doors.append({"conf": conf, "box": box})
            elif cls_name == "person":
                persons.append({"conf": conf, "box": box})

        if not doors:
            return {"status": "NO_DOOR", "box": None, "passable": False, "conf": 0.0}

        # Select largest candidate
        best_door = max(doors, key=lambda d: d["box"][2] * d["box"][3])
        bx, by, bw, bh = best_door["box"]

        # Map to depth coordinates
        dx1 = max(0, int(bx / sx))
        dy1 = max(0, int(by / sy))
        dx2 = min(dw - 1, int((bx + bw) / sx))
        dy2 = min(dh - 1, int((by + bh) / sy))

        if dx2 - dx1 < 6 or dy2 - dy1 < 10:
            return {"status": "NO_DOOR", "box": best_door["box"], "passable": False, "conf": best_door["conf"]}

        # Measure inner depth
        # upper/middle 60% of door height (ignore ground at bottom)
        inner_y1 = dy1 + int((dy2 - dy1) * 0.15)
        inner_y2 = dy1 + int((dy2 - dy1) * 0.75)
        inner_depth = float(np.median(norm_depth[inner_y1:inner_y2, dx1:dx2]))

        # Flanking wall strips
        flank_w = max(4, int((dx2 - dx1) * 0.35))
        l_wall_x1, l_wall_x2 = max(0, dx1 - flank_w), dx1
        r_wall_x1, r_wall_x2 = dx2, min(dw, dx2 + flank_w)

        wall_samples = []
        if l_wall_x2 > l_wall_x1:
            wall_samples.append(norm_depth[inner_y1:inner_y2, l_wall_x1:l_wall_x2].ravel())
        if r_wall_x2 > r_wall_x1:
            wall_samples.append(norm_depth[inner_y1:inner_y2, r_wall_x1:r_wall_x2].ravel())

        if wall_samples:
            wall_depth = float(np.median(np.concatenate(wall_samples)))
        else:
            wall_depth = inner_depth

        depth_step = wall_depth - inner_depth

        # Person check
        blocked = False
        for p in persons:
            px, py, pw, ph = p["box"]
            ix1, iy1 = max(bx, px), max(by, py)
            ix2, iy2 = min(bx + bw, px + pw), min(by + bh, py + ph)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter > 0.20 * (bw * bh):
                    blocked = True
                    break

        cx = bx + bw / 2.0
        cy = by + bh * 0.45
        cx_norm = float(cx / fw - 0.5)
        cy_norm = float(cy / fh - 0.5)

        if blocked:
            status = "DOORWAY_BLOCKED"
            passable = False
        elif depth_step >= OPEN_DOOR_MIN_CONTRAST and inner_depth <= OPEN_DOOR_MAX_INNER:
            status = "OPEN_DOORWAY"
            passable = True
        else:
            status = "CLOSED_DOOR"
            passable = False

        return {
            "status": status,
            "passable": passable,
            "box": best_door["box"],
            "conf": best_door["conf"],
            "depth_step": depth_step,
            "inner_depth": inner_depth,
            "wall_depth": wall_depth,
            "center_x_norm": cx_norm,
            "center_y_norm": cy_norm,
        }
