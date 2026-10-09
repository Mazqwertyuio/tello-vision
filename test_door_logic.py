"""Test script to evaluate Geometric + Semantic Room & Door Navigation."""
import cv2
import numpy as np
from ultralytics import YOLO
from tello_connect import MonoDepthEstimator

estimator = MonoDepthEstimator(input_size=(128, 96))
yolo = YOLO("yolov8s-worldv2.pt")
yolo.set_classes(["door", "doorway", "closed door", "open door", "person", "window", "corridor"])


def evaluate_room_geometry(frame, norm_depth):
    """
    Evaluates 3D room geometry and door passage.
    Returns:
        dict with room geometry interpretation:
          - state: 'WALL_AHEAD' | 'CLOSED_DOOR' | 'OPEN_DOORWAY' | 'CORRIDOR' | 'BLOCKED_BY_OBSTACLE'
          - target: (cx, cy) or None
          - boxes: list of detected items
    """
    fh, fw = frame.shape[:2]
    dh, dw = norm_depth.shape[:2]
    scale_x = fw / dw
    scale_y = fh / dh

    # 1. Semantic detection with YOLO
    yolo_res = yolo(frame, conf=0.15, verbose=False)[0]
    
    doors = []
    obstacles = []
    
    for b in yolo_res.boxes:
        cls_idx = int(b.cls[0])
        cls_name = yolo.names[cls_idx]
        conf = float(b.conf[0])
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        
        if cls_name in ["door", "doorway", "closed door", "open door", "corridor"]:
            doors.append({"cls": cls_name, "conf": conf, "box": (x1, y1, x2 - x1, y2 - y1)})
        elif cls_name in ["person"]:
            obstacles.append({"cls": cls_name, "conf": conf, "box": (x1, y1, x2 - x1, y2 - y1)})

    # 2. Geometric analysis of candidate doorways
    results = []
    
    # Analyze any semantic door candidates
    for d in doors:
        bx, by, bw, bh = d["box"]
        # Convert to depth coords
        dx1 = max(0, int(bx / scale_x))
        dy1 = max(0, int(by / scale_y))
        dx2 = min(dw - 1, int((bx + bw) / scale_x))
        dy2 = min(dh - 1, int((by + bh) / scale_y))
        
        if dx2 - dx1 < 6 or dy2 - dy1 < 10:
            continue
            
        # Inner depth of door opening
        inner_depth = float(np.median(norm_depth[dy1:dy2, dx1:dx2]))
        
        # Flanking walls: left flank and right flank
        flank_w = max(4, int((dx2 - dx1) * 0.4))
        left_flank_x1 = max(0, dx1 - flank_w)
        left_flank_x2 = dx1
        right_flank_x1 = dx2
        right_flank_x2 = min(dw, dx2 + flank_w)
        
        left_depth = float(np.median(norm_depth[dy1:dy2, left_flank_x1:left_flank_x2])) if left_flank_x2 > left_flank_x1 else inner_depth
        right_depth = float(np.median(norm_depth[dy1:dy2, right_flank_x1:right_flank_x2])) if right_flank_x2 > right_flank_x1 else inner_depth
        flank_depth = max(left_depth, right_depth) # Closer wall has higher disparity in norm_depth
        
        # Disparity step: In norm_depth, larger value = closer, smaller value = deeper
        depth_step = flank_depth - inner_depth
        
        # Check if an obstacle (person) overlaps with this doorway
        has_obstacle = False
        for obs in obstacles:
            ox, oy, ow, oh = obs["box"]
            # Intersection over door area
            ix1 = max(bx, ox)
            iy1 = max(by, oy)
            ix2 = min(bx + bw, ox + ow)
            iy2 = min(by + bh, oy + oh)
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                if inter_area > (bw * bh) * 0.20:
                    has_obstacle = True
                    break
                    
        if has_obstacle:
            status = "BLOCKED_BY_OBSTACLE"
        elif depth_step > 25 and inner_depth < 90:
            status = "OPEN_DOORWAY"
        else:
            status = "CLOSED_DOOR"
            
        results.append({
            "type": "SEMANTIC_DOOR",
            "status": status,
            "conf": d["conf"],
            "box": d["box"],
            "inner_depth": inner_depth,
            "flank_depth": flank_depth,
            "depth_step": depth_step
        })

    # 3. Geometric Room Aperture Analysis (Pure Depth Discontinuity)
    # Check if there is an open passage even if YOLO didn't detect a door frame
    mid_y0, mid_y1 = int(dh * 0.25), int(dh * 0.75)
    mid_band = norm_depth[mid_y0:mid_y1, :]
    col_profile = mid_band.mean(axis=0)
    col_smooth = cv2.GaussianBlur(col_profile.reshape(1, -1), (1, 5), 1.0)[0]
    grad_x = np.diff(col_smooth)
    
    # Find step discontinuities (left wall drops into passage, passage rises back to right wall)
    left_jambs = [x for x in range(5, dw - 25) if grad_x[x] < -3.5]
    right_jambs = [x for x in range(25, dw - 5) if grad_x[x] > 3.5]
    
    geom_openings = []
    if left_jambs and right_jambs:
        for lj in left_jambs:
            for rj in right_jambs:
                if 12 <= (rj - lj) <= int(dw * 0.65): # Width between 10% and 65% of screen
                    # Check depth inside vs outside
                    inside_val = float(col_smooth[lj:rj].mean())
                    outside_left = float(col_smooth[max(0, lj - 8):lj].mean())
                    outside_right = float(col_smooth[rj:min(dw, rj + 8)].mean())
                    if (outside_left - inside_val > 20) and (outside_right - inside_val > 20):
                        geom_openings.append((lj, rj, inside_val, min(outside_left, outside_right) - inside_val))

    if geom_openings:
        # Best opening with largest depth contrast
        best_geom = max(geom_openings, key=lambda g: g[3])
        lj, rj, in_val, step = best_geom
        gbx = int(lj * scale_x)
        gbw = int((rj - lj) * scale_x)
        gby = int(mid_y0 * scale_y)
        gbh = int((mid_y1 - mid_y0) * scale_y)
        results.append({
            "type": "GEOMETRIC_PASSAGE",
            "status": "OPEN_CORRIDOR",
            "conf": 1.0,
            "box": (gbx, gby, gbw, gbh),
            "inner_depth": in_val,
            "depth_step": step
        })

    return results


print("Testing evaluation on frames...")
for name in ['inspect_103.jpg', 'inspect_309.jpg', 'inspect_412.jpg', 'inspect_515.jpg']:
    img = cv2.imread(f"debug/{name}")
    norm_depth, _, _, _, _ = estimator.predict(img)
    res = evaluate_room_geometry(img, norm_depth)
    print(f"\n=== {name} ===")
    if not res:
        print("  Result: FLAT WALL / ENCLOSED ROOM (No passage)")
    for r in res:
        print(f"  [{r['type']}] Status: {r['status']}, Step: {r.get('depth_step', 0):.1f}, Inner: {r['inner_depth']:.1f}, Box: {r['box']}")
