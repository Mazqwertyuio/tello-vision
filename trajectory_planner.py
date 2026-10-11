"""Trajectory Planner & Visualization for Tello Indoor Flight.

Compares:
1. Old Reactive Route: Bang-bang / proportional steering based on coarse 3-sector thresholding.
2. New DA2 Planned Route: Continuous free-space corridor optimization, obstacle avoidance,
   and smooth polynomial path heading toward doorway apertures and deep valleys.

Provides:
- Top-Down Bird's-Eye View (BEV) Radar Mini-Map
- 3D AR Perspective Path Corridor projection directly on the camera feed
"""

import math
import cv2
import numpy as np


class FlightTrajectoryPlanner:
    """Plans and visualizes comparative flight paths."""

    def __init__(self, fov_deg=82.6, cam_height=0.35):
        self.fov_deg = fov_deg
        self.fov_rad = math.radians(fov_deg)
        self.cam_height = cam_height  # altitude in meters
        self.window_cfg = None

    def find_deepest_clearance(self, norm_depth):
        """Find the heading angle with the maximum open distance clearance, ensuring corridor width."""
        dh, dw = norm_depth.shape[:2]
        cfg = self.window_cfg
        if cfg:
            y1 = int(dh * (cfg.get("y_min_pct", 18) / 100.0))
            y2 = int(dh * (cfg.get("y_max_pct", 54) / 100.0))
        else:
            y1, y2 = int(dh * 0.18), int(dh * 0.54)
        y1, y2 = max(0, y1), min(dh, max(y1 + 4, y2))
        depth_slice = norm_depth[y1:y2, :]
        col_disp = np.percentile(depth_slice, 60, axis=0)

        k_size = max(5, int(dw * 0.14))
        if k_size % 2 == 0:
            k_size += 1
        smoothed_disp = cv2.GaussianBlur(col_disp.astype(np.float32), (k_size, 1), 0).ravel()

        col_indices = np.arange(dw)
        center_norm = np.abs(col_indices - (dw / 2.0)) / (dw / 2.0)
        cost = smoothed_disp + (center_norm * 14.0)

        best_col = int(np.argmin(cost))
        best_angle = (best_col / float(dw) - 0.5) * self.fov_rad

        best_d = float(smoothed_disp[best_col])
        clamped_d = np.clip(best_d, 15.0, 240.0)
        dist_m = 0.40 + 3.8 * ((240.0 - clamped_d) / 225.0) ** 1.3
        return best_angle, dist_m

    def plan_trajectories(self, norm_depth, sectors, eval_res, telem_row=None):
        """Compute both Old Reactive Route and New DA2 Planned Route.

        Returns:
            old_path: list of (x, z) points in meters (z = forward, x = right)
            new_path: list of (x, z) points in meters
            polar_obstacles: list of (x, z) obstacle boundary points
        """
        dh, dw = norm_depth.shape[:2]
        dl, dc, dr = sectors
        cmds = eval_res["commands"]
        state = eval_res["state"]
        door = eval_res.get("door_info", {})
        passages = eval_res.get("passages", [])

        # 1. Extract 2D polar obstacle profile from DA2 depth (Eye-level, Floor Excluded)
        mid_y1, mid_y2 = int(dh * 0.18), int(dh * 0.54)
        depth_slice = norm_depth[mid_y1:mid_y2, :]
        col_depth = np.median(depth_slice, axis=0)

        polar_obstacles = []
        num_cols = len(col_depth)
        step_col = max(1, num_cols // 45)

        for i in range(0, num_cols, step_col):
            d_val = float(col_depth[i])
            # Map [0..255] disparity to approximate distance [0.4m .. 4.5m]
            # Higher disparity = closer object
            clamped_d = np.clip(d_val, 15.0, 240.0)
            dist_m = 0.40 + 3.8 * ((240.0 - clamped_d) / 225.0) ** 1.3

            angle = (i / float(num_cols) - 0.5) * self.fov_rad
            ox = dist_m * math.sin(angle)
            oz = dist_m * math.cos(angle)
            polar_obstacles.append((ox, oz))

        # 2. OLD ROUTE: What the previous drone autopilot did
        # Based on recorded telemetry or coarse sector bias
        old_fb = float(telem_row.get("fb", 0)) if telem_row else float(cmds["fb"])
        old_lr = float(telem_row.get("lr", 0)) if telem_row else float(cmds["lr"])
        old_yaw = float(telem_row.get("yaw", 0)) if telem_row else float(cmds["yaw"])

        old_path = []
        # Simulate 2.5 seconds trajectory under old reactive commands
        dt = 0.2
        steps = 12
        cur_x, cur_z, cur_psi = 0.0, 0.0, 0.0

        # Speeds in m/s (Tello speed command ~ 18 corresponds to ~0.35 m/s)
        old_v = max(0.0, old_fb * 0.020)
        old_w = -old_yaw * 0.015  # rad/s

        for _ in range(steps):
            old_path.append((cur_x, cur_z))
            if old_v < 0.05:
                break
            cur_psi += old_w * dt
            cur_z += old_v * math.cos(cur_psi) * dt
            cur_x += (old_v * math.sin(cur_psi) + old_lr * 0.015) * dt

        # 3. NEW DA2 ROUTE: Look-ahead Free Space & Geometric Corridor
        # Determine target heading angle
        target_heading = 0.0
        target_speed = 0.40  # m/s

        if state == "OPEN_DOORWAY" and door and door.get("passable"):
            cx_norm = door.get("center_x_norm", 0.0)
            target_heading = cx_norm * (self.fov_rad * 0.75)
            target_speed = 0.45
        elif state == "OPEN_CORRIDOR" and passages:
            cx_norm = passages[0]["cx_norm"]
            target_heading = cx_norm * (self.fov_rad * 0.65)
            target_speed = 0.42
        elif state in ("BLOCKED_AHEAD", "CLOSED_DOOR"):
            target_speed = 0.0  # Stop path
            target_heading = 0.0
        elif state == "FLAT_WALL_BRAKE":
            target_speed = 0.05
            target_heading = math.radians(28.0) if dl >= dr else math.radians(-28.0)
        elif state == "CAUTION":
            bias = dl - dr
            target_heading = math.radians(np.clip(bias * 0.35, -20.0, 20.0))
            target_speed = 0.22
        else:  # CLEAR_AISLE / EXPLORE: Target deepest room space with clearance
            deep_angle, deep_dist = self.find_deepest_clearance(norm_depth)
            target_heading = float(np.clip(deep_angle, -self.fov_rad * 0.45, self.fov_rad * 0.45))
            target_speed = 0.42 if deep_dist > 2.0 else 0.26

        new_path = []
        cur_x, cur_z, cur_psi = 0.0, 0.0, 0.0
        # Smooth pursuit toward target heading
        for _ in range(steps):
            new_path.append((cur_x, cur_z))
            if target_speed < 0.03:
                break
            # Gentle heading convergence
            heading_err = target_heading - cur_psi
            turn_rate = np.clip(heading_err * 2.5, -0.6, 0.6)
            cur_psi += turn_rate * dt
            cur_z += target_speed * math.cos(cur_psi) * dt
            cur_x += target_speed * math.sin(cur_psi) * dt

        return old_path, new_path, polar_obstacles


def render_bev_minimap(old_path, new_path, polar_obstacles, state, size=(260, 260)):
    """Render 2D Bird's-Eye View (Top-Down Radar) Mini-Map."""
    mw, mh = size
    bev = np.zeros((mh, mw, 3), dtype=np.uint8)
    bev[:] = (18, 22, 30)

    # Coordinate mapping: Drone is at (mw/2, mh - 30)
    # Range: 3.5 meters forward, +/- 1.8 meters laterally
    cx = mw // 2
    cz = mh - 28
    scale = 62.0  # pixels per meter

    def to_map(x, z):
        u = int(cx + x * scale)
        v = int(cz - z * scale)
        return (u, v)

    # 1. Draw distance grid rings (1m, 2m, 3m)
    for dist in (1.0, 2.0, 3.0):
        r = int(dist * scale)
        cv2.ellipse(bev, (cx, cz), (r, r), 0, 180, 360, (40, 50, 65), 1)
        cv2.putText(bev, f"{dist:.0f}m", (cx + r - 18, cz - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 95, 115), 1)

    # FOV cone lines
    fov_half = math.radians(41.3)
    cv2.line(bev, (cx, cz), to_map(3.2 * math.sin(-fov_half), 3.2 * math.cos(-fov_half)), (45, 55, 75), 1)
    cv2.line(bev, (cx, cz), to_map(3.2 * math.sin(fov_half), 3.2 * math.cos(fov_half)), (45, 55, 75), 1)

    # 2. Draw Obstacle contour from DA2
    if len(polar_obstacles) > 2:
        obs_pts = [to_map(x, z) for x, z in polar_obstacles]
        for i in range(len(obs_pts) - 1):
            cv2.line(bev, obs_pts[i], obs_pts[i + 1], (70, 70, 230), 2)
        for pt in obs_pts:
            cv2.circle(bev, pt, 2, (100, 100, 255), -1)

    # 3. Draw OLD ROUTE (dashed/thin orange line)
    if len(old_path) > 1:
        old_pts = [to_map(x, z) for x, z in old_path]
        for i in range(len(old_pts) - 1):
            cv2.line(bev, old_pts[i], old_pts[i + 1], (0, 140, 255), 2)
        for pt in old_pts:
            cv2.circle(bev, pt, 2, (0, 160, 255), -1)

    # 4. Draw NEW DA2 ROUTE (glowing thick green path)
    if len(new_path) > 1:
        new_pts = [to_map(x, z) for x, z in new_path]
        # Glowing shadow
        for i in range(len(new_pts) - 1):
            cv2.line(bev, new_pts[i], new_pts[i + 1], (0, 180, 0), 4)
            cv2.line(bev, new_pts[i], new_pts[i + 1], (100, 255, 100), 2)
        for pt in new_pts:
            cv2.circle(bev, pt, 3, (0, 255, 0), -1)

    # 5. Draw Drone Marker at (cx, cz)
    d_col = (0, 255, 255)
    cv2.circle(bev, (cx, cz), 6, d_col, -1)
    cv2.line(bev, (cx - 8, cz), (cx + 8, cz), (255, 255, 255), 2)
    cv2.line(bev, (cx, cz - 10), (cx, cz + 4), (255, 255, 255), 2)

    # 6. Title and Legend
    cv2.rectangle(bev, (0, 0), (mw, 36), (12, 16, 22), -1)
    cv2.putText(bev, "RADAR: TRAJECTORY COMPARISON", (8, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)

    # Legend items
    cv2.line(bev, (8, 26), (22, 26), (0, 140, 255), 2)
    cv2.putText(bev, "OLD (Reactive)", (26, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 160, 255), 1)

    cv2.line(bev, (130, 26), (146, 26), (0, 255, 0), 2)
    cv2.putText(bev, "NEW (DA2 Planned)", (150, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 100), 1)

    # Border
    cv2.rectangle(bev, (0, 0), (mw - 1, mh - 1), (70, 85, 105), 1)
    return bev


def draw_ar_perspective_path(frame, new_path, old_path, cam_height=0.35):
    """Draw AR 3D navigation corridor on camera feed."""
    fh, fw = frame.shape[:2]
    cx = fw / 2.0
    cy = fh * 0.52
    fx = fw * 0.65
    fy = fw * 0.65

    def project_pt(x, z, y_offset=0.25):
        if z <= 0.2:
            return None
        u = int(cx + (x / z) * fx)
        v = int(cy + (y_offset / z) * fy)
        return (u, v)

    # 1. Draw OLD Path in Camera View (thin orange line with waypoints)
    if len(old_path) > 1:
        prev_p = None
        for x, z in old_path:
            p = project_pt(x, z, y_offset=cam_height)
            if p and (0 <= p[0] < fw and 0 <= p[1] < fh):
                if prev_p:
                    cv2.line(frame, prev_p, p, (0, 140, 255), 2)
                cv2.circle(frame, p, 3, (0, 140, 255), -1)
                prev_p = p

    # 2. Draw NEW DA2 Path Corridor (semi-transparent green ribbon)
    if len(new_path) > 2:
        ribbon_left = []
        ribbon_right = []
        drone_half_w = 0.18  # meters

        for x, z in new_path:
            pl = project_pt(x - drone_half_w, z, y_offset=cam_height)
            pr = project_pt(x + drone_half_w, z, y_offset=cam_height)
            if pl and pr and (0 <= pl[1] < fh) and (0 <= pr[1] < fh):
                ribbon_left.append(pl)
                ribbon_right.append(pr)

        if len(ribbon_left) >= 2:
            poly_pts = np.array(ribbon_left + ribbon_right[::-1], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly_pts], (0, 200, 50))
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            # Draw glowing edges
            for i in range(len(ribbon_left) - 1):
                cv2.line(frame, ribbon_left[i], ribbon_left[i + 1], (0, 255, 0), 2)
                cv2.line(frame, ribbon_right[i], ribbon_right[i + 1], (0, 255, 0), 2)

            # Draw central flight path & waypoint beads
            for i in range(len(new_path) - 1):
                p1 = project_pt(new_path[i][0], new_path[i][1], y_offset=cam_height)
                p2 = project_pt(new_path[i + 1][0], new_path[i + 1][1], y_offset=cam_height)
                if p1 and p2:
                    cv2.line(frame, p1, p2, (255, 255, 255), 2)
                    cv2.circle(frame, p2, 4, (0, 255, 0), -1)
                    cv2.circle(frame, p2, 2, (255, 255, 255), -1)
