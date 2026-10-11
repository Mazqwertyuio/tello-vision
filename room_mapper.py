"""Room Occupancy Mapper & Frontier Exploration for Tello & Depth Anything V2.

Builds a real-time 2D/2.5D global occupancy grid map of the room geometry:
1. Tracks drone odometry (x, y, yaw, altitude) from flight telemetry / commands.
2. Raycasts monocular depth rays from DA2 into a metric 2D grid:
     - Free space: cleared cells between drone and obstacles
     - Occupied cells: obstacles, walls, furniture, table legs
     - Apertures: doors and open passages
3. Frontier Exploration & Deepest Clearance Targeting:
     - Finds the deepest open room sector (maximum distance away from obstacles)
     - Validates safe passage clearance (drone diameter width clearance)
     - Generates heading and waypoint commands for autonomous room exploration.
4. Real-time rendering: High-tech HUD map with drone FOV cone, obstacles,
   traveled trajectory, and frontier target vector.
"""

import math
import cv2
import numpy as np


class RoomOccupancyMapper:
    """Real-time 2D metric occupancy grid map with frontier exploration targeting."""

    def __init__(
        self,
        map_size_m=12.0,       # 12x12 meters room canvas
        resolution_m=0.05,     # 5 cm per cell -> 240x240 grid
        fov_deg=82.6,
        drone_radius_m=0.22,   # DJI Tello radius with prop guards ~ 20cm
        max_ray_dist_m=5.0,    # Maximum reliable depth range
    ):
        self.map_size_m = map_size_m
        self.res = resolution_m
        self.grid_size = int(round(map_size_m / resolution_m))  # 240 cells
        self.center_cell = self.grid_size // 2
        self.fov_rad = math.radians(fov_deg)
        self.drone_radius_m = drone_radius_m
        self.max_ray_dist_m = max_ray_dist_m

        # Grid layers (values between 0.0 and 1.0)
        # log-odds or simple occupancy counts
        self.free_counts = np.zeros((self.grid_size, self.grid_size), dtype=np.int16)
        self.occ_counts = np.zeros((self.grid_size, self.grid_size), dtype=np.int16)

        # Drone estimated global pose
        self.drone_x = 0.0     # meters, right = +X
        self.drone_y = 0.0     # meters, forward = +Y
        self.drone_yaw = 0.0   # radians, 0 = forward along +Y, clockwise = positive
        self.trajectory = [(0.0, 0.0)]
        self.last_timestamp = None

        # Deepest target found in global and local frames
        self.deepest_target_local = (0.0, 2.5)  # (x, y) meters
        self.deepest_target_global = (0.0, 2.5)
        self.deepest_target_angle = 0.0         # radians relative to heading
        self.deepest_target_dist = 2.5          # meters

    def reset(self):
        """Reset map to empty state."""
        self.free_counts.fill(0)
        self.occ_counts.fill(0)
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_yaw = 0.0
        self.trajectory = [(0.0, 0.0)]
        self.last_timestamp = None

    def world_to_grid(self, wx, wy):
        """Convert metric coordinates (meters, relative to origin) to grid indices (col, row)."""
        gx = int(round(self.center_cell + (wx / self.res)))
        gy = int(round(self.center_cell - (wy / self.res)))  # +Y is up on map
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Convert grid cell (col, row) to metric world coordinates."""
        wx = (gx - self.center_cell) * self.res
        wy = (self.center_cell - gy) * self.res
        return wx, wy

    def update_odometry(self, dt=0.033, rc_lr=0, rc_fb=0, rc_yaw=0):
        """Dead-reckoning motion estimation based on RC commands and elapsed time.

        rc_fb: forward (+)/back (-) speed command (-100..100), ~0.02 m/s per unit
        rc_lr: lateral right (+)/left (-) speed command (-100..100)
        rc_yaw: clockwise (+)/counter-clockwise (-) rotation (-100..100), ~0.02 rad/s per unit
        """
        if dt <= 0 or dt > 0.5:
            dt = 0.033

        # Speed scaling factors (empirically calibrated for DJI Tello indoor flight)
        vx_body = float(rc_lr) * 0.016   # lateral velocity m/s
        vy_body = float(rc_fb) * 0.018   # forward velocity m/s
        omega = float(rc_yaw) * 0.022     # yaw rate rad/s

        self.drone_yaw += omega * dt
        # Normalize yaw to [-pi, pi]
        self.drone_yaw = (self.drone_yaw + math.pi) % (2 * math.pi) - math.pi

        # Transform body velocity to world coordinates
        cos_y = math.cos(self.drone_yaw)
        sin_y = math.sin(self.drone_yaw)

        dx = (vx_body * cos_y + vy_body * sin_y) * dt
        dy = (-vx_body * sin_y + vy_body * cos_y) * dt

        self.drone_x += dx
        self.drone_y += dy

        # Record trajectory waypoint if moved noticeably
        last_x, last_y = self.trajectory[-1]
        dist_moved = math.hypot(self.drone_x - last_x, self.drone_y - last_y)
        if dist_moved >= 0.05:  # Every 5 cm
            self.trajectory.append((self.drone_x, self.drone_y))

    def update_from_telemetry(self, telem_row, dt=0.033):
        """Update odometry from a telemetry dictionary."""
        if not telem_row:
            return
        lr = float(telem_row.get("lr", 0) or 0)
        fb = float(telem_row.get("fb", 0) or 0)
        yaw = float(telem_row.get("yaw", 0) or 0)
        self.update_odometry(dt=dt, rc_lr=lr, rc_fb=fb, rc_yaw=yaw)

    def extract_polar_rays(self, norm_depth, win_cfg=None, num_rays=64):
        """Extract radial distance profile from DA2 depth map across horizontal corridor window."""
        dh, dw = norm_depth.shape[:2]

        if win_cfg:
            y1 = int(dh * (win_cfg.get("y_min_pct", 18) / 100.0))
            y2 = int(dh * (win_cfg.get("y_max_pct", 54) / 100.0))
        else:
            y1, y2 = int(dh * 0.18), int(dh * 0.54)

        y1 = max(0, min(dh - 5, y1))
        y2 = max(y1 + 4, min(dh, y2))

        # Horizontal slice excluding floor and ceiling
        depth_slice = norm_depth[y1:y2, :]
        # Robust 30th percentile disparity (prioritizes closest obstacles in the slice)
        col_disp = np.percentile(depth_slice, 60, axis=0)

        rays = []
        step = max(1, dw // num_rays)
        for i in range(0, dw, step):
            d_val = float(col_disp[i])
            # Higher disparity = closer object. Map [0..255] to metric distance:
            clamped_d = np.clip(d_val, 15.0, 245.0)
            # Calibration curve: 245 -> 0.35m, 15 -> 4.5m
            dist_m = 0.35 + 4.15 * ((245.0 - clamped_d) / 230.0) ** 1.35
            dist_m = min(self.max_ray_dist_m, dist_m)

            rel_angle = (i / float(dw) - 0.5) * self.fov_rad  # negative = left, positive = right
            rays.append((rel_angle, dist_m))

        return rays

    def update_map(self, rays):
        """Project depth rays into the global 2D grid and update occupancy cells."""
        drone_gx, drone_gy = self.world_to_grid(self.drone_x, self.drone_y)
        cos_yaw = math.cos(self.drone_yaw)
        sin_yaw = math.sin(self.drone_yaw)

        for rel_angle, dist_m in rays:
            # Global ray angle
            global_angle = self.drone_yaw + rel_angle
            gx_hit_m = self.drone_x + dist_m * math.sin(global_angle)
            gy_hit_m = self.drone_y + dist_m * math.cos(global_angle)

            hit_gx, hit_gy = self.world_to_grid(gx_hit_m, gy_hit_m)

            # Bresenham line raycasting to mark free cells
            free_cells = self._bresenham_line(drone_gx, drone_gy, hit_gx, hit_gy)
            for cgx, cgy in free_cells[:-1]:
                if 0 <= cgx < self.grid_size and 0 <= cgy < self.grid_size:
                    if self.free_counts[cgy, cgx] < 200:
                        self.free_counts[cgy, cgx] += 1

            # Mark obstacle cell at hit location
            if 0 <= hit_gx < self.grid_size and 0 <= hit_gy < self.grid_size:
                if dist_m < (self.max_ray_dist_m - 0.15):
                    if self.occ_counts[hit_gy, hit_gx] < 250:
                        self.occ_counts[hit_gy, hit_gx] += 3

    def _bresenham_line(self, x0, y0, x1, y1):
        """Fast Bresenham line algorithm returning list of (x, y) cells along the ray."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cur_x, cur_y = x0, y0
        while True:
            cells.append((cur_x, cur_y))
            if cur_x == x1 and cur_y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cur_x += sx
            if e2 < dx:
                err += dx
                cur_y += sy
        return cells

    def find_deepest_frontier_corridor(self, rays):
        """Find the direction with the maximum distance clearance, safely passable for the drone.

        Returns:
            best_rel_angle (rad): Heading angle relative to current drone direction
            best_dist_m (float): Distance clearance in that direction
            target_local (x, y): Metric waypoint in drone frame (x=right, y=forward)
            target_global (x, y): Metric waypoint in world map frame
        """
        if not rays:
            return 0.0, 1.0, (0.0, 1.0), (self.drone_x, self.drone_y + 1.0)

        # 1. Clearance smoothing: evaluate each ray taking into account neighboring rays
        # (drone needs width clearance ~ 0.40m)
        num_rays = len(rays)
        safe_distances = []

        for i in range(num_rays):
            angle_i, dist_i = rays[i]
            # Check angular window corresponding to drone width at distance dist_i
            # angular span = 2 * atan(half_width / dist_i)
            span_rad = 2.0 * math.atan((self.drone_radius_m * 1.1) / max(0.4, dist_i))
            # Find indices within this angular window
            window_dists = []
            for j in range(num_rays):
                angle_j, dist_j = rays[j]
                if abs(angle_j - angle_i) <= (span_rad / 2.0):
                    window_dists.append(dist_j)

            # Safe distance for this heading is the MINIMUM distance in the corridor window
            corridor_bottleneck = min(window_dists) if window_dists else dist_i
            safe_distances.append(corridor_bottleneck)

        safe_distances = np.array(safe_distances)

        # 2. Select heading with maximum clearance
        # Prefer headings closer to straight ahead if distance difference is small (<15%)
        center_bias = 1.0 - 0.15 * (np.abs([r[0] for r in rays]) / (self.fov_rad / 2.0))
        scores = safe_distances * center_bias

        best_idx = int(np.argmax(scores))
        best_angle = rays[best_idx][0]
        best_dist = float(safe_distances[best_idx])

        # Waypoint placed 65% of the distance along the corridor
        nav_dist = max(0.5, min(best_dist * 0.70, 3.0))
        local_x = nav_dist * math.sin(best_angle)
        local_y = nav_dist * math.cos(best_angle)

        # Transform to global frame
        global_angle = self.drone_yaw + best_angle
        global_x = self.drone_x + nav_dist * math.sin(global_angle)
        global_y = self.drone_y + nav_dist * math.cos(global_angle)

        self.deepest_target_angle = best_angle
        self.deepest_target_dist = best_dist
        self.deepest_target_local = (local_x, local_y)
        self.deepest_target_global = (global_x, global_y)

        return best_angle, best_dist, (local_x, local_y), (global_x, global_y)

    def render_map_hud(self, hud_width=320, hud_height=320):
        """Render a cybernetic high-contrast HUD map of the room, trajectory, and exploration goal."""
        # Occupancy probabilities
        occ_prob = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        # Background: unknown space = dark slate
        canvas = np.full((self.grid_size, self.grid_size, 3), (22, 28, 36), dtype=np.uint8)

        # Free space: cells seen with free_counts > occ_counts
        is_free = (self.free_counts > 1) & (self.free_counts > self.occ_counts)
        canvas[is_free] = (195, 205, 215)  # Light grey-blue for open navigable space

        # Obstacle space: cells with confirmed obstacle hits
        is_occ = self.occ_counts >= 3
        # Dilate obstacles slightly for visibility
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated_occ = cv2.dilate(is_occ.astype(np.uint8), kernel) > 0
        canvas[dilated_occ] = (30, 40, 240)  # Bright Coral-Red for walls / table obstacles

        # Resize canvas to target HUD dimensions
        hud = cv2.resize(canvas, (hud_width, hud_height), interpolation=cv2.INTER_NEAREST)

        scale_x = hud_width / float(self.grid_size)
        scale_y = hud_height / float(self.grid_size)

        def w_to_hud(wx, wy):
            gx, gy = self.world_to_grid(wx, wy)
            hx = int(round(gx * scale_x))
            hy = int(round(gy * scale_y))
            return hx, hy

        # 1. Draw Traveled Flight Path (cyan glowing ribbon)
        if len(self.trajectory) > 1:
            traj_pts = [w_to_hud(x, y) for x, y in self.trajectory]
            for i in range(len(traj_pts) - 1):
                cv2.line(hud, traj_pts[i], traj_pts[i + 1], (255, 180, 0), 2)  # Cyan
            for pt in traj_pts[::3]:
                cv2.circle(hud, pt, 2, (255, 220, 100), -1)

        # 2. Draw Drone Position & Heading Cone (FOV)
        dhx, dhy = w_to_hud(self.drone_x, self.drone_y)

        # FOV ray endpoints
        cone_len = 1.8  # meters
        fov_half = self.fov_rad / 2.0
        left_angle = self.drone_yaw - fov_half
        right_angle = self.drone_yaw + fov_half

        lx = self.drone_x + cone_len * math.sin(left_angle)
        ly = self.drone_y + cone_len * math.cos(left_angle)
        rx = self.drone_x + cone_len * math.sin(right_angle)
        ry = self.drone_y + cone_len * math.cos(right_angle)

        lhx, lhy = w_to_hud(lx, ly)
        rhx, rhy = w_to_hud(rx, ry)

        # FOV semi-transparent triangle
        fov_poly = np.array([(dhx, dhy), (lhx, lhy), (rhx, rhy)], dtype=np.int32)
        fov_overlay = hud.copy()
        cv2.fillPoly(fov_overlay, [fov_poly], (80, 140, 60))
        cv2.addWeighted(fov_overlay, 0.40, hud, 0.60, 0, hud)
        cv2.line(hud, (dhx, dhy), (lhx, lhy), (120, 220, 100), 1)
        cv2.line(hud, (dhx, dhy), (rhx, rhy), (120, 220, 100), 1)

        # 3. Draw Deepest Frontier Target Waypoint (Emerald Crosshair & Vector)
        tx, ty = self.deepest_target_global
        thx, thy = w_to_hud(tx, ty)

        # Guidance vector from drone to frontier target
        cv2.line(hud, (dhx, dhy), (thx, thy), (0, 255, 120), 2)
        # Target Crosshair
        cv2.circle(hud, (thx, thy), 7, (0, 255, 120), 2)
        cv2.circle(hud, (thx, thy), 2, (0, 255, 255), -1)
        cv2.line(hud, (thx - 10, thy), (thx + 10, thy), (0, 255, 120), 1)
        cv2.line(hud, (thx, thy - 10), (thx, thy + 10), (0, 255, 120), 1)

        # 4. Drone Body Marker
        cv2.circle(hud, (dhx, dhy), 6, (0, 240, 255), -1)  # Yellow center
        cv2.circle(hud, (dhx, dhy), 7, (0, 0, 0), 1)
        # Drone heading tip
        tip_x = int(dhx + 10 * math.sin(self.drone_yaw))
        tip_y = int(dhy - 10 * math.cos(self.drone_yaw))
        cv2.line(hud, (dhx, dhy), (tip_x, tip_y), (0, 0, 0), 3)
        cv2.line(hud, (dhx, dhy), (tip_x, tip_y), (255, 255, 255), 2)

        # 5. Map Header & Telemetry
        cv2.rectangle(hud, (0, 0), (hud_width, 36), (14, 18, 24), -1)
        cv2.putText(hud, "ROOM MAP: DA2 OCCUPANCY & FRONTIER", (8, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)

        deg_angle = math.degrees(self.deepest_target_angle)
        cv2.putText(
            hud,
            f"MAX DEPTH: {self.deepest_target_dist:.1f}m | TGT: {deg_angle:+.0f} deg",
            (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 120), 1
        )

        # Map Grid Scale Footer
        cv2.putText(hud, f"Grid: {self.map_size_m:.0f}x{self.map_size_m:.0f}m", (hud_width - 80, hud_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (140, 150, 160), 1)

        # Outer border
        cv2.rectangle(hud, (0, 0), (hud_width - 1, hud_height - 1), (60, 75, 90), 1)
        return hud
