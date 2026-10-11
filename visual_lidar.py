"""Visual LiDAR & Obstacle Avoidance Collision Shield for DJI Tello.

Uses Depth Anything V2 (DA2) as a high-speed Monocular Visual LiDAR:
- Extracts radial metric distance rays in the forward eye-level flight window (ignoring floor and ceiling).
- Multi-sector distance monitoring: Left, Center flight corridor, Right.
- Real-time Forward Collision Shield:
    * 🟢 SAFE / CLEAR (> 1.45m): Free flight allowed forward.
    * 🟡 CAUTION (0.85m - 1.45m): Approaching obstacle (table, wall, chair); speed clamped, steer to open gap.
    * 🔴 EMERGENCY BRAKE (< 0.85m): Immediate stop / reverse command to prevent crash.
- Dynamic escape gap recommendation: identifies the widest open passage.
- Aerospace-style Polar LiDAR Radar HUD with range rings, obstacle contours, collision tunnel, and escape vectors.
"""

import argparse
import csv
import math
from pathlib import Path
import time
import cv2
import numpy as np

from da2_depth import DA2DepthEstimator
from flight_window_config import load_flight_window_config, get_pixel_bounds


class VisualLiDARScanner:
    """Monocular Visual LiDAR scanner using dense neural depth geometry."""

    def __init__(
        self,
        fov_deg=82.6,
        num_rays=90,
        brake_dist_m=0.85,
        caution_dist_m=1.45,
        depth_scale_factor=3.8,
        corridor_half_width_m=0.25,
    ):
        self.fov_deg = fov_deg
        self.fov_rad = math.radians(fov_deg)
        self.half_fov_rad = self.fov_rad / 2.0
        self.num_rays = num_rays
        self.brake_dist_m = brake_dist_m
        self.caution_dist_m = caution_dist_m
        self.depth_scale_factor = depth_scale_factor
        self.corridor_half_width_m = corridor_half_width_m

        # Temporal filter for ray distances to eliminate high-frequency jitter
        self.smoothed_dists = None
        self.alpha_smooth = 0.70

    def reset(self):
        """Reset temporal smoothing."""
        self.smoothed_dists = None

    def scan(self, norm_depth_uint8, window_cfg=None, forward_speed_mps=0.0):
        """Perform radial LiDAR sweep across horizontal FOV in the eye-level flight window.

        Args:
            norm_depth_uint8: uint8 disparity map [0..255] from DA2 (255=closest, 0=farthest)
            window_cfg: flight window calibration dict (cuts off floor and ceiling)
            forward_speed_mps: drone forward speed in m/s (for Time-To-Collision)

        Returns:
            dict containing rays, sectors, closest obstacle, safety state, and recommended action.
        """
        dh, dw = norm_depth_uint8.shape[:2]
        cfg = window_cfg or load_flight_window_config()

        # 1. Vertical slice: eye-level corridor flight window (strictly cuts floor plane)
        y1 = int(dh * (cfg.get("y_min_pct", 20) / 100.0))
        y2 = int(dh * (cfg.get("y_max_pct", 52) / 100.0))
        y1 = max(0, min(dh - 4, y1))
        y2 = max(y1 + 4, min(dh, y2))

        depth_slice = norm_depth_uint8[y1:y2, :]
        # Use 70th percentile across column: conservative obstacle detection (sees table edges & chair backs)
        col_disp = np.percentile(depth_slice, 70, axis=0)

        # Smooth column disparity to filter single-pixel camera sensor noise
        ksize = max(5, int(dw * 0.04))
        if ksize % 2 == 0:
            ksize += 1
        col_disp_smooth = cv2.GaussianBlur(col_disp.astype(np.float32), (ksize, 1), 0).ravel()

        # 2. Sample radial rays across the FOV
        step_col = max(1, dw // self.num_rays)
        sample_cols = np.arange(0, dw, step_col)
        raw_dists = []
        ray_angles_rad = []

        for col_idx in sample_cols:
            disp_val = float(col_disp_smooth[col_idx])
            disp_clamped = np.clip(disp_val, 12.0, 248.0)
            # Metric curve: [0.40m .. 4.5m]
            dist_m = 0.40 + self.depth_scale_factor * ((248.0 - disp_clamped) / 236.0) ** 1.32
            raw_dists.append(dist_m)

            # Ray angle relative to camera optical axis: [-fov/2 .. +fov/2]
            ang = (col_idx / float(dw) - 0.5) * self.fov_rad
            ray_angles_rad.append(ang)

        raw_dists = np.array(raw_dists, dtype=np.float32)

        # Temporal smoothing
        if self.smoothed_dists is None or len(self.smoothed_dists) != len(raw_dists):
            self.smoothed_dists = raw_dists.copy()
        else:
            self.smoothed_dists = self.alpha_smooth * raw_dists + (1.0 - self.alpha_smooth) * self.smoothed_dists

        # 3. Analyze Rays & Sectors
        rays = []
        left_dists = []
        center_dists = []
        right_dists = []
        corridor_danger_points = []

        # Center flight cone: [-14 deg .. +14 deg]
        center_cone_rad = math.radians(14.0)

        for i, (ang, d_m) in enumerate(zip(ray_angles_rad, self.smoothed_dists)):
            deg = math.degrees(ang)
            # Drone body coordinates (X = right, Z = forward)
            xb = float(d_m * math.sin(ang))
            zb = float(d_m * math.cos(ang))

            # Severity classification for this individual ray
            if d_m < self.brake_dist_m:
                sev = "danger"
            elif d_m < self.caution_dist_m:
                sev = "caution"
            else:
                sev = "safe"

            # Check if this ray is inside the drone's forward physical collision corridor
            in_corridor = (abs(xb) <= self.corridor_half_width_m) and (zb < self.caution_dist_m)
            if in_corridor and zb < self.brake_dist_m:
                corridor_danger_points.append((xb, zb, d_m))

            rays.append({
                "index": i,
                "angle_rad": ang,
                "angle_deg": deg,
                "dist_m": float(d_m),
                "x_m": xb,
                "z_m": zb,
                "severity": sev,
                "in_corridor": in_corridor,
            })

            if ang < -center_cone_rad:
                left_dists.append(float(d_m))
            elif ang > center_cone_rad:
                right_dists.append(float(d_m))
            else:
                center_dists.append(float(d_m))

        # Sector metrics
        def get_sector_stats(arr):
            if not arr:
                return {"min_m": 5.0, "avg_m": 5.0, "severity": "safe"}
            min_v = float(np.min(arr))
            avg_v = float(np.mean(arr))
            if min_v < self.brake_dist_m:
                s = "danger"
            elif min_v < self.caution_dist_m:
                s = "caution"
            else:
                s = "safe"
            return {"min_m": min_v, "avg_m": avg_v, "severity": s}

        sec_left = get_sector_stats(left_dists)
        sec_center = get_sector_stats(center_dists)
        sec_right = get_sector_stats(right_dists)

        # Global closest obstacle
        all_dists = [r["dist_m"] for r in rays]
        min_idx = int(np.argmin(all_dists))
        closest_ray = rays[min_idx]

        # 4. Collision Shield Safety Decision
        c_min = sec_center["min_m"]
        has_corridor_intrusion = len(corridor_danger_points) > 0

        if c_min < self.brake_dist_m or has_corridor_intrusion:
            shield_state = "EMERGENCY_BRAKE"
            recommended_action = "HARD_BRAKE_STOP"
        elif c_min < self.caution_dist_m or closest_ray["dist_m"] < 0.65:
            shield_state = "CAUTION_SLOW"
            recommended_action = "SLOW_AND_STEER"
        else:
            shield_state = "CLEAR_FORWARD"
            recommended_action = "FLY_FORWARD"

        # 5. Dynamic Escape Gap Recommendation
        # Find the clearest angle (widest opening with maximum clearance)
        best_heading_deg = 0.0
        if shield_state in ("EMERGENCY_BRAKE", "CAUTION_SLOW"):
            # Compare left vs right clearance
            if sec_left["avg_m"] > sec_right["avg_m"] + 0.25:
                best_heading_deg = -28.0  # Steer Left
            elif sec_right["avg_m"] > sec_left["avg_m"] + 0.25:
                best_heading_deg = +28.0  # Steer Right
            else:
                # Find maximum open ray distance outside center
                outer_rays = [r for r in rays if abs(r["angle_deg"]) > 10.0]
                if outer_rays:
                    best_ray = max(outer_rays, key=lambda r: r["dist_m"])
                    best_heading_deg = best_ray["angle_deg"]
        else:
            # If clear, follow the deepest central valley
            best_ray = max(rays, key=lambda r: r["dist_m"])
            best_heading_deg = np.clip(best_ray["angle_deg"], -20.0, 20.0)

        # Time-To-Collision (TTC)
        ttc_sec = None
        if forward_speed_mps > 0.05:
            ttc_sec = c_min / forward_speed_mps

        return {
            "rays": rays,
            "closest": closest_ray,
            "sectors": {
                "left": sec_left,
                "center": sec_center,
                "right": sec_right,
            },
            "shield_state": shield_state,
            "recommended_action": recommended_action,
            "recommended_heading_deg": float(best_heading_deg),
            "ttc_sec": ttc_sec,
            "window_bounds": (y1, y2),
        }


def render_visual_lidar_hud(frame_bgr, norm_depth_uint8, scan_res, frame_idx=0, total_frames=0, telem_row=None, infer_ms=16.0):
    """Render combined dual-panel HUD:
    - Left panel: Camera View with Flight Window, collision alerts, and distance tags.
    - Right panel: High-Precision Polar Visual LiDAR Radar with range rings, rays, and escape vector.
    """
    fh, fw = frame_bgr.shape[:2]
    vis_cam = frame_bgr.copy()

    shield_state = scan_res["shield_state"]
    sec = scan_res["sectors"]
    closest = scan_res["closest"]
    c_min = sec["center"]["min_m"]
    l_min = sec["left"]["min_m"]
    r_min = sec["right"]["min_m"]
    esc_deg = scan_res["recommended_heading_deg"]
    act = scan_res["recommended_action"]
    ttc = scan_res.get("ttc_sec")

    # State colors
    COLORS = {
        "CLEAR_FORWARD": (0, 255, 100),    # Vibrant Green
        "CAUTION_SLOW": (0, 220, 255),     # Vibrant Yellow
        "EMERGENCY_BRAKE": (0, 40, 255),   # Bright Red
    }
    state_color = COLORS.get(shield_state, (200, 200, 200))

    # Center box color specifically based on forward center corridor clearance
    if c_min < 0.85:
        center_color = (0, 40, 255)
    elif c_min < 1.45:
        center_color = (0, 220, 255)
    else:
        center_color = (0, 255, 100)

    # =========================================================================
    # LEFT PANEL: CAMERA VIEW & COLLISION SHIELD OVERLAY
    # =========================================================================
    # Top banner
    cv2.rectangle(vis_cam, (0, 0), (fw, 64), (16, 20, 26), -1)
    cv2.putText(vis_cam, f"VISUAL LIDAR & OBSTACLE SHIELD [Frame {frame_idx}/{total_frames}]", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    dist_str = f"Center: {c_min:.2f}m | Left: {l_min:.2f}m | Right: {r_min:.2f}m"
    cv2.putText(vis_cam, dist_str, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, state_color, 2)

    # Status badge at top right of camera
    cv2.rectangle(vis_cam, (fw - 250, 8), (fw - 10, 56), (28, 34, 44), -1)
    cv2.putText(vis_cam, f"SHIELD: {shield_state}", (fw - 240, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, state_color, 1)
    cv2.putText(vis_cam, f"Action: {act}", (fw - 240, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    # Forward Flight Window (eye-level obstacle corridor)
    cfg = load_flight_window_config()
    wx1, wy1, wx2, wy2 = get_pixel_bounds(cfg, fw, fh)

    # Box outline colored by forward corridor safety
    box_thick = 3 if c_min < 0.85 else 2
    cv2.rectangle(vis_cam, (wx1, wy1), (wx2, wy2), center_color, box_thick)
    # Corner brackets for HUD aesthetic
    corner_len = 18
    for (cx, cy, dx, dy) in [(wx1, wy1, 1, 1), (wx2, wy1, -1, 1), (wx1, wy2, 1, -1), (wx2, wy2, -1, -1)]:
        cv2.line(vis_cam, (cx, cy), (cx + dx * corner_len, cy), (255, 255, 255), 2)
        cv2.line(vis_cam, (cx, cy), (cx, cy + dy * corner_len), (255, 255, 255), 2)

    # Flight Window Distance Label
    lbl = f"FORWARD: {c_min:.2f}m"
    cv2.rectangle(vis_cam, (wx1, wy1 - 22), (wx1 + 140, wy1), center_color, -1)
    cv2.putText(vis_cam, lbl, (wx1 + 6, wy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

    # Floor Cutoff dashed line
    cv2.line(vis_cam, (0, wy2), (fw, wy2), (80, 80, 110), 1)
    cv2.putText(vis_cam, "FLOOR CUTOFF (IGNORED)", (12, wy2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 170), 1)

    # Collision Warning Banner if Dangerous
    if shield_state == "EMERGENCY_BRAKE":
        cv2.rectangle(vis_cam, (0, fh - 65), (fw, fh), (0, 20, 200), -1)
        disp_dist = c_min if c_min < 1.45 else closest["dist_m"]
        cv2.putText(vis_cam, f"!!! COLLISION WARNING: BRAKE NOW ({disp_dist:.2f}m) !!!",
                    (fw // 2 - 240, fh - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        esc_str = f"Recommended: Turn {abs(esc_deg):.0f} deg {'LEFT' if esc_deg < 0 else 'RIGHT'}"
        cv2.putText(vis_cam, esc_str, (fw // 2 - 170, fh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    elif shield_state == "CAUTION_SLOW":
        cv2.rectangle(vis_cam, (0, fh - 50), (fw, fh), (0, 100, 200), -1)
        side_txt = "LEFT" if esc_deg < 0 else "RIGHT"
        cv2.putText(vis_cam, f"CAUTION: Obstacle at {closest['dist_m']:.2f}m ({closest['angle_deg']:+.0f} deg) -> Steer {abs(esc_deg):.0f} deg {side_txt}",
                    (20, fh - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    elif telem_row:
        t_state = telem_row.get("state", "N/A")[:24]
        t_cmd = f"FB:{telem_row.get('fb', 0)} Yaw:{telem_row.get('yaw', 0)}"
        cv2.rectangle(vis_cam, (0, fh - 36), (fw, fh), (16, 20, 26), -1)
        cv2.putText(vis_cam, f"Sensors: {t_state} | Cmd: {t_cmd} | Inf: {infer_ms:.1f}ms", (15, fh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # =========================================================================
    # RIGHT PANEL: HIGH-PRECISION POLAR VISUAL LIDAR RADAR
    # =========================================================================
    rw, rh = fw, fh
    vis_radar = np.zeros((rh, rw, 3), dtype=np.uint8)
    vis_radar[:] = (14, 18, 24)  # Deep Navy / Charcoal

    # Radar center (Drone position at bottom center)
    radar_cx = rw // 2
    radar_cy = int(rh * 0.88)
    # Pixels per meter scale (e.g. 100 pixels = 1.0m)
    r_scale = 105.0

    def r2p(rx_m, rz_m):
        """Drone body coordinates (rx=right, rz=forward) to Radar pixel (u, v)."""
        u = int(radar_cx + rx_m * r_scale)
        v = int(radar_cy - rz_m * r_scale)
        return (u, v)

    # 1. Concentric Range Rings (0.5m, 1.0m, 1.5m, 2.0m, 3.0m, 4.0m)
    rings = [
        (0.5, (30, 30, 180), "0.5m [BRAKE]"),
        (1.0, (30, 150, 190), "1.0m [CAUTION]"),
        (1.5, (40, 120, 60), "1.5m [CLEAR]"),
        (2.0, (40, 50, 65), "2.0m"),
        (3.0, (35, 45, 55), "3.0m"),
        (4.0, (28, 36, 46), "4.0m"),
    ]

    for dist_ring, ring_color, ring_label in rings:
        radius_px = int(dist_ring * r_scale)
        # Draw upper semi-circle arc (forward FOV)
        cv2.circle(vis_radar, (radar_cx, radar_cy), radius_px, ring_color, 1, cv2.LINE_AA)
        cv2.putText(vis_radar, ring_label, (radar_cx + radius_px - 45, radar_cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 140, 160), 1)

    # 2. Radial Spokes & FOV Boundaries
    fov_half = math.radians(41.3)
    c1_u = int(radar_cx + math.sin(-fov_half) * (4.2 * r_scale))
    c1_v = int(radar_cy - math.cos(-fov_half) * (4.2 * r_scale))
    c2_u = int(radar_cx + math.sin(fov_half) * (4.2 * r_scale))
    c2_v = int(radar_cy - math.cos(fov_half) * (4.2 * r_scale))

    cv2.line(vis_radar, (radar_cx, radar_cy), (c1_u, c1_v), (45, 60, 75), 1)
    cv2.line(vis_radar, (radar_cx, radar_cy), (c2_u, c2_v), (45, 60, 75), 1)
    # Center forward line (0 deg)
    cv2.line(vis_radar, (radar_cx, radar_cy), (radar_cx, int(radar_cy - 4.2 * r_scale)), (50, 70, 85), 1)

    # 3. Forward Drone Collision Tunnel (+- 0.25m physical width)
    tun_left_u, _ = r2p(-0.25, 0.0)
    tun_right_u, _ = r2p(0.25, 0.0)
    tun_top_v = int(radar_cy - 1.8 * r_scale)

    tun_color = (0, 30, 200) if shield_state == "EMERGENCY_BRAKE" else (40, 80, 50)
    cv2.line(vis_radar, (tun_left_u, radar_cy), (tun_left_u, tun_top_v), tun_color, 1)
    cv2.line(vis_radar, (tun_right_u, radar_cy), (tun_right_u, tun_top_v), tun_color, 1)
    cv2.putText(vis_radar, "COLLISION TUNNEL", (tun_left_u - 30, tun_top_v - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, tun_color, 1)

    # 4. Laser Rays Fan & Obstacle Contour
    rays = scan_res["rays"]
    contour_pts = []

    for idx, r in enumerate(rays):
        ru, rv = r2p(r["x_m"], r["z_m"])
        if 0 <= ru < rw and 0 <= rv < rh:
            contour_pts.append((ru, rv))

            # Ray lines (draw every 3rd ray for clean aesthetic)
            if idx % 3 == 0:
                if r["severity"] == "danger":
                    ray_col = (0, 40, 180)
                elif r["severity"] == "caution":
                    ray_col = (0, 120, 150)
                else:
                    ray_col = (30, 70, 45)
                cv2.line(vis_radar, (radar_cx, radar_cy), (ru, rv), ray_col, 1)

            # Hit point on obstacle surface
            if r["severity"] == "danger":
                pt_col = (0, 60, 255)
                pt_r = 3
            elif r["severity"] == "caution":
                pt_col = (0, 220, 255)
                pt_r = 2
            else:
                pt_col = (0, 255, 120)
                pt_r = 2
            cv2.circle(vis_radar, (ru, rv), pt_r, pt_col, -1)

    # Draw continuous obstacle contour line (table / wall / chair edge)
    if len(contour_pts) > 1:
        for i in range(len(contour_pts) - 1):
            cv2.line(vis_radar, contour_pts[i], contour_pts[i + 1], (0, 255, 230), 2)

    # 5. Highlight Closest Obstacle Reticle
    cru, crv = r2p(closest["x_m"], closest["z_m"])
    if 0 <= cru < rw and 0 <= crv < rh:
        cv2.circle(vis_radar, (cru, crv), 8, (0, 0, 255), 2)
        cv2.circle(vis_radar, (cru, crv), 12, (0, 150, 255), 1)
        cv2.putText(vis_radar, f"{closest['dist_m']:.2f}m", (cru + 12, crv - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 255), 1)

    # 6. Recommended Escape / Steering Vector Arrow
    esc_rad = math.radians(esc_deg)
    esc_len_m = 1.3
    eu, ev = r2p(esc_len_m * math.sin(esc_rad), esc_len_m * math.cos(esc_rad))
    cv2.arrowedLine(vis_radar, (radar_cx, radar_cy), (eu, ev), (0, 255, 100), 3, tipLength=0.25)
    cv2.putText(vis_radar, f"ESCAPE GAP ({esc_deg:+.0f} deg)", (eu - 40, ev - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 100), 1)

    # 7. Drone Body Icon at Radar Origin
    cv2.circle(vis_radar, (radar_cx, radar_cy), 8, (255, 255, 255), -1)
    cv2.circle(vis_radar, (radar_cx, radar_cy), 12, (0, 200, 255), 2)
    # Heading indicator
    cv2.line(vis_radar, (radar_cx, radar_cy), (radar_cx, radar_cy - 22), (0, 255, 255), 2)

    # 8. Top Status & Telemetry Header on Radar
    cv2.rectangle(vis_radar, (0, 0), (rw, 64), (16, 20, 26), -1)
    cv2.putText(vis_radar, "POLAR VISUAL LIDAR RADAR (BEV)", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    ttc_txt = f" | TTC: {ttc:.1f}s" if ttc is not None else ""
    summary_txt = f"Min Obstacle: {closest['dist_m']:.2f}m ({closest['angle_deg']:+.0f} deg){ttc_txt}"
    cv2.putText(vis_radar, summary_txt, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, state_color, 1)

    # Legend at top right of Radar
    leg_x = rw - 220
    cv2.circle(vis_radar, (leg_x, 16), 4, (0, 40, 255), -1)
    cv2.putText(vis_radar, "< 0.85m (Brake)", (leg_x + 12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.circle(vis_radar, (leg_x, 32), 4, (0, 220, 255), -1)
    cv2.putText(vis_radar, "0.85-1.45m (Caution)", (leg_x + 12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.circle(vis_radar, (leg_x, 48), 4, (0, 255, 100), -1)
    cv2.putText(vis_radar, "> 1.45m (Clear)", (leg_x + 12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Combine Left and Right panels
    combined = np.hstack([vis_cam, vis_radar])
    cv2.line(combined, (fw, 0), (fw, fh), (60, 70, 85), 2)
    return combined


def run_visual_lidar_stream(video_path, telem_path=None, show_gui=True, output_video=None, target_fps=None):
    """Run full Visual LiDAR processing loop on video."""
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    print("=" * 75)
    print("TELLO VISUAL LIDAR & OBSTACLE COLLISION SHIELD")
    print(f"Video Source: {video_path.name}")
    print(f"Telemetry:    {telem_path.name if telem_path and Path(telem_path).exists() else 'None'}")
    print("=" * 75)

    print("Initializing Depth Anything V2...")
    da2 = DA2DepthEstimator(input_size=252)
    scanner = VisualLiDARScanner()

    # Load telemetry if available
    telemetry = {}
    if telem_path and Path(telem_path).exists():
        with open(telem_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    telemetry[int(row.get("frame_idx", 0))] = row
                except ValueError:
                    pass

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = target_fps or native_fps

    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_writer = None
    if output_video:
        out_p = Path(output_video).resolve()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_p), fourcc, fps, (vw * 2, vh))
        print(f"Exporting annotated Visual LiDAR video to: {out_p}")

    win_name = "Tello Vision: Visual LiDAR & Obstacle Shield"
    if show_gui:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 480)

    frame_idx = 0
    t_start = time.perf_counter()
    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # 1. Depth Estimation with DA2 (~15ms on CUDA)
                t0 = time.perf_counter()
                norm_depth, _, _, _, da2_ms = da2.predict(frame)

                # Get drone forward speed from telemetry if available
                telem_row = telemetry.get(frame_idx)
                fwd_speed = 0.0
                if telem_row:
                    fb_cmd = float(telem_row.get("fb", 0))
                    fwd_speed = max(0.0, fb_cmd * 0.0155)

                # 2. Radial Visual LiDAR Scan
                scan_res = scanner.scan(norm_depth, forward_speed_mps=fwd_speed)
                infer_ms = (time.perf_counter() - t0) * 1000

                # 3. Dual-Panel HUD
                hud = render_visual_lidar_hud(
                    frame, norm_depth, scan_res,
                    frame_idx=frame_idx, total_frames=total_frames,
                    telem_row=telem_row, infer_ms=infer_ms
                )

                if out_writer:
                    out_writer.write(hud)

                if frame_idx % 60 == 0 or frame_idx == total_frames:
                    c_dist = scan_res["sectors"]["center"]["min_m"]
                    sh_state = scan_res["shield_state"]
                    print(f"  Frame {frame_idx:3d}/{total_frames} | {infer_ms:.1f}ms | Center: {c_dist:.2f}m | Shield: {sh_state}")

            if show_gui:
                cv2.imshow(win_name, hud)
                delay = 1 if not paused else 30
                key = cv2.waitKey(delay) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                elif key == 32:  # Space to pause
                    paused = not paused
                elif key == ord('r'):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    scanner.reset()
    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        if show_gui:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print("=" * 75)
    print(f"Visual LiDAR complete! Processed {frame_idx} frames in {elapsed:.2f}s ({frame_idx / max(0.01, elapsed):.1f} FPS)")
    if output_video:
        print(f"Output saved to: {output_video}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Tello Visual LiDAR & Forward Obstacle Shield")
    parser.add_argument("flight", nargs="?", default="latest", help="Flight folder name in debug/ or full video path")
    parser.add_argument("--no-gui", action="store_true", help="Run headlessly without GUI window")
    parser.add_argument("--save-video", action="store_true", help="Save annotated mp4 video to flight directory")
    parser.add_argument("--fps", type=float, default=None, help="Playback speed FPS")
    args = parser.parse_args()

    base_debug = Path(__file__).resolve().parent / "debug"
    cand_path = Path(args.flight)

    if cand_path.is_file():
        video_path = cand_path
        telem_path = cand_path.parent / "flight_telemetry.csv"
        out_vid = cand_path.parent / "visual_lidar_output.mp4" if args.save_video else None
    else:
        if args.flight == "latest":
            subdirs = [p for p in base_debug.iterdir() if p.is_dir()]
            subdirs.sort(key=lambda p: p.name, reverse=True)
            flight_dir = subdirs[0] if subdirs else None
        else:
            flight_dir = (base_debug / args.flight).resolve() if not cand_path.exists() else cand_path.resolve()

        if not flight_dir or not flight_dir.exists():
            print(f"Error: Flight directory '{args.flight}' not found.")
            return

        video_path = flight_dir / "flight_raw.mp4"
        if not video_path.exists():
            video_path = flight_dir / "flight_airborne_clean.mp4"

        telem_path = flight_dir / "flight_telemetry.csv"
        if not telem_path.exists():
            telem_path = flight_dir / "commands.csv"

        out_vid = flight_dir / "visual_lidar_output.mp4" if args.save_video else None

    run_visual_lidar_stream(
        video_path=video_path,
        telem_path=telem_path,
        show_gui=not args.no_gui,
        output_video=out_vid,
        target_fps=args.fps,
    )


if __name__ == "__main__":
    main()
