"""DeepSLAM: Monocular Deep Visual SLAM & 3D Trajectory Reconstructor for DJI Tello.

Integrates:
1. Depth Anything V2 (DA2) dense depth estimation.
2. Lucas-Kanade optical flow feature tracking with sub-pixel precision.
3. 3D-to-2D Perspective-n-Point (PnP-RANSAC) 6-DoF camera pose estimation.
4. Metric scale recovery calibrated with Tello's altitude hold / ToF sensing.
5. Global 3D trajectory mapping and obstacle landmark point cloud reconstruction.
6. Interactive real-time visualizer with Global Top-Down Map and Camera feature overlay.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

from da2_depth import DA2DepthEstimator
from test_da2_flight_offline import find_latest_flight_dir, load_telemetry


def parse_sensor_reading(telem_row, prev_state=None, dt=0.033):
    """Extract and calibrate sensor measurements from Tello telemetry row.

    Combines:
    1. ToF distance sensor (vertical ground height)
    2. IMU Gyroscope (Yaw angle or integrated yaw rate)
    3. Speed odometry (optical flow vgx/vgy or calibrated RC fb/lr velocity)
    """
    if not telem_row:
        return {
            "tof_m": None,
            "imu_yaw": None,
            "yaw_rate": 0.0,
            "speed_forward": 0.0,
            "speed_lateral": 0.0,
            "speed_vertical": 0.0,
            "has_sensors": False,
        }

    # 1. ToF Distance Sensor (Ground distance in meters)
    tof_m = None
    if telem_row.get("tof_cm"):
        try:
            val = float(telem_row["tof_cm"])
            if 5.0 < val < 400.0:
                tof_m = val / 100.0
        except ValueError:
            pass
    if tof_m is None:
        state_str = str(telem_row.get("state", ""))
        m = re.search(r"\[ToF=(\d+)cm\]", state_str)
        if m:
            tof_m = float(m.group(1)) / 100.0

    # 2. IMU Gyroscope (Yaw in degrees)
    imu_yaw = None
    if telem_row.get("imu_yaw") is not None and telem_row.get("imu_yaw") != "":
        try:
            imu_yaw = float(telem_row["imu_yaw"])
        except ValueError:
            pass

    yaw_rate = 0.0
    try:
        yaw_cmd = float(telem_row.get("yaw", 0) or 0)
        # Tello yaw rate scaling: ~1.45 deg/sec per command unit
        yaw_rate = yaw_cmd * 1.45
    except ValueError:
        pass

    # 3. Horizontal and Vertical Velocity Odometry (m/s)
    speed_forward = 0.0
    speed_lateral = 0.0
    speed_vertical = 0.0

    if telem_row.get("speed_vz") is not None and telem_row.get("speed_vz") != "":
        try:
            # vgx, vgy, vgz are cm/s in Tello state packet
            speed_forward = float(telem_row.get("speed_vy", 0) or 0) / 100.0
            speed_lateral = float(telem_row.get("speed_vx", 0) or 0) / 100.0
            speed_vertical = float(telem_row.get("speed_vz", 0) or 0) / 100.0
        except ValueError:
            pass
    else:
        # Fallback to calibrated RC velocity commands
        try:
            fb = float(telem_row.get("fb", 0) or 0)
            lr = float(telem_row.get("lr", 0) or 0)
            ud = float(telem_row.get("ud", 0) or 0)
            # Tello velocity gain: cmd 18 ~ 0.28 m/s
            speed_forward = fb * 0.0155
            speed_lateral = lr * 0.0155
            speed_vertical = ud * 0.0155
        except ValueError:
            pass

    has_sensors = (tof_m is not None) or (imu_yaw is not None) or (abs(speed_forward) > 0.01) or (abs(yaw_rate) > 0.01)

    return {
        "tof_m": tof_m,
        "imu_yaw": imu_yaw,
        "yaw_rate": yaw_rate,
        "speed_forward": speed_forward,
        "speed_lateral": speed_lateral,
        "speed_vertical": speed_vertical,
        "has_sensors": has_sensors,
    }


class DeepSLAMTracker:
    """Monocular Deep Visual SLAM engine using Depth Anything V2 for metric 3D lifting."""

    def __init__(
        self,
        img_w=960,
        img_h=720,
        fov_deg=82.6,
        max_features=250,
        depth_scale_factor=3.5,
    ):
        self.img_w = img_w
        self.img_h = img_h
        self.fov_deg = fov_deg
        self.max_features = max_features
        self.depth_scale_factor = depth_scale_factor

        # Camera Intrinsics Matrix K
        # Tello Camera: ~82.6 deg horizontal FOV
        fx = (img_w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        fy = fx
        cx = img_w / 2.0
        cy = img_h / 2.0
        self.K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # SLAM State
        self.prev_gray = None
        self.prev_pts = None
        self.prev_depth = None

        # Global Camera Pose (4x4 Transformation Matrix in World Frame)
        # World coordinate frame: +Z forward, +X right, +Y down (or standard navigation)
        self.cur_pose_w = np.eye(4, dtype=np.float64)

        # Trajectory History: list of (x, y, z, yaw_deg, pitch_deg, roll_deg, timestamp)
        self.trajectory = []

        # 3D Voxel Grid Filter for Sharp Walls: {(vx, vy, vz): [count, sum_x, sum_y, sum_z]}
        self.voxel_size = 0.06  # 6 cm voxel resolution
        self.voxel_map = {}

        # Optical Flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        self.total_distance = 0.0
        self.tracking_inliers = 0

        # Sensor Fusion state (IMU Gyroscope + ToF Laser + Odometry/Velocity)
        self.initial_tof_m = None
        self.integrated_gyro_yaw = 0.0
        self.prev_timestamp = None
        self.fused_x = 0.0
        self.fused_y = 0.0
        self.fused_z = 0.0
        self.fused_yaw = 0.0
        self.sensor_status = "INITIALIZING"
        self.live_fov_scan = []

    @property
    def landmarks(self):
        """Return clean 3D landmark points averaged within each voxel (filters out 1-frame visual noise)."""
        pts = []
        for v in self.voxel_map.values():
            cnt = v[0]
            if cnt >= 2:
                pts.append((v[1] / cnt, v[2] / cnt, v[3] / cnt))
            elif len(self.voxel_map) < 350:
                pts.append((v[1] / cnt, v[2] / cnt, v[3] / cnt))
        return pts

    def reset(self):
        """Reset SLAM tracker to initial origin (0, 0, 0)."""
        self.prev_gray = None
        self.prev_pts = None
        self.prev_depth = None
        self.cur_pose_w = np.eye(4, dtype=np.float64)
        self.trajectory.clear()
        self.voxel_map.clear()
        self.live_fov_scan.clear()
        self.total_distance = 0.0
        self.tracking_inliers = 0
        self.initial_tof_m = None
        self.integrated_gyro_yaw = 0.0
        self.prev_timestamp = None
        self.fused_x = 0.0
        self.fused_y = 0.0
        self.fused_z = 0.0
        self.fused_yaw = 0.0
        self.sensor_status = "INITIALIZING"

    def detect_bucketed_features(self, gray, existing_pts=None, grid_cols=4, grid_rows=3):
        """Detect features uniformly across a 4x3 grid with sub-pixel refinement.

        Guarantees spatial coverage across the entire FOV (left wall, right wall, ceiling, floor),
        maximizing the triangulation baseline for PnP-RANSAC.
        """
        h, w = gray.shape[:2]
        cell_w = w // grid_cols
        cell_h = h // grid_rows
        pts_per_cell = max(10, self.max_features // (grid_cols * grid_rows))

        mask = np.full((h, w), 255, dtype=np.uint8)
        if existing_pts is not None and len(existing_pts) > 0:
            for pt in existing_pts:
                px, py = int(round(pt[0])), int(round(pt[1]))
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(mask, (px, py), 12, 0, -1)

        all_points = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                y0 = r * cell_h
                y1 = (r + 1) * cell_h if r < grid_rows - 1 else h
                x0 = c * cell_w
                x1 = (c + 1) * cell_w if c < grid_cols - 1 else w

                cell_gray = gray[y0:y1, x0:x1]
                cell_mask = mask[y0:y1, x0:x1]

                pts = cv2.goodFeaturesToTrack(
                    cell_gray,
                    maxCorners=pts_per_cell,
                    qualityLevel=0.015,
                    minDistance=10,
                    blockSize=7,
                    mask=cell_mask
                )
                if pts is not None and len(pts) > 0:
                    pts[:, 0, 0] += x0
                    pts[:, 0, 1] += y0
                    all_points.append(pts)

        if not all_points:
            return np.empty((0, 1, 2), dtype=np.float32)

        combined = np.vstack(all_points)

        # Sub-pixel corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.02)
        try:
            refined = cv2.cornerSubPix(gray, combined, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)
            return refined
        except cv2.error:
            return combined

    def add_landmarks(self, local_pts, pose_world, inlier_indices):
        """Add 3D points to voxel grid filter to produce crisp, clean wall planes without clutter."""
        for in_idx in inlier_indices[::2]:
            p_local = np.array([local_pts[in_idx][0], local_pts[in_idx][1], local_pts[in_idx][2], 1.0])
            p_world = pose_world @ p_local
            wx, wy, wz = float(p_world[0]), float(p_world[1]), float(p_world[2])

            vx = int(math.floor(wx / self.voxel_size))
            vy = int(math.floor(wy / self.voxel_size))
            vz = int(math.floor(wz / self.voxel_size))
            key = (vx, vy, vz)

            if key in self.voxel_map:
                v = self.voxel_map[key]
                v[0] += 1
                v[1] += wx
                v[2] += wy
                v[3] += wz
            else:
                if len(self.voxel_map) < 5000:
                    self.voxel_map[key] = [1, wx, wy, wz]

    def project_in_frame_scan(self, norm_depth_uint8, cur_x, cur_z, cur_yaw_deg, num_rays=85):
        """Project dense in-frame Depth Anything V2 rays within the current camera FOV.

        Only scans obstacles currently in the active camera frame (eye-level flight window).
        """
        dh, dw = norm_depth_uint8.shape[:2]
        # Eye-level corridor flight window: vertical slice 25% to 45% (above floor)
        y_slice_top = int(dh * 0.25)
        y_slice_bot = int(dh * 0.45)
        depth_slice = norm_depth_uint8[y_slice_top:y_slice_bot, :]
        col_disp = np.median(depth_slice, axis=0)

        step_col = max(1, dw // num_rays)
        sample_cols = np.arange(0, dw, step_col)

        yaw_rad = math.radians(cur_yaw_deg)
        fx = self.K[0, 0]
        cx = self.K[0, 2]

        scan_points = []
        for col_idx in sample_cols:
            disp_val = float(col_disp[col_idx])
            disp_clamped = np.clip(disp_val, 10.0, 250.0)
            dist_m = 0.40 + self.depth_scale_factor * ((250.0 - disp_clamped) / 240.0) ** 1.35

            if 0.35 <= dist_m <= 5.5:
                img_u = col_idx * (float(self.img_w) / dw)
                ray_angle_cam = math.atan2(img_u - cx, fx)
                world_ray_angle = yaw_rad + ray_angle_cam

                wx = cur_x + dist_m * math.sin(world_ray_angle)
                wz = cur_z + dist_m * math.cos(world_ray_angle)

                scan_points.append((wx, wz, dist_m))

        self.live_fov_scan = scan_points
        return scan_points

    def disparity_to_metric_depth(self, norm_depth_uint8):
        """Convert DA2 disparity map (0..255; higher = closer) to metric depth (meters)."""
        disp = norm_depth_uint8.astype(np.float32)
        disp = np.clip(disp, 10.0, 250.0)
        # Empirical metric curve for indoor corridor spaces (0.4m to 5.0m)
        depth_m = 0.40 + self.depth_scale_factor * ((250.0 - disp) / 240.0) ** 1.35
        return depth_m

    def track_frame(self, frame_bgr, norm_depth_uint8, timestamp=0.0, sensor_reading=None):
        """Process one video frame, track features, and update 6-DoF camera pose with sensor fusion.

        Fuses:
        - DA2 dense metric depth lifting
        - Lucas-Kanade optical flow feature points
        - EPnP-RANSAC camera transform
        - IMU Gyroscope (drift-free yaw anchor)
        - ToF Laser distance (vertical ground height)
        - Velocity/command odometry (motion prior and outlier gating)
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        metric_depth = self.disparity_to_metric_depth(norm_depth_uint8)

        dt = 0.033
        if self.prev_timestamp is not None and timestamp > self.prev_timestamp:
            dt = min(0.20, max(0.01, timestamp - self.prev_timestamp))
        self.prev_timestamp = timestamp

        sensor = sensor_reading or {}
        tof_m = sensor.get("tof_m")
        imu_yaw = sensor.get("imu_yaw")
        yaw_rate = sensor.get("yaw_rate", 0.0)
        speed_fwd = sensor.get("speed_forward", 0.0)
        speed_lat = sensor.get("speed_lateral", 0.0)
        has_sensors = sensor.get("has_sensors", False)

        # 1. Update Gyro Heading State
        if imu_yaw is not None:
            self.integrated_gyro_yaw = imu_yaw
        else:
            self.integrated_gyro_yaw += yaw_rate * dt

        # 2. Update ToF Ground Reference
        if self.initial_tof_m is None and tof_m is not None:
            self.initial_tof_m = tof_m

        # Sensor displacement prior in body coordinates (+Z forward, +X right)
        d_z_sens = speed_fwd * dt
        d_x_sens = speed_lat * dt

        tracked_prev = []
        tracked_curr = []
        inliers_count = 0

        # First frame initialization
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_depth = metric_depth
            pts = self.detect_bucketed_features(gray)
            self.prev_pts = pts if pts is not None else np.empty((0, 1, 2), dtype=np.float32)
            self.fused_x = 0.0
            self.fused_y = 0.0
            self.fused_z = 0.0
            self.fused_yaw = self.integrated_gyro_yaw
            self.trajectory.append((0.0, 0.0, 0.0, self.fused_yaw, 0.0, 0.0, timestamp))
            live_scan = self.project_in_frame_scan(norm_depth_uint8, self.fused_x, self.fused_z, self.fused_yaw)
            return {
                "pose": (0.0, 0.0, 0.0, self.fused_yaw, 0.0, 0.0),
                "tracked_pts_prev": [],
                "tracked_pts_curr": [],
                "inliers_count": len(self.prev_pts),
                "total_distance": 0.0,
                "sensor_status": "TAKEOFF_ORIGIN",
                "tof_m": tof_m,
                "gyro_yaw": self.integrated_gyro_yaw,
                "live_scan": live_scan,
            }

        # 1. Lucas-Kanade Optical Flow from prev_gray to gray
        pnp_success = False
        d_x_vis = 0.0
        d_y_vis = 0.0
        d_z_vis = 0.0
        vis_dyaw = 0.0
        good_curr = np.empty((0, 2), dtype=np.float32)
        pts_3d = []
        inliers = None

        if self.prev_pts is not None and len(self.prev_pts) >= 6:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, **self.lk_params
            )

            # Filter valid tracked points
            good_prev = self.prev_pts[st == 1]
            good_curr = p1[st == 1]

            # 2. Lift valid points in prev frame to 3D using Depth Anything V2
            dh, dw = metric_depth.shape[:2]
            scale_x = dw / float(self.img_w)
            scale_y = dh / float(self.img_h)

            pts_3d = []
            pts_2d = []
            valid_prev_pts = []
            valid_curr_pts = []

            fx = self.K[0, 0]
            fy = self.K[1, 1]
            cx = self.K[0, 2]
            cy = self.K[1, 2]

            for pt_prev, pt_curr in zip(good_prev, good_curr):
                u, v = pt_prev.ravel()
                u_c, v_c = pt_curr.ravel()

                du = int(np.clip(u * scale_x, 0, dw - 1))
                dv = int(np.clip(v * scale_y, 0, dh - 1))
                z = float(self.prev_depth[dv, du])

                if 0.35 <= z <= 6.5:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    pts_3d.append([x, y, z])
                    pts_2d.append([u_c, v_c])
                    valid_prev_pts.append((u, v))
                    valid_curr_pts.append((u_c, v_c))

            # 3. Solve 6-DoF Pose with PnP-RANSAC
            if len(pts_3d) >= 6:
                pts_3d_np = np.array(pts_3d, dtype=np.float64)
                pts_2d_np = np.array(pts_2d, dtype=np.float64)

                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_3d_np,
                    pts_2d_np,
                    self.K,
                    distCoeffs=None,
                    iterationsCount=250,
                    reprojectionError=2.0,
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_EPNP
                )

                if success and inliers is not None and len(inliers) >= 6:
                    inliers_count = len(inliers)
                    inl_idx = inliers.ravel()

                    # Levenberg-Marquardt Non-Linear Pose Refinement
                    try:
                        pts_3d_inl = pts_3d_np[inl_idx]
                        pts_2d_inl = pts_2d_np[inl_idx]
                        lm_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        rvec, tvec = cv2.solvePnPRefineLM(
                            pts_3d_inl, pts_2d_inl, self.K, distCoeffs=None,
                            rvec=rvec, tvec=tvec, criteria=lm_crit
                        )
                    except cv2.error:
                        pass

                    R_c, _ = cv2.Rodrigues(rvec)

                    T_rel = np.eye(4, dtype=np.float64)
                    T_rel[:3, :3] = R_c
                    T_rel[:3, 3] = tvec.ravel()

                    try:
                        T_cam_motion = np.linalg.inv(T_rel)
                        vis_step = float(np.linalg.norm(T_cam_motion[:3, 3]))
                        if vis_step < 0.45:
                            pnp_success = True
                            d_x_vis = float(T_cam_motion[0, 3])
                            d_y_vis = float(T_cam_motion[1, 3])
                            d_z_vis = float(T_cam_motion[2, 3])

                            R_mot = T_cam_motion[:3, :3]
                            vis_dyaw = math.degrees(math.atan2(R_mot[0, 2], R_mot[2, 2]))
                    except np.linalg.LinAlgError:
                        pass

            tracked_prev = valid_prev_pts
            tracked_curr = valid_curr_pts

        # 4. SENSOR-DOMINANT FUSION ENGINE: 70% SENSOR + 30% VISUAL SLAM
        # 4a. YAW HEADING FUSION (80% Gyro + 20% SLAM alignment)
        if has_sensors:
            target_yaw = self.integrated_gyro_yaw
            if pnp_success and inliers_count >= 40 and abs(yaw_rate) < 4.0:
                # 80% Gyro, 20% Visual fine alignment
                w_gyro = 0.80
                w_vis = 0.20
                self.fused_yaw = w_gyro * target_yaw + w_vis * (self.fused_yaw + vis_dyaw)
            else:
                # Active turn or low inliers: 100% Gyro heading
                self.fused_yaw = target_yaw
        else:
            self.fused_yaw += vis_dyaw

        # 4b. TRANSLATION FUSION (Strict 70% Sensor Odometry + 30% Visual SLAM)
        step_body_x = 0.0
        step_body_z = 0.0

        if has_sensors and (abs(speed_fwd) > 0.005 or abs(speed_lat) > 0.005):
            if pnp_success and inliers_count >= 8:
                # 70% Sensor Odometry + 30% Visual SLAM
                w_sens = 0.70
                w_vis = 0.30

                # Consistency check: if motor driving forward, reject backward visual jump
                if speed_fwd > 0.05 and d_z_vis < -0.05:
                    step_body_z = d_z_sens
                    step_body_x = d_x_sens
                    self.sensor_status = "GATE 100% SENS (Slip Filtered)"
                else:
                    step_body_z = w_sens * d_z_sens + w_vis * d_z_vis
                    step_body_x = w_sens * d_x_sens + w_vis * d_x_vis
                    self.sensor_status = "FUSED (70% SENS + 30% SLAM)"
            else:
                step_body_z = d_z_sens
                step_body_x = d_x_sens
                self.sensor_status = "100% SENS ODO (PnP Fail)"
        else:
            # Hover / station-keeping
            if pnp_success and inliers_count >= 8:
                step_body_z = d_z_vis
                step_body_x = d_x_vis
                self.sensor_status = "HOVER: VISUAL_PNP"
            else:
                step_body_z = 0.0
                step_body_x = 0.0
                self.sensor_status = "HOVER: HOLD"

        # 4c. ALTITUDE / VERTICAL FUSION (80% ToF Sensor + 20% Visual)
        if tof_m is not None and self.initial_tof_m is not None:
            target_y = -(tof_m - self.initial_tof_m)
            self.fused_y = 0.80 * target_y + 0.20 * (self.fused_y + d_y_vis)
        else:
            self.fused_y += d_y_vis

        # 4d. ROTATE BODY STEP TO WORLD FRAME USING FUSED YAW
        yaw_rad = math.radians(self.fused_yaw)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        delta_world_x = cos_y * step_body_x + sin_y * step_body_z
        delta_world_z = -sin_y * step_body_x + cos_y * step_body_z

        self.fused_x += delta_world_x
        self.fused_z += delta_world_z

        fused_step_dist = math.hypot(delta_world_x, delta_world_z)
        self.total_distance += fused_step_dist

        # 4e. RECONSTRUCT 4x4 WORLD POSE MATRIX
        R_world = np.array([
            [cos_y,  0.0, sin_y],
            [0.0,    1.0, 0.0  ],
            [-sin_y, 0.0, cos_y]
        ], dtype=np.float64)
        self.cur_pose_w = np.eye(4, dtype=np.float64)
        self.cur_pose_w[:3, :3] = R_world
        self.cur_pose_w[:3, 3] = [self.fused_x, self.fused_y, self.fused_z]

        # Register persistent visual landmarks in the 3D voxel map
        if pnp_success and inliers is not None and len(inliers) >= 8:
            self.add_landmarks(pts_3d, self.cur_pose_w, inliers.ravel())

        # 4f. ACTIVE IN-FRAME FOV SCANNING (Only update what is visible in the current camera frame)
        live_scan = self.project_in_frame_scan(norm_depth_uint8, self.fused_x, self.fused_z, self.fused_yaw)

        self.trajectory.append((self.fused_x, self.fused_y, self.fused_z, self.fused_yaw, 0.0, 0.0, timestamp))

        # Temporal Feature Replenishment with spatial coverage
        if len(good_curr) < 130:
            existing = [p for p in good_curr]
            new_pts = self.detect_bucketed_features(gray, existing_pts=existing)
            if len(new_pts) > 0:
                if len(good_curr) > 0:
                    kept = good_curr.reshape(-1, 1, 2)
                    self.prev_pts = np.vstack([kept, new_pts])
                else:
                    self.prev_pts = new_pts
            else:
                self.prev_pts = good_curr.reshape(-1, 1, 2) if len(good_curr) > 0 else np.empty((0, 1, 2), dtype=np.float32)
        else:
            self.prev_pts = good_curr.reshape(-1, 1, 2)

        self.prev_gray = gray
        self.prev_depth = metric_depth
        self.tracking_inliers = inliers_count

        return {
            "pose": (self.fused_x, self.fused_y, self.fused_z, self.fused_yaw, 0.0, 0.0),
            "tracked_pts_prev": tracked_prev,
            "tracked_pts_curr": tracked_curr,
            "inliers_count": inliers_count,
            "total_distance": self.total_distance,
            "sensor_status": self.sensor_status,
            "tof_m": tof_m,
            "gyro_yaw": self.integrated_gyro_yaw,
            "live_scan": live_scan,
        }


def render_slam_hud(frame_bgr, norm_depth_uint8, slam_res, slam_tracker, frame_idx, total_frames, telem_row=None, infer_ms=18.0):
    """Render combined dual-panel SLAM HUD: Camera with feature tracks (left) and Global 3D SLAM Map (right)."""
    fh, fw = frame_bgr.shape[:2]
    vis_cam = frame_bgr.copy()

    # 1. Draw SLAM Feature Tracks on Camera View
    pts_curr = slam_res.get("tracked_pts_curr", [])
    pts_prev = slam_res.get("tracked_pts_prev", [])
    inliers = slam_res.get("inliers_count", 0)

    for p_cur, p_prv in zip(pts_curr, pts_prev):
        cx, cy = int(p_cur[0]), int(p_cur[1])
        px, py = int(p_prv[0]), int(p_prv[1])
        cv2.circle(vis_cam, (cx, cy), 3, (0, 255, 120), -1)
        # Small motion trail line
        cv2.line(vis_cam, (px, py), (cx, cy), (0, 200, 255), 1)

    # Top status bar on Camera View
    cv2.rectangle(vis_cam, (0, 0), (fw, 64), (18, 22, 28), -1)
    cv2.putText(vis_cam, f"DEEPSLAM + SENSOR FUSION: Tello Vision [Frame {frame_idx}/{total_frames}]", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    x, y, z, yaw, pitch, roll = slam_res["pose"]
    dist = slam_res["total_distance"]

    status_str = f"Pose 3D: X={x:+.2f}m Y={y:+.2f}m Z={z:+.2f}m | Yaw={yaw:+.1f} deg"
    cv2.putText(vis_cam, status_str, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 2)

    # Inliers & Sensor Mode badge
    cv2.rectangle(vis_cam, (fw - 275, 8), (fw - 12, 56), (30, 36, 46), -1)
    mode_str = slam_res.get("sensor_status", "FUSED")[:24]
    cv2.putText(vis_cam, f"Mode: {mode_str}", (fw - 265, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 120), 1)

    tof_val = slam_res.get("tof_m")
    tof_txt = f"ToF:{int(tof_val * 100)}cm" if tof_val is not None else "ToF:Hold"
    cv2.putText(vis_cam, f"{tof_txt} | Inl:{inliers} | D:{dist:.2f}m", (fw - 265, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 255), 1)

    # Telemetry compare if present
    if telem_row:
        t_state = telem_row.get("state", "N/A")[:26]
        t_cmd = f"FB:{telem_row.get('fb', 0)} Yaw:{telem_row.get('yaw', 0)}"
        cv2.rectangle(vis_cam, (0, fh - 40), (fw, fh), (15, 18, 24), -1)
        cv2.putText(vis_cam, f"Sensors: {t_state} | Cmd: {t_cmd}", (15, fh - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # 2. Render Global 3D SLAM Map (Top-Down Bird's Eye View)
    map_w, map_h = fw, fh
    vis_map = np.zeros((map_h, map_w, 3), dtype=np.uint8)
    vis_map[:] = (16, 20, 26)

    # Map Scale & Center
    # Origin (Takeoff point) placed at bottom-center of the map
    origin_u = map_w // 2
    origin_v = int(map_h * 0.82)
    # Pixels per meter (e.g. 80 pixels = 1 meter)
    scale = 85.0

    def w2m(wx, wz):
        """World (x=right, z=forward) to Map (u, v)"""
        u = int(origin_u + wx * scale)
        v = int(origin_v - wz * scale)
        return (u, v)

    # Draw 1m metric grid lines
    grid_col = (30, 38, 48)
    for gm in range(-10, 11):
        # Vertical grid lines (every 1m)
        gu, _ = w2m(gm, 0)
        if 0 <= gu < map_w:
            cv2.line(vis_map, (gu, 0), (gu, map_h), grid_col, 1)
            cv2.putText(vis_map, f"{gm}m", (gu + 3, map_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (70, 85, 105), 1)

    for gm in range(-3, 15):
        # Horizontal grid lines (every 1m forward)
        _, gv = w2m(0, gm)
        if 0 <= gv < map_h:
            cv2.line(vis_map, (0, gv), (map_w, gv), grid_col, 1)
            cv2.putText(vis_map, f"{gm}m", (10, gv - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (70, 85, 105), 1)

    # Origin Takeoff marker
    cv2.circle(vis_map, (origin_u, origin_v), 6, (0, 140, 255), -1)
    cv2.putText(vis_map, "START (0,0)", (origin_u + 10, origin_v + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 140, 255), 1)

    # Draw 3D Landmark Point Cloud (sparse map of walls/tables)
    if slam_tracker.landmarks:
        for lx, ly, lz in slam_tracker.landmarks[::2]:
            lu, lv = w2m(lx, lz)
            if 0 <= lu < map_w and 0 <= lv < map_h:
                cv2.circle(vis_map, (lu, lv), 1, (80, 120, 220), -1)

    # Draw Reconstructed 3D Flight Trajectory (continuous glowing curve)
    traj = slam_tracker.trajectory
    if len(traj) > 1:
        pts = [w2m(tx, tz) for tx, ty, tz, _, _, _, _ in traj]
        # Glowing shadow
        for i in range(len(pts) - 1):
            cv2.line(vis_map, pts[i], pts[i + 1], (0, 160, 50), 3)
            cv2.line(vis_map, pts[i], pts[i + 1], (0, 255, 120), 2)
        for pt in pts[::5]:
            cv2.circle(vis_map, pt, 2, (180, 255, 200), -1)

    # Draw Current Drone Position & Heading Frustum
    cur_u, cur_v = w2m(x, z)
    if 0 <= cur_u < map_w and 0 <= cur_v < map_h:
        # Heading vector
        yaw_rad = math.radians(yaw)
        # World +Z is forward (up on map), +X is right (right on map)
        hx = math.sin(yaw_rad)
        hz = math.cos(yaw_rad)
        fwd_len = 38
        end_u = int(cur_u + hx * fwd_len)
        end_v = int(cur_v - hz * fwd_len)

        # Field of view cone
        fov_half = math.radians(41.3)
        c1_u = int(cur_u + math.sin(yaw_rad - fov_half) * 55)
        c1_v = int(cur_v - math.cos(yaw_rad - fov_half) * 55)
        c2_u = int(cur_u + math.sin(yaw_rad + fov_half) * 55)
        c2_v = int(cur_v - math.cos(yaw_rad + fov_half) * 55)

        cone_pts = np.array([[cur_u, cur_v], [c1_u, c1_v], [c2_u, c2_v]], dtype=np.int32)
        cv2.fillPoly(vis_map, [cone_pts], (30, 60, 45))
        cv2.line(vis_map, (cur_u, cur_v), (c1_u, c1_v), (0, 200, 100), 1)
        cv2.line(vis_map, (cur_u, cur_v), (c2_u, c2_v), (0, 200, 100), 1)

        # Draw Active In-Frame LiDAR Scan (Live FOV raycast)
        live_scan = slam_res.get("live_scan", [])
        for idx_r, (sx, sz, sdist) in enumerate(live_scan):
            su, sv = w2m(sx, sz)
            if 0 <= su < map_w and 0 <= sv < map_h:
                # Faint laser ray from drone to obstacle (draw every 4th ray for crisp display)
                if idx_r % 4 == 0:
                    cv2.line(vis_map, (cur_u, cur_v), (su, sv), (35, 80, 50), 1)
                # Bright glowing cyan obstacle hit point
                cv2.circle(vis_map, (su, sv), 2, (255, 255, 0), -1)

        # Heading arrow & drone body
        cv2.line(vis_map, (cur_u, cur_v), (end_u, end_v), (0, 255, 255), 2)
        cv2.circle(vis_map, (cur_u, cur_v), 7, (0, 220, 255), -1)
        cv2.circle(vis_map, (cur_u, cur_v), 9, (255, 255, 255), 1)

    # Top title and legend on Map View
    cv2.rectangle(vis_map, (0, 0), (map_w, 64), (18, 22, 28), -1)
    cv2.putText(vis_map, "GLOBAL 3D SLAM MAP (TOP-DOWN BEV)", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    live_pts_cnt = len(slam_res.get("live_scan", []))
    cv2.putText(vis_map, f"Scale: 1m grid | Live Scan: {live_pts_cnt} pts | Voxel Map: {len(slam_tracker.landmarks)} pts", (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 190, 215), 1)

    # Map Legend
    legend_x = map_w - 290
    cv2.circle(vis_map, (legend_x, 18), 4, (0, 255, 120), -1)
    cv2.putText(vis_map, "Trajectory (70% Sens + 30% SLAM)", (legend_x + 12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)
    cv2.circle(vis_map, (legend_x, 35), 4, (255, 255, 0), -1)
    cv2.putText(vis_map, "Live In-Frame Scan (DA2 LiDAR)", (legend_x + 12, 39),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
    cv2.circle(vis_map, (legend_x, 52), 3, (80, 120, 220), -1)
    cv2.putText(vis_map, "Voxel Room Map (Historical)", (legend_x + 12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 190, 210), 1)

    # Border dividing the two panels
    combined = np.hstack([vis_cam, vis_map])
    cv2.line(combined, (fw, 0), (fw, fh), (60, 70, 85), 2)
    return combined


def run_deep_slam(flight_dir, step=1, max_frames=None, export_video=True, show_gui=True):
    """Run complete DeepSLAM pipeline on recorded flight."""
    flight_dir = Path(flight_dir).resolve()
    print("=" * 75)
    print("DEEPSLAM: MONOCULAR DEEP VISUAL SLAM & TRAJECTORY RECONSTRUCTOR")
    print(f"Flight Directory: {flight_dir}")

    video_path = flight_dir / "flight_raw.mp4"
    if not video_path.exists():
        video_path = flight_dir / "flight_airborne_clean.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found in {flight_dir}")

    csv_path = flight_dir / "flight_telemetry.csv"
    if not csv_path.exists():
        csv_path = flight_dir / "commands.csv"

    print(f"Video Source:     {video_path.name}")
    print(f"Telemetry File:   {csv_path.name if csv_path.exists() else 'None'}")
    print("=" * 75)

    telemetry = load_telemetry(csv_path)

    # Initialize Depth Anything V2
    print("Initializing Depth Anything V2 (Dense 3D Geometry)...")
    da2 = DA2DepthEstimator(input_size=252)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize DeepSLAM Tracker
    slam = DeepSLAMTracker(img_w=fw, img_h=fh, fov_deg=82.6)

    out_video_path = flight_dir / "deep_slam_reconstruction.mp4"
    writer = None
    if export_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps / max(1, step), (fw * 2, fh))

    window_name = f"DeepSLAM Visualizer: {flight_dir.name}"
    if show_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    processed_count = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if (frame_idx - 1) % step != 0:
            continue

        if max_frames and processed_count >= max_frames:
            break

        # 1. Depth Anything V2 Dense Depth
        norm_depth, _, _, _, infer_ms = da2.predict(frame)

        # 2. Extract Sensor Reading (ToF Laser + IMU Gyro + Odometry)
        telem_row = telemetry.get(frame_idx, {})
        sensor_reading = parse_sensor_reading(telem_row, dt=float(step) / max(1.0, fps))

        # 3. DeepSLAM 6-DoF Motion Tracking with Sensor Fusion
        t_slam0 = time.perf_counter()
        slam_res = slam.track_frame(frame, norm_depth, timestamp=(frame_idx / fps), sensor_reading=sensor_reading)
        slam_ms = (time.perf_counter() - t_slam0) * 1000

        # 4. Render Combined SLAM View (Camera Feature Tracks + Global Top-Down Map)
        combined = render_slam_hud(
            frame, norm_depth, slam_res, slam, frame_idx, total_frames,
            telem_row=telem_row, infer_ms=infer_ms
        )

        if writer is not None:
            writer.write(combined)

        if show_gui:
            # Scale down if too wide for 1080p desktop
            ch, cw = combined.shape[:2]
            scaled = cv2.resize(combined, (int(cw * 0.75), int(ch * 0.75))) if cw > 1500 else combined
            cv2.imshow(window_name, scaled)
            key = cv2.waitKey(1)
            if key in (ord("q"), 27):
                break

        processed_count += 1
        if processed_count % 30 == 0 or processed_count == total_frames:
            x, y, z, yaw, _, _ = slam_res["pose"]
            mode_s = slam_res.get("sensor_status", "FUSED")
            tof_pr = f"{int(slam_res['tof_m']*100)}cm" if slam_res.get("tof_m") else "Hold"
            print(f"  Frame {frame_idx}/{total_frames} | DA2: {infer_ms:.1f}ms | SLAM: {slam_ms:.1f}ms | "
                  f"Pos: [{x:+.2f}, {z:+.2f}]m | Yaw: {yaw:+.1f} deg | ToF: {tof_pr} | Mode: {mode_s}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nReconstruction video exported to: {out_video_path}")

    if show_gui:
        cv2.destroyAllWindows()

    # Export Trajectory CSV
    traj_csv_path = flight_dir / "deep_slam_trajectory.csv"
    with open(traj_csv_path, "w", newline="", encoding="utf-8") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["timestamp_s", "x_m", "y_m", "z_m", "yaw_deg", "pitch_deg", "roll_deg"])
        for row in slam.trajectory:
            w_csv.writerow([round(row[6], 3), round(row[0], 4), round(row[1], 4), round(row[2], 4),
                            round(row[3], 2), round(row[4], 2), round(row[5], 2)])

    total_time = time.time() - t0
    final_x, final_y, final_z, final_yaw, _, _, _ = slam.trajectory[-1] if slam.trajectory else (0, 0, 0, 0, 0, 0, 0)

    print("\n" + "=" * 75)
    print("DEEPSLAM RECONSTRUCTION SUMMARY:")
    print(f"  Processed Frames:       {processed_count}")
    print(f"  Elapsed Time:           {total_time:.2f}s ({processed_count / max(0.01, total_time):.1f} FPS)")
    print(f"  Total Path Length:      {slam.total_distance:.2f} meters")
    print(f"  Final Position (X, Z):  [{final_x:+.2f}m, {final_z:+.2f}m]")
    print(f"  Final Yaw Angle:        {final_yaw:+.1f} degrees")
    print(f"  3D Map Landmarks:       {len(slam.landmarks)} points")
    print(f"  Trajectory CSV Saved:   {traj_csv_path}")
    print(f"  Reconstruction Video:   {out_video_path}")
    print("=" * 75)

    return {
        "frames": processed_count,
        "distance_m": slam.total_distance,
        "final_pose": (final_x, final_y, final_z, final_yaw),
        "landmarks": len(slam.landmarks),
        "trajectory_file": str(traj_csv_path),
        "video_file": str(out_video_path)
    }


def main():
    parser = argparse.ArgumentParser(description="DeepSLAM: Monocular Deep Visual SLAM for Tello")
    parser.add_argument("flight", nargs="?", default="latest", help="Flight folder name in debug/ or full path")
    parser.add_argument("--step", type=int, default=1, help="Frame step/stride (default 1)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames")
    parser.add_argument("--no-gui", action="store_true", help="Run headless without OpenCV window")
    parser.add_argument("--no-video", action="store_true", help="Do not export reconstruction video")
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

    run_deep_slam(
        flight_dir=flight_dir,
        step=args.step,
        max_frames=args.max_frames,
        export_video=not args.no_video,
        show_gui=not args.no_gui
    )


if __name__ == "__main__":
    main()
