# Tello Vision & Autonomous Navigation

Autonomous vision-guided flight system for DJI Tello featuring:
- **Monocular Visual LiDAR & Collision Shield** (`visual_lidar.py`): Real-time (~26 FPS) radial 90-ray obstacle scanner, eye-level flight window (floor cutoff), 3-tier proximity brake (<0.85m), and polar BEV radar.
- **Deep Visual-Inertial SLAM (70% Sensors / 30% SLAM)** (`deep_slam.py`): Monocular dense SLAM combining DA2 depth, Lucas-Kanade optical flow, Levenberg-Marquardt PnP refinement, and IMU/odometry sensor fusion.
- **Depth Anything V2 (DA2) Monocular Estimation**: High-speed CUDA depth prediction (~15–18 ms).
- **2D/2.5D Room Occupancy Grid Mapping & Frontier Exploration** (`room_mapper.py`): Real-time spatial mapping of walls, tables, and free space with automatic targeting toward the deepest room opening.
- **3D AR Perspective Path Ribbon & Radar Mini-Map** (`trajectory_planner.py`): Projected flight corridor directly on video feed.
- **Low-altitude Table Corridor Autopilot (~30–35 cm)** with downward ToF ground distance sensing.
- **Monochrome Wall Guard**: Texture variance and gradient analysis to prevent fly-aways near featureless walls.
- **Semantic Doorway Detection & Traversal**: YOLO-World aperture detection validated with local depth steps.
- **Real-Time Interactive Flight Viewer** (`live_da2_navigator.py`): Scrubbing, calibration, and live room map reconstruction.

## Architecture & Modules
- `visual_lidar.py`: Standalone Visual LiDAR scanner, collision shield, polar radar HUD, and safe escape gap finder.
- `deep_slam.py`: 6-DoF Visual-Inertial SLAM, 70/30 sensor fusion, 3D voxel filtering, and flight trajectory reconstructor.
- `flight_window_config.py` & `flight_window_config.json`: Interactive flight corridor & floor cutoff calibration.
- `tello_connect.py`: Real-time drone control loop, autopilot state machines, and HUD.
- `room_mapper.py`: Real-time 2D occupancy grid mapper, raycasting, and deepest clearance targeting.
- `da2_depth.py`: Depth Anything V2 wrapper with temporal EMA flicker suppression.
- `trajectory_planner.py`: Smooth polynomial path planner, 3D AR ribbon projection, and BEV radar.
- `room_geometry_da2.py`: Corridor and monochrome wall evaluator.
- `door_navigator.py`: YOLO-World doorway detector and local depth verifier.
- `live_da2_navigator.py`: Interactive flight replay with DA2 depth and real-time room map.
- `test_da2_flight_offline.py`: Offline evaluation benchmark and automated video generator.

## Requirements
```bash
conda activate yolo_env
pip install -r requirements.txt
```

## Quick Start
- **Visual LiDAR & Obstacle Shield (Real-Time Radar):**
  ```bash
  python visual_lidar.py latest
  ```
- **Deep Visual-Inertial SLAM & Trajectory Reconstructor:**
  ```bash
  python deep_slam.py latest
  ```
- **Real-Time Interactive Viewer & Room Map Replayer:**
  ```bash
  python live_da2_navigator.py latest
  ```
  * Or double-click `start_realtime_viewer.bat`.
- **Run Drone Autopilot (Live Flight):**
  ```bash
  python tello_connect.py
  ```
  - Press `a`: Low-Altitude Table Corridor Autopilot (30cm)
  - Press `e`: Room Exploration & Door Navigation (steers into deepest room clearance)
  - Press `SPACE`: Instant hover
  - Press `l`: Land
- **Run Offline Benchmark & Video Export:**
  ```bash
  .\run_da2_test.bat --flight <flight_folder>
  ```
