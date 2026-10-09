# Tello Vision & Autonomous Navigation

Autonomous vision-guided flight system for DJI Tello featuring:
- Monocular depth estimation and forward corridor navigation between rows of obstacles/tables.
- Low-altitude hold (~30–35 cm) with downward ToF ground distance sensing.
- Monochrome Wall Guard with gradient/texture variance analysis.
- Semantic doorway detection and traversal with YOLO-World and depth step validation.
- Interactive flight playback, telemetry analysis, and DA3 dataset export.

## Architecture & Modules
- `tello_connect.py`: Real-time drone control loop, autopilot state machines, and HUD.
- `door_navigator.py`: Forward corridor analyzer, doorway verification, and visual HUD overlays.
- `replay_flight.py`: Offline replay tool with telemetry scrubbing and depth map inspection.
- `export_for_da3.py`: Clean footage exporter for Depth Anything 3D reconstruction.

## Requirements
```bash
pip install -r requirements.txt
pip install torch torchvision
```

## Quick Start
- Run drone autopilot:
  ```bash
  python tello_connect.py
  ```
  - Press `a`: Low-Altitude Table Corridor Autopilot (30cm)
  - Press `e`: Room Exploration & Door Navigation
  - Press `SPACE`: Instant hover
  - Press `l`: Land
- Replay flight telemetry:
  ```bash
  python replay_flight.py debug/<flight_folder>
  ```
