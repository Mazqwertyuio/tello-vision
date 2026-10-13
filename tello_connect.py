"""DJI Tello Autopilot Entry Point.

Modular architecture with:
- Visual LiDAR (Depth Anything V2)
- Autonomous Collision Prevention Shield
- Mode 1: Autonomous Room Exploration
- Mode 2: Human Target Following
"""

import sys
from main import main

if __name__ == "__main__":
    sys.exit(main())
