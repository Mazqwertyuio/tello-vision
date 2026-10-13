"""360-degree Persistent Spatial Obstacle Memory.

Maintains a polar grid around the drone in world compass frame (Yaw).
Retains obstacle clearances across time so walls, furniture, and boundaries
that rotate out of camera FOV remain remembered for route planning.
"""

from typing import Tuple, List, Dict, Any
import numpy as np

from config import LIDAR_MAX_RANGE_M


class SpatialObstacleMemory:
    """Polar memory grid (36 bins = 10 degrees each) in world frame."""

    def __init__(self, num_bins: int = 36, default_dist_m: float = LIDAR_MAX_RANGE_M):
        self.num_bins = num_bins
        self.bin_size_deg = 360.0 / num_bins
        self.default_dist_m = default_dist_m
        self.distances = np.full(num_bins, default_dist_m, dtype=np.float32)
        self.timestamps = np.zeros(num_bins, dtype=np.float64)

    def reset(self):
        """Reset polar memory to default empty room clearances."""
        self.distances.fill(self.default_dist_m)
        self.timestamps.fill(0.0)

    def update_from_scan(self, rays: List[Dict[str, Any]], current_yaw_deg: float, now: float):
        """Integrate new radial sweep from LiDAR into world-frame memory."""
        for r in rays:
            rel_ang = r["angle_deg"]
            dist_m = r["dist_m"]
            world_ang = (current_yaw_deg + rel_ang) % 360.0
            idx = int(world_ang / self.bin_size_deg) % self.num_bins

            old_d = self.distances[idx]
            # Fast attack on new closer obstacles, slower decay on openings
            if dist_m < old_d:
                self.distances[idx] = 0.80 * dist_m + 0.20 * old_d
            else:
                self.distances[idx] = 0.35 * dist_m + 0.65 * old_d
            self.timestamps[idx] = now

    def decay(self, now: float):
        """Slowly relax older unseen obstacle bins towards open space."""
        for i in range(self.num_bins):
            age = now - self.timestamps[i]
            if age > 2.0:
                self.distances[i] = min(
                    self.default_dist_m,
                    self.distances[i] + 0.04 * min(1.0, age - 2.0),
                )

    def get_flank_clearance(
        self, current_yaw_deg: float, relative_deg: float, spread_deg: float = 25.0
    ) -> float:
        """Query obstacle clearance on left (-90°) or right (+90°) flank."""
        center_w = (current_yaw_deg + relative_deg) % 360.0
        min_d = self.default_dist_m
        for off in np.linspace(-spread_deg, spread_deg, 5):
            w_ang = (center_w + off) % 360.0
            idx = int(w_ang / self.bin_size_deg) % self.num_bins
            min_d = min(min_d, float(self.distances[idx]))
        return min_d

    def find_best_open_heading(self, current_yaw_deg: float) -> Tuple[float, float]:
        """Find the world compass heading with maximum open clearance.

        Applies 3-bin smoothing and slight penalty to 180° backtracks.

        Returns:
            best_heading_deg: Heading angle [-180..+180].
            clearance_m: Distance to nearest obstacle in that corridor.
        """
        smoothed = np.zeros(self.num_bins, dtype=np.float32)
        for i in range(self.num_bins):
            prev_i = (i - 1) % self.num_bins
            next_i = (i + 1) % self.num_bins
            smoothed[i] = (
                0.25 * self.distances[prev_i]
                + 0.50 * self.distances[i]
                + 0.25 * self.distances[next_i]
            )

        # Penalize reverse direction (>135 deg) slightly to avoid unnecessary 180 backtrack
        for i in range(self.num_bins):
            bin_ang = i * self.bin_size_deg
            rel_ang = abs((bin_ang - current_yaw_deg + 180.0) % 360.0 - 180.0)
            if rel_ang > 135.0:
                smoothed[i] *= 0.85

        best_idx = int(np.argmax(smoothed))
        best_heading = (best_idx * self.bin_size_deg + self.bin_size_deg / 2.0) % 360.0
        if best_heading > 180.0:
            best_heading -= 360.0
        return float(best_heading), float(smoothed[best_idx])
