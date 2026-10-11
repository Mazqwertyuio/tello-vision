"""Depth Anything V2 (DA2) Depth Estimator for Tello Vision & Indoor Navigation.

High-accuracy monocular depth estimation powered by DINOv2-based Depth Anything V2.
Provides:
  - Real-time CUDA disparity / depth map generation (~14-16ms on GPU).
  - Standardized [0..255] uint8 disparity representation (255 = closest, 0 = farthest).
  - Temporal exponential moving average (EMA) filter to eliminate frame flicker.
  - 3-Sector obstacle awareness (Left, Center corridor, Right) and Center Max danger level.
  - Dedicated crop prediction for doorway aperture verification.
"""

from pathlib import Path
import time
import cv2
import numpy as np
import torch

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    HAS_DA2 = True
except ImportError:
    HAS_DA2 = False


class DA2DepthEstimator:
    """Depth Anything V2 monocular depth estimator."""

    def __init__(
        self,
        checkpoint_path=None,
        encoder="vits",
        input_size=252,
        temporal_alpha=0.80,
        device=None,
    ):
        self.encoder = encoder
        self.input_size = input_size
        self.temporal_alpha = temporal_alpha

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not HAS_DA2:
            raise ImportError(
                "DepthAnythingV2 package not found. Ensure 'depth_anything_v2' is installed or in PYTHONPATH."
            )

        # Locate checkpoint
        if checkpoint_path is None:
            candidates = [
                Path("checkpoints/depth_anything_v2_vits.pth"),
                Path(__file__).parent / "checkpoints" / "depth_anything_v2_vits.pth",
                Path(r"C:\Users\User\Desktop\bd_3d\Depth-Anything-V2\checkpoints\depth_anything_v2_vits.pth"),
            ]
            for c in candidates:
                if c.exists():
                    checkpoint_path = str(c.resolve())
                    break

        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Depth Anything V2 checkpoint not found. Looked in: {candidates}"
            )

        print(f"Loading Depth Anything V2 ({encoder}) on {self.device}...")
        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        cfg = model_configs.get(encoder, model_configs["vits"])
        self.model = DepthAnythingV2(**cfg)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu")
        )
        self.model = self.model.to(self.device).eval()

        # Stabilization state
        self.smooth_depth = None
        self.running_min = None
        self.running_max = None
        self.smooth_sectors = None
        print(f"DA2 initialized successfully! Checkpoint: {checkpoint_path}")

    def reset(self):
        """Reset temporal state when video is scrubbed or jumped."""
        self.smooth_depth = None
        self.running_min = None
        self.running_max = None
        self.smooth_sectors = None

    def predict(self, frame_bgr):
        """Predict depth map for frame, apply temporal stabilization,
        and calculate corridor sectors.

        Returns:
            norm_depth: uint8 [0..255] (255 = nearest obstacle, 0 = deepest/farthest)
            color_map: BGR visualization (Inferno)
            sectors: (left_val, center_val, right_val) in [0..255]
            center_max: float 85th percentile of center corridor
            infer_ms: inference latency in milliseconds
        """
        t0 = time.perf_counter()

        with torch.no_grad():
            raw_depth = self.model.infer_image(frame_bgr, input_size=self.input_size)

        infer_ms = (time.perf_counter() - t0) * 1000

        # Temporal filter on raw depth map
        if self.smooth_depth is None or self.smooth_depth.shape != raw_depth.shape:
            self.smooth_depth = raw_depth.copy()
        else:
            self.smooth_depth = (
                self.temporal_alpha * raw_depth
                + (1.0 - self.temporal_alpha) * self.smooth_depth
            )

        # Smooth dynamic range bounds (prevents global luminance pulsing)
        curr_min = float(np.percentile(self.smooth_depth, 2))
        curr_max = float(np.percentile(self.smooth_depth, 98))
        if self.running_min is None:
            self.running_min = curr_min
            self.running_max = curr_max
        else:
            self.running_min = 0.25 * curr_min + 0.75 * self.running_min
            self.running_max = 0.25 * curr_max + 0.75 * self.running_max

        scale = max(self.running_max - self.running_min, 1e-4)
        norm_depth = np.clip(
            (self.smooth_depth - self.running_min) / scale * 255.0, 0, 255
        ).astype(np.uint8)

        # Subtle spatial filtering
        norm_depth = cv2.GaussianBlur(norm_depth, (3, 3), 0)
        color_map = cv2.applyColorMap(norm_depth, cv2.COLORMAP_INFERNO)

        # 3 Sectors based on user-calibrated Flight Window (Eye-level, Floor Excluded)
        from flight_window_config import load_flight_window_config, get_pixel_bounds
        h, w = norm_depth.shape[:2]
        cfg = getattr(self, "window_cfg", None) or load_flight_window_config()
        x1, y1, x2, y2 = get_pixel_bounds(cfg, w, h)

        roi_l = norm_depth[y1:y2, 0:x1]
        roi_c = norm_depth[y1:y2, x1:x2]
        roi_r = norm_depth[y1:y2, x2:w]

        l_raw = float(np.mean(roi_l)) if roi_l.size else 0.0
        c_raw = float(np.mean(roi_c)) if roi_c.size else 0.0
        r_raw = float(np.mean(roi_r)) if roi_r.size else 0.0
        curr_sectors = np.array([l_raw, c_raw, r_raw], dtype=np.float32)

        if self.smooth_sectors is None:
            self.smooth_sectors = curr_sectors
        else:
            self.smooth_sectors = 0.70 * curr_sectors + 0.30 * self.smooth_sectors

        left_val, center_val, right_val = self.smooth_sectors
        center_max = float(np.percentile(roi_c, 85)) if roi_c.size else 0.0

        return norm_depth, color_map, (float(left_val), float(center_val), float(right_val)), float(center_max), infer_ms

    def predict_crop(self, crop_bgr):
        """Stateless high-contrast depth for a single crop (e.g. door aperture).
        Returns float32 disparity map in [0, 1] (1 = closest wall, 0 = deepest passage).
        """
        ch, cw = crop_bgr.shape[:2]
        if ch < 16 or cw < 16:
            return np.zeros((ch, cw), dtype=np.float32)

        with torch.no_grad():
            raw_depth = self.model.infer_image(crop_bgr, input_size=self.input_size)

        lo = float(np.percentile(raw_depth, 2))
        hi = float(np.percentile(raw_depth, 98))
        disp = np.clip((raw_depth - lo) / max(hi - lo, 1e-4), 0.0, 1.0).astype(np.float32)
        return disp
