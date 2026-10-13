"""Depth Anything V2 Neural Network Interface for Monocular LiDAR Depth.

Predicts dense metric-correlated disparity map [0..255] on GPU.
Higher values (255) indicate immediate proximity, lower values (0) indicate distant space.
Includes temporal EMA stabilization to eliminate frame-to-frame noise.
"""

from pathlib import Path
import time
from typing import Tuple, Optional
import cv2
import numpy as np
import torch

from config import DA2_CHECKPOINT, DEPTH_INPUT_SIZE, DEPTH_EMA_ALPHA, get_safe_path

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    HAS_DA2 = True
except ImportError:
    HAS_DA2 = False


class DepthEstimator:
    """High-speed neural monocular depth estimator powered by Depth Anything V2."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        encoder: str = "vits",
        input_size: int = DEPTH_INPUT_SIZE,
        temporal_alpha: float = DEPTH_EMA_ALPHA,
        device: Optional[str] = None,
    ):
        self.input_size = input_size
        self.temporal_alpha = temporal_alpha
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.smooth_depth = None

        if checkpoint_path is None:
            checkpoint_path = str(DA2_CHECKPOINT)

        if not HAS_DA2:
            raise ImportError(
                "DepthAnythingV2 package is not installed. Ensure depth_anything_v2 is available."
            )

        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(f"DA2 checkpoint not found at: {checkpoint_path}")

        print(f"[AI DEPTH] Loading Depth Anything V2 ({encoder}) on {self.device}...")
        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        cfg = model_configs.get(encoder, model_configs["vits"])
        self.model = DepthAnythingV2(**cfg)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        self.model = self.model.to(self.device).eval()
        print("[AI DEPTH] DA2 model loaded and ready on " + self.device)

    def reset(self):
        """Reset temporal smoothing."""
        self.smooth_depth = None

    def estimate(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Infer depth map from BGR frame.

        Returns:
            norm_depth: uint8 [0..255] where 255 = closest obstacle, 0 = farthest open space.
            color_map: Inferno-colored visualization (BGR).
            elapsed_ms: Execution time in milliseconds.
        """
        t0 = time.perf_counter()
        orig_h, orig_w = frame_bgr.shape[:2]

        # Model expects BGR/RGB resized to input_size
        with torch.no_grad():
            raw_depth = self.model.infer_image(frame_bgr, input_size=self.input_size)

        # In DA2, higher raw_depth values correspond to closer objects.
        # Normalize to robust 0..255 range using 2nd and 98th percentiles
        p_low = np.percentile(raw_depth, 2)
        p_high = np.percentile(raw_depth, 98)
        denom = max(p_high - p_low, 1e-5)
        norm_float = np.clip((raw_depth - p_low) / denom, 0.0, 1.0) * 255.0

        # Temporal EMA smoothing to remove high-frequency noise
        if self.smooth_depth is None or self.smooth_depth.shape != norm_float.shape:
            self.smooth_depth = norm_float.copy()
        else:
            self.smooth_depth = (
                self.temporal_alpha * norm_float
                + (1.0 - self.temporal_alpha) * self.smooth_depth
            )

        norm_depth = self.smooth_depth.astype(np.uint8)

        # Resize back to original frame dimensions if needed for HUD display
        if norm_depth.shape != (orig_h, orig_w):
            norm_depth_resized = cv2.resize(
                norm_depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            norm_depth_resized = norm_depth

        color_map = cv2.applyColorMap(norm_depth_resized, cv2.COLORMAP_INFERNO)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return norm_depth_resized, color_map, elapsed_ms
