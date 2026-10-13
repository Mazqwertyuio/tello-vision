"""Human Face & Person Detection Neural Module for Target Following.

Combines high-accuracy OpenCV YuNet DNN with Haar Cascades fallback.
Includes target selection with spatial hysteresis to avoid flickering between targets.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np

from config import YUNET_MODEL, FACE_DETECTION_SCALE, FACE_CONFIDENCE_THRESH, get_safe_path


class PersonDetector:
    """Robust face and person detector for drone tracking."""

    def __init__(self, model_path: Optional[Path] = None):
        self.yunet = None
        self.haars = []
        self.name = "Haar Cascade"
        self.target_face = None

        target_model = model_path or YUNET_MODEL
        if target_model.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                safe_model_str = get_safe_path(target_model)
                self.yunet = cv2.FaceDetectorYN.create(
                    model=safe_model_str,
                    config="",
                    input_size=(320, 240),
                    score_threshold=FACE_CONFIDENCE_THRESH,
                    nms_threshold=0.3,
                    top_k=5000,
                    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                    target_id=cv2.dnn.DNN_TARGET_CPU,
                )
                self.name = "OpenCV YuNet (DNN)"
                print(f"[AI DETECTOR] Initialized YuNet detector from: {safe_model_str}")
            except Exception as exc:
                print(f"[AI DETECTOR WARNING] Failed to load YuNet: {exc}. Falling back to Haar.")
                self.yunet = None

        if self.yunet is None:
            self._load_haar_cascades()

    def _load_haar_cascades(self):
        """Load Haar cascades from cv2 data."""
        try:
            p_default = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            c1 = cv2.CascadeClassifier(str(p_default))
            if not c1.empty():
                self.haars.append(c1)
            p_alt = Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
            if p_alt.exists():
                c2 = cv2.CascadeClassifier(str(p_alt))
                if not c2.empty():
                    self.haars.append(c2)
            print("[AI DETECTOR] Haar Cascades loaded.")
        except Exception as exc:
            print(f"[AI DETECTOR WARNING] Could not load Haar cascades: {exc}")

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces in BGR frame.

        Returns:
            List of (x, y, w, h, confidence) tuples in original frame coordinates.
        """
        h, w = frame_bgr.shape[:2]
        if self.yunet is not None:
            # Downscale frame for fast inference
            scale = FACE_DETECTION_SCALE
            dw = max(16, int(w * scale))
            dh = max(16, int(h * scale))
            small = cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_LINEAR)

            self.yunet.setInputSize((dw, dh))
            _, faces = self.yunet.detect(small)

            if faces is None or len(faces) == 0:
                return []

            results = []
            inv_scale = 1.0 / scale
            for f in faces:
                x = int(f[0] * inv_scale)
                y = int(f[1] * inv_scale)
                bw = int(f[2] * inv_scale)
                bh = int(f[3] * inv_scale)
                conf = float(f[14])
                # Bounds check
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                bw = max(1, min(w - x, bw))
                bh = max(1, min(h - y, bh))
                results.append((x, y, bw, bh, conf))
            return results

        # Fallback to Haar Cascade
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        for cascade in self.haars:
            rects = cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30)
            )
            if len(rects) > 0:
                return [(int(x), int(y), int(bw), int(bh), 0.85) for (x, y, bw, bh) in rects]

        return []

    def select_target(
        self,
        faces: List[Tuple[int, int, int, int, float]],
        frame_shape: Tuple[int, int],
        last_target: Optional[Tuple[int, int, int, int, float]] = None,
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """Select best candidate face with spatial hysteresis."""
        if not faces:
            return None

        h, w = frame_shape[:2]
        center_x = w / 2.0
        center_y = h / 2.0

        def score(face):
            fx, fy, fw, fh = face[:4]
            fc_x = fx + fw / 2.0
            fc_y = fy + fh / 2.0
            # Center offset penalty (normalized)
            dist_to_center = np.hypot(fc_x - center_x, fc_y - center_y) / (w * 0.5)
            # Area score (larger face = closer = higher priority)
            area_ratio = (fw * fh) / float(w * h)
            base_score = area_ratio * 2.0 - dist_to_center * 0.5

            # Hysteresis bonus for tracking the same target
            if last_target is not None:
                lx, ly, lw, lh = last_target[:4]
                lc_x = lx + lw / 2.0
                lc_y = ly + lh / 2.0
                shift = np.hypot(fc_x - lc_x, fc_y - lc_y) / (w * 0.5)
                if shift < 0.25:
                    base_score += 1.5

            return base_score

        return max(faces, key=score)
