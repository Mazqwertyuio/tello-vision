"""Person Silhouette Detector using YOLOv8n (with OpenCV HOG Fallback).

Detects full-body and upper-body human silhouettes in real-time on GPU.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

from config import YOLO_MODEL_PATH

try:
    from ultralytics import YOLO
    import torch
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class PersonDetector:
    """Detects human silhouettes for drone tracking."""

    def __init__(self, model_path: Optional[Path] = None):
        self.yolo = None
        self.hog = None
        self.name = "None"
        self.target_person = None

        m_path = model_path or YOLO_MODEL_PATH
        if HAS_YOLO and m_path.exists():
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[AI PERSON] Loading YOLOv8n silhouette detector on {device}...")
                self.yolo = YOLO(str(m_path.resolve()))
                self.name = f"YOLOv8n ({device})"
                print(f"[AI PERSON] YOLOv8n ready.")
            except Exception as exc:
                print(f"[AI PERSON WARNING] Failed to initialize YOLO: {exc}")
                self.yolo = None

        if self.yolo is None:
            # Fallback to OpenCV HOG People Detector
            print("[AI PERSON] Using OpenCV HOG People Detector fallback.")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.name = "OpenCV HOG People Detector"

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect human silhouettes in BGR frame.

        Returns:
            List of (x, y, w, h, confidence) tuples.
        """
        h, w = frame_bgr.shape[:2]

        if self.yolo is not None:
            # Run inference for class 0 (person)
            results = self.yolo(
                frame_bgr, classes=[0], conf=0.35, imgsz=320, verbose=False
            )
            persons = []
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = xyxy
                    bx = int(max(0, x1))
                    by = int(max(0, y1))
                    bw = int(min(w - bx, x2 - x1))
                    bh = int(min(h - by, y2 - y1))
                    if bw > 20 and bh > 30:
                        persons.append((bx, by, bw, bh, conf))
            return persons

        # HOG fallback
        if self.hog is not None:
            small = cv2.resize(frame_bgr, (320, 240))
            sx = w / 320.0
            sy = h / 240.0
            boxes, weights = self.hog.detectMultiScale(
                small, winStride=(8, 8), padding=(4, 4), scale=1.05
            )
            persons = []
            for (bx, by, bw, bh), weight in zip(boxes, weights):
                if float(weight) > 0.2:
                    persons.append((
                        int(bx * sx), int(by * sy),
                        int(bw * sx), int(bh * sy),
                        float(weight)
                    ))
            return persons

        return []

    def select_target(
        self,
        persons: List[Tuple[int, int, int, int, float]],
        frame_shape: Tuple[int, int],
        last_target: Optional[Tuple[int, int, int, int, float]] = None,
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """Select best candidate silhouette with center/size priority and hysteresis."""
        if not persons:
            return None

        h, w = frame_shape[:2]
        center_x = w / 2.0
        center_y = h / 2.0

        def score(p):
            px, py, pw, ph = p[:4]
            pc_x = px + pw / 2.0
            pc_y = py + ph / 2.0
            # Center offset penalty
            dist_to_center = np.hypot(pc_x - center_x, pc_y - center_y) / (w * 0.5)
            # Area score (closer person is larger)
            area_ratio = (pw * ph) / float(w * h)
            base_score = area_ratio * 3.0 - dist_to_center * 0.6

            # Tracking hysteresis for same target
            if last_target is not None:
                lx, ly, lw, lh = last_target[:4]
                lc_x = lx + lw / 2.0
                lc_y = ly + lh / 2.0
                shift = np.hypot(pc_x - lc_x, pc_y - lc_y) / (w * 0.5)
                if shift < 0.30:
                    base_score += 1.5

            return base_score

        return max(persons, key=score)
