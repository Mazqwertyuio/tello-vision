"""Neural network inference models for monocular depth and human detection."""

from .depth_model import DepthEstimator
from .person_detector import PersonDetector

__all__ = ["DepthEstimator", "PersonDetector"]
