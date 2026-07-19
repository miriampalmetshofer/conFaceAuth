"""Data models for face detection components."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Represents a rectangular bounding box for a detected face."""
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class DetectionResult:
    """Result of face detection and extraction."""
    face_image: np.ndarray  # Cropped face region in RGB format
    bounding_box: BoundingBox
