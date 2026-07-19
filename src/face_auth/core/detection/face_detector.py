"""Face detection functionality."""
from typing import Optional
import cv2
import numpy as np
from face_auth.core.detection.backend.detector_backend import DetectorBackend
from face_auth.core.detection.models import DetectionResult
from face_auth.core.detection.detector_factory import create_detector


class FaceDetector:
    """Detects and extracts faces from images."""

    def __init__(self, detector_backend: str):
        """Initialize face detector with specified backend.

        Args:
            detector_backend: Name of backend to use (e.g., "mtcnn", "mediapipe")

        Raises:
            ValueError: If detector_backend is not recognized
        """
        self._detector: DetectorBackend = create_detector(detector_backend)

    def detect(self, image_bgr: np.ndarray) -> Optional[DetectionResult]:
        """Detect and extract the largest face in an image.

        Args:
            image_bgr: Image in BGR format (OpenCV default)

        Returns:
            DetectionResult with extracted face and bounding box,
            or None if no face was found
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        bounding_box = self._detector.detect(image_rgb)
        if bounding_box is None:
            return None

        face_region = image_rgb[
            bounding_box.y:bounding_box.y + bounding_box.height,
            bounding_box.x:bounding_box.x + bounding_box.width
        ]
        return DetectionResult(face_image=face_region, bounding_box=bounding_box)
