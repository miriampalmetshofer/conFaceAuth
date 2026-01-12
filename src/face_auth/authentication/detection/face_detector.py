"""Face detection functionality."""
from typing import Optional
import cv2
import numpy as np
from face_auth.authentication.detection.backend.detector_backend import DetectorBackend
from face_auth.authentication.detection.models import BoundingBox
from face_auth.authentication.detection.detector_factory import create_detector


class FaceDetector:
    """Detects faces in images and returns bounding boxes."""

    def __init__(self, detector_backend: str):
        """Initialize face detector with specified backend.

        Args:
            detector_backend: Name of backend to use (e.g., "mtcnn", "mediapipe")

        Raises:
            ValueError: If detector_backend is not recognized
        """
        self._detector: DetectorBackend = create_detector(detector_backend)

    def detect(self, image_bgr: np.ndarray) -> Optional[BoundingBox]:
        """Detect the largest face in an image.

        Args:
            image_bgr: Image in BGR format (OpenCV default)

        Returns:
            BoundingBox of largest detected face, or None if no face found
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self._detector.detect(image_rgb)
