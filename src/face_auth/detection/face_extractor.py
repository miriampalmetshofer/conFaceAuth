"""Face extraction and preprocessing functionality."""
from typing import Optional
import cv2
import numpy as np

from face_auth.detection.models import BoundingBox, DetectionResult
from face_auth.detection.face_detector import FaceDetector


class FaceExtractor:
    """Extracts and preprocesses face regions from images."""

    def __init__(
        self,
        target_width: int,
        target_height: int
    ):
        """Initialize face extractor.

        Args:
            target_width: Width to resize extracted faces to
            target_height: Height to resize extracted faces to
        """
        self.target_width = target_width
        self.target_height = target_height

    def extract(
        self,
        image_bgr: np.ndarray,
        bounding_box: BoundingBox
    ) -> np.ndarray:
        """Extract and preprocess face region from image.

        Args:
            image_bgr: Source image in BGR format
            bounding_box: Location of face in image

        Returns:
            Preprocessed face image in RGB format, resized to target dimensions
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        face_region = image_rgb[
            bounding_box.y:bounding_box.y + bounding_box.height,
            bounding_box.x:bounding_box.x + bounding_box.width
        ]

        return cv2.resize(face_region, (self.target_width, self.target_height))

    def detect_and_extract(
        self,
        image_bgr: np.ndarray,
        detector: FaceDetector
    ) -> Optional[DetectionResult]:
        """Convenience method to detect and extract face in one call.

        Args:
            image_bgr: Source image in BGR format
            detector: Face detector to use

        Returns:
            DetectionResult with preprocessed face and bounding box,
            or None if no face was detected
        """
        bounding_box = detector.detect(image_bgr)
        if bounding_box is None:
            return None

        face_image = self.extract(image_bgr, bounding_box)
        return DetectionResult(face_image=face_image, bounding_box=bounding_box)
