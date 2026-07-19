"""Face detector backend interface."""
from typing import Protocol, Optional
import numpy as np

from face_auth.core.detection.models import DetectionResult


class DetectorBackend(Protocol):
    """Interface that all face detector backend must implement."""

    def detect(self, image_rgb: np.ndarray) -> Optional[DetectionResult]:
        """Detect and extract the largest face in an RGB image.

        Args:
            image_rgb: Image in RGB format

        Returns:
            DetectionResult with extracted face and bounding box,
            or None if no face found
        """
        ...
