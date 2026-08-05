"""Face detector backend interface."""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from face_auth.core.detection.models import DetectionResult


class DetectorBackend(ABC):
    """Interface that all face detector backend must implement."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """The detector backend name used in configuration."""
        pass

    @abstractmethod
    def detect(self, image_rgb: np.ndarray) -> Optional[DetectionResult]:
        """Detect and extract the largest face in an RGB image.

        Args:
            image_rgb: Image in RGB format

        Returns:
            DetectionResult with extracted face and bounding box,
            or None if no face found
        """
        pass
