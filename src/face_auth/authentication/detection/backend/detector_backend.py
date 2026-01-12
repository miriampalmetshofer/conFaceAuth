"""Face detector backend interface."""
from typing import Protocol, Optional
import numpy as np

from face_auth.authentication.detection.models import BoundingBox


class DetectorBackend(Protocol):
    """Interface that all face detector backend must implement."""

    def detect(self, image_rgb: np.ndarray) -> Optional[BoundingBox]:
        """Detect the largest face in an RGB image.

        Args:
            image_rgb: Image in RGB format

        Returns:
            BoundingBox of largest detected face, or None if no face found
        """
        ...
