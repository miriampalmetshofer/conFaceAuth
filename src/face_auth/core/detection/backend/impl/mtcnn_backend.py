"""MTCNN face detector backend."""
from typing import Optional
import numpy as np
from mtcnn import MTCNN

from face_auth.core.detection.models import BoundingBox


class MTCNNBackend:
    """Face detector using MTCNN."""

    def __init__(self):
        """Initialize MTCNN detector."""
        self._detector = MTCNN()

    def detect(self, image_rgb: np.ndarray) -> Optional[BoundingBox]:
        """Detect the largest face in an RGB image.

        Args:
            image_rgb: Image in RGB format

        Returns:
            BoundingBox of largest detected face, or None if no face found
        """
        try:
            results = self._detector.detect_faces(image_rgb)
        except ValueError as e:
            # MTCNN bug: sometimes P-Net finds candidates that get filtered out before R-Net,
            # leaving an empty batch that R-Net can't handle
            if "empty output" in str(e).lower() or "shape=(0," in str(e):
                return None
            raise

        if not results:
            return None

        # Find largest face by area
        largest_face = max(results, key=lambda face: face['box'][2] * face['box'][3])
        box_dict = largest_face['box']

        return BoundingBox(
            x=max(0, box_dict[0]),
            y=max(0, box_dict[1]),
            width=box_dict[2],
            height=box_dict[3]
        )
