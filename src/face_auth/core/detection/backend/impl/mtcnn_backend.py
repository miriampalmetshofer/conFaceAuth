"""MTCNN face detector backend."""
from typing import Optional
import numpy as np
from mtcnn import MTCNN

from face_auth.core.detection.backend.detector_backend import DetectorBackend
from face_auth.core.detection.models import BoundingBox, DetectionResult


class MTCNNBackend(DetectorBackend):
    """Face detector using MTCNN."""

    def __init__(self):
        """Initialize MTCNN detector."""
        self._detector = MTCNN()

    def get_backend_name(self) -> str:
        """Return the detector backend name used in configuration."""
        return "mtcnn"

    def detect(self, image_rgb: np.ndarray) -> Optional[DetectionResult]:
        """Detect and extract the largest face in an RGB image.

        Args:
            image_rgb: Image in RGB format

        Returns:
            DetectionResult with extracted face and bounding box,
            or None if no face found
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

        bounding_box = BoundingBox(
            x=max(0, box_dict[0]),
            y=max(0, box_dict[1]),
            width=box_dict[2],
            height=box_dict[3]
        )

        face_region = image_rgb[
            bounding_box.y:bounding_box.y + bounding_box.height,
            bounding_box.x:bounding_box.x + bounding_box.width
        ]

        return DetectionResult(face_image=face_region, bounding_box=bounding_box)
