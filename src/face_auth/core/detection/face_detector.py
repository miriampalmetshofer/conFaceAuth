"""Face detection functionality."""
from typing import Optional
import cv2
import numpy as np
from mtcnn import MTCNN

from face_auth.core.detection.models import BoundingBox


class FaceDetector:
    """Detects faces in images and returns bounding boxes."""

    def __init__(self, detector_backend: str):
        if detector_backend == "mtcnn":
            self._detector = MTCNN()
        else:
            raise ValueError(
                f"Unsupported detector backend: {detector_backend}. "
                f"Supported backends: ['mtcnn']"
            )

    def detect(self, image_bgr: np.ndarray) -> Optional[BoundingBox]:
        """Detect the first face in an image.

        Args:
            image_bgr: Image in BGR format (OpenCV default)

        Returns:
            BoundingBox of detected face, or None if no face found
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

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

        # Take first detected face
        box_dict = results[0]['box']
        return BoundingBox(
            x=max(0, box_dict[0]),
            y=max(0, box_dict[1]),
            width=box_dict[2],
            height=box_dict[3]
        )
