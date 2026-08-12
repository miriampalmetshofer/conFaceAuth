"""FaceNet embedder backend."""
from typing import Any, Dict, Tuple

import numpy as np
import cv2
from keras_facenet import FaceNet
from face_auth.core.detection.backend.detector_backend import DetectorBackend
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.models import EmbeddingResult


class FaceNetBackend(EmbedderBackend):
    """Face embedder using FaceNet model with integrated face detection."""

    def __init__(
        self,
        detector: DetectorBackend,
        target_size: Tuple[int, int],
    ):
        """Initialize FaceNet model with face detection components.

        Args:
            detector: Face detector instance
            target_size: Width and height requested for FaceNet input
        """
        self._detector = detector
        self._target_size = target_size
        self._model = FaceNet()

    def embed(
        self,
        frame_rgb: np.ndarray,
    ) -> EmbeddingResult:
        """Generate embedding vector using FaceNet.

        Args:
            frame_rgb: Full frame image in RGB format

        Returns:
            EmbeddingResult with 512-dimensional embedding vector or None if no face detected
        """
        detection_result = self._detector.detect(frame_rgb)

        if detection_result is None:
            return EmbeddingResult.no_face()

        face_resized = cv2.resize(detection_result.face_image, self._target_size)

        embeddings = self._model.embeddings([face_resized])
        return EmbeddingResult.success(embeddings[0], detection_result.bounding_box)

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serializable description of this embedder configuration."""
        return {
            "backend": "facenet",
            "config": {
                "detector": self._detector.backend_name,
                "target_size": list(self._target_size),
            },
        }
