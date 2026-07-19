"""FaceNet embedder backend."""
from typing import Tuple
import numpy as np
import cv2
from keras_facenet import FaceNet
from face_auth.core.detection.face_detector import FaceDetector
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.models import EmbeddingResult


class FaceNetBackend(EmbedderBackend):
    """Face embedder using FaceNet model with integrated face detection."""

    def __init__(self, detector: FaceDetector, target_size: Tuple[int, int]):
        """Initialize FaceNet model with face detection components.

        Args:
            detector: Face detector instance
            target_size: Width and height requested for FaceNet input
        """
        self._detector = detector
        self._target_size = target_size
        self._model = FaceNet()

    def get_embedding(self, frame_rgb: np.ndarray) -> EmbeddingResult:
        """Generate embedding vector using FaceNet.

        Args:
            frame_rgb: Full frame image in RGB format

        Returns:
            EmbeddingResult with 512-dimensional embedding vector or None if no face detected
        """
        # Convert RGB to BGR for detector
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Detect and extract face
        detection_result = self._detector.detect(frame_bgr)

        if detection_result is None:
            return EmbeddingResult.no_face()

        face_resized = cv2.resize(detection_result.face_image, self._target_size)

        # Generate embedding
        embeddings = self._model.embeddings([face_resized])
        return EmbeddingResult.success(embeddings[0])
