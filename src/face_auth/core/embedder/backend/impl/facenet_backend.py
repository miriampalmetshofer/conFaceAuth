"""FaceNet embedder backend."""
import numpy as np
import cv2
from keras_facenet import FaceNet
from face_auth.core.detection import FaceDetector, FaceExtractor
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.models import EmbeddingResult


class FaceNetBackend(EmbedderBackend):
    """Face embedder using FaceNet model with integrated face detection."""

    def __init__(self, detector: FaceDetector, extractor: FaceExtractor):
        """Initialize FaceNet model with face detection components.

        Args:
            detector: Face detector instance
            extractor: Face extractor instance
        """
        self._detector = detector
        self._extractor = extractor
        self._model = FaceNet()
        # FaceNet expects 160x160 input
        self._target_size = (160, 160)

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
        detection_result = self._extractor.detect_and_extract(frame_bgr, self._detector)

        if detection_result is None:
            return EmbeddingResult.no_face()

        # Crop face region using detection bbox
        face_region = frame_rgb[
            detection_result.bounding_box.y:detection_result.bounding_box.y + detection_result.bounding_box.height,
            detection_result.bounding_box.x:detection_result.bounding_box.x + detection_result.bounding_box.width
        ]

        # Resize to FaceNet input size
        face_resized = cv2.resize(face_region, self._target_size)

        # Generate embedding
        embeddings = self._model.embeddings([face_resized])
        return EmbeddingResult.success(embeddings[0])