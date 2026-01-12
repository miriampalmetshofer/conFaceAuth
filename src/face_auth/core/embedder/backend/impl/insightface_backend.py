"""InsightFace (ArcFace) embedder backend."""
import numpy as np
import cv2
import logging

from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.models import EmbeddingResult


class InsightFaceBackend(EmbedderBackend):
    """Face embedder using InsightFace ArcFace model with integrated detection."""

    def __init__(self, model_name: str = 'buffalo_sc', det_size: tuple = (640, 640), min_detection_confidence: float = 0.5):
        """Initialize InsightFace model.

        Args:
            model_name: Name of the InsightFace model pack to use
                       Options: 'buffalo_l' (ResNet50, default, best accuracy),
                               'buffalo_s' (smaller/faster),
                               'buffalo_sc' (smallest/fastest)
            det_size: Detection size for the model (width, height)
                     Larger = more accurate but slower
                     Default: (640, 640)
            min_detection_confidence: Minimum detection confidence threshold (0-1)
                                     Faces with confidence below this are treated as "no face"
                                     Default: 0.5
        """
        logging.getLogger('onnxruntime').setLevel(logging.ERROR)

        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface is not installed. "
                "Install it with: pip install insightface onnxruntime"
            )

        self.model_name = model_name
        self.det_size = det_size
        self.min_detection_confidence = min_detection_confidence

        # Initialize FaceAnalysis with detection + recognition
        self._app = FaceAnalysis(
            name=model_name,
            providers=['CPUExecutionProvider'],
            min_detection_confidence=self.min_detection_confidence,
        )
        self._app.prepare(ctx_id=-1, det_size=det_size)

    def get_embedding(self, frame_rgb: np.ndarray) -> EmbeddingResult:
        """Generate embedding vector using InsightFace.

        Args:
            frame_rgb: Full frame image in RGB format

        Returns:
            EmbeddingResult with 512-dimensional L2-normalized embedding or None if no face detected

        Note: InsightFace performs detection, alignment, and embedding in one pipeline.
              This is optimal as the model was trained with this specific alignment.
        """
        # InsightFace expects BGR format
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        faces = self._app.get(frame_bgr)

        if not faces or len(faces) == 0:
            return EmbeddingResult.no_face()

        face = faces[0]
        embedding = face.embedding

        normalized_embedding = self._l2_normalize_embeddings(embedding)

        return EmbeddingResult.success(normalized_embedding)

    def _l2_normalize_embeddings(self, embedding):
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding