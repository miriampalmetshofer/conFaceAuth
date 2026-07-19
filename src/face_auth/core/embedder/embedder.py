"""Face embedding generation."""
from typing import Dict, Any
import numpy as np

from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.embedder_factory import create_embedder
from face_auth.core.embedder.models import EmbeddingResult

class Embedder:
    """Generates face embeddings from frames."""

    def __init__(
        self,
        model_name: str,
        model_config: Dict[str, Any] = None
    ):
        """Initialize embedder with specified model.

        Args:
            model_name: Name of embedding model to use ('facenet', 'insightface')
            model_config: Optional configuration dict for the model backend
                         For FaceNet:
                           - detector: str (detector backend name, default: 'mediapipe')
                           - target_size: list[int, int] (default: [160, 160])
                         For InsightFace:
                           - model_name: str (default: 'buffalo_sc')
                           - det_size: list[int, int] (default: [640, 640])
                           - min_detection_confidence: float (default: 0.5)

        Raises:
            ValueError: If model_name is not supported
        """
        self._backend: EmbedderBackend = create_embedder(model_name, model_config)

    def get_embedding(self, frame_rgb: np.ndarray) -> EmbeddingResult:
        """Generate embedding vector from a frame.

        Args:
            frame_rgb: Full frame image in RGB format

        Returns:
            EmbeddingResult with embedding vector and face detection status
        """
        return self._backend.get_embedding(frame_rgb)
