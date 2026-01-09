"""Face embedding generation."""
from typing import Dict, Any
import numpy as np
from face_auth.core.authentication.backend.embedder_backend import EmbedderBackend
from face_auth.core.authentication.backend.embedder_factory import create_embedder


class Embedder:
    """Generates face embeddings from preprocessed face images."""

    def __init__(self, model_name: str, model_config: Dict[str, Any] = None):
        """Initialize embedder with specified model.

        Args:
            model_name: Name of embedding model to use ('facenet', 'insightface', 'arcface')
            model_config: Optional configuration dict for the model backend
                         For insightface/arcface:
                           - model_name: str (default: 'buffalo_l')
                           - det_size: tuple (default: (640, 640))

        Raises:
            ValueError: If model_name is not supported
        """
        self._backend: EmbedderBackend = create_embedder(model_name, model_config)

    def get_embedding(self, face_rgb: np.ndarray) -> np.ndarray:
        """Generate embedding vector for a face image.

        Args:
            face_rgb: Preprocessed face image in RGB format

        Returns:
            Embedding vector for the face
        """
        return self._backend.get_embedding(face_rgb)
