"""Face embedding generation."""
import numpy as np
from keras_facenet import FaceNet


class Embedder:
    """Generates face embeddings from preprocessed face images."""

    def __init__(self, model_name: str):
        """Initialize embedder with specified model.

        Args:
            model_name: Name of embedding model to use

        Raises:
            ValueError: If model_name is not supported
        """
        if model_name == "facenet":
            self._model = FaceNet()
        else:
            raise ValueError(
                f"Unsupported embedding model: {model_name}. "
                f"Supported models: ['facenet']"
            )

    def get_embedding(self, face_rgb: np.ndarray) -> np.ndarray:
        """Generate embedding vector for a face image.

        Args:
            face_rgb: Preprocessed face image in RGB format

        Returns:
            Embedding vector for the face
        """
        embeddings = self._model.embeddings([face_rgb])
        return embeddings[0]
