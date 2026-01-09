"""FaceNet embedder backend."""
import numpy as np
from keras_facenet import FaceNet


class FaceNetBackend:
    """Face embedder using FaceNet model."""

    def __init__(self):
        """Initialize FaceNet model."""
        self._model = FaceNet()

    def get_embedding(self, face_rgb: np.ndarray) -> np.ndarray:
        """Generate embedding vector using FaceNet.

        Args:
            face_rgb: Preprocessed face image in RGB format (160x160)

        Returns:
            512-dimensional embedding vector
        """
        embeddings = self._model.embeddings([face_rgb])
        return embeddings[0]