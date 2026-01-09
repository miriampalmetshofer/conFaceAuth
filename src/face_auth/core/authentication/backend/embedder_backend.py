"""Face embedder backend interface."""
from typing import Protocol
import numpy as np


class EmbedderBackend(Protocol):
    """Interface that all face embedder backends must implement."""

    def get_embedding(self, face_rgb: np.ndarray) -> np.ndarray:
        """Generate embedding vector for a face image.

        Args:
            face_rgb: Preprocessed face image in RGB format

        Returns:
            Embedding vector for the face
        """
        ...