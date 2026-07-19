"""Face embedder backend interface."""
from abc import ABC, abstractmethod
import numpy as np

from face_auth.core.embedder.models import EmbeddingResult


class EmbedderBackend(ABC):
    """Abstract base class that all face embedder backends must implement.

    All backends handle their own face detection and preprocessing internally.
    """

    @abstractmethod
    def get_embedding(self, frame_rgb: np.ndarray) -> EmbeddingResult:
        """Generate embedding vector from a frame.

        Args:
            frame_rgb: Full frame image in RGB format

        Returns:
            EmbeddingResult with embedding vector and face detection status
        """
        pass