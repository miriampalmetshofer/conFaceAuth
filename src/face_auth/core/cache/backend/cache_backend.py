"""Abstract base class for embedding caches."""
from abc import ABC, abstractmethod
import numpy as np

from face_auth.core.embedder.models import EmbeddingResult


class EmbeddingCache(ABC):
    """Abstract interface for embedding caches."""

    @abstractmethod
    def get_or_compute(
        self,
        frame_rgb: np.ndarray,
    ) -> EmbeddingResult:
        """Return a cached embedding result or compute one."""
        pass
