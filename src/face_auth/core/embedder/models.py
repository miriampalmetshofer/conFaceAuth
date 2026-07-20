"""Data models for face embedding components."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass
class EmbeddingResult:
    """Frame-level face embedding result.

    A no-face result means the embedding provider did not produce an embedding,
    either because its internal detector found no face or because its delegated
    detector found no face.
    """
    embedding: Optional[np.ndarray]
    face_detected: bool

    @classmethod
    def no_face(cls) -> 'EmbeddingResult':
        """Create result for when no face was detected."""
        return cls(embedding=None, face_detected=False)

    @classmethod
    def success(cls, embedding: np.ndarray) -> 'EmbeddingResult':
        """Create result for successful embedding generation."""
        return cls(embedding=embedding, face_detected=True)
