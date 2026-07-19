"""Data models for face authentication core components."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass
class EmbeddingResult:
    """Result of embedding generation, explicitly handling no-face cases."""
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