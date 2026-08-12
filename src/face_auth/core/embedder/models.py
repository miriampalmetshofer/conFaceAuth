"""Data models for face embedding components."""
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from face_auth.core.detection.models import BoundingBox


class FaceNetEmbedderConfig(BaseModel):
    """Configuration for the FaceNet embedder backend."""
    type: Literal["facenet"]
    detector: str = Field(..., min_length=1)
    target_size: Tuple[int, int] = Field(...)

    model_config = ConfigDict(frozen=True)


class InsightFaceEmbedderConfig(BaseModel):
    """Configuration for the InsightFace embedder backend."""
    type: Literal["insightface"]
    model_name: str = Field(..., min_length=1)
    det_size: Tuple[int, int] = Field(...)
    min_detection_confidence: float = Field(..., ge=0, le=1)

    model_config = ConfigDict(frozen=True)


@dataclass
class EmbeddingResult:
    """Frame-level face embedding result.

    A no-face result means the embedding provider did not produce an embedding,
    either because its internal detector found no face or because its delegated
    detector found no face.

    ``bounding_box`` locates the face the embedding was computed from, so callers
    that want to visualize the detection do not need to run detection a second time.
    """
    embedding: Optional[np.ndarray]
    face_detected: bool
    bounding_box: Optional[BoundingBox] = None

    @classmethod
    def no_face(cls) -> 'EmbeddingResult':
        """Create result for when no face was detected."""
        return cls(embedding=None, face_detected=False, bounding_box=None)

    @classmethod
    def success(cls, embedding: np.ndarray, bounding_box: BoundingBox) -> 'EmbeddingResult':
        """Create result for successful embedding generation."""
        return cls(embedding=embedding, face_detected=True, bounding_box=bounding_box)
