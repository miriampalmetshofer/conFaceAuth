"""Data models for trust-based authenticator."""
from pydantic import BaseModel, Field, ConfigDict


class TrustBasedConfig(BaseModel):
    """Configuration for windowed trust-based authentication using cosine similarity."""
    threshold: float = Field(..., ge=-1.0, le=1.0)
    window_size: int = Field(..., gt=0)
    similarity_percentile: float = Field(..., ge=0, le=100)
    alpha: float = Field(..., ge=0, le=1)
    no_face_penalty: float = Field(..., ge=-1.0, le=1.0)

    model_config = ConfigDict(frozen=True)
