"""Data models for temporal decay authenticator."""
from pydantic import BaseModel, Field, ConfigDict


class TemporalDecayConfig(BaseModel):
    """Configuration for temporal decay authentication using time-weighted confidence."""
    threshold: float = Field(..., ge=-1.0, le=1.0)
    similarity_percentile: float = Field(..., ge=0, le=100)
    k_weight: float = Field(..., gt=0)
    k_decay: float = Field(..., gt=0)
    initial_confidence: float = Field(..., ge=-1.0, le=1.0)

    model_config = ConfigDict(frozen=True)
