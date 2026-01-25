"""Data models for temporal decay authenticator."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from face_auth.authentication.core.backend.authenticator_backend import AuthenticationState


@dataclass(frozen=True)
class TemporalDecayConfig:
    """Configuration for temporal decay authentication."""
    threshold: float
    similarity_percentile: float
    k_weight: float
    k_decay: float
    initial_confidence: float


@dataclass
class TemporalDecayState(AuthenticationState):
    """State for temporal decay authenticator."""
    confidence_score: float
    last_timestamp: Optional[datetime]
