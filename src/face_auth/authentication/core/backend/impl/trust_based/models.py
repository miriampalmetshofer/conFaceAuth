"""Data models for risk-based authenticator."""
from dataclasses import dataclass
from face_auth.authentication.core.backend.authenticator_backend import AuthenticationState


@dataclass(frozen=True)
class TrustBasedConfig:
    """Configuration for windowed risk-based authentication."""
    threshold: float
    window_size: int
    similarity_percentile: float
    alpha: float
    no_face_penalty: float


@dataclass
class TrustBasedState(AuthenticationState):
    """State for risk-based authenticator."""
    similarity_window: list[float]
    trust_score: float
