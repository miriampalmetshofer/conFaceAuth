"""Data models for risk-based authenticator."""
from dataclasses import dataclass
from face_auth.authentication.core.backend.authenticator_backend import AuthenticationState


@dataclass(frozen=True)
class RiskBasedConfig:
    """Configuration for windowed risk-based authentication."""
    threshold: float
    window_size: int
    similarity_percentile: float
    alpha: float
    no_face_penalty: float


@dataclass
class RiskBasedState(AuthenticationState):
    """State for risk-based authenticator."""
    distance_window: list[float]
    risk_score: float
