"""Data models for risk-based authenticator."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrustBasedConfig:
    """Configuration for windowed risk-based authentication."""
    threshold: float
    window_size: int
    similarity_percentile: float
    alpha: float
    no_face_penalty: float
