"""Data models for temporal decay authenticator."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalDecayConfig:
    """Configuration for temporal decay authentication."""
    threshold: float
    similarity_percentile: float
    k_weight: float
    k_decay: float
    initial_confidence: float
