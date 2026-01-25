"""Converts pydantic config models to backend dataclass configs."""
from face_auth.config.models import TrustBasedConfigModel, TemporalDecayConfigModel
from face_auth.authentication.core.backend.impl.trust_based.models import TrustBasedConfig
from face_auth.authentication.core.backend.impl.temporal_decay.models import TemporalDecayConfig


def convert_trust_based_config(model: TrustBasedConfigModel) -> TrustBasedConfig:
    """Convert pydantic model to RiskBasedConfig dataclass."""
    return TrustBasedConfig(
        threshold=model.threshold,
        window_size=model.window_size,
        similarity_percentile=model.similarity_percentile,
        alpha=model.alpha,
        no_face_penalty=model.no_face_penalty
    )


def convert_temporal_decay_config(model: TemporalDecayConfigModel) -> TemporalDecayConfig:
    """Convert pydantic model to TemporalDecayConfig dataclass."""
    return TemporalDecayConfig(
        threshold=model.threshold,
        similarity_percentile=model.similarity_percentile,
        k_weight=model.k_weight,
        k_decay=model.k_decay,
        initial_confidence=model.initial_confidence
    )

