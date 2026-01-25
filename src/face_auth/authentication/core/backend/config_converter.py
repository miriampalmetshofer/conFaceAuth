"""Converts pydantic config models to backend dataclass configs."""
from face_auth.config.models import TrustBasedConfigModel
from face_auth.authentication.core.backend.impl.trust_based.models import TrustBasedConfig


def convert_trust_based_config(model: TrustBasedConfigModel) -> TrustBasedConfig:
    """Convert pydantic model to RiskBasedConfig dataclass."""
    return TrustBasedConfig(
        threshold=model.threshold,
        window_size=model.window_size,
        similarity_percentile=model.similarity_percentile,
        alpha=model.alpha,
        no_face_penalty=model.no_face_penalty
    )

