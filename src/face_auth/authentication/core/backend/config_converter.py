"""Converts pydantic config models to backend dataclass configs."""
from face_auth.config.models import RiskBasedConfigModel
from face_auth.authentication.core.backend.impl.risk_based.models import RiskBasedConfig


def convert_risk_based_config(model: RiskBasedConfigModel) -> RiskBasedConfig:
    """Convert pydantic model to RiskBasedConfig dataclass."""
    return RiskBasedConfig(
        threshold=model.threshold,
        window_size=model.window_size,
        similarity_percentile=model.similarity_percentile,
        alpha=model.alpha,
        no_face_penalty=model.no_face_penalty
    )
