"""Configuration for the authentication feature."""

from typing import Annotated, Union

from pydantic import BaseModel, Field, ConfigDict

from face_auth.core.authenticator.backend.impl.temporal_decay.models import TemporalDecayConfig
from face_auth.core.authenticator.backend.impl.trust_based.models import TrustBasedConfig

AuthenticationBackendConfig = Annotated[
    Union[TrustBasedConfig, TemporalDecayConfig],
    Field(discriminator="type"),
]


class AuthenticationConfig(BaseModel):
    """Authentication parameters configuration."""

    backend: AuthenticationBackendConfig
    authentication_hz: float = Field(
        ...,
        gt=0,
        description="How many times per second a frame is authenticated (1.0 = once per second)",
    )

    model_config = ConfigDict(frozen=True)
