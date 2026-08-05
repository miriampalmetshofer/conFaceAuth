"""Factory for creating authenticator backends."""
import numpy as np

from face_auth.core.authenticator.backend.authenticator_backend import AuthenticatorBackend
from face_auth.config.models import AuthenticationConfig


def create_authenticator(
    config: AuthenticationConfig,
    enrollment_embeddings: list[np.ndarray]
) -> AuthenticatorBackend:
    """Create the configured authenticator backend.

    Args:
        config: Authentication configuration selecting the backend and its settings
        enrollment_embeddings: Reference embeddings for enrolled user

    Returns:
        Configured authenticator backend instance

    Raises:
        ValueError: If the configured backend is not supported
    """
    if config.backend == "trust_based":
        from face_auth.core.authenticator.backend.impl.trust_based.trust_based_authenticator import (
            RiskBasedAuthenticator
        )
        return RiskBasedAuthenticator(config.backend_config, enrollment_embeddings)

    if config.backend == "temporal_decay":
        from face_auth.core.authenticator.backend.impl.temporal_decay.temporal_decay_authenticator import (
            TemporalDecayAuthenticator
        )
        return TemporalDecayAuthenticator(config.backend_config, enrollment_embeddings)

    raise ValueError(f"Unsupported authentication backend: {config.backend}")
