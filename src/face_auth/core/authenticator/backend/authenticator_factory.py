"""Factory for creating authenticator backends."""
import numpy as np

from face_auth.core.authenticator.backend.authenticator_backend import AuthenticatorBackend
from face_auth.authentication.config import AuthenticationBackendConfig


def create_authenticator(
    backend: AuthenticationBackendConfig,
    enrollment_embeddings: list[np.ndarray]
) -> AuthenticatorBackend:
    """Create the configured authenticator backend.

    Takes the backend config directly rather than the full authentication config,
    because building an authenticator does not depend on the sampling rate. This
    lets the live demo, which authenticates every camera frame, build one without
    supplying a rate it never uses.

    Args:
        backend: Authentication backend configuration selecting the backend and its settings
        enrollment_embeddings: Reference embeddings for enrolled user

    Returns:
        Configured authenticator backend instance

    Raises:
        ValueError: If the configured backend is not supported
    """
    if backend.type == "trust_based":
        from face_auth.core.authenticator.backend.impl.trust_based.trust_based_authenticator import (
            RiskBasedAuthenticator
        )
        return RiskBasedAuthenticator(backend, enrollment_embeddings)

    if backend.type == "temporal_decay":
        from face_auth.core.authenticator.backend.impl.temporal_decay.temporal_decay_authenticator import (
            TemporalDecayAuthenticator
        )
        return TemporalDecayAuthenticator(backend, enrollment_embeddings)

    raise ValueError(f"Unsupported authentication backend: {backend.type}")
