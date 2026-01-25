"""Factory for creating authenticator backends."""
from enum import Enum
import numpy as np

from face_auth.authentication.core.backend.authenticator_backend import AuthenticatorBackend
from face_auth.authentication.core.backend.impl.trust_based.models import TrustBasedConfig


class AuthenticatorBackendType(Enum):
    """Available authenticator backend types."""
    TRUST_BASED = "trust_based"


def create_authenticator(
    backend_type: AuthenticatorBackendType,
    config: TrustBasedConfig,
    enrollment_embeddings: list[np.ndarray]
) -> AuthenticatorBackend:
    """Create an authenticator backend based on type and configuration.

    Args:
        backend_type: Type of authenticator to create
        config: Backend-specific configuration
        enrollment_embeddings: Reference embeddings for enrolled user

    Returns:
        Configured authenticator backend instance

    Raises:
        ValueError: If backend_type is not supported
    """
    if backend_type == AuthenticatorBackendType.TRUST_BASED:
        from face_auth.authentication.core.backend.impl.trust_based.trust_based_authenticator import (
            RiskBasedAuthenticator
        )
        return RiskBasedAuthenticator(config, enrollment_embeddings)
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")
