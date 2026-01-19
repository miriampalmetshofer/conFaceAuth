"""Factory for creating authenticator backends."""
from enum import Enum
import numpy as np

from face_auth.authentication.core.backend.authenticator_backend import AuthenticatorBackend
from face_auth.authentication.core.backend.impl.risk_based.models import RiskBasedConfig


class AuthenticatorBackendType(Enum):
    """Available authenticator backend types."""
    RISK_BASED = "risk_based"


def create_authenticator(
    backend_type: AuthenticatorBackendType,
    config: RiskBasedConfig,
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
    if backend_type == AuthenticatorBackendType.RISK_BASED:
        from face_auth.authentication.core.backend.impl.risk_based.risk_based_authenticator import (
            RiskBasedAuthenticator
        )
        return RiskBasedAuthenticator(config, enrollment_embeddings)
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")
