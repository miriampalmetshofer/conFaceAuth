"""Core authentication logic."""
from face_auth.authentication.core.constants import FACENET_INPUT_WIDTH, FACENET_INPUT_HEIGHT
from face_auth.authentication.core.frame_authenticator import FrameAuthenticator
from face_auth.authentication.core.models import AuthenticationStatus, AuthenticationResult, FrameAuthenticationResult
from face_auth.authentication.core.backend.authenticator_backend import AuthenticatorBackend

__all__ = [
    # Main components
    'FrameAuthenticator',
    'AuthenticatorBackend',
    # Models
    'AuthenticationStatus',
    'AuthenticationResult',
    'FrameAuthenticationResult',
    # Constants
    'FACENET_INPUT_WIDTH',
    'FACENET_INPUT_HEIGHT',
]

