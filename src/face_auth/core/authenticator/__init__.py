"""Core authentication logic."""
from face_auth.core.authenticator.constants import FACENET_INPUT_WIDTH, FACENET_INPUT_HEIGHT
from face_auth.core.authenticator.frame_authenticator import FrameAuthenticator
from face_auth.core.authenticator.models import AuthenticationStatus, AuthenticationResult, FrameAuthenticationResult, FrameAuthenticationInput, SourceFrame
from face_auth.core.authenticator.temporal_logic.backend import AuthenticatorBackend

__all__ = [
    # Main components
    'FrameAuthenticator',
    'AuthenticatorBackend',
    # Models
    'AuthenticationStatus',
    'AuthenticationResult',
    'FrameAuthenticationInput',
    'SourceFrame',
    'FrameAuthenticationResult',
    # Constants
    'FACENET_INPUT_WIDTH',
    'FACENET_INPUT_HEIGHT',
]
