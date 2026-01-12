"""Core authentication logic."""
from face_auth.authentication.core.continuous_authenticator import ContinuousAuthenticator
from face_auth.authentication.core.constants import FACENET_INPUT_WIDTH, FACENET_INPUT_HEIGHT
from face_auth.authentication.core.frame_authenticator import FrameAuthenticator
from face_auth.authentication.core.models import AuthenticationStatus, AuthenticationResult, FrameAuthenticationResult
from face_auth.authentication.core.percentile_filter import PercentileFilter
from face_auth.authentication.core.risk_scorer import RiskScorer
from face_auth.authentication.core.similarity_calculator import SimilarityCalculator
from face_auth.authentication.core.temporal_window import TemporalWindow

__all__ = [
    # Main components
    'ContinuousAuthenticator',
    'FrameAuthenticator',
    # Models
    'AuthenticationStatus',
    'AuthenticationResult',
    'FrameAuthenticationResult',
    # Constants
    'FACENET_INPUT_WIDTH',
    'FACENET_INPUT_HEIGHT',
    # Low-level components
    'SimilarityCalculator',
    'PercentileFilter',
    'TemporalWindow',
    'RiskScorer',
]

