"""Core authentication logic."""
from face_auth.core.authentication.continuous_authenticator import ContinuousAuthenticator
from face_auth.core.authentication.constants import FACENET_INPUT_WIDTH, FACENET_INPUT_HEIGHT
from face_auth.core.authentication.frame_authenticator import FrameAuthenticator
from face_auth.core.authentication.models import AuthenticationStatus, AuthenticationResult, FrameAuthenticationResult
from face_auth.core.authentication.percentile_filter import PercentileFilter
from face_auth.core.authentication.risk_scorer import RiskScorer
from face_auth.core.authentication.similarity_calculator import SimilarityCalculator
from face_auth.core.authentication.temporal_window import TemporalWindow

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

