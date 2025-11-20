"""Core authentication logic."""

# Main components (high-level API)
from face_auth.core.embedder import Embedder
from face_auth.core.authenticator import ContinuousAuthenticator
from face_auth.core.frame_processor import FrameProcessor

# Models and enums
from face_auth.core.models import (
    AuthenticationState,
    FrameAuthenticationResult,
)

# Constants
from face_auth.core.constants import (
    FACENET_INPUT_WIDTH,
    FACENET_INPUT_HEIGHT,
)

# Low-level components (for advanced usage)
from face_auth.core.similarity_calculator import SimilarityCalculator
from face_auth.core.percentile_filter import PercentileFilter
from face_auth.core.temporal_window import TemporalWindow
from face_auth.core.risk_scorer import RiskScorer

__all__ = [
    # Main components
    'Embedder',
    'ContinuousAuthenticator',
    'FrameProcessor',
    # Models
    'AuthenticationState',
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
