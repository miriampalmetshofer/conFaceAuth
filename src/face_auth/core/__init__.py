"""Core authentication logic."""
from face_auth.core.authenticator import ContinuousAuthenticator
from face_auth.core.detector import FaceDetector
from face_auth.core.embedder import Embedder
from face_auth.core.frame_processor import FrameProcessor, FrameAuthenticationResult

__all__ = [
    'ContinuousAuthenticator',
    'FaceDetector',
    'Embedder',
    'FrameProcessor',
    'FrameAuthenticationResult',
]
