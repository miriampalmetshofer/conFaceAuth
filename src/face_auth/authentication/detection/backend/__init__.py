"""Face detector backend implementations."""
from face_auth.authentication.detection.backend.impl.mtcnn_backend import MTCNNBackend
from face_auth.authentication.detection.backend.impl.mediapipe_backend import MediaPipeBackend

__all__ = [
    'MTCNNBackend',
    'MediaPipeBackend',
]
