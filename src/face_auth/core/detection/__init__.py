"""Face detection and extraction components."""

from face_auth.core.detection.face_detector import FaceDetector
from face_auth.core.detection.face_extractor import FaceExtractor
from face_auth.core.detection.models import BoundingBox, DetectionResult

__all__ = [
    'FaceDetector',
    'FaceExtractor',
    'BoundingBox',
    'DetectionResult',
]
