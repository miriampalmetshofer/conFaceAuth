"""Face detection and extraction components."""

from face_auth.core.detection.face_detector import FaceDetector
from face_auth.core.detection.face_extractor import FaceExtractor
from face_auth.core.detection.models import BoundingBox, DetectionResult
from face_auth.core.detection.backend.detector_backend import DetectorBackend
from face_auth.core.detection.detector_factory import create_detector, DETECTOR_REGISTRY

__all__ = [
    'FaceDetector',
    'FaceExtractor',
    'BoundingBox',
    'DetectionResult',
    'DetectorBackend',
    'create_detector',
    'DETECTOR_REGISTRY',
]
