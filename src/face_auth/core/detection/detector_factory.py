"""Factory for creating face detector backend."""
from face_auth.core.detection.backend import MTCNNBackend, MediaPipeBackend
from face_auth.core.detection.backend.detector_backend import DetectorBackend

# Registry of available detector backend
DETECTOR_REGISTRY = {
    "mtcnn": MTCNNBackend,
    "mediapipe": MediaPipeBackend,
}


def create_detector(backend_name: str) -> DetectorBackend:
    """Create a face detector backend instance.

    Args:
        backend_name: Name of the backend to create (e.g., "mtcnn", "mediapipe")

    Returns:
        Detector backend instance

    Raises:
        ValueError: If backend_name is not recognized
    """
    if backend_name not in DETECTOR_REGISTRY:
        supported = ", ".join(DETECTOR_REGISTRY.keys())
        raise ValueError(
            f"Unsupported detector backend: {backend_name}. "
            f"Supported backend: [{supported}]"
        )

    backend_class = DETECTOR_REGISTRY[backend_name]
    return backend_class()
