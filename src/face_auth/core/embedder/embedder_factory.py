"""Factory for creating face embedder backends."""
from typing import Dict, Any
from face_auth.core.detection import FaceDetector, FaceExtractor
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.backend.impl.facenet_backend import FaceNetBackend
from face_auth.core.embedder.backend.impl.insightface_backend import InsightFaceBackend

EMBEDDER_REGISTRY = {
    "facenet": FaceNetBackend,
    "insightface": InsightFaceBackend,
}


def create_embedder(
    backend_name: str,
    backend_config: Dict[str, Any] = None
) -> EmbedderBackend:
    """Create a face embedder backend instance with all its dependencies.

    Args:
        backend_name: Name of the backend to create (e.g., "facenet", "insightface")
        backend_config: Optional configuration dict for the backend

    Returns:
        Embedder backend instance with all dependencies initialized

    Raises:
        ValueError: If backend_name is not recognized
    """
    if backend_name not in EMBEDDER_REGISTRY:
        supported = ", ".join(EMBEDDER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported embedder backend: {backend_name}. "
            f"Supported backends: [{supported}]"
        )

    backend_config = backend_config or {}

    # FaceNet backend needs a face detector and extractor
    if backend_name == "facenet":
        detector = FaceDetector(
            detector_backend=backend_config.get("detector", "mediapipe")
        )

        target_size = backend_config.get("target_size", [160, 160])
        extractor = FaceExtractor(
            target_width=target_size[0],
            target_height=target_size[1]
        )

        return FaceNetBackend(detector=detector, extractor=extractor)

    # InsightFace backend - handles detection internally
    else:
        return InsightFaceBackend(
            model_name=backend_config.get("model_name", "buffalo_sc"),
            det_size=tuple(backend_config.get("det_size", [640, 640])),
            min_detection_confidence=backend_config.get("min_detection_confidence", 0.5)
        )