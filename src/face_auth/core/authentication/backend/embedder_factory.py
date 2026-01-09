"""Factory for creating face embedder backends."""
from typing import Dict, Any
from face_auth.core.authentication.backend import FaceNetBackend, InsightFaceBackend
from face_auth.core.authentication.backend.embedder_backend import EmbedderBackend

# Registry of available embedder backends
EMBEDDER_REGISTRY = {
    "facenet": FaceNetBackend,
    "insightface": InsightFaceBackend,
}


def create_embedder(backend_name: str, backend_config: Dict[str, Any] = None) -> EmbedderBackend:
    """Create a face embedder backend instance.

    Args:
        backend_name: Name of the backend to create (e.g., "facenet", "insightface", "arcface")
        backend_config: Optional configuration dict for the backend
                       For insightface:
                         - model_name: str (default: 'buffalo_l')
                         - det_size: tuple (default: (640, 640))

    Returns:
        Embedder backend instance

    Raises:
        ValueError: If backend_name is not recognized
    """
    if backend_name not in EMBEDDER_REGISTRY:
        supported = ", ".join(EMBEDDER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported embedder backend: {backend_name}. "
            f"Supported backends: [{supported}]"
        )

    backend_class = EMBEDDER_REGISTRY[backend_name]
    backend_config = backend_config or {}

    # Instantiate with config if provided
    return backend_class(**backend_config)