"""Factory for creating face embedder backends."""
from pathlib import Path
from typing import Optional, Union

from face_auth.core.cache.backend.impl.persistent_embedding_cache import (
    PersistentEmbeddingCache,
)
from face_auth.core.detection.backend.detector_factory import create_detector
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.backend.impl.facenet_backend import FaceNetBackend
from face_auth.core.embedder.backend.impl.insightface_backend import InsightFaceBackend
from face_auth.core.embedder.cached_embedder import CachedEmbedder
from face_auth.core.embedder.models import FaceNetEmbedderConfig, InsightFaceEmbedderConfig

EMBEDDER_REGISTRY = {
    "facenet": FaceNetBackend,
    "insightface": InsightFaceBackend,
}


def create_embedder(
    backend_name: str,
    backend_config: Union[FaceNetEmbedderConfig, InsightFaceEmbedderConfig],
    cache_enabled: bool,
    cache_dir: Optional[Path],
) -> EmbedderBackend:
    """Create a face embedder backend instance with all its dependencies.

    Args:
        backend_name: Name of the backend to create (e.g., "facenet", "insightface")
        backend_config: Typed configuration for the selected backend
        cache_enabled: Whether to wrap the embedder with a persistent embedding cache
        cache_dir: Directory where cache files are stored when caching is enabled

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

    # FaceNet backend uses the detector and requests its own input size.
    if backend_name == "facenet":
        detector = create_detector(backend_config.detector)

        embedder = FaceNetBackend(
            detector=detector,
            target_size=backend_config.target_size,
        )

    # InsightFace backend - handles detection internally
    else:
        embedder = InsightFaceBackend(
            model_name=backend_config.model_name,
            det_size=backend_config.det_size,
            min_detection_confidence=backend_config.min_detection_confidence,
        )

    if not cache_enabled:
        return embedder

    if cache_dir is None:
        raise ValueError("cache_dir is required when cache_enabled=True")

    embedding_cache = PersistentEmbeddingCache(
        cache_dir=cache_dir,
        embedder=embedder,
        embedder_config=embedder.get_config(),
    )
    return CachedEmbedder(
        embedder=embedder,
        embedding_cache=embedding_cache,
    )
