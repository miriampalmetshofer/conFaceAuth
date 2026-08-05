"""Factory for creating face embedder backends."""
from pathlib import Path
from typing import Optional

from face_auth.core.cache.backend.impl.persistent_embedding_cache import (
    PersistentEmbeddingCache,
)
from face_auth.core.detection.backend.detector_factory import create_detector
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.backend.impl.facenet_backend import FaceNetBackend
from face_auth.core.embedder.backend.impl.insightface_backend import InsightFaceBackend
from face_auth.core.embedder.cached_embedder import CachedEmbedder
from face_auth.config.models import EmbedderConfig


def create_embedder(
    config: EmbedderConfig,
    cache_enabled: bool,
    cache_dir: Optional[Path],
) -> EmbedderBackend:
    """Create a face embedder backend instance with all its dependencies.

    Args:
        config: Embedder configuration selecting the model and its settings
        cache_enabled: Whether to wrap the embedder with a persistent embedding cache
        cache_dir: Directory where cache files are stored when caching is enabled

    Returns:
        Embedder backend instance with all dependencies initialized

    Raises:
        ValueError: If the configured model is not recognized
    """
    backend_config = config.backend_config

    # FaceNet backend uses the detector and requests its own input size.
    if config.model == "facenet":
        detector = create_detector(backend_config.detector)

        embedder = FaceNetBackend(
            detector=detector,
            target_size=backend_config.target_size,
        )

    # InsightFace backend - handles detection internally
    elif config.model == "insightface":
        embedder = InsightFaceBackend(
            model_name=backend_config.model_name,
            det_size=backend_config.det_size,
            min_detection_confidence=backend_config.min_detection_confidence,
        )

    else:
        raise ValueError(f"Unsupported embedder model: {config.model}")

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
