"""Factory for creating face embedder backends."""
from pathlib import Path
from typing import Optional

from face_auth.core.embedder.cache.backend.impl.persistent_embedding_cache import (
    PersistentEmbeddingCache,
)
from face_auth.core.detection.backend.detector_factory import create_detector
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.backend.impl.facenet_backend import FaceNetBackend
from face_auth.core.embedder.backend.impl.insightface_backend import InsightFaceBackend
from face_auth.core.embedder.cached_embedder import CachedEmbedder
from face_auth.config import EmbedderModelConfig


def create_embedder(
    config: EmbedderModelConfig,
    cache_enabled: bool,
    cache_dir: Optional[Path],
) -> EmbedderBackend:
    """Create a face embedder backend instance with all its dependencies.

    Args:
        config: Embedder model configuration selecting the model and its settings
        cache_enabled: Whether to wrap the embedder with a persistent embedding cache
        cache_dir: Directory where cache files are stored when caching is enabled

    Returns:
        Embedder backend instance with all dependencies initialized

    Raises:
        ValueError: If the configured model is not recognized
    """
    # FaceNet backend uses the detector and requests its own input size.
    if config.type == "facenet":
        detector = create_detector(config.detector)

        embedder = FaceNetBackend(
            detector=detector,
            target_size=config.target_size,
        )

    # InsightFace backend - handles detection internally
    elif config.type == "insightface":
        embedder = InsightFaceBackend(
            model_name=config.model_name,
            det_size=config.det_size,
            min_detection_confidence=config.min_detection_confidence,
        )

    else:
        raise ValueError(f"Unsupported embedder model: {config.type}")

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
