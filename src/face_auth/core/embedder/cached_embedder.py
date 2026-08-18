"""Optional embedder wrapper that hides cache access behind the embedder interface."""
from typing import Any, Dict

import numpy as np

from face_auth.core.embedder.cache.backend.cache_backend import EmbeddingCache
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.models import EmbeddingResult


class CachedEmbedder(EmbedderBackend):
    """Wraps an embedder with optional persistent caching."""

    def __init__(
        self,
        embedder: EmbedderBackend,
        embedding_cache: EmbeddingCache,
    ):
        """Initialize cached embedder.

        Args:
            embedder: Embedder used when no cache entry exists
            embedding_cache: Cache for frame embeddings
        """
        self.embedder = embedder
        self.embedding_cache = embedding_cache

    def embed(
        self,
        frame_rgb: np.ndarray,
    ) -> EmbeddingResult:
        """Return an embedding result, using a content-addressed cache."""
        return self.embedding_cache.get_or_compute(frame_rgb)

    def get_config(self) -> Dict[str, Any]:
        """Return the wrapped embedder's configuration."""
        return self.embedder.get_config()
