"""Persistent content-addressed embedding cache."""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from face_auth.logging_config import get_logger
from face_auth.core.embedder.cache.backend.cache_backend import EmbeddingCache
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.models import EmbeddingResult

logger = get_logger(__name__)


class PersistentEmbeddingCache(EmbeddingCache):
    """Caches expensive face embeddings by frame content.

    Each cached frame is stored as its own small ``.npz`` shard named after the
    frame's content fingerprint, under a directory keyed by the embedder
    configuration. Reading a frame loads only its own shard and writing one adds
    a single file, so cost stays flat as the cache grows and concurrent workers
    never block on a shared file. A shard's name is a pure function of its
    content, so two workers computing the same frame simply write identical
    bytes to the same path.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        cache_dir: Path,
        embedder: EmbedderBackend,
        embedder_config: Dict[str, Any],
    ):
        """Initialize the embedding cache.

        Args:
            cache_dir: Directory where cache files are stored
            embedder: Embedding provider used for cache misses
            embedder_config: JSON-serializable configuration identity for cached embeddings
        """
        self.embedder = embedder
        self.embedder_config = embedder_config
        self._reported_cache_miss = False
        self._shard_dir = cache_dir / self._config_fingerprint()
        self._shard_dir.mkdir(parents=True, exist_ok=True)

    def get_or_compute(
        self,
        frame_rgb: np.ndarray,
    ) -> EmbeddingResult:
        """Return cached embedding for a frame or compute and persist it."""
        frame_fingerprint = self._frame_fingerprint(frame_rgb)
        shard_path = self._shard_path(frame_fingerprint)

        cached = self._load_shard(shard_path)
        if cached is not None:
            logger.debug(f"Embedding cache hit for frame {frame_fingerprint}")
            return cached

        if not self._reported_cache_miss:
            logger.info("Embedding cache miss; computing missing frames")
            self._reported_cache_miss = True

        computed = self.embedder.embed(frame_rgb)
        self._save_shard(shard_path, computed)
        return computed

    def _shard_path(self, frame_fingerprint: str) -> Path:
        return self._shard_dir / f"{frame_fingerprint}.npz"

    def _config_fingerprint(self) -> str:
        manifest = {
            "cache_schema_version": self.SCHEMA_VERSION,
            "cache_type": "frame_content_embeddings",
            "embedder": self.embedder_config,
        }
        return self._fingerprint(manifest)

    def _fingerprint(self, payload: Dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def _frame_fingerprint(self, frame_rgb: np.ndarray) -> str:
        payload = {
            "shape": list(frame_rgb.shape),
            "dtype": str(frame_rgb.dtype),
            "content_hash": hashlib.sha256(frame_rgb.tobytes()).hexdigest(),
        }
        return self._fingerprint(payload)

    def _load_shard(self, shard_path: Path) -> EmbeddingResult | None:
        if not shard_path.exists():
            return None

        try:
            with np.load(shard_path, allow_pickle=False) as data:
                if not bool(data["face_detected"]):
                    return EmbeddingResult.no_face()
                # The cache stores only the embedding, not the face location.
                # Batch processing never draws overlays, and the live path runs
                # uncached, so a cached hit legitimately has no bounding box.
                return EmbeddingResult.success(
                    data["embedding"].copy(),
                    bounding_box=None,
                )
        except Exception as exc:
            logger.warning(f"Ignoring unreadable embedding cache shard {shard_path}: {exc}")
            return None

    def _save_shard(self, shard_path: Path, result: EmbeddingResult) -> None:
        if result.face_detected and result.embedding is not None:
            embedding = result.embedding.astype(np.float32)
        else:
            embedding = np.zeros(0, dtype=np.float32)

        temp_path = shard_path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(
            temp_path,
            face_detected=np.array(result.face_detected, dtype=bool),
            embedding=embedding,
        )
        os.replace(temp_path, shard_path)
