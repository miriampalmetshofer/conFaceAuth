"""Persistent content-addressed embedding cache."""
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import numpy as np

from face_auth.logging_config import get_logger
from face_auth.core.cache.backend.cache_backend import EmbeddingCache
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.models import EmbeddingResult

logger = get_logger(__name__)


class PersistentEmbeddingCache(EmbeddingCache):
    """Caches expensive face embeddings by frame content."""

    SCHEMA_VERSION = 1
    LOCK_STALE_SECONDS = 300

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
        self.cache_dir = cache_dir
        self.embedder = embedder
        self.embedder_config = embedder_config
        self._reported_cache_miss = False
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_or_compute(
        self,
        frame_rgb: np.ndarray,
    ) -> EmbeddingResult:
        """Return cached embedding for a frame or compute and persist it."""
        manifest = self._build_manifest()
        cache_path = self._cache_path(manifest)
        frame_fingerprint = self._frame_fingerprint(frame_rgb)

        with self._file_lock(cache_path):
            records = self._load_records(cache_path)
            cached = records.get(frame_fingerprint)
            if cached is not None:
                logger.debug(f"Embedding cache hit for frame {frame_fingerprint}")
                return cached

        if not self._reported_cache_miss:
            logger.info("Embedding cache miss; computing missing frames")
            self._reported_cache_miss = True

        computed = self.embedder.embed(frame_rgb)

        with self._file_lock(cache_path):
            records = self._load_records(cache_path)
            # Another worker may have written the same frame while this worker computed it.
            cached = records.get(frame_fingerprint)
            if cached is not None:
                return cached
            records[frame_fingerprint] = computed
            self._save_records(cache_path, manifest, records)

        return computed

    def _cache_path(
        self,
        manifest: Dict[str, Any],
    ) -> Path:
        cache_key = self._fingerprint(manifest)
        return self.cache_dir / f"embeddings_{cache_key}.npz"

    def _build_manifest(self) -> Dict[str, Any]:
        return {
            "cache_schema_version": self.SCHEMA_VERSION,
            "cache_type": "frame_content_embeddings",
            "embedder": self.embedder_config,
        }

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

    def _load_records(self, cache_path: Path) -> Dict[str, EmbeddingResult]:
        if not cache_path.exists():
            return {}

        try:
            with np.load(cache_path, allow_pickle=False) as data:
                frame_fingerprints = data["frame_fingerprints"].astype(str)
                face_detected = data["face_detected"].astype(bool)
                embeddings = data["embeddings"]
                records = {}

                for row_index, frame_fingerprint in enumerate(frame_fingerprints):
                    if face_detected[row_index]:
                        # The cache stores only the embedding, not the face location.
                        # Batch processing never draws overlays, and the live path runs
                        # uncached, so a cached hit legitimately has no bounding box.
                        records[str(frame_fingerprint)] = EmbeddingResult.success(
                            embeddings[row_index].copy(),
                            bounding_box=None,
                        )
                    else:
                        records[str(frame_fingerprint)] = EmbeddingResult.no_face()

                return records
        except Exception as exc:
            logger.warning(f"Ignoring unreadable embedding cache {cache_path}: {exc}")
            return {}

    def _save_records(
        self,
        cache_path: Path,
        manifest: Dict[str, Any],
        records: Dict[str, EmbeddingResult],
    ) -> None:
        frame_fingerprints = sorted(records)
        detected_embeddings = [
            records[index].embedding
            for index in frame_fingerprints
            if records[index].face_detected and records[index].embedding is not None
        ]
        embedding_dim = len(detected_embeddings[0]) if detected_embeddings else 0

        frame_fingerprint_values = np.array(frame_fingerprints, dtype=str)
        face_detected = np.array(
            [records[index].face_detected for index in frame_fingerprints],
            dtype=bool,
        )
        embeddings = np.zeros((len(frame_fingerprints), embedding_dim), dtype=np.float32)

        if embedding_dim > 0:
            for row_index, frame_fingerprint in enumerate(frame_fingerprints):
                result = records[frame_fingerprint]
                if result.face_detected and result.embedding is not None:
                    embeddings[row_index] = result.embedding.astype(np.float32)

        temp_path = cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(
            temp_path,
            manifest_json=np.array(json.dumps(manifest, sort_keys=True)),
            frame_fingerprints=frame_fingerprint_values,
            face_detected=face_detected,
            embeddings=embeddings,
        )
        os.replace(temp_path, cache_path)

    @contextmanager
    def _file_lock(self, cache_path: Path):
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        lock_fd = None
        try:
            while True:
                try:
                    lock_fd = os.open(
                        lock_path,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    )
                    break
                except FileExistsError:
                    if self._is_stale_lock(lock_path):
                        logger.warning(f"Removing stale embedding cache lock {lock_path}")
                        try:
                            os.unlink(lock_path)
                        except FileNotFoundError:
                            pass
                        continue
                    time.sleep(0.05)
            yield
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass

    def _is_stale_lock(self, lock_path: Path) -> bool:
        try:
            return (time.time() - lock_path.stat().st_mtime) > self.LOCK_STALE_SECONDS
        except FileNotFoundError:
            return False
